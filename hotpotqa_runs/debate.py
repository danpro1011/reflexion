import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel

from llm import AnyOpenAILLM
from prompts import debate_meta_reflection_prompt, debator_response_prompt, debate_affirmative_reflection_prompt, debate_negative_reflection_prompt, judge_meta_reflection_prompt, judge_end_of_round_reflection_prompt
from fewshots import REFLECTIONS

try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    from langchain.prompts import PromptTemplate
try:
    from langchain.schema import HumanMessage, SystemMessage
except ImportError:
    from langchain_core.messages import HumanMessage, SystemMessage


FINISH_PATTERN = re.compile(r"Finish\\[(.*?)\\]", re.IGNORECASE)


def extract_answer(completion: str) -> str:
    """
    Best-effort extraction of the argument passed to Finish[...].
    Falls back to the raw completion if no Finish token is found.
    """
    for line in reversed(completion.splitlines()):
        match = FINISH_PATTERN.search(line)
        if match:
            return match.group(1).strip()
    match = FINISH_PATTERN.search(completion)
    if match:
        return match.group(1).strip()
    return completion.strip()

class DebateLLM:
    def __init__(
        self,
        question: str,
        scratchpad: str,
        llm: Optional[AnyOpenAILLM] = None,
        system_prompt: PromptTemplate =  debate_meta_reflection_prompt.format(examples = REFLECTIONS) #Should I be calling .format()?
    ) -> None:
        self.question = question
        self.scratchpad = scratchpad
        self.llm = llm or AnyOpenAILLM(
            temperature=0.2,
            max_tokens=256,
            model_name="gpt-3.5-turbo",
            model_kwargs={"stop": "\n"}, #Why???
        )

        # 'System prompt' so that LLM actually debates when prompted. Response doesn't matter
        self.llm([SystemMessage(content=system_prompt)])

    def initial_response(self, initial_response_prompt: PromptTemplate, prompt_kwargs) -> str:
        #NOTE: kwags doing a lot of heavy lifting here, maybe don't want it designed in exactly this manner
        prompt = initial_response_prompt.format(kwargs=prompt_kwargs)
        
        completion = self.llm(prompt)
        return self._build_response(completion, round_idx=1)

    # The prompt for this should pretty much always be the same, so I don't see the need to pass it in as an argument
    def debate_response(self, peer_responses: str) -> str:
        #NOTE: This is just written for two agents, things gotta change to run with multiple agents
        prompt = debator_response_prompt.format(
                opponent_response = peer_responses
        )
        completion = self.llm(prompt)
        return self._build_response(completion)

    #TODO: Look again at what information we actually need
    # def _build_response(self, completion: str, round_idx: int) -> DebateResponse:
    #     answer = extract_answer(completion)
    #     return DebateResponse(
    #         agent_id=self.agent_id,
    #         round=round_idx,
    #         raw_response=completion.strip(),
    #         answer=answer,
    #         normalized_answer=normalize_answer(answer),
    #     )


@dataclass
class JudgeResponse(BaseModel):
    #{\"Whether there is a preference\": \"Yes or No\", \"Supported Side\": \"Affirmative or Negative\", \"Reason\": \"\", \"debate_answer\": \"\"}. Please strictly output in JSON format, do not output irrelevant content.
    # I'm really struggling to see the difference between 'supported side' and 'debate answer' here
    debate_finished: bool
    reason: str
    debate_answer: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "round": self.round,
            "raw_response": self.raw_response,
            "answer": self.answer,
            "normalized_answer": self.normalized_answer,
        }

#TODO: This class is written around the assumption that we're just using the openai llm, but it really should work for any LLM type
class DebateCoordinator:
    def __init__(
        self,
        question: str,
        answer_key: str,
        # scratchpad: str,
        num_debators: int = 2,
        max_num_rounds: int = 10, #This is the hard max, not recommended or average number of runs
        llm_kwargs: Optional[Dict[str, Any]] = None,
        llm: AnyOpenAILLM = None
    ) -> None:
        if num_debators < 1:
            raise ValueError("num_debators must be >= 1 for debate.")
        
        self.question = question
        self.answer_key = answer_key
        self.max_num_rounds = max(1, max_num_rounds)
        self.debators = []

        # Enforcing that all judge outputs follow the desired format
        #I think don't use this llm wrapper just for here
        self.llm = llm or AnyOpenAILLM(
            temperature=0,
            max_tokens=256,
            model_name="gpt-3.5-turbo",
        ).with_structured_output(JudgeResponse)

        
        #Setup the system prompt for the judge
        #NOTE: Way this is currently setup, judge isn't able to view the reasoning traces, which is probbaly not optimal
        self.llm.invoke([SystemMessage(judge_meta_reflection_prompt.format())])
            

    def _build_debators(self, num_debators: int, scratchpad:str, llm_kwargs: Dict[str, Any]) -> List[DebateLLM]:
        debators: List[DebateLLM] = []
        #TODO: question scratchpad can either be added here or in inital repsonse 
        for idx in range(num_debators):
            llm = AnyOpenAILLM(**llm_kwargs) if llm_kwargs else None
            debators.append(
                DebateLLM(
                    question=self.question,
                    scratchpad=scratchpad,
                    llm=llm,
                )
            )
        return debators

    #TODO: Add some type of logging for this
    def run(self, scratchpad) -> str:
        #Context has to be passed into this and debators have to be build just for that instance, 
        # just based off of how its designed right now
        rounds: List[List] = []

        self.debators = self._build_debators(2,scratchpad) 

        # Debators first have to propose their ideas
        first_round = []
        for idx, debator in enumerate(self.debators):
            if idx == 0:
                #First agent is the only one that doesn't need to respond to the others
                kwargs = {"question": debator.question, "scratchpad": debator.scratchpad}
                response = debator.inital_response(initial_response_template = debate_affirmative_reflection_prompt, kwags=kwargs)
                first_round.append(response)
            else:
                #NOTE: Better formatting when adding actual multiple debators
                kwargs = {"question": debator.question, "scratchpad": debator.scratchpad, "opponent_response": first_round}
                debator.inital_response(initial_response_template = debate_negative_reflection_prompt, kwags=kwargs)
                
        # Then they get to argue against each other
        # Way its described in paper, rounds continue until the judge finds the current debate satisfactory
        debate_finished = False
        num_debate_rounds = 0
        prev_response = first_round[-1] 
        while (not debate_finished or num_debate_rounds > self.max_num_rounds):
            curr_round = []

            #Prev responses *should still be within the context window so just need to pass in the most recent one
            for indx, debator in enumerate(self.debators):
                response = debator.debate_response(opponent_response = prev_response)
                curr_round.append(response) 
                prev_response = response

            verdict = self.llm.invoke(judge_end_of_round_reflection_prompt.format(
                                                            affirmative_response = curr_round[0], 
                                                            negative_response = curr_round[1]))
            debate_finished = verdict.debate_finished

            num_debate_rounds += 1

        final_reflection = ""
        #If no consensus has been reached, just take the most recent argument
        if not debate_finished:
           final_reflection = prev_response
        else:
            final_reflection = verdict.debate_answer

        return final_reflection

       
    # def _format_peer_responses(self, rounds: List[List[DebateResponse]], exclude_agent: int) -> str:
    #     snippets: List[str] = []
    #     for round_turns in rounds:
    #         for resp in round_turns:
    #             if resp.agent_id == exclude_agent:
    #                 continue
    #             snippets.append(f"Round {resp.round} Agent {resp.agent_id}: {resp.raw_response}")
    #     return "\n".join(snippets) if snippets else "No peer responses yet."

    # @staticmethod
    # def _majority_vote(round_responses: List[DebateResponse]) -> Tuple[str, str, int]:
    #     counts: Dict[str, Dict[str, Any]] = {}
    #     for resp in round_responses:
    #         norm = resp.normalized_answer
    #         if norm not in counts:
    #             counts[norm] = {"count": 0, "answer": resp.answer}
    #         counts[norm]["count"] += 1

    #     best_norm, stats = max(counts.items(), key=lambda kv: (kv[1]["count"], kv[0]))
    #     return stats["answer"], best_norm, stats["count"]
