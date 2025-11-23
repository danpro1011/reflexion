import re
import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TypedDict
import pprint as pp

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

try:
    from langchain_openai import ChatOpenAI, OpenAI
except ImportError:
    from langchain.chat_models import ChatOpenAI
    from langchain.llms import OpenAI

from dotenv import load_dotenv
load_dotenv()


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
        debate_id: int,
        llm: Optional[AnyOpenAILLM] = None,
        system_prompt: PromptTemplate =  debate_meta_reflection_prompt
    ) -> None:
        self.question = question
        self.scratchpad = scratchpad
        self.llm = llm or AnyOpenAILLM(
            temperature=0.2,
            max_tokens=256,
            model_name="gpt-3.5-turbo",
            model_kwargs={"stop": "\n"}, #Makes it stop after a new line, to limit long responses
        )
        self.debate_id = debate_id

        #This stays constant so convienient to just save it here 
        self.system_prompt = SystemMessage(content=system_prompt.format(examples=REFLECTIONS))

    def initial_response(self, initial_response_prompt: PromptTemplate, prompt_kwargs) -> str:
        #NOTE: kwags doing a lot of heavy lifting here, maybe don't want it designed in exactly this manner.
        #Reason I'm doing it like this is because number of kwargs changes based on whether is the affirmative or disagreeing debator debating
        prompt = initial_response_prompt.format(**prompt_kwargs)
        initial_message = HumanMessage(content = prompt)
        
        response = self.llm.query([self.system_prompt, initial_message])

        return response

    # The prompt for this should pretty much always be the same, so I don't see the need to pass it in as an argument
    def debate_response(self, debator_response: str, debate_history: str) -> str:
        #NOTE: This is just written for two agents, things gotta change to run with multiple agents
        debate_history = self._format_debate_history(debate_history)
        prompt = debator_response_prompt.format(
                debator_response = debator_response,
                debate_log = debate_history
        )

        debate_history = HumanMessage(content = prompt)
        response = self.llm.query([self.system_prompt, debate_history])

        return response
    
    def _format_debate_history(self, debate_history):
        pattern = rf"(Debator {self.debate_id}:\s*)(.*?)(?=\nDebator \d+:|\Z)"
        
        def repl(m):
            response = m.group(2)    # original response text
            return "Your response: " + response.strip()
        
        return re.sub(pattern, repl, debate_history, flags=re.DOTALL)


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


class JudgeResponse(TypedDict):
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
        self.round_number = 0
        self.debate_history = ""

        # Enforcing that all judge outputs follow the desired format
        #I think don't use this llm wrapper just for here
        self.judge_llm = llm or AnyOpenAILLM(
            temperature=0,
            max_tokens=256,
            model_name="gpt-3.5-turbo",
        )
        #NOTE: Do we want the 'stop' thingy here too or not


    def _build_debators(self, num_debators: int, scratchpad:str, llm_kwargs: Dict[str, Any] = None) -> List[DebateLLM]:
        debators: List[DebateLLM] = []
        #TODO: question scratchpad can either be added here or in inital repsonse 
        for indx in range(num_debators):
            llm = AnyOpenAILLM(**llm_kwargs) if llm_kwargs else None
            debators.append(
                DebateLLM(
                    question=self.question,
                    scratchpad=scratchpad,
                    llm=llm,
                    debate_id=indx
                )
            )
        return debators

    #TODO: Add some type of logging for this
    def run(self, scratchpad) -> str:
        #Context has to be passed into this and debators have to be build just for that instance, 
        # just based off of how its designed right now
        rounds: List[List] = []

        debators = self._build_debators(2,scratchpad) 
        
        #TODO: Manage the context of the debator agents as well

        # Debators first have to propose their ideas
        first_round = []
        for idx, debator in enumerate(debators):
            if idx == 0:
                #First agent is the only one that doesn't need to respond to the others
                kwargs = {"question": debator.question, "scratchpad": debator.scratchpad}
    # def initial_response(self, initial_response_prompt: PromptTemplate, prompt_kwargs) -> str:
                response = debator.initial_response(initial_response_prompt = debate_affirmative_reflection_prompt, prompt_kwargs=kwargs)
                first_round.append(response)
            else:
                #NOTE: Better formatting when adding actual multiple debators
                kwargs = {"question": debator.question, "scratchpad": debator.scratchpad, "debator_response": first_round}
                response = debator.initial_response(initial_response_prompt = debate_negative_reflection_prompt, prompt_kwargs=kwargs)
                first_round.append(response)

        self._update_debate_history(first_round)

        rounds.append(first_round)
                
        # Then they get to argue against each other
        # Way its described in paper, rounds continue until the judge finds the current debate satisfactory
        debate_finished = False
        num_debate_rounds = 0
        prev_response = first_round[-1] 
        while (not debate_finished or num_debate_rounds > self.max_num_rounds):
            curr_round = []

    
            for indx, debator in enumerate(debators):
                                # input_variables=["debate_log","debator_response"],
                response = debator.debate_response(debator_response = prev_response, debate_history = self.debate_history)
                curr_round.append(response) 
                prev_response = response

            rounds.append(curr_round)

            #Way I'm currently writing this, the log gets updated *after the round is done, maybe not the best way to do this
            self._update_debate_history(curr_round)           
            
            # Old langchain library requires you to wrap the prompt templates like this before querying the LLM
            system_prompt = SystemMessage(content=judge_meta_reflection_prompt.format())
            judgement_question = HumanMessage(content=judge_end_of_round_reflection_prompt.format(
                                                            affirmative_response = curr_round[0], 
                                                            negative_response = curr_round[1],
                                                            round_num = len(rounds)))

            verdict = self.judge_llm.query([system_prompt, judgement_question])

            verdict = json.loads(verdict)
            
            #So long as LLM doesn't mess up its output, this should parse correctly
            print(verdict)
            debate_finished = verdict["preference_found"]

            num_debate_rounds += 1

        final_reflection = ""
        #If no consensus has been reached, just take the most recent argument
        if not debate_finished:
           final_reflection = prev_response
        else:
            final_reflection = verdict["debate_answer"]

        print("--"*20 + "Full debate log" + "--"*20)
        pp.pprint(rounds)

        return final_reflection
    
    def _update_debate_history(self, new_round):
        #Tracking the round may not be necessary
        self.debate_history += "--"*5 + f"Start of round {self.round_number}" + "--"*5

        for indx, response in enumerate(new_round):
            self.debate_history += f"Debator {indx}: " + response + "\n"
            
        self.round_number += 1

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
