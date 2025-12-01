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
    from langchain.schema import HumanMessage, SystemMessage, AIMessage
except ImportError:
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

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
        persona: str = "Helpful Assistant",  # Added persona
        llm: Optional[AnyOpenAILLM] = None,
        system_prompt: PromptTemplate =  debate_meta_reflection_prompt
    ) -> None:
        self.question = question
        self.scratchpad = scratchpad
        self.llm = llm or AnyOpenAILLM(
            temperature=0.2,
            max_tokens=256,
            model_name="gpt-3.5-turbo",
            model_kwargs={"stop": "\n"}, 
        )
        self.debate_id = debate_id
        self.persona = persona

        # Inject persona into the system prompt
        # We append the persona instruction to the existing system prompt content
        base_content = system_prompt.format(examples=REFLECTIONS)
        persona_instruction = f"\n\nYou are adopting the persona of: {self.persona}. Act accordingly in your reasoning and responses."
        
        safety_instruction = (
            "\n\nCRITICAL INSTRUCTION: Do not overthink or hallucinate. "
            "If a previous answer was factually correct but marked wrong, assume it is a formatting issue "
            "(e.g., needs full definition instead of acronym) rather than a factual error. "
            "Do not invent new facts to satisfy the prompt. Stick strictly to the evidence."
        )

        self.system_prompt = SystemMessage(content=base_content + persona_instruction + safety_instruction)

    def initial_response(self, initial_response_prompt: PromptTemplate, prompt_kwargs) -> str:
        prompt = initial_response_prompt.format(**prompt_kwargs)
        initial_message = HumanMessage(content = prompt)
        
        response = self.llm.query([self.system_prompt, initial_message])

        return response

    def debate_response(self, debator_response: str, debate_history: str) -> str:
        debate_history_msgs = self._format_debate_history(debate_history)
        prompt = debator_response_prompt.format(debator_responses=debate_history)

        response_question = HumanMessage(content = prompt)
        response = self.llm.query([self.system_prompt, *debate_history_msgs, response_question])

        return response
    
    def _format_debate_history(self, debate_history) -> List:
        pattern = re.compile(r"Debator (\d+):\s*(.*?)\n(?=Debator \d+:|$)", re.DOTALL)
        matches = pattern.findall(debate_history)

        formatted_message = []
        for debator_id, text in matches:
            if int(debator_id) == self.debate_id:
               formatted_message.append(AIMessage(content=text)) 
            else:
               formatted_message.append(HumanMessage(content= f"Debator {debator_id}" + text)) 
        
        return formatted_message


class JudgeResponse(TypedDict):
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

class DebateCoordinator:
    def __init__(
        self,
        question: str,
        context: str, # Added context argument to match run_debate_experiment.py
        answer_key: str,
        num_agents: int = 2, 
        num_rounds: int = 5, 
        llm_kwargs: Optional[Dict[str, Any]] = None,
        llm: AnyOpenAILLM = None,
        personas: List[str] = None # Added personas list
    ) -> None:
        if num_agents < 1:
            raise ValueError("num_agents must be >= 1 for debate.")
        
        self.question = question
        self.context = context
        self.answer_key = answer_key
        self.max_num_rounds = max(1, num_rounds)
        self.num_agents = num_agents
        self.round_number = 0
        self.debate_history = ""
        self.llm_kwargs = llm_kwargs
        
        # Default personas if none provided
        if not personas:
            self.personas = ["Helpful Assistant"] * num_agents
        else:
            self.personas = personas
            # Ensure we have enough personas for agents by cycling if necessary
            while len(self.personas) < num_agents:
                self.personas.extend(personas)
            self.personas = self.personas[:num_agents]
        
        self.model_name = "gpt-3.5-turbo" 

        self.judge_llm = llm or AnyOpenAILLM(
            temperature=0,
            max_tokens=256,
            model_name=self.model_name,
        )


    def _build_debators(self, num_debators: int, scratchpad:str, llm_kwargs: Dict[str, Any] = None) -> List[DebateLLM]:
        debators: List[DebateLLM] = []
        for indx in range(num_debators):
            llm = AnyOpenAILLM(**llm_kwargs) if llm_kwargs else None
            debators.append(
                DebateLLM(
                    question=self.question,
                    scratchpad=scratchpad,
                    llm=llm,
                    debate_id=indx,
                    persona=self.personas[indx] # Pass the specific persona
                )
            )
        return debators

    def run(self) -> Dict[str, Any]: # Changed return type to match usage
        # Using context as scratchpad for now based on previous usage
        scratchpad = self.context 
        rounds: List[List] = []

        debators = self._build_debators(self.num_agents, scratchpad, self.llm_kwargs) 
        
        # Debators first have to propose their ideas
        first_round = []
        for idx, debator in enumerate(debators):
            if idx == 0:
                kwargs = {"question": debator.question, "scratchpad": debator.scratchpad}
                response = debator.initial_response(initial_response_prompt = debate_affirmative_reflection_prompt, prompt_kwargs=kwargs)
                first_round.append(response)
            else:
                kwargs = {"question": debator.question, "scratchpad": debator.scratchpad, "debator_response": first_round[0]} # Fixed to pass string not list
                response = debator.initial_response(initial_response_prompt = debate_negative_reflection_prompt, prompt_kwargs=kwargs)
                first_round.append(response)

        self._update_debate_history(first_round)
        rounds.append(first_round)
                
        debate_finished = False
        num_debate_rounds = 0
        prev_response = first_round[-1] 

        final_answer = ""
        is_correct = False

        while (not debate_finished and num_debate_rounds < self.max_num_rounds):
            curr_round = []
    
            for indx, debator in enumerate(debators):
                response = debator.debate_response(debator_response = prev_response, debate_history = self.debate_history)
                curr_round.append(response) 
                prev_response = response

            rounds.append(curr_round)
            self._update_debate_history(curr_round)           
            
            system_prompt = SystemMessage(content=judge_meta_reflection_prompt.format())
            
            # Handle case where we might have more than 2 agents, but prompt expects 2
            # For now, just taking first two for the judge prompt or last two
            aff_resp = curr_round[0] if len(curr_round) > 0 else ""
            neg_resp = curr_round[1] if len(curr_round) > 1 else curr_round[0]

            judgement_question = HumanMessage(content=judge_end_of_round_reflection_prompt.format(
                                                            affirmative_response = aff_resp, 
                                                            negative_response = neg_resp,
                                                            round_num = len(rounds)))

            try:
                verdict_str = self.judge_llm.query([system_prompt, judgement_question])
                # Clean up potential markdown code blocks in JSON response
                verdict_str = verdict_str.replace("```json", "").replace("```", "").strip()
                verdict = json.loads(verdict_str)
                print(verdict)
                debate_finished = verdict.get("preference_found", False) or verdict.get("Whether there is a preference", "No") == "Yes"
            except Exception as e:
                print(f"Error parsing judge output: {e}")
                debate_finished = False
                verdict = {}

            num_debate_rounds += 1

        # Determine final answer
        if not debate_finished:
           final_answer = prev_response
        else:
            final_answer = verdict.get("summary_of_winning_position", prev_response)
            if not final_answer:
                final_answer = verdict.get("debate_answer", prev_response)

        print("--"*20 + "Full debate log" + "--"*20)
        pp.pprint(rounds)

        # Simple check for correctness (exact match or substring)
        # In a real scenario, you might want a more robust check
        normalized_final = extract_answer(final_answer)
        is_correct = self.answer_key.lower() in normalized_final.lower() or normalized_final.lower() in self.answer_key.lower()

        return {
            "final_answer": final_answer,
            "normalized_final_answer": normalized_final,
            "rounds": rounds,
            "is_correct": is_correct,
            "majority_votes": 0 # Placeholder as we aren't doing strict voting yet
        }
    
    def _update_debate_history(self, new_round):
        self.debate_history += "--"*5 + f"Start of round {self.round_number}" + "--"*5 + "\n"

        for indx, response in enumerate(new_round):
            self.debate_history += f"Debator {indx}: " + response + "\n"
            
        self.round_number += 1