import re
import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict
import pprint as pp

from llm import AnyOpenAILLM
# We reuse the prompt templates, but we will override the system instructions heavily
from prompts import (
    debate_meta_reflection_prompt, 
    debator_response_prompt, 
    debate_affirmative_reflection_prompt, 
    debate_negative_reflection_prompt, 
    judge_meta_reflection_prompt, 
    judge_end_of_round_reflection_prompt
)
from fewshots import REFLECTIONS

try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    from langchain.prompts import PromptTemplate
try:
    from langchain.schema import HumanMessage, SystemMessage, AIMessage
except ImportError:
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from dotenv import load_dotenv
load_dotenv()

def extract_code(completion: str) -> str:
    """
    Helper to extract python code blocks from debate responses.
    """
    pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
    match = pattern.search(completion)
    if match:
        return match.group(1).strip()
    return completion.strip()

class DebateLLM:
    def __init__(
        self,
        question: str,
        scratchpad: str,
        debate_id: int,
        persona: str = "Senior Engineer",
        llm: Optional[AnyOpenAILLM] = None,
        system_prompt: PromptTemplate = debate_meta_reflection_prompt
    ) -> None:
        self.question = question
        self.scratchpad = scratchpad # In programming, this is Code + Error Traceback
        self.llm = llm or AnyOpenAILLM(
            temperature=0.2,
            max_tokens=512, # Increased for code generation
            model_name="gpt-3.5-turbo",
            model_kwargs={"stop": "\n"} # Be careful with stop tokens in code
        )
        # Remove stop token for code generation if it interferes
        if self.llm.model_kwargs.get("stop") == "\n":
             del self.llm.model_kwargs["stop"]

        self.debate_id = debate_id
        self.persona = persona

        # --- CUSTOM SYSTEM PROMPT FOR PROGRAMMING ---
        # We override the generic prompt to ensure they focus on CODE, not QA.
        
        base_content = (
            f"You are participating in a technical code review debate.\n"
            f"Your Persona: {self.persona}\n"
            f"Task: Analyze the failed code and the error message to propose a fix.\n"
        )
        
        persona_instruction = (
            f"\nAct strictly according to your persona ({self.persona}). "
            "Critique the code's logic, syntax, and edge case handling. "
            "When proposing a fix, you may provide the corrected code block or pseudocode."
        )
        
        safety_instruction = (
            "\n\nCRITICAL INSTRUCTION: "
            "1. Trust the error message provided in the context. "
            "2. Do not assume the test case is wrong unless proven otherwise. "
            "3. Focus on the specific function implementation. "
            "4. Be concise and technical. Do not be polite, be efficient."
        )

        self.system_prompt = SystemMessage(content=base_content + persona_instruction + safety_instruction)

    def initial_response(self, initial_response_prompt: PromptTemplate, prompt_kwargs) -> str:
        # prompt_kwargs usually contains 'question' and 'scratchpad'
        prompt = initial_response_prompt.format(**prompt_kwargs)
        initial_message = HumanMessage(content=prompt)
        response = self.llm.query([self.system_prompt, initial_message])
        return response

    def debate_response(self, debator_response: str, debate_history: str) -> str:
        debate_history_msgs = self._format_debate_history(debate_history)
        prompt = debator_response_prompt.format(debator_responses=debate_history)
        response_question = HumanMessage(content=prompt)
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
               formatted_message.append(HumanMessage(content= f"Debator {debator_id}: " + text)) 
        return formatted_message

class DebateCoordinator:
    def __init__(
        self,
        question: str, # Function signature / prompt
        context: str,  # Failed Code + Error Message
        answer_key: str, # "Passing Tests"
        num_agents: int = 2, 
        num_rounds: int = 3, 
        llm_kwargs: Optional[Dict[str, Any]] = None,
        llm: AnyOpenAILLM = None,
        personas: List[str] = None
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
        
        # Default Coding Personas
        if not personas:
            self.personas = ["Senior Engineer", "QA Engineer"] * num_agents
        else:
            self.personas = personas
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
                    persona=self.personas[indx]
                )
            )
        return debators

    def run(self) -> Dict[str, Any]:
        # In programming, 'scratchpad' passed to agents is the Context (Code + Error)
        scratchpad = self.context 
        rounds: List[List] = []

        debators = self._build_debators(self.num_agents, scratchpad, self.llm_kwargs) 
        
        # --- Round 1: Initial Proposals ---
        first_round = []
        for idx, debator in enumerate(debators):
            if idx == 0:
                kwargs = {"question": debator.question, "scratchpad": debator.scratchpad}
                response = debator.initial_response(initial_response_prompt=debate_affirmative_reflection_prompt, prompt_kwargs=kwargs)
            else:
                kwargs = {"question": debator.question, "scratchpad": debator.scratchpad, "debator_response": first_round[0] if first_round else ""}
                response = debator.initial_response(initial_response_prompt=debate_negative_reflection_prompt, prompt_kwargs=kwargs)
            first_round.append(response)

        self._update_debate_history(first_round)
        rounds.append(first_round)
                
        debate_finished = False
        num_debate_rounds = 0
        prev_response = first_round[-1] 
        final_answer = ""

        # --- Debate Loop ---
        while (not debate_finished and num_debate_rounds < self.max_num_rounds):
            curr_round = []
            for indx, debator in enumerate(debators):
                response = debator.debate_response(debator_response=prev_response, debate_history=self.debate_history)
                curr_round.append(response) 
                prev_response = response

            rounds.append(curr_round)
            self._update_debate_history(curr_round)           
            
            # --- Judge Step ---
            system_prompt = SystemMessage(content=judge_meta_reflection_prompt.format())
            
            aff_resp = curr_round[0] if len(curr_round) > 0 else ""
            neg_resp = curr_round[1] if len(curr_round) > 1 else curr_round[0]

            judgement_question = HumanMessage(content=judge_end_of_round_reflection_prompt.format(
                                                            affirmative_response=aff_resp, 
                                                            negative_response=neg_resp,
                                                            round_num=len(rounds)))

            try:
                verdict_str = self.judge_llm.query([system_prompt, judgement_question])
                verdict_str = verdict_str.replace("```json", "").replace("```", "").strip()
                verdict = json.loads(verdict_str)
                debate_finished = verdict.get("preference_found", False) or verdict.get("Whether there is a preference", "No") == "Yes"
            except Exception as e:
                print(f"Error parsing judge output: {e}")
                debate_finished = False
                verdict = {}

            num_debate_rounds += 1

        # --- Final Extraction ---
        if not debate_finished:
           final_answer = prev_response
        else:
            final_answer = verdict.get("summary_of_winning_position", prev_response)
            if not final_answer:
                final_answer = verdict.get("debate_answer", prev_response)

        print("--"*20 + " Full Programming Debate Log " + "--"*20)
        pp.pprint(rounds)

        return {
            "final_answer": final_answer,
            "rounds": rounds,
            "is_correct": False # Unknown until execution
        }
    
    def _update_debate_history(self, new_round):
        self.debate_history += "--"*5 + f"Start of round {self.round_number}" + "--"*5 + "\n"
        for indx, response in enumerate(new_round):
            self.debate_history += f"Debator {indx} ({self.personas[indx]}): " + response + "\n"
        self.round_number += 1