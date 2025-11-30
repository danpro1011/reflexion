import re
import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TypedDict
import pprint as pp

from llm import AnyOpenAILLM
from prompts import debate_meta_reflection_prompt, debator_response_prompt, debator_initial_prompt, consensus_reached_prompt, determine_consensus_prompt
from fewshots import REFLECTIONS
from environment import normalize_answer, EM

try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    from langchain.prompts import PromptTemplate
try:
    from langchain.schema import HumanMessage, SystemMessage, AIMessage, ChatMessage
except ImportError:
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ChatMessage

try:
    from langchain_openai import ChatOpenAI, OpenAI
except ImportError:
    from langchain.chat_models import ChatOpenAI
    from langchain.llms import OpenAI

from dotenv import load_dotenv
load_dotenv()


def extract_json_from_text(text: str) -> Optional[Dict]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    matches = re.findall(json_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    json_obj_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_obj_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    brace_count = 0
    start_idx = -1
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                try:
                    return json.loads(text[start_idx:i+1])
                except json.JSONDecodeError:
                    start_idx = -1

    return None


FINISH_PATTERN = re.compile(r"Finish\\[(.*?)\\]", re.IGNORECASE)
ANSWER_PATTERN = re.compile(r"Answer:\s*(.+)", re.IGNORECASE)


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


def extract_qa_answer(completion: str) -> str:
    """
    Extract answer from QA debate format: "Thought: ... Answer: ..."
    Falls back to extracting from Thought if Answer is not found.
    """
    # Try to find "Answer:" in the response
    for line in reversed(completion.splitlines()):
        match = ANSWER_PATTERN.search(line)
        if match:
            return match.group(1).strip()
    match = ANSWER_PATTERN.search(completion)
    if match:
        return match.group(1).strip()

    # Fallback: extract from "Thought:" section
    thought_pattern = re.compile(r"Thought:\s*(.+)", re.IGNORECASE | re.DOTALL)
    thought_match = thought_pattern.search(completion)
    if thought_match:
        thought_text = thought_match.group(1).strip()

        # Extract the last sentence, which typically contains the answer
        sentences = [s.strip() for s in thought_text.split('.') if s.strip()]
        if sentences:
            last_sentence = sentences[-1]
            # Strip common reasoning prefixes
            for prefix in ["Therefore,", "Thus,", "So,", "Hence,"]:
                if last_sentence.startswith(prefix):
                    last_sentence = last_sentence[len(prefix):].strip()
            return last_sentence

    return completion.strip()

class DebateLLM:
    def __init__(
        self,
        question: str,
        scratchpad: str,
        debate_id: int,
        llm: Optional[AnyOpenAILLM] = None,
        system_prompt: PromptTemplate =  debate_meta_reflection_prompt,
    ) -> None:
        self.question = question
        self.scratchpad = scratchpad
        self.llm = llm or AnyOpenAILLM(
            temperature=.25,
            max_tokens=256,
            model_name="gpt-3.5-turbo",
        )
        self.debate_id = debate_id

        # System prompt stays constant across debate rounds
        self.system_prompt = SystemMessage(content=system_prompt.format(examples=REFLECTIONS))

    def initial_response(self) -> str:
        prompt = debator_initial_prompt.format(question=self.question, scratchpad = self.scratchpad)
        initial_message = HumanMessage(content = prompt)
        
        response = self.llm.query([self.system_prompt, initial_message])

        return response

    def debate_response(self, debate_history) -> str:
        question_context = f"Question: {self.question}\n\nContext: {self.scratchpad}\n\nThese are the answers that other agents provided:"
        question_context = HumanMessage(content = question_context)

        formatted_history = self._format_debate_history(debate_history)

        response_question = HumanMessage(content = "Using the opinions of other agents as additional advice, can you give an updated response?\nThink carefully and provide your updated reasoning and answer.\nFormat: Thought: <your reasoning>\nAnswer: <your answer>")

        response = self.llm.query([self.system_prompt, question_context, *formatted_history, response_question])

        return response
    
    def _format_debate_history(self, debate_history) -> List:
        """
        Format debate history for LangChain: marks own responses as AIMessage
        and other agents' responses as ChatMessage.
        """
        pattern = re.compile(
            r"Debator\s+(\d+):\s*(.*?)\n(?=Debator\s+\d+:|$)",
            re.DOTALL
        )

        matches = pattern.findall(debate_history)
        formatted_messages = []

        for debator_id, content in matches:
            debator_id = int(debator_id)

            if debator_id == self.debate_id:
                formatted_messages.append(
                    AIMessage(content=content.strip())
                )
            else:
                # Use ChatMessage for other agents' responses
                formatted_messages.append(
                    ChatMessage(
                        role="assistant",
                        name=f"debator_{debator_id}",
                        content=content.strip()
                    )
                )

        return formatted_messages

class DebateCoordinator:
    def __init__(
        self,
        question: str,
        answer_key: str,
        num_debators: int = 3,
        max_num_rounds: int = 5,
        llm: AnyOpenAILLM = None,
        llm_kwargs: Dict = None
    ) -> None:
        self.question = question
        self.answer_key = answer_key
        self.num_debators = num_debators
        self.max_num_rounds = max(1, max_num_rounds)
        self.round_number = 0
        self.debate_history = ""

        # Configure LLM with provided kwargs or use defaults
        if llm_kwargs:
            self.model_name = llm_kwargs.get("model_name", "gpt-3.5-turbo")
            self.llm = llm or AnyOpenAILLM(**llm_kwargs)
        else:
            self.model_name = "gpt-3.5-turbo"
            self.llm = llm or AnyOpenAILLM(
                temperature=0,
                max_tokens=256,
                model_name=self.model_name,
            )


    def _build_debators(self, num_debators: int, scratchpad:str, llm= None) -> List[DebateLLM]:
        debators: List[DebateLLM] = []
        for indx in range(num_debators):
            # Use varying temperatures to encourage diverse perspectives
            llm = AnyOpenAILLM(
                temperature=.33*(indx),
                max_tokens=256,
                model_name="gpt-3.5-turbo",
            )
            debators.append(
                DebateLLM(
                    question=self.question,
                    scratchpad=scratchpad,
                    llm=llm,
                    debate_id=indx
                )
            )
        return debators

    def run(self, scratchpad) -> Dict[str, Any]:
        """
        Run the multi-agent debate process.

        Returns:
            Dict containing final_answer, normalized_final_answer, is_correct,
            rounds, and majority_votes.
        """
        rounds: List[List] = []

        debators = self._build_debators(self.num_debators, scratchpad)

        num_debate_rounds = 0
        curr_round = []
        consensus = ""
        consensus_reached = False

        while num_debate_rounds < self.max_num_rounds:
            if num_debate_rounds == 0:
                for debator in debators:
                    response = debator.initial_response()
                    curr_round.append(response)

            else:
                for debator in debators:
                    response = debator.debate_response(self.debate_history)
                    curr_round.append(response)

            self._update_debate_history(curr_round)
            num_debate_rounds += 1
            rounds.append(curr_round)
            curr_round = []

            consensus_reached, debator_id = self._find_consensus()

            consensus = self._extract_debator_response(debator_id)

            if consensus_reached:
                break

        # If max rounds reached without consensus, have judge select best answer
        if not consensus_reached:
            print("No consensus reached")
            prompt = HumanMessage(content = determine_consensus_prompt.format(debate_log = self.debate_history))
            consensus = self.llm.query([prompt])

        print("=" * 50)
        print("Full Debate Log")
        print("=" * 50)
        pp.pprint(rounds)

        # Extract and evaluate the final answer
        final_answer = extract_qa_answer(consensus)
        normalized_final = normalize_answer(final_answer)
        normalized_key = normalize_answer(self.answer_key)

        # Check correctness using substring matching (suitable for QA tasks)
        is_correct = normalized_key in normalized_final or EM(normalized_final, normalized_key)

        # Determine majority vote from final round
        majority_votes = final_answer
        if rounds:
            last_round_answers = [extract_qa_answer(resp) for resp in rounds[-1]]
            if last_round_answers:
                majority_votes = max(set(last_round_answers), key=last_round_answers.count)

        return {
            "final_answer": final_answer,
            "normalized_final_answer": normalized_final,
            "is_correct": is_correct,
            "rounds": rounds,
            "majority_votes": majority_votes,
        }
    

    def _find_consensus(self) -> tuple[bool, str]:
        """
        Check if agents have reached consensus on an answer.
        """
        verdict = self.llm(consensus_reached_prompt.format(debate_log=self.debate_history))
        # Normalize Python booleans to JSON format
        safe = verdict.replace("True", "true").replace("False", "false").replace("None", "null")

        try:
            verdict = json.loads(safe)
        except json.JSONDecodeError:
            print(f"Warning: Unable to parse consensus verdict: {safe}")
            verdict = {"consensus_reached": False, "debator_id": -1}

        return verdict["consensus_reached"], verdict.get("debator_id",-1)
    
    def _extract_reflection(self, text) -> str:
        """Extract content from Reflection[...] format."""
        match = re.search(r"Reflection\[(.*?)\]", text, re.DOTALL)
        if not match:
            print(f"Warning: Could not extract reflection from: {text}")
            return ""
        return match.group(1).strip()
        
    def _extract_debator_response(self, target_debator_id) -> str:
        pattern = re.compile(r"Debator (\d+):\s*(.*?)\n(?=Debator \d+:|$)", re.DOTALL)
        matches = pattern.findall(self.debate_history)

        for debator_id, text in matches:
            if int(debator_id) == target_debator_id:
                return text

        print(f"Warning: Debator {target_debator_id} not found in history")
        return ""
    
    def _update_debate_history(self, new_round):
        self.debate_history += f"\n{'='*20} Round {self.round_number} {'='*20}\n"
        for indx, response in enumerate(new_round):
            self.debate_history += f"Debator {indx}: {response}\n"
        self.round_number += 1