import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from llm import AnyOpenAILLM

try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    from langchain.prompts import PromptTemplate

try:
    from langchain.schema import HumanMessage, SystemMessage, AIMessage
except ImportError:
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


# --- Helper: extract python code blocks from responses ---

def extract_code_block(completion: str) -> str:
    """
    Extract the first ```python ... ``` block from a string.
    Falls back to returning the whole string if no block found.
    """
    if not isinstance(completion, str):
        return ""

    pattern = re.compile(r"```python\s*(.*?)```", re.DOTALL | re.IGNORECASE)
    match = pattern.search(completion)
    if match:
        return match.group(1).strip()

    # Fallback: sometimes the model just outputs code without fences
    return completion.strip()


# --- Debater LLM wrapper ---

@dataclass
class DebateLLM:
    llm: AnyOpenAILLM
    persona: str
    debate_id: int
    question: str
    context: str

    def _system_message(self) -> SystemMessage:
        return SystemMessage(
            content=(
                "You are participating in a multi-agent code review and fixing debate.\n"
                f"Persona: {self.persona}\n\n"
                "You will be given:\n"
                "  - The original problem description (with function signature / docstring).\n"
                "  - The current failed implementation.\n"
                "  - The error trace from running unit tests.\n\n"
                "Your job is to ANALYZE THE BUG and PROPOSE A FIX.\n"
                "CRITICAL RULES:\n"
                "  1. Read the original problem carefully and respect the specification.\n"
                "  2. Assume tests are correct; treat the error trace as truth.\n"
                "  3. At the end of your answer, you MUST output a full corrected function\n"
                "     implementation in this exact format:\n\n"
                "```python\n"
                "# your full corrected function here\n"
                "def ...:\n"
                "    ...\n"
                "```\n"
            )
        )

    def initial_proposal(self) -> str:
        """
        First round: respond directly to question + context.
        """
        prompt = (
            f"Task:\n{self.question}\n\n"
            "Context (problem + failed code + error):\n"
            f"{self.context}\n\n"
            "Explain what is wrong and then provide the corrected function.\n"
        )

        messages = [self._system_message(), HumanMessage(content=prompt)]
        return self.llm.query(messages)

    def respond_to_debate(self, debate_history: str) -> str:
        """
        Subsequent rounds: see debate history and propose an improved fix.
        """
        prompt = (
            f"Task:\n{self.question}\n\n"
            "You are in a debate with other engineers. Here is the debate so far:\n"
            f"{debate_history}\n\n"
            "Your goal in this round:\n"
            "  - Critique previous proposals.\n"
            "  - Fix any logical errors or spec violations you see.\n"
            "  - If a previous proposal is mostly correct, you may refine it.\n"
            "  - At the end, output your best corrected function.\n\n"
            "Remember to end with a full function in a ```python``` code block."
        )

        messages = [self._system_message(), HumanMessage(content=prompt)]
        return self.llm.query(messages)


# --- Debate Coordinator ---

class DebateCoordinator:
    """
    Multi-agent programming debate + judge that produces a final code snippet.

    Returns from run():
      {
        "summary": <text summary of bug / fix>,
        "code": <string of python code (function)>,
        "rounds": [[debator_0_round1, debator_1_round1, ...], [...]],
        "is_correct": False  # we don't know until tests are run
      }
    """

    def __init__(
        self,
        question: str,
        context: str,
        answer_key: str,
        num_agents: int = 2,
        num_rounds: int = 2,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        llm: Optional[AnyOpenAILLM] = None,
        personas: Optional[List[str]] = None,
    ) -> None:
        if num_agents < 1:
            raise ValueError("num_agents must be >= 1")

        self.question = question
        self.context = context
        self.answer_key = answer_key
        self.num_agents = num_agents
        self.num_rounds = max(1, num_rounds)

        self.llm_kwargs = llm_kwargs or {"model_name": "gpt-3.5-turbo", "temperature": 0.2}
        self.personas = personas or ["Generic Engineer"] * num_agents
        while len(self.personas) < num_agents:
            self.personas.extend(self.personas)
        self.personas = self.personas[:num_agents]

        # Judge model: lower temperature and slightly shorter outputs
        self.judge_llm = llm or AnyOpenAILLM(
            temperature=0.0,
            max_tokens=512,
            model_name=self.llm_kwargs.get("model_name", "gpt-3.5-turbo"),
        )

    def _build_debaters(self) -> List[DebateLLM]:
        debaters: List[DebateLLM] = []
        for idx in range(self.num_agents):
            deb_llm = AnyOpenAILLM(**self.llm_kwargs)
            debaters.append(
                DebateLLM(
                    llm=deb_llm,
                    persona=self.personas[idx],
                    debate_id=idx,
                    question=self.question,
                    context=self.context,
                )
            )
        return debaters

    def _build_judge_prompt(self, debate_rounds: List[List[str]]) -> str:
        """
        Flatten debate rounds into a readable transcript to feed to the judge.
        """
        lines: List[str] = []
        for r_idx, round_responses in enumerate(debate_rounds, start=1):
            lines.append(f"=== Round {r_idx} ===")
            for d_idx, resp in enumerate(round_responses):
                lines.append(f"Debater {d_idx} ({self.personas[d_idx]}):\n{resp}\n")
        transcript = "\n".join(lines)

        return (
            "You are the judge of a multi-agent programming debate.\n"
            "Each debater proposes a corrected implementation of the SAME Python function.\n"
            "Your goals:\n"
            "  1. Identify which debater's final code is MOST likely to be correct\n"
            "     w.r.t. the original problem specification.\n"
            "  2. Summarize the core bug and the key fix in plain language.\n"
            "  3. Output a JSON object with the following fields ONLY:\n"
            '       {\n'
            '         "summary": "<1-3 sentence explanation of the bug & fix>",\n'
            '         "code": "<the full, best corrected function as plain text>"\n'
            "       }\n\n"
            "IMPORTANT:\n"
            "  - The `code` field MUST contain a standalone Python function definition.\n"
            "  - Do NOT include backticks or markdown in the JSON.\n"
            "  - Do NOT include any other keys besides `summary` and `code`.\n\n"
            "Here is the full debate transcript:\n\n"
            f"{transcript}\n\n"
            "Now output ONLY the JSON object described above."
        )

    def run(self) -> Dict[str, Any]:
        debaters = self._build_debaters()
        rounds: List[List[str]] = []
        debate_history_str = ""

        # --- Multi-round debate ---
        for r in range(self.num_rounds):
            round_responses: List[str] = []
            for idx, debater in enumerate(debaters):
                if r == 0:
                    resp = debater.initial_proposal()
                else:
                    resp = debater.respond_to_debate(debate_history_str)

                round_responses.append(resp)

            rounds.append(round_responses)

            # Append to running string transcript
            debate_history_str += f"\n=== Round {r+1} ===\n"
            for d_idx, resp in enumerate(round_responses):
                debate_history_str += f"Debater {d_idx} ({self.personas[d_idx]}):\n{resp}\n"

        # --- Judge picks the best code and summarizes ---
        judge_prompt = self._build_judge_prompt(rounds)
        judge_messages = [
            SystemMessage(
                content=(
                    "You are a strict but fair judge of programming debates. "
                    "You will output JSON only, no additional commentary."
                )
            ),
            HumanMessage(content=judge_prompt),
        ]
        judge_raw = self.judge_llm.query(judge_messages).strip()

        # Clean up any ```json fences if the model ignores instructions
        judge_raw = judge_raw.replace("```json", "").replace("```", "").strip()

        summary = ""
        code = ""

        try:
            judge_obj = json.loads(judge_raw)
            summary = str(judge_obj.get("summary", "")).strip()
            code = str(judge_obj.get("code", "")).strip()
        except Exception:
            # Fallback: if JSON parsing fails, try to salvage code & summary heuristically
            # Try to find a code block in the raw text
            code = extract_code_block(judge_raw)
            summary = "Judge output could not be parsed as JSON; using best-effort extracted code."

        # As a final safety: if judge didn't provide code, extract from last debater of last round
        if not code.strip():
            last_round = rounds[-1] if rounds else []
            if last_round:
                code = extract_code_block(last_round[0])

        return {
            "summary": summary,
            "code": code,
            "rounds": rounds,
            "is_correct": False,  # tests decide this later
        }
