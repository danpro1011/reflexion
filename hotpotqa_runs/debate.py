import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from agents import normalize_answer
from llm import AnyOpenAILLM
from prompts import debate_followup_prompt, debate_initial_prompt


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


@dataclass
class DebateResponse:
    agent_id: int
    round: int
    raw_response: str
    answer: str
    normalized_answer: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "round": self.round,
            "raw_response": self.raw_response,
            "answer": self.answer,
            "normalized_answer": self.normalized_answer,
        }


class DebateAgent:
    def __init__(
        self,
        agent_id: int,
        question: str,
        context: str,
        llm: Optional[AnyOpenAILLM] = None,
    ) -> None:
        self.agent_id = agent_id
        self.question = question
        self.context = context
        self.llm = llm or AnyOpenAILLM(
            temperature=0.2,
            max_tokens=256,
            model_name="gpt-3.5-turbo",
            model_kwargs={"stop": "\n"},
        )

    def initial_response(self) -> DebateResponse:
        prompt = debate_initial_prompt.format(
            agent_id=self.agent_id, context=self.context, question=self.question
        )
        completion = self.llm(prompt)
        return self._build_response(completion, round_idx=1)

    def debate_response(self, peer_responses: str, round_idx: int) -> DebateResponse:
        prompt = debate_followup_prompt.format(
            agent_id=self.agent_id,
            peer_responses=peer_responses or "No peer responses yet.",
            context=self.context,
            question=self.question,
        )
        completion = self.llm(prompt)
        return self._build_response(completion, round_idx=round_idx)

    def _build_response(self, completion: str, round_idx: int) -> DebateResponse:
        answer = extract_answer(completion)
        return DebateResponse(
            agent_id=self.agent_id,
            round=round_idx,
            raw_response=completion.strip(),
            answer=answer,
            normalized_answer=normalize_answer(answer),
        )


class DebateCoordinator:
    def __init__(
        self,
        question: str,
        context: str,
        answer_key: str,
        num_agents: int = 3,
        num_rounds: int = 3,
        llm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if num_agents < 1:
            raise ValueError("num_agents must be >= 1 for debate.")
        self.question = question
        self.context = context
        self.answer_key = answer_key
        self.num_rounds = max(1, num_rounds)
        self.agents = self._build_agents(num_agents, llm_kwargs or {})

    def _build_agents(self, num_agents: int, llm_kwargs: Dict[str, Any]) -> List[DebateAgent]:
        agents: List[DebateAgent] = []
        for idx in range(num_agents):
            llm = AnyOpenAILLM(**llm_kwargs) if llm_kwargs else None
            agents.append(
                DebateAgent(
                    agent_id=idx + 1,
                    question=self.question,
                    context=self.context,
                    llm=llm,
                )
            )
        return agents

    def run(self) -> Dict[str, Any]:
        rounds: List[List[DebateResponse]] = []

        # Round 1: independent proposals
        first_round = [agent.initial_response() for agent in self.agents]
        rounds.append(first_round)

        # Subsequent rounds: debate using peer responses
        for round_idx in range(2, self.num_rounds + 1):
            next_round: List[DebateResponse] = []
            for agent in self.agents:
                peer_responses = self._format_peer_responses(rounds, exclude_agent=agent.agent_id)
                next_round.append(agent.debate_response(peer_responses, round_idx))
            rounds.append(next_round)

        final_round = rounds[-1]
        majority_answer, normalized_answer, vote_count = self._majority_vote(final_round)

        return {
            "rounds": [[resp.to_dict() for resp in round_turn] for round_turn in rounds],
            "final_answer": majority_answer,
            "normalized_final_answer": normalized_answer,
            "majority_votes": vote_count,
            "is_correct": normalize_answer(self.answer_key) == normalized_answer,
        }

    def _format_peer_responses(self, rounds: List[List[DebateResponse]], exclude_agent: int) -> str:
        snippets: List[str] = []
        for round_turns in rounds:
            for resp in round_turns:
                if resp.agent_id == exclude_agent:
                    continue
                snippets.append(f"Round {resp.round} Agent {resp.agent_id}: {resp.raw_response}")
        return "\n".join(snippets) if snippets else "No peer responses yet."

    @staticmethod
    def _majority_vote(round_responses: List[DebateResponse]) -> Tuple[str, str, int]:
        counts: Dict[str, Dict[str, Any]] = {}
        for resp in round_responses:
            norm = resp.normalized_answer
            if norm not in counts:
                counts[norm] = {"count": 0, "answer": resp.answer}
            counts[norm]["count"] += 1

        best_norm, stats = max(counts.items(), key=lambda kv: (kv[1]["count"], kv[0]))
        return stats["answer"], best_norm, stats["count"]
