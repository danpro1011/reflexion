import argparse
import json
import os
from typing import Any, Dict, List
import pandas as pd

# Import the Agent that attempts to solve the problem
from agents import CoTAgent, ReflexionStrategy
from fewshots import COT, COT_REFLECT
from prompts import cot_agent_prompt, cot_reflect_prompt

# Import the Debate Coordinator
from debate_personas import DebateCoordinator
from run_cot_experiment import load_dataset, build_summary

# RICHER PERSONAS
PERSONA_PROMPTS = {
    "Skeptic": (
        "a Skeptic. Your role is to strictly verify every claim against the context. "
        "Assume the previous reasoning contains hallucinations or assumptions not supported by the text. "
        "Point out exactly where the logic skips a step."
    ),
    "Strategist": (
        "a Strategist. Your focus is on the 'process'. Don't just critique the facts; suggest better search queries, "
        "alternative logical paths, and specific steps to avoid the previous error. "
        "Think: 'How would a better agent solve this?'"
    ),
    "Logician": (
        "a Strict Logician. You care about valid inference. Check if the answer actually matches the question type "
        "(e.g., does it ask for a name, a date, or a yes/no?). Ensure the conclusion follows inevitably from the premises."
    ),
    "Creative": (
        "a Lateral Thinker. You look for alternative interpretations of the question. "
        "If the obvious path failed, suggest a completely different angle or possibility that might have been overlooked. "
        "Consider if the question is a trick."
    )
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CoT with Multi-Agent Debate Reflexion.")
    parser.add_argument("--dataset_path", default="hotpotqa_runs/data/hotpot-qa-distractor-sample.joblib")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_trials", type=int, default=3, help="Number of retry attempts.")
    parser.add_argument("--num_agents", type=int, default=2, help="Number of debating agents.")
    parser.add_argument("--num_rounds", type=int, default=2, help="Number of debate rounds.")
    parser.add_argument("--model_name", default="gpt-3.5-turbo")
    parser.add_argument("--max_examples", type=int, default=None, help="Limit the number of questions to run.")
    #parser.add_argument("--personas", nargs="+", default=["Helpful Assistant"], help="Personas for the debaters.")
    parser.add_argument(
        "--personas", 
        nargs="+", 
        default=["Skeptic", "Strategist"], 
        help="Personas for the debaters (keys in PERSONA_PROMPTS or raw text)."
    )
    return parser.parse_args()

def build_agents(df: pd.DataFrame) -> List[CoTAgent]:
    agents: List[CoTAgent] = []
    for _, row in df.iterrows():
        agents.append(
            CoTAgent(
                question=row["question"],
                context=row["supporting_paragraphs"],
                key=row["answer"],
                agent_prompt=cot_agent_prompt,
                reflect_prompt=cot_reflect_prompt, 
                cot_examples=COT,
                reflect_examples=COT_REFLECT,
            )
        )
    return agents

def run_mad_reflexion_trials(
    agents: List[CoTAgent],
    metadata: List[Dict[str, Any]],
    args: argparse.Namespace
) -> Dict[str, Any]:
    
    attempt_logs = []
    per_agent_attempts = [[] for _ in agents]

    # Prepare personas
    base_personas = args.personas
    agent_personas = [base_personas[i % len(base_personas)] for i in range(args.num_agents)]
    print(f"[*] Debater Personas: {agent_personas}")

    for trial in range(1, args.num_trials + 1):
        print(f"\n--- Starting Trial {trial} ---")
        
        for idx, agent in enumerate(agents):
            if agent.is_correct():
                continue

            # 1. Run the CoT Agent (The Actor)
            agent.run(reflexion_strategy=ReflexionStrategy.REFLEXION)
            print(f"Q{idx+1} Correct: {agent.is_correct()} | Ans: {agent.answer}")
            print(f"    Prediction: {agent.answer}")
            print(f"    Ground Truth: {agent.key}")  # <--- Add this line

            # Log the attempt
            attempt = {
                "trial": trial,
                "question_id": metadata[idx]["id"],
                "question": metadata[idx]["question"],
                "ground_truth": metadata[idx]["answer"],
                "prediction": agent.answer,
                "is_correct": agent.is_correct(),
                "scratchpad": agent.scratchpad,
                "reflections": agent.reflections.copy(),
            }
            per_agent_attempts[idx].append(attempt)
            attempt_logs.append(attempt)

            # 2. If failed, trigger Debate to generate a reflection
            if not agent.is_correct() and trial < args.num_trials:
                print(f"  [!] Failed. Starting Debate Reflexion...")
                
                coordinator = DebateCoordinator(
                    question=agent.question,
                    context=agent.scratchpad, # Pass the FAILED scratchpad as context
                    answer_key=agent.key,
                    num_agents=args.num_agents,
                    num_rounds=args.num_rounds,
                    llm_kwargs={"model_name": args.model_name, "temperature": 0.2},
                    personas=agent_personas
                )

                # The debate output becomes the reflection
                debate_result = coordinator.run()
                generated_reflection = debate_result["final_answer"]
                
                agent.reflections.append(generated_reflection)
                print(f"  [+] Debate finished. Reflection added.")

    return {
        "attempt_logs": attempt_logs,
        "per_agent_attempts": per_agent_attempts
    }

def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = load_dataset(args.dataset_path)
    metadata = df[["id", "question", "answer"]].to_dict("records")
    agents = build_agents(df)
    
    if args.max_examples:
        agents = agents[:args.max_examples]
        metadata = metadata[:args.max_examples]

    results = run_mad_reflexion_trials(agents, metadata, args)

    summary_df = build_summary(agents, metadata, results["per_agent_attempts"])
    summary_path = os.path.join(args.output_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    attempts_path = os.path.join(args.output_dir, "attempts.jsonl")
    with open(attempts_path, "w") as f:
        for record in results["attempt_logs"]:
            f.write(json.dumps(record) + "\n")

    print(f"[+] Saved summary: {summary_path}")

if __name__ == "__main__":
    main()