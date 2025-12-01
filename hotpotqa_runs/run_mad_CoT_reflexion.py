import argparse
import json
import os
from typing import Any, Dict, List
import pandas as pd
from datetime import datetime, timezone

# Import the Agent that attempts to solve the problem
from agents import CoTAgent, ReflexionStrategy
from fewshots import COT, COT_REFLECT
from prompts import cot_agent_prompt, cot_reflect_prompt

# Import the Debate Coordinator
from debate_personas import DebateCoordinator
from run_cot_experiment import load_dataset, build_summary

# RICHER PERSONAS
PERSONA_PROMPTS = {
    "Verifier": """
You are a Verifier. Your job is to check each claim carefully for factual correctness and internal logical consistency.
For each assertion the Actor or another agent makes, ask yourself: “Is this backed by evidence or context? Could this be wrong?”
Discard any reasoning steps that lack justification, ambiguous references, or unsupported assumptions.
If you find an error or gap — call it out explicitly and explain why it might be wrong.
""",

    "Planner": """
You are a Planner. You care about the high-level structure of the reasoning.
Before diving into low-level details, outline a strategy: what steps to take, what subproblems to solve, and in what order.
If earlier attempts failed, propose a different overall plan (alternative breakdown).
Your reflections should focus on planning, not just individual mistakes.
""",

    "Skeptic": """
You are a Skeptic. Assume earlier reasoning may have hallucinations or leaps.
Critique every assumption, spec, and inference.
Ask: “How do I know this is true?”, “What if the premise is wrong?”, “Is there another possible interpretation?”.
Your goal is to prevent overconfidence and surface plausible failure modes.
""",

    "Logician": """
You are a Strict Logician. Evaluate whether the answer exactly matches the specification or asked question.
Do not accept vague matches, implied meanings, or partially correct statements.
If the requirement asks for a full definition, full proof, or exact formatting — check strictly for compliance.
""",

    "Creative": """
You are a Creative Thinker. If conventional reasoning fails or stalls, propose unforeseen angles.
Look for edge cases, trick questions, alternative interpretations, or unusual solutions.
Your reflections should expand the search space rather than refine within the existing pattern.
""",

    "Meta-Reflector": """
You are a Meta-Reflector. After seeing multiple failed attempts, reflect not just on code / reasoning errors but on the overall process.
Ask: “Why did we keep failing?”, “Are we stuck in a loop of similar mistakes?”, “Should we change the memory buffer, retry policy, or strategy type?”
Suggest meta-changes: different prompting style, more memory, switching reasoning mode, or abandoning this approach.
"""
}

# Example meta-prompt to wrap the debate  
DEBATE_META_PROMPT = """
You are a debater in a multi-agent debate. There will be several agents, each with a distinct persona.
Each agent will take turns arguing. You don’t have to fully agree — the goal is to explore different reasoning paths and find the best solution.
Discuss your reasoning step by step. Respond to previous agents’ arguments, highlight flaws or alternative paths, and aim to converge to a correct, well-justified answer.
"""

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
                
                # construct coordinator with the core required args (aligns with debate.py)
                coordinator = DebateCoordinator(
                    question=agent.question,
                    context=agent.scratchpad, # Pass the FAILED scratchpad as context
                    answer_key=agent.key,
                    num_agents=args.num_agents,
                    num_rounds=args.num_rounds,
                    llm_kwargs={"model_name": args.model_name, "temperature": 0.2},
                    personas=agent_personas
                )

                # Call run with expected signature; accept either string or dict return
                debate_result = coordinator.run()
                if isinstance(debate_result, dict):
                    generated_reflection = debate_result.get("final_answer") or debate_result.get("consensus") or json.dumps(debate_result)
                    debate_text = json.dumps(debate_result, indent=2)
                else:
                    generated_reflection = str(debate_result)
                    debate_text = generated_reflection

                # Append a structured reflection containing:
                #  - consensus (final answer)
                #  - short_summary (truncated summary safe for next prompt)
                #  - full_debate_log (pointer/inline if small; stored in jsonl above)
                full_log = debate_entry.get("full_debate_log") if 'debate_entry' in locals() else (debate_result if isinstance(debate_result, (dict, list, str)) else str(debate_result))
                short_summary = None
                try:
                    # create a short token-safe summary (simple heuristic truncate of consensus + first N chars of full log)
                    fl_text = json.dumps(full_log, ensure_ascii=False) if not isinstance(full_log, str) else full_log
                    short_summary = (generated_reflection or "").strip()
                    if not short_summary:
                        short_summary = (fl_text[:800] + "...") if len(fl_text) > 800 else fl_text
                    else:
                        # append a small context snippet
                        snippet = (fl_text[:500] + "...") if len(fl_text) > 500 else fl_text
                        short_summary = short_summary + "\n\nContext-snippet:\n" + snippet
                except Exception:
                    short_summary = (generated_reflection or "")[:1000]

                # store structured reflection (safe size) and leave full log in file / debate_jsonl
                reflection_record = {
                    "consensus": generated_reflection,
                    "short_summary": short_summary,
                    "full_debate_ref": {
                        "jsonl_path": os.path.join(args.output_dir, "debate_outputs.jsonl"),
                        "question_id": metadata[idx]["id"],
                        "trial": trial,
                    }
                }
                agent.reflections.append(reflection_record)
                print(f"  [+] Debate finished. Reflection added (consensus + short summary). Full log saved to debate_outputs.jsonl")

                # Write a neat, timestamped debate log entry to output_dir/debate_outputs.txt
                os.makedirs(args.output_dir, exist_ok=True)
                debate_log_path = os.path.join(args.output_dir, "debate_outputs.txt")
                debate_jsonl_path = os.path.join(args.output_dir, "debate_outputs.jsonl")
                
                # Ensure JSONL file exists
                if not os.path.exists(debate_jsonl_path):
                    with open(debate_jsonl_path, "w", encoding="utf-8") as _:
                        pass

                ts = datetime.now(timezone.utc).isoformat()
                personas_str = ", ".join(agent_personas)

                # Log debate to JSONL
                debate_entry = {
                    "timestamp": ts,
                    "trial": trial,
                    "agent_index": idx,
                    "question_id": metadata[idx]['id'],
                    "question": metadata[idx]['question'],
                    "ground_truth": metadata[idx]['answer'],
                    "personas": agent_personas,
                    "scratchpad": agent.scratchpad,
                    "debate_consensus": generated_reflection,
                    "debate_rounds": debate_result.get("rounds") if isinstance(debate_result, dict) else [],
                    "full_debate_log": debate_result.get("full_debate_log") if isinstance(debate_result, dict) else debate_text
                }
                try:
                    with open(debate_jsonl_path, "a", encoding="utf-8") as jf:
                        jf.write(json.dumps(debate_entry, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"WARNING: failed to write debate JSONL: {e}")

                with open(debate_log_path, "a", encoding="utf-8") as df:
                    df.write("\n" + "="*80 + "\n")
                    df.write(f"Timestamp: {ts}\n")
                    df.write(f"Trial: {trial} | Agent index: {idx} | Question ID: {metadata[idx]['id']}\n")
                    df.write(f"Personas: {personas_str}\n")
                    df.write(f"Question: {metadata[idx]['question']}\n")
                    df.write(f"Ground truth: {metadata[idx]['answer']}\n")
                    df.write("-"*80 + "\n")
                    df.write("Scratchpad (failed attempt):\n")
                    df.write(agent.scratchpad + "\n")
                    df.write("-"*80 + "\n")
                    df.write("Debate output / consensus:\n")
                    df.write(debate_text + "\n")
                    df.write("-"*80 + "\n")
                    df.write("Full Debate Rounds:\n")
                    if isinstance(debate_result, dict) and "rounds" in debate_result:
                        df.write(json.dumps(debate_result["rounds"], indent=2) + "\n")
                    df.write("="*80 + "\n\n")

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
    
    run_meta = {
        "num_agents": args.num_agents,
        "num_rounds": args.num_rounds,
        "dataset_path": os.path.abspath(args.dataset_path),
        "model_name": args.model_name,
        "num_questions": len(df),
        "num_correct": int(summary_df["is_correct"].sum()),
    }
    with open(os.path.join(args.output_dir, "run_meta.json"), "w") as f:
        json.dump(run_meta, f, indent=2)

    print(f"[+] Saved run metadata: {os.path.join(args.output_dir, 'run_meta.json')}")


if __name__ == "__main__":
    main()