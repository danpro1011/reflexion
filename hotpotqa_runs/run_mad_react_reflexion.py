import argparse
import json
import os
from typing import Any, Dict, List
import pandas as pd
from datetime import datetime, timezone

# Import agents and strategies
from agents import ReactAgent, ReactReflectAgent, ReflexionStrategy
# Import ReAct specific fewshots and prompts
from fewshots import REACT, REACT_REFLECT
from prompts import react_agent_prompt, react_reflect_agent_prompt

# Import the Debate Coordinator
from debate_personas import DebateCoordinator
from run_cot_experiment import load_dataset, build_summary

# --- CUSTOM AGENT CLASS ---
class DebateReactAgent(ReactReflectAgent):
    """
    A wrapper around ReactReflectAgent that disables the built-in self-reflection loop.
    We want the 'Debate' process to generate reflections, not the agent itself.
    """
    def run(self, reset=True, reflect_strategy=ReflexionStrategy.REFLEXION):
        # We call the GRANDPARENT's run method (ReactAgent.run).
        # This runs the ReAct reasoning loop (Thought/Action/Obs) using the 
        # reflection-aware prompt from ReactReflectAgent, but it SKIPS the 
        # automatic 'self.reflect()' call that ReactReflectAgent.run() usually does.
        ReactAgent.run(self, reset)

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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ReAct with Multi-Agent Debate Reflexion.")
    parser.add_argument("--dataset_path", default="hotpotqa_runs/data/hotpot-qa-distractor-sample.joblib")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_trials", type=int, default=3, help="Number of retry attempts.")
    parser.add_argument("--num_agents", type=int, default=2, help="Number of debating agents.")
    parser.add_argument("--num_rounds", type=int, default=2, help="Number of debate rounds.")
    parser.add_argument("--model_name", default="gpt-3.5-turbo")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for the Actor agent.")
    parser.add_argument("--max_tokens", type=int, default=256, help="Max tokens for the Actor agent.")
    parser.add_argument("--max_examples", type=int, default=None, help="Limit the number of questions to run.")
    parser.add_argument(
        "--personas", 
        nargs="+", 
        default=["Skeptic", "Strategist"], 
        help="Personas for the debaters (keys in PERSONA_PROMPTS or raw text)."
    )
    return parser.parse_args()

def build_agents(df: pd.DataFrame) -> List[DebateReactAgent]:
    agents: List[DebateReactAgent] = []
    for _, row in df.iterrows():
        # Initialize our custom agent
        agent = DebateReactAgent(
            question=row["question"],
            key=row["answer"],
            agent_prompt=react_reflect_agent_prompt, # Use the prompt that supports reflections
            reflect_prompt=react_reflect_agent_prompt, # Dummy, we won't use internal reflection
        )
        # Manually inject the correct examples
        agent.react_examples = REACT
        agent.reflect_examples = REACT_REFLECT
        agents.append(agent)
    return agents

def run_mad_reflexion_trials(
    agents: List[DebateReactAgent],
    metadata: List[Dict[str, Any]],
    args: argparse.Namespace
) -> Dict[str, Any]:
    
    attempt_logs = []
    per_agent_attempts = [[] for _ in agents]

    # Prepare personas
    base_personas = args.personas
    agent_personas = []
    for i in range(args.num_agents):
        key = base_personas[i % len(base_personas)]
        desc = PERSONA_PROMPTS.get(key, key)
        agent_personas.append(desc)
    
    print(f"[*] Debater Personas: {args.personas}")

    for trial in range(1, args.num_trials + 1):
        print(f"\n--- Starting Trial {trial} ---")
        
        for idx, agent in enumerate(agents):
            if agent.is_correct():
                continue

            # 1. Run the ReAct Agent (The Actor)
            # We pass REFLEXION strategy, but our custom class ignores the self-reflect part
            agent.run(reflect_strategy=ReflexionStrategy.REFLEXION)
            
            print(f"Q{idx+1} Correct: {agent.is_correct()} | Ans: {agent.answer}")
            print(f"    Prediction: {agent.answer}")
            print(f"    Ground Truth: {agent.key}")

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
                    context=agent.scratchpad, 
                    answer_key=agent.key,
                    num_agents=args.num_agents,
                    num_rounds=args.num_rounds,
                    llm_kwargs={"model_name": args.model_name, "temperature": 0.2},
                    personas=agent_personas
                )

                debate_result = coordinator.run()
                
                # Handle different return types from debate
                if isinstance(debate_result, dict):
                    generated_reflection = debate_result.get("final_answer") or debate_result.get("consensus") or json.dumps(debate_result)
                    debate_text = json.dumps(debate_result, indent=2)
                else:
                    generated_reflection = str(debate_result)
                    debate_text = generated_reflection

                agent.reflections.append(generated_reflection)
                # Update the string representation for the next prompt
                agent.reflections_str = agent.reflections_str + "\n- " + generated_reflection
                
                print(f"  [+] Debate finished. Reflection added.")

                # Log debate details
                os.makedirs(args.output_dir, exist_ok=True)
                debate_log_path = os.path.join(args.output_dir, "debate_outputs.txt")
                ts = datetime.now(timezone.utc).isoformat()
                personas_str = ", ".join(args.personas)
                with open(debate_log_path, "a", encoding="utf-8") as df_log:
                    df_log.write("\n" + "="*80 + "\n")
                    df_log.write(f"Timestamp: {ts}\n")
                    df_log.write(f"Trial: {trial} | Agent index: {idx} | Question ID: {metadata[idx]['id']}\n")
                    df_log.write(f"Personas: {personas_str}\n")
                    df_log.write(f"Question: {metadata[idx]['question']}\n")
                    df_log.write(f"Ground truth: {metadata[idx]['answer']}\n")
                    df_log.write("-"*80 + "\n")
                    df_log.write("Scratchpad (failed attempt):\n")
                    df_log.write(agent.scratchpad + "\n")
                    df_log.write("-"*80 + "\n")
                    df_log.write("Debate output / consensus:\n")
                    df_log.write(debate_text + "\n")
                    df_log.write("="*80 + "\n\n")

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
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "num_questions": len(df),
        "num_correct": int(summary_df["is_correct"].sum()),
    }
    with open(os.path.join(args.output_dir, "run_meta.json"), "w") as f:
        json.dump(run_meta, f, indent=2)

    print(f"[+] Saved run metadata: {os.path.join(args.output_dir, 'run_meta.json')}")

if __name__ == "__main__":
    main()