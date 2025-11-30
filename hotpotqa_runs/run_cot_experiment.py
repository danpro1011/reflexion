import argparse
import json
import os
from typing import Any, Dict, List

import importlib
import joblib
import numpy as np
import pandas as pd

from agents import CoTAgent, ReflexionStrategy
from fewshots import COT, COT_REFLECT
from prompts import cot_agent_prompt, cot_reflect_agent_prompt, cot_reflect_prompt
from util import log_trial, save_agents


def load_dataset(path: str) -> pd.DataFrame:
    import sys
    import types
    try:
        importlib.import_module("pandas.core.indexes.numeric")
    except ModuleNotFoundError:
        numeric_module = types.ModuleType('pandas.core.indexes.numeric')
        numeric_module.Int64Index = pd.Index
        numeric_module.UInt64Index = pd.Index
        numeric_module.Float64Index = pd.Index
        sys.modules['pandas.core.indexes.numeric'] = numeric_module
    df = joblib.load(path)
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected DataFrame, got {type(df)}")
    df = df.reset_index(drop=True)
    if "supporting_paragraphs" in df.columns:
        return df

    supporting_paragraphs: List[str] = []
    for _, row in df.iterrows():
        titles = row["context"]["title"]
        sentences = row["context"]["sentences"]
        selected = row["supporting_facts"]["title"]
        paragraphs: List[str] = []
        for article in selected:
            idx = np.where(titles == article)
            if len(idx[0]) == 0:
                continue
            paragraphs.append("".join(sentences[idx[0][0]]))
        supporting_paragraphs.append("\n\n".join(paragraphs))
    df["supporting_paragraphs"] = supporting_paragraphs
    return df


def build_agents(df: pd.DataFrame, strategy: ReflexionStrategy) -> List[CoTAgent]:
    prompt = cot_agent_prompt if strategy == ReflexionStrategy.NONE else cot_reflect_agent_prompt
    agents: List[CoTAgent] = []
    for _, row in df.iterrows():
        agents.append(
            CoTAgent(
                question=row["question"],
                context=row["supporting_paragraphs"],
                key=row["answer"],
                agent_prompt=prompt,
                reflect_prompt=cot_reflect_prompt,
                cot_examples=COT,
                reflect_examples=COT_REFLECT,
            )
        )
    return agents


def run_trials(
    agents: List[CoTAgent],
    metadata: List[Dict[str, Any]],
    strategy: ReflexionStrategy,
    num_trials: int,
) -> Dict[str, Any]:
    attempt_logs: List[Dict[str, Any]] = []
    per_agent_attempts: List[List[Dict[str, Any]]] = [[] for _ in agents]
    aggregate_log = ""

    for trial in range(1, num_trials + 1):
        for idx, agent in enumerate(agents):
            if agent.is_correct():
                continue
            agent.run(reflexion_strategy=strategy)
            attempt = {
                "trial": trial,
                "question_id": metadata[idx]["id"],
                "question": metadata[idx]["question"],
                "ground_truth": metadata[idx]["answer"],
                "prediction": agent.answer,
                "is_correct": agent.is_correct(),
                "is_finished": agent.is_finished(),
                "scratchpad": agent.scratchpad,
                "reflections": getattr(agent, "reflections", []),
            }
            per_agent_attempts[idx].append(attempt)
            attempt_logs.append(attempt)
        aggregate_log += log_trial(agents, trial)

    return {
        "attempt_logs": attempt_logs,
        "per_agent_attempts": per_agent_attempts,
        "aggregate_log": aggregate_log,
    }


def build_summary(
    agents: List[CoTAgent],
    metadata: List[Dict[str, Any]],
    per_agent_attempts: List[List[Dict[str, Any]]],
) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for meta, agent, attempts in zip(metadata, agents, per_agent_attempts):
        records.append(
            {
                "question_id": meta["id"],
                "question": meta["question"],
                "ground_truth": meta["answer"],
                "num_attempts": len(attempts),
                "is_correct": agent.is_correct(),
                "final_answer": agent.answer,
                "reflections": getattr(agent, "reflections", []),
            }
        )
    return pd.DataFrame(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CoT Reflexion experiment with structured logging.")
    parser.add_argument("--dataset_path", default="hotpotqa_runs/data/hotpot-qa-distractor-sample.joblib")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--strategy",
        default="NONE",
        choices=[s.name for s in ReflexionStrategy],
        help="Reflexion strategy to apply.",
    )
    parser.add_argument("--num_trials", type=int, default=5, help="Number of Reflexion trials.")
    parser.add_argument("--save_agents", action="store_true", help="Persist agent joblib snapshots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    strategy = ReflexionStrategy[args.strategy]
    os.makedirs(args.output_dir, exist_ok=True)

    df = load_dataset(args.dataset_path)
    metadata = df[["id", "question", "answer"]].to_dict("records")
    agents = build_agents(df, strategy)

    results = run_trials(agents, metadata, strategy, args.num_trials)
    summary_df = build_summary(agents, metadata, results["per_agent_attempts"])

    summary_path = os.path.join(args.output_dir, "summary.csv")
    attempts_path = os.path.join(args.output_dir, "attempts.jsonl")
    log_path = os.path.join(args.output_dir, "trial_log.txt")

    summary_df.to_csv(summary_path, index=False)
    with open(attempts_path, "w") as f:
        for record in results["attempt_logs"]:
            f.write(json.dumps(record) + "\n")
    with open(log_path, "w") as f:
        f.write(results["aggregate_log"])

    run_meta = {
        "strategy": strategy.name,
        "num_trials": args.num_trials,
        "dataset_path": os.path.abspath(args.dataset_path),
        "num_questions": len(agents),
        "num_correct": int(summary_df["is_correct"].sum()),
    }
    with open(os.path.join(args.output_dir, "run_meta.json"), "w") as f:
        json.dump(run_meta, f, indent=2)

    if args.save_agents:
        save_agents(agents, os.path.join(args.output_dir, "agents"))

    print(f"[+] Saved summary: {summary_path}")
    print(f"[+] Saved attempts: {attempts_path}")
    print(f"[+] Saved log: {log_path}")

if __name__ == "__main__":
    main()
