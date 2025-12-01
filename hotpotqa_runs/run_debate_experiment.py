import argparse
import json
import os
from typing import Any, Dict

import pandas as pd

from debate import DebateCoordinator
from run_cot_experiment import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HotPotQA with multi-agent debate.")
    parser.add_argument("--dataset_path", default="hotpotqa_runs/data/hotpot-qa-distractor-sample.joblib")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_agents", type=int, default=3, help="Number of debating agents.")
    parser.add_argument("--num_rounds", type=int, default=3, help="Debate rounds (>=1).")
    parser.add_argument("--model_name", default="gpt-3.5-turbo", help="OpenAI model to use.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--max_tokens", type=int, default=256, help="Max tokens per agent reply.")
    return parser.parse_args()


def build_llm_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "model_name": args.model_name,
    }


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = load_dataset(args.dataset_path)
    metadata = df[["id", "question", "answer"]].to_dict("records")
    contexts = df["supporting_paragraphs"].tolist()
    llm_kwargs = build_llm_kwargs(args)

    summary_records = []
    attempts_path = os.path.join(args.output_dir, "debate_attempts.jsonl")

    with open(attempts_path, "w") as attempts_file:
        for meta, context in zip(metadata, contexts):
            coordinator = DebateCoordinator(
                question=meta["question"],
                answer_key=meta["answer"],
                num_debators=args.num_agents,
                max_num_rounds=args.num_rounds,
                llm_kwargs=llm_kwargs,
            )
            # TODO: context should be used as scratchpad, but need to clarify the architecture
            outcome = coordinator.run(scratchpad=context)

            summary_records.append(
                {
                    "question_id": meta["id"],
                    "question": meta["question"],
                    "ground_truth": meta["answer"],
                    "final_answer": outcome["final_answer"],
                    "is_correct": outcome["is_correct"],
                    "majority_votes": outcome["majority_votes"],
                    "num_agents": args.num_agents,
                    "num_rounds": args.num_rounds,
                }
            )

            attempts_file.write(
                json.dumps(
                    {
                        "question_id": meta["id"],
                        "question": meta["question"],
                        "ground_truth": meta["answer"],
                        "rounds": outcome["rounds"],
                        "final_answer": outcome["final_answer"],
                        "normalized_final_answer": outcome["normalized_final_answer"],
                        "majority_votes": outcome["majority_votes"],
                        "is_correct": outcome["is_correct"],
                    }
                )
                + "\n"
            )

    summary_df = pd.DataFrame(summary_records)
    summary_path = os.path.join(args.output_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)

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

    print(f"[+] Saved summary: {summary_path}")
    print(f"[+] Saved attempts: {attempts_path}")


if __name__ == "__main__":
    main()
