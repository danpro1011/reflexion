"""
Small helper to summarize a reflexion run JSONL log and (optionally) rebuild a
readable failures-only log.

Usage:
  python3 compute_output_info.py --log_path programming_runs/root/test_reflexion/humaneval-py._reflexion_2_gpt-4_pass_at_k_1_py.jsonl --failed_log programming_runs/root/test_reflexion/failed_examples.log
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple, Dict, Any


def load_log(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def summarize_run(items: List[Dict[str, Any]]) -> Tuple[int, int, int, float]:
    total = len(items)
    solved = sum(1 for it in items if it.get("is_solved"))
    api_calls = sum(len(it.get("implementations", [])) for it in items)
    avg_api_calls = round(api_calls / total, 2) if total else 0.0
    return total, solved, api_calls, avg_api_calls


def write_failure_log(items: List[Dict[str, Any]], out_path: str) -> int:
    failures = 0
    with open(out_path, "w") as out:
        for i, item in enumerate(items, 1):
            if item.get("is_solved"):
                continue
            failures += 1
            prompt = item.get("prompt", "").strip()
            impls = item.get("implementations", []) or []
            feedbacks = item.get("test_feedback", [])
            reflections = item.get("reflections", [])
            last_impl_raw = next(
                (impl for impl in reversed(impls) if isinstance(impl, str) and impl.strip()),
                impls[-1] if impls else ""
            )
            last_impl = last_impl_raw.strip() if isinstance(last_impl_raw, str) else str(last_impl_raw)

            out.write("\n==================== FAILED EXAMPLE ====================\n")
            out.write(f"Index: {i} | Entry point: {item.get('entry_point', 'unknown')}\n")
            out.write("--------------------------------------------------------\n")
            out.write("PROMPT\n------\n")
            out.write(prompt + "\n\n")
            out.write(f"Attempts: {len(impls)} | Passes tried: {item.get('pass_at_k', 1)}\n\n")
            out.write("UNIT TEST FEEDBACK\n------------------\n")
            for idx, fb in enumerate(feedbacks, 1):
                out.write(f"[Attempt {idx}]\n{fb}\n\n")
            if reflections:
                out.write("REFLECTIONS\n-----------\n")
                for idx, ref in enumerate(reflections, 1):
                    out.write(f"[Reflection {idx}]\n{ref}\n\n")
            out.write("LAST IMPLEMENTATION\n-------------------\n")
            out.write(last_impl + "\n")
            out.write("========================================================\n")
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", required=True, help="Path to run JSONL log")
    parser.add_argument("--failed_log", help="Optional path to write a readable failed-examples log")
    args = parser.parse_args()

    items = load_log(args.log_path)
    total, solved, api_calls, avg_api_calls = summarize_run(items)
    failures = total - solved

    print("===== RUN SUMMARY =====")
    print(f"Log path: {args.log_path}")
    print(f"Total examples: {total}")
    print(f"Successful: {solved}")
    print(f"Failed: {failures}")
    print(f"Final accuracy: {round(solved / total, 3) if total else 0}")
    print(f"Total API calls: {api_calls}")
    print(f"Average API calls per example: {avg_api_calls}")

    if args.failed_log:
        os.makedirs(os.path.dirname(args.failed_log), exist_ok=True)
        failures_written = write_failure_log(items, args.failed_log)
        print(f"Wrote {failures_written} failed examples to {args.failed_log}")


if __name__ == "__main__":
    main()
