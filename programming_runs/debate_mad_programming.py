import argparse
import os
import sys
import json
from io import StringIO
from datetime import datetime, timezone
from typing import List, Dict, Any

from debate_coordinator_programming import DebateCoordinator
from utils import enumerate_resume, write_jsonl
from generators import generator_factory, model_factory


# --- CODING PERSONAS ---

CODING_PERSONAS = {
    "Senior Engineer": (
        "You are a Senior Software Engineer. Your goal is to write clean, "
        "efficient, and correct code that passes all unit tests."
    ),
    "QA Engineer": (
        "You are a QA Test Engineer. You focus on edge cases, input validation, "
        "and making sure the implementation truly satisfies the specification."
    ),
    "Algorithm Expert": (
        "You are an Algorithm Specialist. Focus on correctness and time/space "
        "complexity of the underlying algorithm."
    ),
    "Code Reviewer": (
        "You are a Strict Code Reviewer. You check for logical bugs, syntax "
        "errors, Pythonic style, and maintainability."
    ),
}


# --- EXECUTION / EVAL HELPERS ---

def run_humaneval_test(code: str, test_code: str) -> Dict[str, Any]:
    """
    Execute HumanEval-style code + tests in a sandboxed namespace.
    Returns {passed: bool, result: str}.
    """
    full_program = f"from typing import *\nimport math\n\n{code}\n\n{test_code}"

    capture = StringIO()
    sys_stdout, sys_stderr = sys.stdout, sys.stderr
    sys.stdout = capture
    sys.stderr = capture

    try:
        exec_globals: Dict[str, Any] = {}
        exec(full_program, exec_globals)
        return {"passed": True, "result": "Tests Passed"}
    except Exception:
        import traceback
        return {"passed": False, "result": traceback.format_exc()}
    finally:
        sys.stdout = sys_stdout
        sys.stderr = sys_stderr


# --- CLI ARGS ---

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data/humaneval-py.jsonl",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./results/mad_python_results.jsonl",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="python",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=3,
        help="Maximum attempts per problem (simple + MAD refinements).",
    )
    parser.add_argument(
        "--num_agents",
        type=int,
        default=3,
        help="Number of debating agents per round.",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=2,
        help="Number of debate rounds before judge picks a winner.",
    )
    parser.add_argument(
        "--personas",
        nargs="+",
        default=["Senior Engineer", "QA Engineer", "Code Reviewer"],
        help="Keys in CODING_PERSONAS to use as agent personas.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Stop after this many examples (for quick runs). Default: run all.",
    )
    return parser.parse_args()


# --- MAIN EXPERIMENT LOOP ---

def run_mad_programming_trials(args):
    # --- Load dataset ---
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset not found at {args.dataset_path}")
        return

    dataset: List[Dict[str, Any]] = []
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            dataset.append(json.loads(line))

    # --- Ensure output dirs / log files exist ---
    out_dir = os.path.dirname(os.path.abspath(args.output_path)) or "."
    os.makedirs(out_dir, exist_ok=True)

    debate_jsonl_path = os.path.join(out_dir, "debate_outputs.jsonl")
    debate_txt_path = os.path.join(out_dir, "debate_outputs.txt")
    attempts_txt_path = os.path.join(out_dir, "attempts_summary.txt")
    accuracy_txt_path = os.path.join(out_dir, "final_accuracy.txt")  # New accuracy log file
    # create files if missing
    for p in (debate_jsonl_path, debate_txt_path, attempts_txt_path, accuracy_txt_path):
        if not os.path.exists(p):
            with open(p, "w", encoding="utf-8"):
                pass

    # Build persona strings for agents
    agent_personas: List[str] = []
    for key in args.personas:
        persona_text = CODING_PERSONAS.get(key, key)
        agent_personas.append(persona_text)

    # Generators / model factory
    gen = generator_factory(args.language)
    model = model_factory(args.model)

    processed = 0
    passed_count = 0  # Counter for passed tests
    for i, item in enumerate_resume(dataset, args.output_path):
        if args.max_examples is not None and processed >= args.max_examples:
            print(f"[+] Reached max_examples={args.max_examples}. Stopping.")
            break
        processed += 1

        task_name = item.get("name") or item.get("entry_point") or f"task_{i}"
        entry_point = item.get("entry_point") or item.get("name") or "solution"

        print(f"\n=== Task: {task_name} ({entry_point}) ===")

        # --- Initial SIMPLE generation (no debate) ---
        code = gen.func_impl(item["prompt"], model, "simple")

        # Fallback stub if generator gave nothing usable
        if not isinstance(code, str) or not code.strip():
            print(f"[!] Empty initial code for {entry_point}. Inserting stub.")
            code = f"def {entry_point}(*args, **kwargs):\n    raise NotImplementedError\n"

        tests = item.get("test", "") or item.get("example_test", "")

        last_execution_result: Dict[str, Any] = {"passed": False, "result": ""}
        final_debate_consensus = None
        trial_used = 0

        for trial in range(1, args.num_trials + 1):
            print(f"[Trial {trial}/{args.num_trials}] Running tests...")
            last_execution_result = run_humaneval_test(code, tests)

            if execution_result['passed']:
                print(f"Trial {trial}: PASS")
                passed_count += 1  # Increment passed count
                break

            print(f"  ❌ FAIL (trial {trial})")
            trial_used = trial

            # Build context for debate: original prompt, failed code, traceback
            context = (
                "### Original Problem:\n"
                f"{item['prompt']}\n\n"
                "### Failed Code:\n"
                f"{code}\n\n"
                "### Execution Output / Error:\n"
                f"{last_execution_result['result']}\n"
            )

            # --- Run Multi-Agent Debate + Judge ---
            coordinator = DebateCoordinator(
                question=f"Fix the Python function `{entry_point}` so that it passes the tests.",
                context=context,
                answer_key="Passing Unit Tests",
                num_agents=args.num_agents,
                num_rounds=args.num_rounds,
                llm_kwargs={"model_name": args.model, "temperature": 0.2},
                personas=agent_personas,
            )

            debate_result = coordinator.run()
            final_debate_consensus = debate_result.get("summary")
            patched_code = debate_result.get("code")  # full function string
            rounds = debate_result.get("rounds", [])

            # --- Log debate to JSONL and TXT ---
            debate_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "task_name": task_name,
                "task_entry": entry_point,
                "trial": trial,
                "personas": args.personas,
                "scratchpad": context,
                "consensus": final_debate_consensus,
                "proposed_code": patched_code,
                "raw_debate": rounds,
            }

            # JSONL log
            try:
                with open(debate_jsonl_path, "a", encoding="utf-8") as df:
                    df.write(json.dumps(debate_entry, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[WARN] failed to write debate JSONL: {e}")

            # Human-readable log
            try:
                with open(debate_txt_path, "a", encoding="utf-8") as tf:
                    ts = debate_entry["timestamp"]
                    tf.write("\n" + "=" * 100 + "\n")
                    tf.write(f"Timestamp: {ts}\n")
                    tf.write(f"Task: {task_name} ({entry_point})\n")
                    tf.write(f"Trial: {trial}\n")
                    tf.write(f"Personas: {', '.join(args.personas)}\n")
                    tf.write("-" * 100 + "\n")
                    tf.write("Scratchpad / Context:\n")
                    tf.write(context + "\n")
                    tf.write("-" * 100 + "\n")
                    tf.write("Debate Summary:\n")
                    tf.write((final_debate_consensus or "") + "\n")
                    tf.write("-" * 100 + "\n")
                    tf.write("Proposed Code:\n")
                    tf.write((patched_code or "") + "\n")
                    tf.write("-" * 100 + "\n")
                    tf.write("Raw Debate Rounds:\n")
                    tf.write(json.dumps(rounds, indent=2, ensure_ascii=False) + "\n")
                    tf.write("=" * 100 + "\n\n")
            except Exception as e:
                print(f"[WARN] failed to write debate TXT: {e}")

            # --- Use debate's patched code as the new implementation, if any ---
            if isinstance(patched_code, str) and patched_code.strip():
                print("  [+] Applying debate-patched code directly.")
                code = patched_code
                continue  # next trial will run tests on this new code

            # --- Fallback path: use Reflexion-like generator if no code from debate ---
            print("  [!] No explicit code found from debate; falling back to Reflexion generator.")
            prev_impl = code
            feedback_text = last_execution_result.get("result", "")

            self_reflection_text = (
                "The previous attempt failed the unit tests.\n"
                "Here is a high-level critique and fix plan from a multi-agent debate:\n"
                f"{final_debate_consensus or str(debate_result)}\n\n"
                "Please use this critique to carefully rewrite the function so that it passes the tests."
            )

            code = gen.func_impl(
                func_sig=item["prompt"],
                model=model,
                strategy="reflexion",
                prev_func_impl=prev_impl,
                feedback=feedback_text,
                self_reflection=self_reflection_text,
            )

        # --- Final record for this task ---
        result_record = {
            "task_name": task_name,
            "task_entry": entry_point,
            "prompt": item.get("prompt"),
            "completion": code,
            "passed": last_execution_result.get("passed", False),
            "trials_used": trial_used,
            "execution_result": last_execution_result.get("result", ""),
            "debate_consensus": final_debate_consensus,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        write_jsonl(args.output_path, [result_record], append=True)

        # Concise text summary
        try:
            with open(attempts_txt_path, "a", encoding="utf-8") as af:
                af.write("\n" + "-" * 80 + "\n")
                af.write(f"Timestamp: {result_record['timestamp']}\n")
                af.write(f"Task: {task_name} ({entry_point})\n")
                af.write(
                    f"Passed: {result_record['passed']}  |  Trials used: {result_record['trials_used']}\n"
                )
                af.write("Execution Result (truncated):\n")
                ex = result_record["execution_result"] or ""
                af.write((ex[:800] + "...") if len(ex) > 800 else ex)
                af.write("\n")
                af.write("Debate Consensus (truncated):\n")
                dc = result_record.get("debate_consensus") or ""
                af.write((dc[:800] + "...") if len(dc) > 800 else dc)
                af.write("\n" + "-" * 80 + "\n\n")
        except Exception as e:
            print(f"WARNING: failed to write attempts summary txt: {e}")

    # Calculate and write final accuracy
    if processed > 0:
        accuracy = (passed_count / processed) * 100
        with open(accuracy_txt_path, "w", encoding="utf-8") as af:
            af.write(f"Final Accuracy: {accuracy:.2f}%\n")
            af.write(f"Total Tests Processed: {processed}\n")
            af.write(f"Total Passed: {passed_count}\n")
    else:
        print("No tests processed, accuracy cannot be calculated.")

if __name__ == "__main__":
    args = parse_args()
    run_mad_programming_trials(args)