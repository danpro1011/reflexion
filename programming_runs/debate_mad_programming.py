import argparse
import os
import sys
import json
import multiprocessing
from io import StringIO
from typing import List
from debate_coordinator_programming import DebateCoordinator
from utils import enumerate_resume, write_jsonl
from generators import generator_factory, model_factory
from datetime import datetime, timezone

# --- CODING PERSONAS ---
CODING_PERSONAS = {
    "Senior Engineer": "You are a Senior Software Engineer. Your goal is to write clean, efficient, and correct code.",
    "QA Engineer": "You are a QA Test Engineer. You focus on edge cases and input validation.",
    "Algorithm Expert": "You are an Algorithm Specialist. Focus on the complexity and the underlying algorithm.",
    "Code Reviewer": "You are a Strict Code Reviewer. Check for syntax errors and Pythonic practices."
}

def run_humaneval_test(code: str, test_code: str) -> dict:
    full_program = f"from typing import *\nimport math\n\n{code}\n\n{test_code}"

    capture = StringIO()
    sys.stdout = capture
    sys.stderr = capture
    
    try:
        exec_globals = {}
        exec(full_program, exec_globals)
        return {"passed": True, "result": "Tests Passed"}
    except Exception:
        import traceback
        return {"passed": False, "result": traceback.format_exc()}
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./data/humaneval-py.jsonl")
    parser.add_argument("--output_path", type=str, default="./results/mad_python_results.jsonl")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--language", type=str, default="python")
    parser.add_argument("--num_trials", type=int, default=3)
    parser.add_argument("--num_agents", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=2)
    parser.add_argument("--personas", nargs="+", default=["Senior Engineer", "QA Engineer"])
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Stop after this many examples (for quick runs). Default: run all")
    return parser.parse_args()

def run_mad_programming_trials(args):
    dataset = []
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset not found at {args.dataset_path}")
        return

    with open(args.dataset_path, 'r') as f:
        for line in f:
            dataset.append(json.loads(line))

    # ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(args.output_path)) or '.'
    os.makedirs(out_dir, exist_ok=True)

    # debate log path (JSONL)
    debate_jsonl_path = os.path.join(out_dir, "debate_outputs.jsonl")
    debate_txt_path = os.path.join(out_dir, "debate_outputs.txt")
    attempts_txt_path = os.path.join(out_dir, "attempts_summary.txt")
    accuracy_txt_path = os.path.join(out_dir, "final_accuracy.txt")  # New accuracy log file
    # create files if missing
    for p in (debate_jsonl_path, debate_txt_path, attempts_txt_path, accuracy_txt_path):
        if not os.path.exists(p):
            with open(p, "w", encoding="utf-8") as _:
                pass
    
    agent_personas = [CODING_PERSONAS[key] for key in args.personas]

    gen = generator_factory(args.language)
    model = model_factory(args.model)

    processed = 0
    passed_count = 0  # Counter for passed tests
    for i, item in enumerate_resume(dataset, args.output_path):
        if args.max_examples is not None and processed >= args.max_examples:
            print(f"[+] Reached max_examples={args.max_examples}. Stopping.")
            break
        processed += 1

        print(f"\n--- Task {item.get('name') or item.get('entry_point') or i} ---")
        code = gen.func_impl(item["prompt"], model, "simple")
        # sanitize generator output: ensure we have a non-empty string to exec
        if not isinstance(code, str) or not code.strip():
            entry = item.get("entry_point") or item.get("name") or "solution"
            print(f"WARNING: generator returned empty code for task '{entry}'. Inserting stub.")
            code = f"def {entry}(*args, **kwargs):\n    pass\n"
        
        for trial in range(1, args.num_trials + 1):
            tests = item.get("test", "") or item.get("example_test", "")
            execution_result = run_humaneval_test(code, tests)

            if execution_result['passed']:
                print(f"Trial {trial}: PASS")
                passed_count += 1  # Increment passed count
                break
            
            print(f"Trial {trial}: FAIL")
            context = f"### Code:\n{code}\n\n### Execution Output/Error:\n{execution_result['result']}"
            coordinator = DebateCoordinator(
                question=f"Fix the python function `{item.get('entry_point')}` to pass the tests.",
                context=context,
                answer_key="Passing Unit Tests",
                num_agents=args.num_agents,
                num_rounds=args.num_rounds,
                llm_kwargs={"model_name": args.model, "temperature": 0.2},
                personas=agent_personas
            )

            debate_result = coordinator.run()
            # consensus string and raw serializable debate object
            consensus = None
            try:
                consensus = debate_result.get("final_answer") or debate_result.get("consensus") or None
            except Exception:
                # debate_result may be a string or other type
                pass
            if consensus is None:
                # fallback to string
                consensus = str(debate_result)

            print(f"  [+] Debate Consensus: {consensus[:100]}...")

            # log debate output (JSONL)
            debate_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "task_name": item.get("name"),
                "task_entry": item.get("entry_point"),
                "trial": trial,
                "personas": args.personas,
                "scratchpad": context,
                "consensus": consensus,
                "raw_debate": debate_result if isinstance(debate_result, (dict, list, str, int, float, bool, type(None))) else str(debate_result)
            }
            try:
                with open(debate_jsonl_path, "a", encoding="utf-8") as df:
                    df.write(json.dumps(debate_entry, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"WARNING: failed to write debate log: {e}")

            # also append a human-readable text log (neat formatting)
            try:
                with open(debate_txt_path, "a", encoding="utf-8") as tf:
                    ts = debate_entry["timestamp"]
                    tf.write("\n" + "="*100 + "\n")
                    tf.write(f"Timestamp: {ts}\n")
                    tf.write(f"Task: {debate_entry['task_name'] or debate_entry['task_entry']}\n")
                    tf.write(f"Trial: {debate_entry['trial']}\n")
                    tf.write(f"Personas: {', '.join(debate_entry['personas'])}\n")
                    tf.write("-"*100 + "\n")
                    tf.write("Scratchpad / Context:\n")
                    tf.write(debate_entry["scratchpad"] + "\n")
                    tf.write("-"*100 + "\n")
                    tf.write("Debate Consensus:\n")
                    tf.write(debate_entry["consensus"] + "\n")
                    tf.write("-"*100 + "\n")
                    tf.write("Raw Debate Output:\n")
                    # try pretty-print if JSON-serializable
                    try:
                        tf.write(json.dumps(debate_entry["raw_debate"], indent=2, ensure_ascii=False) + "\n")
                    except Exception:
                        tf.write(str(debate_entry["raw_debate"]) + "\n")
                    tf.write("="*100 + "\n\n")
            except Exception as e:
                print(f"WARNING: failed to write human-readable debate log: {e}")

            # ensure consensus and feedback are strings (not None) before calling generator
            prev_impl = code or ""
            feedback_text = execution_result.get('result') if isinstance(execution_result, dict) else str(execution_result)
            feedback_text = (feedback_text or "")
            self_reflection_text = (consensus or "")

            code = gen.func_impl(
                func_sig=item["prompt"],
                model=model,
                strategy="reflexion",
                prev_func_impl=prev_impl,
                feedback=feedback_text,
                self_reflection=self_reflection_text
            )

        # build enriched attempt record
        cur_result = {
            "task_name": item.get("name"),
            "task_entry": item.get("entry_point"),
            "prompt": item.get("prompt"),
            "completion": code,
            "passed": execution_result['passed'],
            "trials_used": trial,
            "execution_result": execution_result.get("result") if isinstance(execution_result, dict) else str(execution_result),
            "debate_consensus": consensus if 'consensus' in locals() else None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        write_jsonl(args.output_path, [cur_result], append=True)

        # append a concise human-readable attempt summary
        try:
            with open(attempts_txt_path, "a", encoding="utf-8") as af:
                af.write("\n" + "-"*80 + "\n")
                af.write(f"Timestamp: {cur_result['timestamp']}\n")
                af.write(f"Task: {cur_result['task_name'] or cur_result['task_entry']}\n")
                af.write(f"Passed: {cur_result['passed']}  |  Trials used: {cur_result['trials_used']}\n")
                af.write(f"Execution Result (truncated):\n")
                ex = cur_result['execution_result'] or ""
                af.write((ex[:800] + "...") if len(ex) > 800 else ex)
                af.write("\n")
                af.write(f"Debate Consensus (truncated):\n{(cur_result.get('debate_consensus') or '')[:800]}\n")
                af.write("-"*80 + "\n\n")
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