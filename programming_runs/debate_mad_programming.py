import argparse
import os
import sys
import json
import multiprocessing
from io import StringIO
from typing import List

# Add parent dir to path to allow importing from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from debate_coordinator_programming import DebateCoordinator
from utils import enumerate_resume, make_printv, write_jsonl
from generators import generator_factory, model_factory

# --- CODING PERSONAS ---
CODING_PERSONAS = {
    "Senior Engineer": """
You are a Senior Software Engineer. Your goal is to write clean, efficient, and correct code.
Review the failed code and the error message.
Identify logic errors, off-by-one errors, or incorrect assumptions about the API/Input.
Propose a concrete fix in code or pseudocode.
""",
    "QA Engineer": """
You are a QA Test Engineer. You focus on edge cases and input validation.
Look at the test failure. Why did it fail? Was it an empty input? A large number? A type mismatch?
Critique the current implementation's handling of boundary conditions.
""",
    "Algorithm Expert": """
You are an Algorithm Specialist. Focus on the complexity and the underlying algorithm.
Is the current approach too slow (Time Limit Exceeded)? Is the logic fundamentally flawed for the problem type?
Suggest a better algorithm or data structure if necessary.
""",
    "Code Reviewer": """
You are a Strict Code Reviewer. Check for syntax errors, variable naming confusion, and Pythonic practices.
Ensure the code actually matches the function signature provided.
"""
}

def run_humaneval_test(code: str, test_code: str, timeout: int = 5) -> dict:
    """
    Runs the generated code + the HumanEval test harness.
    Returns {'passed': bool, 'result': str}
    """
    # HumanEval tests usually look like:
    # def check(candidate): ...
    # check(solution)
    # So we just need to concatenate them.
    
    full_program = f"from typing import *\nimport math\n\n{code}\n\n{test_code}"
    
    def target(queue):
        # Capture stdout/stderr
        capture = StringIO()
        sys.stdout = capture
        sys.stderr = capture
        
        try:
            # Create a fresh global scope
            exec_globals = {}
            exec(full_program, exec_globals)
            queue.put({"passed": True, "result": "Tests Passed"})
        except Exception:
            import traceback
            # Return the traceback as feedback
            queue.put({"passed": False, "result": traceback.format_exc()})
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=target, args=(queue,))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return {"passed": False, "result": f"TimeoutError: Execution exceeded {timeout}s"}

    if not queue.empty():
        return queue.get()
    return {"passed": False, "result": "Process crashed silently"}

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
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()

def run_mad_programming_trials(args):
    # Load Dataset
    dataset = []
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset not found at {args.dataset_path}")
        return

    with open(args.dataset_path, 'r') as f:
        for line in f:
            dataset.append(json.loads(line))

    # Prepare Personas
    base_personas = args.personas
    agent_personas = []
    for i in range(args.num_agents):
        key = base_personas[i % len(base_personas)]
        desc = CODING_PERSONAS.get(key, key)
        agent_personas.append(desc)

    # Initialize Factories
    gen = generator_factory(args.language)
    model = model_factory(args.model)

    print(f"[*] Starting MAD Programming Run with {len(dataset)} tasks.")
    print(f"[*] Personas: {args.personas}")

    for i, item in enumerate_resume(dataset, args.output_path):
        task_id = item['task_id']
        print(f"\n--- Task {task_id} ---")
        
        # 1. First Attempt (Simple Strategy)
        code = gen.func_impl(item["prompt"], model, "simple")
        
        is_passing = False
        feedback = ""
        trial = 0
        
        # Execution Loop
        for trial in range(1, args.num_trials + 1):
            # Execute Code
            tests = item.get("test", "") or item.get("example_test", "")
            
            execution_result = run_humaneval_test(code, tests)
            is_passing = execution_result['passed']
            feedback = execution_result['result'] # Traceback or error message

            print(f"Trial {trial}: {'PASS' if is_passing else 'FAIL'}")
            
            if is_passing:
                break
            
            if trial < args.num_trials:
                print(f"  [!] Tests failed. Starting Debate...")
                
                # Context for the debate: The Code + The Error
                context = f"### Code:\n{code}\n\n### Execution Output/Error:\n{feedback}"
                
                coordinator = DebateCoordinator(
                    question=f"Fix the python function `{item['entry_point']}` to pass the tests.",
                    context=context,
                    answer_key="Passing Unit Tests", # Abstract key
                    num_agents=args.num_agents,
                    num_rounds=args.num_rounds,
                    llm_kwargs={"model_name": args.model, "temperature": 0.2},
                    personas=agent_personas
                )

                debate_result = coordinator.run()
                
                # Extract consensus
                if isinstance(debate_result, dict):
                    consensus = debate_result.get("final_answer") or debate_result.get("consensus") or str(debate_result)
                else:
                    consensus = str(debate_result)

                print(f"  [+] Debate Consensus: {consensus[:100]}...")

                # 2. Next Attempt (Reflexion Strategy using Debate Consensus)
                # We pass the consensus as 'self_reflection'
                code = gen.func_impl(
                    func_sig=item["prompt"],
                    model=model,
                    strategy="reflexion",
                    prev_func_impl=code,
                    feedback=feedback,
                    self_reflection=consensus
                )

        # Log Result
        cur_result = {
            "task_id": task_id,
            "completion": code,
            "passed": is_passing,
            "trials_used": trial,
            "log": consensus if 'consensus' in locals() else ""
        }
        
        # Save progress
        write_jsonl(args.output_path, [cur_result], append=True)

if __name__ == "__main__":
    args = parse_args()
    run_mad_programming_trials(args)