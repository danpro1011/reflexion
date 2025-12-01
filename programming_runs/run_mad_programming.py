import argparse
import os
import sys
import json
from datetime import datetime, timezone

# Add parent dir to path to allow importing from sibling directories if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from debate_personas_programming import DebateCoordinator

# Import Programming specific modules
# Note: These imports depend on your specific reflexion.py structure in programming_runs
from reflexion import PyReflexionAgent, ReflexionStrategy
from utils import enumerate_resume, make_printv, write_jsonl
from executors import PyExecutor # Or the specific executor for your language

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./data/humaneval-py.jsonl")
    parser.add_argument("--output_path", type=str, default="./results/mad_python_results.jsonl")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--num_trials", type=int, default=3)
    parser.add_argument("--num_agents", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=2)
    parser.add_argument("--personas", nargs="+", default=["Senior Engineer", "QA Engineer"])
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()

def run_mad_programming_trials(args):
    # Load Dataset
    dataset = []
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

    print_v = make_printv(args.verbose)
    results = []

    print(f"[*] Starting MAD Programming Run with {len(dataset)} tasks.")
    print(f"[*] Personas: {args.personas}")

    for i, item in enumerate_resume(dataset, args.output_path):
        task_id = item['task_id']
        print(f"\n--- Task {task_id} ---")
        
        # Initialize Agent
        agent = PyReflexionAgent(
            model=args.model,
            max_iters=args.num_trials,
            strategy=ReflexionStrategy.REFLEXION
        )

        # Initial Attempt
        code = agent.init(item["prompt"], item["entry_point"])
        
        # Execution Loop
        for trial in range(1, args.num_trials + 1):
            # Execute Code
            executor = PyExecutor()
            # Note: You might need to adjust how tests are passed depending on your dataset format
            # HumanEval usually has 'test' field or 'example_test'
            tests = item.get("test", "") or item.get("example_test", "")
            
            execution_result = executor.execute(code, tests)
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

                # Inject consensus as a "Reflection"
                # We manually append to the agent's memory or use a specific method if available
                # Assuming standard ReflexionAgent has a way to add feedback
                # If not, we might need to override the internal prompt
                
                # Standard Reflexion usually generates self-reflection here.
                # We override it by passing our consensus as the 'feedback' for the next step
                code = agent.step(consensus) 

        # Log Result
        results.append({
            "task_id": task_id,
            "completion": code,
            "passed": is_passing,
            "trials_used": trial,
            "log": agent.scratchpad if hasattr(agent, 'scratchpad') else ""
        })
        
        # Save progress
        write_jsonl(args.output_path, results)

if __name__ == "__main__":
    args = parse_args()
    run_mad_programming_trials(args)