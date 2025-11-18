from utils import enumerate_resume, make_printv, write_jsonl
from executors import executor_factory
from generators import generator_factory, model_factory

from typing import List
import logging
import os
api_key = os.environ.get("OPENAI_API_KEY")

# Configure logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("simple_run_leetcode_hard.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

SIMPLE_COMPLETION_INSTRUCTION = "# Write the body of this function only."
SIMPLE_CHAT_INSTRUCTION = "You are a programming assistant. You will be given a function signature and docstring. You should fill in the following text of the missing function body. For example, the first line of the completion should have 4 spaces for the indendation so that it fits syntactically with the preceding signature."

def run_simple(
        dataset: List[dict],
        model_name: str,
        language: str,
        pass_at_k: int,
        log_path: str,
        verbose: bool,
        is_leetcode: bool = False
    ) -> None:
    exe = executor_factory(language, is_leet=is_leetcode)
    gen = generator_factory(language)
    model = model_factory(model_name)

    print_v = make_printv(verbose)
    
    num_items = len(dataset)
    num_success = 0
    total_api_calls = 0
    
    for i, item in enumerate_resume(dataset, log_path):
        cur_pass = 0
        is_solved = False
        cur_func_impl = ""
        print(f"Processing example {i+1}/{num_items}")
        logger.info(f"Processing example {i+1}/{num_items}")
        
        while cur_pass < pass_at_k:
            cur_func_impl = gen.func_impl(item["prompt"], model, "simple")
            total_api_calls += 1
            logger.info(f"API call #{total_api_calls} for example {i+1}")
            logger.info(f"Prompt: {item['prompt'][:100]}...")

            # Log token usage if available
            if hasattr(model, 'last_token_usage'):
                logger.info(f"Token usage: {model.last_token_usage}")

            print(f"cur_func_impl: {cur_func_impl} (type: {type(cur_func_impl)})")
            assert isinstance(cur_func_impl, str)
            print(f"\nGenerated code for example {i+1}:\n{cur_func_impl}\n")

            # Evaluate solution correctness (HumanEval style)
            solution_pass = exe.evaluate(item["entry_point"], cur_func_impl, item["test"], timeout=20 if is_leetcode else 10)
            logger.info(f"Solution pass: {solution_pass}")
            print(f"Solution passed: {solution_pass}")

            # If you have unit tests as a list, run execute for per-test results
            if "test" in item:
                result = exe.execute(cur_func_impl, item["test"], timeout=20 if is_leetcode else 10)
                unit_test_pass = all(result.state)
                logger.info(f"Unit tests passed: {unit_test_pass} ({sum(result.state)}/{len(result.state)})")
                print(f"Unit tests passed: {unit_test_pass} ({sum(result.state)}/{len(result.state)})")
            else:
                unit_test_pass = solution_pass  # fallback for HumanEval

            # TP/FP/FN/TN classification
            if unit_test_pass and solution_pass:
                result_type = "TP"
            elif not unit_test_pass and solution_pass:
                result_type = "FP"
            elif unit_test_pass and not solution_pass:
                result_type = "FN"
            else:
                result_type = "TN"
            logger.info(f"Evaluation result for example {i+1}: {result_type}")

            if solution_pass:
                is_solved = True
                num_success += 1
                break
            cur_pass += 1
            
        item["solution"] = cur_func_impl
        item["is_solved"] = is_solved
        write_jsonl(log_path, [item], append=True)
        accuracy = round(num_success/(i+1), 2)
        logger.info(f'Completed {i+1}/{num_items}: acc = {accuracy}, Total API calls: {total_api_calls}')
        print_v(f'completed {i+1}/{num_items}: acc = {accuracy}')
    
    # Final summary
    logger.info(f"=== FINAL SUMMARY ===")
    logger.info(f"Total examples: {num_items}")
    logger.info(f"Successful: {num_success}")
    logger.info(f"Final accuracy: {round(num_success/num_items, 3)}")
    logger.info(f"Total API calls: {total_api_calls}")
    logger.info(f"Average API calls per example: {round(total_api_calls/num_items, 2)}")