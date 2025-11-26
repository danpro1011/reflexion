import os

from utils import enumerate_resume, make_printv, write_jsonl, resume_success_count
from executors import executor_factory
from generators import generator_factory, model_factory
from generators.model import get_total_tokens_used

from typing import List
# TODO: Put in some logging
import logging

# Configure logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("reflexion_run.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_reflexion(
    dataset: List[dict],
    model_name: str,
    language: str,
    max_iters: int,
    pass_at_k: int,
    log_path: str,
    verbose: bool,
    is_leetcode: bool = False,
    max_examples: int | None = None
) -> None:
    exe = executor_factory(language, is_leet=is_leetcode)
    gen = generator_factory(language)
    model = model_factory(model_name)

    print_v = make_printv(verbose)

    num_skips = 0 # TRACKS HOW MANY WERE FAILED PARSES AND SKIPPED
    num_items = len(dataset)
    num_success = resume_success_count(dataset)
    total_api_calls = 0
    processed_this_run = 0

    failure_log_path = os.path.join(os.path.dirname(log_path), "failed_examples.log")

    for i, item in enumerate_resume(dataset, log_path):
        if max_examples is not None and processed_this_run >= max_examples:
            logger.info(f"Reached max_examples={max_examples}; stopping early.")
            break

        cur_pass = 0
        is_solved = False
        reflections = []
        implementations = []
        test_feedback = []
        cur_func_impl = ""
        while cur_pass < pass_at_k and not is_solved:
            if is_leetcode:
                tests_i = item['visible_tests']
            else:
                tests_i = gen.internal_tests(item["prompt"], model, 1)

            # first attempt
            cur_func_impl = gen.func_impl(item["prompt"], model, "simple")
            total_api_calls += 1
            implementations.append(cur_func_impl)
            logger.info(f"API call #{total_api_calls} for example {i+1}")
            logger.info(f"Prompt: {item['prompt'][:100]}...")
            print("\n--- DEBUG: func_impl output ---")
            print("cur_func_impl:", repr(cur_func_impl), "type:", type(cur_func_impl))
            if not isinstance(cur_func_impl, str) or not cur_func_impl.strip():
                print("WARNING: Failed to parse function implementation. Raw output:")
                print(cur_func_impl)
                cur_pass += 1
                num_skips += 1
                continue

            # Run internal unit tests
            result = exe.execute(cur_func_impl, tests_i)
            unit_test_pass = all(result.state)
            logger.info(f"Unit tests passed: {unit_test_pass} ({sum(result.state)}/{len(result.state)})")
            print(f"Unit tests passed: {unit_test_pass} ({sum(result.state)}/{len(result.state)})")
            test_feedback.append(result.feedback)

            # Evaluate on real/hidden tests (solution correctness)
            solution_pass = exe.evaluate(item["entry_point"], cur_func_impl, item["test"], timeout=10)
            logger.info(f"Solution pass: {solution_pass}")
            print(f"Solution passed: {solution_pass}")

            # TP/FP/FN/TN classification
            if unit_test_pass and solution_pass:
                result_type = "TP"
            elif not unit_test_pass and solution_pass:
                result_type = "FN"
            elif unit_test_pass and not solution_pass:
                result_type = "FP"
            else:
                result_type = "TN"
            logger.info(f"Evaluation result for example {i+1}: {result_type}")

            if solution_pass:
                is_solved = True
                num_success += 1
                break

            # Reflexion loop
            cur_iter = 1
            cur_feedback = result.feedback
            while cur_iter < max_iters:
                reflection = gen.self_reflection(cur_func_impl, cur_feedback, model)
                reflections.append(reflection)

                cur_func_impl = gen.func_impl(
                    func_sig=item["prompt"],
                    model=model,
                    strategy="reflexion",
                    prev_func_impl=cur_func_impl,
                    feedback=cur_feedback,
                    self_reflection=reflection,
                )
                total_api_calls += 1
                implementations.append(cur_func_impl)
                logger.info(f"API call #{total_api_calls} for example {i+1} (reflexion iter {cur_iter})")
                print("cur_func_impl:", repr(cur_func_impl), "type:", type(cur_func_impl))
                if not isinstance(cur_func_impl, str) or not cur_func_impl.strip():
                    print(f"WARNING: Failed to parse function implementation on iteration {cur_iter}.")
                    break

                result = exe.execute(cur_func_impl, tests_i)
                unit_test_pass = all(result.state)
                logger.info(f"Unit tests passed: {unit_test_pass} ({sum(result.state)}/{len(result.state)})")
                print(f"Unit tests passed: {unit_test_pass} ({sum(result.state)}/{len(result.state)})")
                test_feedback.append(result.feedback)

                solution_pass = exe.evaluate(item["entry_point"], cur_func_impl, item["test"], timeout=10)
                logger.info(f"Solution pass: {solution_pass}")
                print(f"Solution passed: {solution_pass}")

                # TP/FP/FN/TN classification
                if unit_test_pass and solution_pass:
                    result_type = "TP"
                elif not unit_test_pass and solution_pass:
                    result_type = "FN"
                elif unit_test_pass and not solution_pass:
                    result_type = "FP"
                else:
                    result_type = "TN"
                logger.info(f"Evaluation result for example {i+1}: {result_type}")

                if solution_pass:
                    item["solution"] = cur_func_impl
                    is_solved = True
                    num_success += 1
                    break

                cur_feedback = result.feedback
                cur_iter += 1
            cur_pass += 1

        item["is_solved"] = is_solved
        item["reflections"] = reflections
        item["implementations"] = implementations
        item["test_feedback"] = test_feedback
        item["solution"] = cur_func_impl
        write_jsonl(log_path, [item], append=True)
        processed_this_run += 1

        # accuracy = round(num_success/(i+1), 2)
        # logger.info(f'Completed {i+1}/{num_items}: acc = {accuracy}, Total API calls: {total_api_calls}')
        # print_v(f'completed {i+1}/{num_items}: acc = {accuracy}')

    # Final summary
    # THERE APPEARS TO BE A BUG HERE FOR SUCCESS COUNTING; USE COMPUTER OUTPUT INFO 

    # logger.info(f"==============================================================")
    # logger.info(f"======================== FINAL SUMMARY =======================")
    # logger.info(f"==============================================================")
    # denom = max_examples if max_examples else num_items
    # logger.info(f"Total examples: {num_items}")
    # logger.info(f"Number actually ran: {denom}")
    # logger.info(f"Successful: {num_success}")
    # logger.info(f"Total API calls: {total_api_calls}")
    # logger.info(f"Final accuracy: {round(num_success/denom, 3)}")
    # logger.info(f"Average API calls per example: {round(total_api_calls/denom, 2)}")
    # logger.info(f"Total skipped examples: {num_skips}")
    # logger.info(f"Total tokens used (OpenAI): {get_total_tokens_used()}")
