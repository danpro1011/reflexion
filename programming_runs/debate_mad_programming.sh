python3 /Users/tinachen/Documents/reflexion/programming_runs/debate_mad_programming.py \
    --dataset_path ./programming_runs/benchmarks/humaneval-py.jsonl \
    --output_path ./programming_runs/mad_programming/mad_python_results.jsonl \
    --model gpt-3.5-turbo \
    --num_trials 2 \
    --num_agents 3 \
    --num_rounds 2 \
    --personas "Senior Engineer" "QA Engineer" "Code Reviewer"\
    --verbose