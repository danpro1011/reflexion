Document dataset verification: include the snippet you just ran (joblib load + sample question) and mention path + schema. That becomes the “Data” section in the replication write-up.


(.venv) (base) danieldosti@Daniels-MacBook-Pro-429 reflexion % python - <<'PY'
import joblib
from pandas.core.indexes import numeric  # register pickled index types

path = "hotpotqa_runs/data/hotpot-qa-distractor-sample.joblib"
df = joblib.load(path)        # df is a DataFrame
print("total examples:", len(df))

first = df.iloc[0]
print("columns:", list(df.columns))
print("question:", first["question"])
print("answer:", first["answer"])
print("supporting_facts:", first.get("supporting_facts"))
PY

total examples: 100
columns: ['id', 'question', 'answer', 'type', 'level', 'supporting_facts', 'context']
question: VIVA Media AG changed it's name in 2004. What does their new acronym stand for?
answer: Gesellschaft mit beschränkter Haftung
supporting_facts: {'title': array(['VIVA Media', 'Gesellschaft mit beschränkter Haftung'],
      dtype=object), 'sent_id': array([0, 0], dtype=int32)}
(.venv) (base) danieldosti@Daniels-MacBook-Pro-429 reflexion % 

## Multi-agent debate runner

We now support a debate loop for HotPotQA. Multiple CoT-style agents propose answers, read peers' responses each round, and then majority-vote on the final round.

Example:

```bash
python hotpotqa_runs/run_debate_experiment.py \
  --output_dir hotpotqa_runs/experiments/debate-demo \
  --num_agents 3 \
  --num_rounds 3
```

Outputs:
- `summary.csv` – per-question final answers and correctness.
- `debate_attempts.jsonl` – round-by-round agent responses and final vote.
- `run_meta.json` – run configuration and aggregate stats.
