from llm import LocalLLM
model = LocalLLM(model_name="meta-llama/Llama-3.2-3B-Instruct")
print(model("def add(a, b):\n    "))