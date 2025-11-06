from llm import LocalLLM
model = LocalLLM(model_name="meta-llama/Llama-3.2-3B-Instruct")
print(model("""
            Prompt to model: '# Write the body of this function only.\ndef strlen(string: str) -> int:\n    n    Return length of given string\n    >>> strlen(\'\')\n    0\n    >>> strlen(\'abc\')\n    3\n    n\n\nUse a Python code block to write your response. For example:\n```python\nprint(\'Hello world!\')\n```'
            """
            ))