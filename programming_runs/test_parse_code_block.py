import re
from generators.parse import parse_code_block
from generators.generator_utils import normalize_code_blocks



RAW_MODEL_OUTPUT = (
    "# Write the body of this function only.\n"
    "def strlen(string: str) -> int:\n"
    '    """ Return length of given string\n'
    "    >>> strlen('')\n"
    "    0\n"
    "    >>> strlen('abc')\n"
    "    3\n"
    '    """\n'
    "\nUse a Python code block to write your response. For example:\n"
    "```python\nprint('Hello world!')\n``` ```\n```python\ndef strlen(string: str) -> int:\n    return len(string)\n``` ``` ``` ``` ``` ```\ndef strlen(string: str) -> int:\n    return len(string)\n``` ``` ``` ``` ``` ```\ndef strlen(string: str) -> int:\n    return len(string)\n``` ``` ``` ``` ``` ```\ndef strlen(string: str) -> int:\n    return len(string)\n``` ``` ``` ``` ``` ```\ndef strlen(string: str) -> int:\n    return len(string)\n``` ``` ``` ``` ``` ```\ndef strlen(string: str) -> int:\n    return len(string)\n``` ``` ``` ``` ``` ```\ndef strlen(string: str) -> int:\n    return len(string)\n``` ``` ``` ``` ``` ```\ndef strlen(string: str) -> int:\n    return len(string)\n``` ``` ``` ``` ``` ```\ndef strlen(string: str) -> int:\n    return len(string)\n``` ``` ``` ``` ``` ```\ndef strlen(string: str) -> int:\n    return len(string)\n``` ``` ``` ``` ``` ```\ndef strlen(string: str) -> int:\n    return len(string)\n``` ``` ``` ``` ``` ```\ndef strlen(string: str) -> int:\n    return len(string)\n``` ``` ``` ``` ``` ```\ndef strlen(string: str) -> int:\n    return len(string)\n``` ``` ``` ``` ``` ```\ndef strlen(string: str) -> int:\n    return len(string)\n``` ``` ``` ``` ``` ```\ndef strlen(string: str) -> int:\n    return len(string)\n``` ``` ``` ``` ``` ```\ndef strlen(string: str) -> int:\n    return len(string)\n``` ``` ``` ``` ``` ```\ndef strlen(string: str) -> int:\n    return len(string)\n``` ``` ``` ``` ``` ```\ndef strlen(string: str) -> int:\n    return len(string)\n``` ``` ``` ``` ``` ```\ndef strlen(string: str) -> int:\n    return len(string)\n``` ``` ``` ``` ``` ```\ndef strlen(string: str) -> int:\n    return len(string)\n``` ``` ``` ``` ``` ```\ndef strlen(string: str) -> int:\n    return len(string)\n``` ``` ``` ``` ``` ```\ndef strlen(string: str) -> int:\n    return len(string)\n``` ``` ``` ``` ``` ```\ndef strlen(string: str) -> int:\n    return len(string)\n``` ``` ``` ``` ``` ```\ndef strlen(string: str) -> int"
)

print("=== RAW MODEL OUTPUT ===")
print(RAW_MODEL_OUTPUT)
print("\n=== NORMALIZED OUTPUT ===")
normalized = normalize_code_blocks(RAW_MODEL_OUTPUT)
print(normalized)

print("\n=== PARSED CODE BLOCK ===")
parsed_code = parse_code_block(normalized, "python")
print(parsed_code)

# Optionally, check if the parsed code matches what you expect
expected_code = "def strlen(string: str) -> int:\n    return len(string)"
if parsed_code and expected_code in parsed_code:
    print("\nSUCCESS: Parsed code matches expected output.")
else:
    print("\nFAILURE: Parsed code does not match expected output.")