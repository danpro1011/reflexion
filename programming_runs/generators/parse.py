import re
from typing import Optional, List

# TODO: DEBUG
def parse_code_block(string: str, lang: str, verbose: bool = True) -> Optional[str]:
    """
    Improved parsing that handles multiple edge cases:
    - Code blocks with/without language specifier
    - Code blocks with/without newlines after backticks
    - Inline code mixed with explanations
    - Multiple code blocks (takes the first valid one with a function definition)
    """

    # Try multiple patterns in order of specificity
    pattern = fr"```{lang}\s*\n(.*?)\n```"
    matches = re.findall(pattern, string, re.DOTALL)
    for match in matches:
        result = match.strip()
        if result and len(result) > 10 and 'def ' in result:
            return result

    pattern = r"```\s*\n(.*?)\n```"
    matches = re.findall(pattern, string, re.DOTALL)
    for match in matches:
        result = match.strip()
        if result and len(result) > 10 and 'def ' in result:
            return result

    pattern = fr"```{lang}\s+(.*?)```"
    matches = re.findall(pattern, string, re.DOTALL)
    for match in matches:
        result = match.strip()
        if result and len(result) > 10 and 'def ' in result:
            return result

    pattern = r"```\s+(.*?)```"
    matches = re.findall(pattern, string, re.DOTALL)
    for match in matches:
        result = match.strip()
        if result and len(result) > 10 and 'def ' in result:
            return result

    # Fallback to parse_first_func
    result = parse_first_func(string, lang)

    # If parsing failed and verbose mode, print diagnostic info
    if result is None and verbose:
        print(f"\n[PARSE DEBUG] Failed to parse. Model output (first 500 chars):")
        print(repr(string[:500]))
        print(f"[PARSE DEBUG] Model output length: {len(string)} chars")
        print(f"[PARSE DEBUG] Contains 'def ': {'def ' in string}")
        print(f"[PARSE DEBUG] Contains code blocks: {'```' in string}\n")

    return result

def parse_first_func(code: str, lang: str) -> Optional[str]:
    """
    Improved function extraction that handles:
    - Functions without explicit returns
    - Functions with multiple empty lines
    - Functions with nested definitions
    - Various indentation levels
    - Comments and test code after function
    """
    assert lang == "python", "Only python is supported for now. TODO: Rust"

    code_lines = code.split("\n")
    def_i = -1
    last_i = 0
    base_indent = None

    for i, line in enumerate(code_lines):
        if def_i == -1:
            if line.strip().startswith("def "):
                def_i = i
                base_indent = len(line) - len(line.lstrip())
        else:
            stripped = line.strip()

            if not stripped or stripped.startswith("#"):
                continue

            current_indent = len(line) - len(line.lstrip())

            if current_indent <= base_indent and stripped:
                if stripped.startswith("def ") or stripped.startswith("class "):
                    last_i = i - 1
                    break
                elif stripped.startswith("print(") or stripped.startswith("assert ") or stripped.startswith("if __name__"):
                    last_i = i - 1
                    break

    if def_i != -1 and last_i == 0:
        last_i = len(code_lines) - 1

        while last_i > def_i and not code_lines[last_i].strip():
            last_i -= 1

    if def_i == -1:
        return None

    func_code = "\n".join(code_lines[def_i:last_i+1])

    func_code = func_code.rstrip("[/PYTHON]")
    func_code = func_code.rstrip("```")
    func_code = func_code.strip()

    # Check if function has actual body (more than just signature and docstring)
    lines = func_code.split('\n')
    if len(lines) <= 1:
        return None  # Just signature, no body

    # Count non-empty, non-comment, non-docstring lines after signature
    in_docstring = False
    body_lines = 0
    for i, line in enumerate(lines[1:]):  # Skip first line (def ...)
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        if '"""' in stripped or "'''" in stripped:
            in_docstring = not in_docstring
            continue
        if not in_docstring:
            body_lines += 1

    # Must have at least one line of actual code
    if body_lines == 0:
        return None

    return func_code if func_code else None


def add_code_block(string: str, lang: str) -> str:
    return f"```{lang}\n{string}\n```"


if __name__ == "__main__":
    CODE = """
aldaas
sub_parser = parser.add_subparsers().add_parser("frf
a")

def my_wonderful_func():
    def useless_helper():
        return 1
    if 1:
        return 1
    else:
        return (
            1,
            2,
        )

sadsadsa
2023-08-04dsa
dsa

def bleh():
    return aaa
"""
    print(parse_code_block(CODE, "python"))
    CODE = """def total_match(lst1: List[str], lst2: List[str]) -> List[str]:
    \"\"\"
    Write a function that accepts two lists of strings and returns the list that has
    total number of chars in the all strings of the list less than the other list.
    
    if the two lists have the same number of chars, return the first list.
    
    Examples
    >>> total_match([], [])
    []
    >>> total_match(['hi', 'admin'], ['hI', 'Hi'])
    ['hI', 'Hi']
    >>> total_match(['hi', 'admin'], ['hi', 'hi', 'admin', 'project'])
    ['hi', 'admin']
    >>> total_match(['hi', 'admin'], ['hI', 'hi', 'hi'])
    ['hI', 'hi', 'hi']
    >>> total_match(['4'], ['1', '2', '3', '4', '5'])
    ['4']
    \"\"\"
    total_chars_lst1 = sum(len(word) for word in lst1)
    total_chars_lst2 = sum(len(word) for word in lst2)
    
    if total_chars_lst1 < total_chars_lst2:
        return lst1
    elif total_chars_lst1 > total_chars_lst2:
        return lst2
    else:
        return lst1
    """
    print(parse_code_block(CODE, "python"))
