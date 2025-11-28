import re
import json

def extract_inline_question_texts(filepath):
    results = []
    pattern = r'^\s*Question:\s*(.*)$'   # grab everything after "Question:" on that line

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            match = re.match(pattern, line, flags=re.IGNORECASE)
            if match:
                results.append(match.group(1).strip())
    return results



if __name__ == "__main__":
    path = "wrong_q.txt"
    extracted = extract_inline_question_texts(path)

    with open("hard_questions.json", "w") as file:
        json.dump(extracted, file, indent=4)

    for i, text in enumerate(extracted, 1):
        print(f"--- Question {i} ---")
        print(text)
        print()
