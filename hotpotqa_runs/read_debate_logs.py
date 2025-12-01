#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any


class DebateLogReader:
    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
        self.entries: List[Dict[str, Any]] = []
        self._load_entries()

    def _load_entries(self):
        with open(self.log_file, 'r') as f:
            for line in f:
                if line.strip():
                    self.entries.append(json.loads(line))

    def print_summary(self):
        total = len(self.entries)
        correct = sum(1 for e in self.entries if e.get('is_correct', False))
        accuracy = (correct / total * 100) if total > 0 else 0

        print("=" * 80)
        print("DEBATE EXPERIMENT SUMMARY")
        print("=" * 80)
        print(f"Total Questions: {total}")
        print(f"Correct Answers: {correct}")
        print(f"Incorrect Answers: {total - correct}")
        print(f"Accuracy: {accuracy:.2f}%")
        print("=" * 80)
        print()

    def print_entry(self, entry: Dict[str, Any], index: int):
        print(f"\n{'=' * 80}")
        print(f"QUESTION #{index + 1}")
        print(f"{'=' * 80}")
        print(f"ID: {entry.get('question_id', 'N/A')}")
        print(f"\nQuestion: {entry.get('question', 'N/A')}")
        print(f"Ground Truth: {entry.get('ground_truth', 'N/A')}")
        print(f"Final Answer: {entry.get('final_answer', 'N/A')}")
        print(f"Correct: {'✓ YES' if entry.get('is_correct') else '✗ NO'}")
        print(f"Majority Votes: {entry.get('majority_votes', 'N/A')}")

        # Print debate rounds
        rounds = entry.get('rounds', [])
        if rounds:
            print(f"\n{'-' * 80}")
            print("DEBATE ROUNDS")
            print(f"{'-' * 80}")

            for round_idx, round_data in enumerate(rounds, 1):
                print(f"\n--- Round {round_idx} ---")

                if isinstance(round_data, list):
                    for agent_idx, agent_response in enumerate(round_data):
                        if isinstance(agent_response, str):
                            print(f"\nAgent {agent_idx}:")
                            print(self._format_text(agent_response, indent=2))
                        elif isinstance(agent_response, list):
                            # Multiple exchanges in this round
                            for exchange_idx, exchange in enumerate(agent_response):
                                print(f"\nAgent {agent_idx} - Exchange {exchange_idx + 1}:")
                                print(self._format_text(exchange, indent=2))

        print(f"\n{'=' * 80}\n")

    def _format_text(self, text: str, indent: int = 0, max_width: int = 76) -> str:
        import textwrap

        indent_str = " " * indent
        wrapper = textwrap.TextWrapper(
            width=max_width,
            initial_indent=indent_str,
            subsequent_indent=indent_str,
            break_long_words=False,
            break_on_hyphens=False
        )

        paragraphs = text.split('\n\n')
        formatted_paragraphs = []

        for para in paragraphs:
            para = para.replace('\n', ' ')
            formatted_paragraphs.append(wrapper.fill(para))

        return '\n\n'.join(formatted_paragraphs)

    def print_entries(self, limit: int = None, correct_only: bool = False,
                     incorrect_only: bool = False):
        entries_to_print = self.entries

        if correct_only:
            entries_to_print = [e for e in entries_to_print if e.get('is_correct', False)]
        elif incorrect_only:
            entries_to_print = [e for e in entries_to_print if not e.get('is_correct', False)]

        if limit:
            entries_to_print = entries_to_print[:limit]

        for idx, entry in enumerate(entries_to_print):
            self.print_entry(entry, idx)


def main():
    parser = argparse.ArgumentParser(
        description="Read and display debate experiment logs in human-readable format"
    )
    parser.add_argument(
        "log_file",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
    )
    parser.add_argument(
        "--correct-only",
        action="store_true",
    )
    parser.add_argument(
        "--incorrect-only",
        action="store_true",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
    )

    args = parser.parse_args()

    if args.correct_only and args.incorrect_only:
        parser.error("Cannot use both --correct-only and --incorrect-only")

    reader = DebateLogReader(args.log_file)
    reader.print_summary()

    if not args.summary_only:
        reader.print_entries(
            limit=args.limit,
            correct_only=args.correct_only,
            incorrect_only=args.incorrect_only
        )


if __name__ == "__main__":
    main()
