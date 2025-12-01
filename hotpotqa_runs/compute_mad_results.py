import json
import os
import sys
from datetime import datetime
from pathlib import Path

def parse_debate_log(log_file_path: str) -> list:
    """
    Parse the debate_outputs.txt file and extract structured debate records.
    Returns a list of debate dictionaries.
    """
    debates = []
    current_debate = {}
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Start of new debate record
        if line.startswith('Timestamp:'):
            if current_debate:  # Save previous debate
                debates.append(current_debate)
            current_debate = {
                'timestamp': line.replace('Timestamp: ', ''),
                'metadata': {},
                'scratchpad': '',
                'debate_output': ''
            }
        
        # Parse metadata
        elif line.startswith('Trial:'):
            parts = line.split(' | ')
            for part in parts:
                if 'Trial:' in part:
                    current_debate['metadata']['trial'] = int(part.split(': ')[1])
                elif 'Agent index:' in part:
                    current_debate['metadata']['agent_index'] = int(part.split(': ')[1])
                elif 'Question ID:' in part:
                    current_debate['metadata']['question_id'] = part.split(': ')[1]
        
        elif line.startswith('Personas:'):
            current_debate['metadata']['personas'] = line.replace('Personas: ', '').split(', ')
        
        elif line.startswith('Question:'):
            current_debate['metadata']['question'] = line.replace('Question: ', '')
        
        elif line.startswith('Ground truth:'):
            current_debate['metadata']['ground_truth'] = line.replace('Ground truth: ', '')
        
        # Capture scratchpad and debate output sections
        elif 'Scratchpad (failed attempt):' in line:
            scratchpad_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('-'):
                scratchpad_lines.append(lines[i].rstrip())
                i += 1
            current_debate['scratchpad'] = ''.join(scratchpad_lines).strip()
            continue
        
        elif 'Debate output / consensus:' in line:
            debate_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('='):
                debate_lines.append(lines[i].rstrip())
                i += 1
            current_debate['debate_output'] = ''.join(debate_lines).strip()
            continue
        
        i += 1
    
    # Don't forget last debate
    if current_debate:
        debates.append(current_debate)
    
    return debates


def parse_attempts_jsonl(jsonl_file_path: str) -> list:
    """
    Parse the attempts.jsonl file and return list of attempt records.
    """
    attempts = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                attempts.append(json.loads(line))
    return attempts


def create_consolidated_report(
    output_dir: str,
    attempts: list,
    debates: list
) -> dict:
    """
    Create a consolidated report combining attempts and debate data.
    """
    report = {
        'summary': {
            'total_questions': len(set(a['question_id'] for a in attempts)),
            'total_trials': max([a['trial'] for a in attempts], default=0),
            'total_correct': 0,
            'accuracy': 0,
            'total_debates': len(debates),
            'correct_per_trial': {}
        },
        'attempts_by_question': {},
        'debates_by_question': {}
    }
    
    # Track correct answers for each question
    correct_answers = {}
    
    # Group attempts by question_id
    for attempt in attempts:
        q_id = attempt['question_id']
        
        # Track if the question was answered correctly in any trial
        if q_id not in correct_answers:
            correct_answers[q_id] = False
        
        if attempt['is_correct']:
            correct_answers[q_id] = True
        
        if q_id not in report['attempts_by_question']:
            report['attempts_by_question'][q_id] = {
                'question': attempt['question'],
                'ground_truth': attempt['ground_truth'],
                'trials': []
            }
        report['attempts_by_question'][q_id]['trials'].append({
            'trial': attempt['trial'],
            'prediction': attempt['prediction'],
            'is_correct': attempt['is_correct'],
            'reflections': attempt.get('reflections', [])
        })
    
    # Calculate total correct answers and accuracy
    report['summary']['total_correct'] = sum(correct_answers.values())
    report['summary']['accuracy'] = round(
        (report['summary']['total_correct'] / report['summary']['total_questions']) * 100, 2
    ) if report['summary']['total_questions'] > 0 else 0
    
    # Track correct answers per trial
    for trial in range(1, report['summary']['total_trials'] + 1):
        correct_count = sum(
            1 for attempt in attempts if attempt['trial'] == trial and attempt['is_correct']
        )
        report['summary']['correct_per_trial'][trial] = correct_count
    
    # Group debates by question_id
    for debate in debates:
        q_id = debate['metadata'].get('question_id')
        if q_id:
            if q_id not in report['debates_by_question']:
                report['debates_by_question'][q_id] = []
            report['debates_by_question'][q_id].append({
                'trial': debate['metadata'].get('trial'),
                'agent_index': debate['metadata'].get('agent_index'),
                'personas': debate['metadata'].get('personas'),
                'timestamp': debate['timestamp'],
                'scratchpad': debate['scratchpad'][:200] + '...',  # First 200 chars
                'debate_summary': debate['debate_output'][:300] + '...'  # First 300 chars
            })
    
    return report


def main(result_dir_name: str = None):
    # Default to 'mad_rich_personas' if not provided
    if not result_dir_name:
        result_dir_name = 'mad_rich_personas'
    
    # Build path to the results directory under persona_results
    base_path = os.path.join('hotpotqa_runs', 'persona_results', result_dir_name)
    
    # Ensure directory exists
    if not os.path.exists(base_path):
        print(f"[!] Directory not found: {base_path}")
        print(f"[!] Creating it...")
        os.makedirs(base_path, exist_ok=True)
    
    # Parse input files
    attempts_file = os.path.join(base_path, "attempts.jsonl")
    debate_log_file = os.path.join(base_path, "debate_outputs.txt")
    
    attempts = []
    debates = []
    
    if os.path.exists(attempts_file):
        attempts = parse_attempts_jsonl(attempts_file)
        print(f"[+] Loaded {len(attempts)} attempt records from {attempts_file}")
    else:
        print(f"[!] attempts.jsonl not found at {attempts_file}")
    
    if os.path.exists(debate_log_file):
        debates = parse_debate_log(debate_log_file)
        print(f"[+] Loaded {len(debates)} debate records from {debate_log_file}")
    else:
        print(f"[!] debate_outputs.txt not found at {debate_log_file}")
    
    if not attempts and not debates:
        print("[!] No data to process. Exiting.")
        return
    
    # Create consolidated report
    report = create_consolidated_report(base_path, attempts, debates)
    
    # Write consolidated report
    report_path = os.path.join(base_path, "consolidated_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[+] Wrote consolidated report to {report_path}")
    
    # Write summary stats
    summary_path = os.path.join(base_path, "summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(report['summary'], f, indent=2)
    print(f"[+] Wrote summary to {summary_path}")
    
    print(f"\n=== Summary ===")
    print(f"Total Questions: {report['summary']['total_questions']}")
    print(f"Total Trials: {report['summary']['total_trials']}")
    print(f"Accuracy: {report['summary']['accuracy']}%")
    print(f"Total Debates: {report['summary']['total_debates']}")


if __name__ == '__main__':
    # Get directory name from command line argument, or use default
    result_dir = sys.argv[1] if len(sys.argv) > 1 else 'mad_rich_personas'
    main(result_dir)