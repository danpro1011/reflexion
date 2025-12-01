#!/usr/bin/env python3
"""
Replication script for HotPotQA experiments
Usage: python replicate_hotpotqa.py --strategy [base|reflexion] --trials 5
"""
import os
import sys
import argparse
import joblib
from datetime import datetime

from util import summarize_react_trial, log_react_trial
from agents import ReactReflectAgent, ReactAgent, ReflexionStrategy

def main():
    parser = argparse.ArgumentParser(description='Replicate HotPotQA experiments')
    parser.add_argument('--strategy', type=str, default='reflexion',
                       choices=['base', 'reflexion', 'last_trial', 'last_trial_and_reflexion'],
                       help='Reflexion strategy to use')
    parser.add_argument('--trials', type=int, default=5,
                       help='Number of trials (default: 5)')
    parser.add_argument('--questions', type=int, default=100,
                       help='Number of questions to test (default: 100, max: 100)')
    parser.add_argument('--output-dir', type=str, default='./replication_results',
                       help='Directory to save results')

    args = parser.parse_args()

    strategy_map = {
        'base': ReflexionStrategy.NONE,
        'reflexion': ReflexionStrategy.REFLEXION,
        'last_trial': ReflexionStrategy.LAST_ATTEMPT,
        'last_trial_and_reflexion': ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION
    }
    strategy = strategy_map[args.strategy]

    print("=" * 80)
    print("HotPotQA Replication Experiment")
    print("=" * 80)
    print(f"Strategy: {args.strategy}")
    print(f"Trials: {args.trials}")
    print(f"Questions: {args.questions}")
    print("=" * 80)

    print("\nLoading HotPotQA data...")
    hotpot = joblib.load('data/hotpot-qa-distractor-sample.joblib').reset_index(drop=True)

    # Filter to hard questions only (as in original paper)
    hard_ones = hotpot[hotpot["level"] == "hard"]
    print(f"Found {len(hard_ones)} hard questions")

    if args.questions < len(hard_ones):
        hard_ones = hard_ones.head(args.questions)
        print(f"Using first {args.questions} questions")

    print("\nInitializing agents...")
    agent_cls = ReactReflectAgent if strategy != ReflexionStrategy.NONE else ReactAgent

    try:
        from langchain_community.docstore.wikipedia import Wikipedia
    except ImportError:
        from langchain import Wikipedia

    try:
        from langchain_classic.agents.react.base import DocstoreExplorer
    except ImportError:
        try:
            from langchain.agents.react.base import DocstoreExplorer
        except ImportError:
            from langchain_community.agent_toolkits.base import DocstoreExplorer

    from environment import QAEnv

    # Create agents with environments
    agents = []
    for _, row in hard_ones.iterrows():
        env = QAEnv(
            question=row['question'],
            key=row['answer'],
            max_steps=6,
            explorer=DocstoreExplorer(Wikipedia())
        )
        agent = agent_cls(question=row['question'], env=env)
        agents.append(agent)

    print(f"Created {len(agents)} agents")

    # Run trials
    log = ''
    trial_results = []

    for trial in range(args.trials):
        print(f"\n{'=' * 80}")
        print(f"TRIAL {trial + 1}/{args.trials}")
        print('=' * 80)

        # Run agents that haven't succeeded yet
        agents_to_run = [a for a in agents if not a.is_correct()]
        print(f"Running {len(agents_to_run)} remaining agents...")

        for i, agent in enumerate(agents_to_run):
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{len(agents_to_run)}")

            if strategy != ReflexionStrategy.NONE:
                agent.run(reflect_strategy=strategy)
            else:
                agent.run()

        # Log results
        log += log_react_trial(agents, trial + 1)
        correct, incorrect, halted = summarize_react_trial(agents)

        accuracy = len(correct) / len(agents)
        trial_results.append({
            'trial': trial + 1,
            'correct': len(correct),
            'incorrect': len(incorrect),
            'halted': len(halted),
            'accuracy': accuracy
        })

        print(f"\nTrial {trial + 1} Results:")
        print(f"  Correct: {len(correct)}")
        print(f"  Incorrect: {len(incorrect)}")
        print(f"  Halted: {len(halted)}")
        print(f"  Accuracy: {accuracy:.2%}")

        # Early stopping if all correct
        if len(correct) == len(agents):
            print("\nAll questions answered correctly! Stopping early.")
            break

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        args.output_dir,
        f'{args.strategy}_{args.questions}q_{args.trials}t_{timestamp}.txt'
    )

    with open(output_file, 'w') as f:
        f.write(log)

    print(f"\n{'=' * 80}")
    print("FINAL RESULTS")
    print('=' * 80)
    print(f"\nResults saved to: {output_file}")
    print("\nTrial progression:")
    for result in trial_results:
        print(f"  Trial {result['trial']}: {result['accuracy']:.2%} "
              f"(Correct: {result['correct']}, Incorrect: {result['incorrect']}, "
              f"Halted: {result['halted']})")

    print(f"\nFinal Accuracy: {trial_results[-1]['accuracy']:.2%}")
    print(f"Improvement: {trial_results[-1]['accuracy'] - trial_results[0]['accuracy']:.2%}")

if __name__ == "__main__":
    main()
