import os
from multiprocessing import Pool
import json
import joblib
from util import summarize_react_trial, log_react_trial, save_agents
from agents import ReactReflectAgent, ReactAgent, ReflexionStrategy, ReactDebateReflectAgent

def run_single_agent_trial(agent):
    agent.run()
    return agent

#TODO: No idea if this actually works Test this on some dummy agent before losing 5$ again
def run_test_multi_process(num_trails = 5, num_debators = 2):
    hotpot = joblib.load('data/hotpot-qa-distractor-sample.joblib').reset_index(drop = True)
    agents = [ReactDebateReflectAgent(question = row['question'], key= row['answer'], num_debators=num_debators) for _, row in hotpot.iterrows()]

    log = ''
    for trial in num_trails:
        with Pool() as pool:
            new_agents = pool.map(run_single_agent_trial, agents)
            log += log_react_trial(new_agents, trial)
            
    root  = 'root/'
    dir_path = os.path.join('root/', 'ReAct', "Debate_v3")
    os.makedirs(dir_path, exist_ok=True)

    with open(os.path.join(dir_path, f'{len(agents)}_questions_{trial}_trials_{num_debators}_debators.txt'), 'w') as f:
        f.write(log)

def run_test_single_process(hard_only = True):
    num_debators = 2
    hotpot = joblib.load('data/hotpot-qa-distractor-sample.joblib').reset_index(drop = True)

    if hard_only:
        with open("hard_questions.json", "r") as file:
            hard_questions = json.load(file)

        hotpot = hotpot[hotpot["question"].isin(hard_questions)]
    
    agents = [ReactDebateReflectAgent(question = row['question'], key= row['answer'], num_debators=num_debators) for _, row in hotpot.iterrows()]
    
    trial = 0
    log = ''
    for i in range(5):
        for agent in [a for a in agents if not a.is_correct()]:
            agent.run()
        trial += 1
        log += log_react_trial(agents, trial)
        correct, incorrect, halted = summarize_react_trial(agents)
        print(f'Finished Trial {trial}, Correct: {len(correct)}, Incorrect: {len(incorrect)}, Halted: {len(halted)}')
        
        
    root  = 'root/'
    dir_path = os.path.join('root/', 'ReAct', "Debate_v3")
    os.makedirs(dir_path, exist_ok=True)

    with open(os.path.join(dir_path, f'{len(agents)}_questions_{trial}_trials_{num_debators}_debators.txt'), 'w') as f:
        f.write(log)
        


if __name__ == "__main__":
    run_test_single_process(hard_only=True)
    # run_test_multi_process()

        