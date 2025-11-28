import os
import joblib
from util import summarize_react_trial, log_react_trial, save_agents
from agents import ReactReflectAgent, ReactAgent, ReflexionStrategy, ReactDebateReflectAgent


if __name__ == "__main__":
    num_debators = 2
    hotpot = joblib.load('data/hotpot-qa-distractor-sample.joblib').reset_index(drop = True)
    agents = [ReactDebateReflectAgent(question = row['question'], key= row['answer'], num_debators=num_debators) for _, row in hotpot.iterrows()]
    
    trial = 0
    log = ''
    for i in range(5):
        for agent in [a for a in agents if not a.is_correct()]:
            agent.run()
            break
        trial += 1
        log += log_react_trial(agents, trial)
        correct, incorrect, halted = summarize_react_trial(agents)
        print(f'Finished Trial {trial}, Correct: {len(correct)}, Incorrect: {len(incorrect)}, Halted: {len(halted)}')
        
        
    root  = 'root/'
    dir_path = os.path.join('root/', 'ReAct', "Debate_v3")
    os.makedirs(dir_path, exist_ok=True)

    with open(os.path.join(dir_path, f'{len(agents)}_questions_{trial}_trials_{num_debators}_debators.txt'), 'w') as f:
        f.write(log)
        
    # # save_agents(agents, os.path.join('ReAct',"Debate_v1", 'agents'))

        