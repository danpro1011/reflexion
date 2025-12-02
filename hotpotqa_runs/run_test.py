import os
from multiprocessing import Pool
import json
import joblib
from util import summarize_react_trial, log_react_trial, save_agents
from prompts import react_agent_prompt
from agents import ReactReflectAgent, ReactAgent, ReflexionStrategy, ReactDebateReflectAgent, ReactVerbalisedSamplingAgent

class DummyAgent(ReactAgent):

    def __init__(self,
             question: str,
             key: str,
             max_steps: int = 6,
             agent_prompt= react_agent_prompt,
             docstore = None,
             react_llm = None,
             agent_num= 0
             ) -> None:
        
        super().__init__(question, key, max_steps, agent_prompt, docstore, react_llm)
        self.agent_num = agent_num

    def run(self):
        self.scratchpad += "-"*self.agent_num + self.agent_num + "-"*self.agent_num + '\n'


def run_single_agent_trial(agent):
    agent.initialize_llm_tokenizer()
    agent.run()
    return agent

#TODO: No idea if this actually works Test this on some dummy agent before losing 5$ again -> Ok it does not
def run_test_multi_process(num_trails = 5, num_debators = 2):
    hotpot = joblib.load('data/hotpot-qa-distractor-sample.joblib').reset_index(drop = True)
    agents = [ReactDebateReflectAgent(question = row['question'], key= row['answer'], num_debators=num_debators) for _, row in hotpot.iterrows()]

    # Deinitialize all the non-pickleable variables before starting the threads
    for agent in agents:
        agent.deinitialize_llm_tokenizer()

    log = ''
    for trial in range(num_trails):
        with Pool() as pool:
            new_agents = pool.map(run_single_agent_trial, agents)
            log += log_react_trial(new_agents, trial)
            
    print(log)
    # root  = 'root/'
    # dir_path = os.path.join('root/', 'ReAct', "Debate_v3")
    # os.makedirs(dir_path, exist_ok=True)

    # with open(os.path.join(dir_path, f'{len(agents)}_questions_{trial}_trials_{num_debators}_debators.txt'), 'w') as f:
    #     f.write(log)


def run_test_single_process(hard_only = False, num_debators = 2, trails = 5):
    num_debators = num_debators
    hotpot = joblib.load('data/hotpot-qa-distractor-sample.joblib').reset_index(drop = True)

    if hard_only:
        with open("hard_questions.json", "r") as file:
            hard_questions = json.load(file)

        hotpot = hotpot[hotpot["question"].isin(hard_questions)]
    
    agents = [ReactVerbalisedSamplingAgent(question = row['question'], key= row['answer']) for _, row in hotpot.iterrows()]
    
    trial = 0
    log = ''
    for i in range(trails):
        for agent in [a for a in agents if not a.is_correct()]:
            agent.run()
        trial += 1
        log += log_react_trial(agents, trial)
        correct, incorrect, halted = summarize_react_trial(agents)
        print(f'Finished Trial {trial}, Correct: {len(correct)}, Incorrect: {len(incorrect)}, Halted: {len(halted)}')
        
        root  = 'root/'
        dir_path = os.path.join('root/', 'ReAct', "VS_all")
        os.makedirs(dir_path, exist_ok=True)

        with open(os.path.join(dir_path, f'{len(agents)}_questions_{trial}_trials.txt'), 'w') as f:
            f.write(log)
            


if __name__ == "__main__":
    run_test_single_process(num_debators=3, trails=5)
    
    # run_test_multi_process()

        