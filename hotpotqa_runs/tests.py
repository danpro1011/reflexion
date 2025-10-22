
import joblib
from react import ReactReflectAgent
from mocks import DocStoreExplorerMock, LLMMock
from environment import QAEnv

test_q = "Who wrote The Great Gatsby?"
test_a = "F. Scott Fitzgerald"

# Adjust args to QAEnv to match its signature in environment.py
env = QAEnv(question=test_q, key=test_a, max_steps=6) 

agent = ReactReflectAgent(question=test_q, env=env)

agent.run()

print(agent._build_agent_prompt())
print(agent._build_reflection_prompt())