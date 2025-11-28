try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    from langchain.prompts import PromptTemplate

COT_INSTRUCTION = """Solve a question answering task by having a Thought, then Finish with your answer. Thought can reason about the current situation. Finish[answer] returns the answer and finishes the task. You will be given context that you should use to help you answer the question.
Here are some examples:
{examples}
(END OF EXAMPLES)
{reflections}
Relevant Context: {context} 
Question: {question}{scratchpad}"""

COT_AGENT_REFLECT_INSTRUCTION = """Solve a question answering task by having a Thought, then Finish with your answer. Thought can reason about the current situation. Finish[answer] returns the answer and finishes the task. You will be given context that you should use to help you answer the question.
Here are some examples:
{examples}
(END OF EXAMPLES)

{reflections}

Relevant Context: {context}
Question: {question}{scratchpad}"""

COT_REFLECT_INSTRUCTION = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to relevant context and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>] or there is a phrasing discrepancy with your provided answer and the answer key. In a few sentences, Diagnose a possible reason for failure or phrasing discrepancy and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  
Here are some examples:
{examples}
(END OF EXAMPLES)

Previous trial:
Relevant Context: {context}
Question: {question}{scratchpad}

Reflection:"""

cot_agent_prompt = PromptTemplate(
                        input_variables=["examples", "reflections", "context", "question", "scratchpad"],
                        template = COT_INSTRUCTION,
                        )

cot_reflect_agent_prompt = PromptTemplate(
                        input_variables=["examples", "reflections", "context", "question", "scratchpad"],
                        template = COT_AGENT_REFLECT_INSTRUCTION,
                        )

cot_reflect_prompt = PromptTemplate(
                        input_variables=["examples", "context", "question", "scratchpad"],
                        template = COT_REFLECT_INSTRUCTION,
                        )

COT_SIMPLE_INSTRUCTION = """Solve a question answering task by having a Thought, then Finish with your answer. Thought can reason about the current situation. Finish[answer] returns the answer and finishes the task.
Here are some examples:
{examples}
(END OF EXAMPLES)
{reflections}
{context}
Question: {question}{scratchpad}"""

COT_SIMPLE_AGENT_REFLECT_INSTRUCTION = """Solve a question answering task by having a Thought, then Finish with your answer. Thought can reason about the current situation. Finish[answer] returns the answer and finishes the task.
Here are some examples:
{examples}
(END OF EXAMPLES)
{context}
{reflections}

Question: {question}{scratchpad}"""

COT_SIMPLE_REFLECT_INSTRUCTION = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>] or there is a phrasing discrepancy with your provided answer and the answer key. In a few sentences, Diagnose a possible reason for failure or phrasing discrepancy and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.
Here are some examples:
{examples}
(END OF EXAMPLES)
{context}
Previous trial:
Question: {question}{scratchpad}

Reflection:"""

cot_simple_agent_prompt = PromptTemplate(
                        input_variables=["examples", "question", "reflections", "context", "scratchpad"],
                        template = COT_SIMPLE_INSTRUCTION,
                        )

cot_simple_reflect_agent_prompt = PromptTemplate(
                        input_variables=["examples", "context", "reflections", "question", "scratchpad"],
                        template = COT_SIMPLE_AGENT_REFLECT_INSTRUCTION,
                        )

cot_simple_reflect_prompt = PromptTemplate(
                        input_variables=["examples", "question", "context", "scratchpad"],
                        template = COT_SIMPLE_REFLECT_INSTRUCTION,
                        )


REACT_INSTRUCTION = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.
You may take as many steps as necessary.
Here are some examples:
{examples}
(END OF EXAMPLES)
Question: {question}{scratchpad}"""

REACT_REFLECT_INSTRUCTION = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.
You may take as many steps as necessary.
Here are some examples:
{examples}
(END OF EXAMPLES)

{reflections}

Question: {question}{scratchpad}"""

REFLECTION_HEADER = 'You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\n'
REFLECTION_AFTER_LAST_TRIAL_HEADER = 'The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\n'
LAST_TRIAL_HEADER = 'You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\n'

REFLECT_INSTRUCTION = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to an Docstore API environment and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  
Here are some examples:
{examples}

Previous trial:
Question: {question}{scratchpad}

Reflection:"""

#Switching the debate style to 'society of mind' thing as desrcibed in this paper (https://openreview.net/pdf?id=zj7YuTE4t8)
#My initial concern is how exactly is consensus extracted, not really properly explained fully, probably can't be worse than what's happening rn though

#Example prompts:
# Task Type | Prompt
# -------------------------------------------------------------

# Starting (Arithmetic)
# What is the result of {} + {} * {} + {} - {} * {}?
# Make sure to state your answer at the end of the response.

# Debate (Arithmetic)
# These are the recent/updated opinions from other agents: <other agent responses>
# Use these opinions carefully as additional advice.
# Can you provide an updated answer? Make sure to state your answer at the end of the response.

# Starting (GSM8K)
# Can you solve the following math problem? <Problem>
# Explain your reasoning.
# Your final answer should be a single numerical number, in the form \boxed{{answer}}, at the end of your response.

# Debate (GSM8K)
# These are the solutions to the problem from other agents: <other agent responses>
# Using the solutions from other agents as additional information, can you provide your answer to the math problem?
# The original math problem is <Problem>.
# Your final answer should be a single numerical number, in the form \boxed{{answer}}, at the end of your response.

# So TLDR, each agent is given a question to answer (ideally some method is used so that they come up with different answers)
#then, the other agents answers are queried, and each agent is given a 'consensus prompt' I guess until they all converge upon an answer:

#(did this really demand a graphic in the paper, lmao)
#               'short debate'
#  "These are the solutions to the problem from other agents: [other answers]
#  Based off the opinion of other agents, can you give an updated response ..."

#               'long debate' 
# "These are the solutions to the problem from other agents: [other answers]
#  Using the opinion of other agents as additional advice, can you give an updated response ..."



# No more 'you are a debator' thing, which IMO probably for the best
# Typical reflection prompt it's labeled as 'your' reasoning traces, I think for the structure of the 'debate' it makes more sense to say its another model's reasoning traces -> maybe not, let's see
DEBATER_META_PROMPT_REFLECTION = """You are a highly sophisticated reasoning agent that's capable of improving through self-reflection. You will be given a previous reasoning trial in which you were given access to an Docstore API environment and a question to answer.
You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  

A few examples of such reflections are:
{examples}'
"""

# Arguably shouldn't be a seperate prompt, but organizing like this for now so that stuff like personas can be added in later
DEBATOR_INITIAL_PROMPT = """
Previous trial:
Question: {question}{scratchpad}

Reflection:"""

#No more affirmative-negative prompts, so exactly how does the paper gaurantee that the agents generate different responses other than temp param??
DEBATOR_REPLY = """
These are the reflections that other agents analyzing your reasoning traces came up with: {debator_responses}
Using the opinion of other agents as additional advice, can you give an updated response ..."""


#Some method of consensus extraction is needed, as stated in paper: 'In cases of disagreement, we took the majority answer across agents at the end of debate.'
#having a seperate judge LLM do this is the easiest solution, but is it the actually most effective??

# One prompt to search for and extract a consensus, if there is one
CONSENSUS_REACHED = """You are a highly capable moderator of a debate between agents. These debators are analyzing the reasoning traces of some other agent that failed to answer the question it was given and trying to determine why it failed.
You will view their arguments and determine whether or not the debators have come to a consensus. If they've reached a consensus, then summarize the consensus view as succinctly as possible. 
Output your response as {{\"consensus_reached\": True or False, \"consensus\": \"\"}} don't include any other sentences or words outside of the json. 

Debate log:
{debate_log}
"""

#Another prompt to choose a verdict even if a consensus hasn't been reached.
# NOTE: This could either summarize the consensus or just choose the 'winner' and we take their last reflection, maybe even for the other prompt too
DETERMINE_CONSENSUS = """You are a highly capable moderator of a debate between agents. These debators are analyzing the reasoning traces of some other agent that failed to answer the question it was given and trying to determine why it failed.
You will view this debate, determine which debator had the most convincing argument, and output their view as succinctly as possible. 

Debate log:
{debate_log}
"""

react_agent_prompt = PromptTemplate(
                        input_variables=["examples", "question", "scratchpad"],
                        template = REACT_INSTRUCTION,
                        )

react_reflect_agent_prompt = PromptTemplate(
                        input_variables=["examples", "reflections", "question", "scratchpad"],
                        template = REACT_REFLECT_INSTRUCTION,
                        )

reflect_prompt = PromptTemplate(
                        input_variables=["examples", "question", "scratchpad"],
                        template = REFLECT_INSTRUCTION,
                        )

# Debate, reflection prompts
debate_meta_reflection_prompt = PromptTemplate(
                                input_variables=["examples"],
                                template=DEBATER_META_PROMPT_REFLECTION
                                )

debator_initial_prompt = PromptTemplate(
                            input_variables = ["question", "scratchpad"],
                            template = DEBATOR_INITIAL_PROMPT
                            )

debator_response_prompt = PromptTemplate( 
                            input_variables=["debator_responses"],
                            template=DEBATOR_REPLY
                            )

consensus_reached_prompt = PromptTemplate(template=CONSENSUS_REACHED, input_variables=["debate_log"])

determine_consensus_prompt = PromptTemplate(
                                input_variables=["debate_log"],
                                template=DETERMINE_CONSENSUS
                                )