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

### Multi-agent debate prompts ###
DEBATE_INITIAL_INSTRUCTION = """You are Agent {agent_id} in a small team debating an answer. Read the context, reason step by step, and end with a single Finish[answer] action.
Context:
{context}

Question: {question}"""

DEBATE_FOLLOWUP_INSTRUCTION = """You are Agent {agent_id} continuing a debate. Other agents have proposed answers and rationales. Identify any mistakes, adopt good ideas, and update your own reasoning. Keep it concise and end with a single Finish[answer].

Other agents said:
{peer_responses}

Context:
{context}
Question: {question}"""


#Version of this that I'm thinking about is using the debate for generating the reflections. Also having the debate will likely be alot more complicated
# Debate prompt using the 'meta' prompts used in original multi-agent debate paper (https://aclanthology.org/2024.emnlp-main.992.pdf)
{
    "debate_topic": "",
    "base_answer": "",
    "debate_answer": "",
    "player_meta_prompt": "You are a debater. Hello and welcome to the debate. It's not necessary to fully agree with each other's perspectives, as our objective is to find the correct answer.\nThe debate topic is stated as follows:\n##debate_topic##",
    "moderator_meta_prompt": "You are a moderator. There will be two debaters involved in a debate. They will present their answers and discuss their perspectives on the following topic: \"##debate_topic##\"\nAt the end of each round, you will evaluate answers and decide which is correct.",
    "affirmative_prompt": "##debate_topic##",
    "negative_prompt": "##aff_ans##\n\nYou disagree with my answer. Provide your answer and reasons.",
    "moderator_prompt": "Now the ##round## round of debate for both sides has ended.\n\nAffirmative side arguing:\n##aff_ans##\n\nNegative side arguing: ##neg_ans##\n\nYou, as the moderator, will evaluate both sides' answers and determine if there is a clear preference for an answer candidate. If so, please summarize your reasons for supporting affirmative/negative side and give the final answer that you think is correct, and the debate will conclude. If not, the debate will continue to the next round. Now please output your answer in json format, with the format as follows: {\"Whether there is a preference\": \"Yes or No\", \"Supported Side\": \"Affirmative or Negative\", \"Reason\": \"\", \"debate_answer\": \"\"}. Please strictly output in JSON format, do not output irrelevant content.",
    "judge_prompt_last1": "Affirmative side arguing: ##aff_ans##\n\nNegative side arguing: ##neg_ans##\n\nNow, what answer candidates do we have? Present them without reasons.",
    "judge_prompt_last2": "Therefore, ##debate_topic##\nPlease summarize your reasons and give the final answer that you think is correct. Now please output your answer in json format, with the format as follows: {\"Reason\": \"\", \"debate_answer\": \"\"}. Please strictly output in JSON format, do not output irrelevant content.",
    "debate_prompt": "##oppo_ans##\n\nDo you agree with my perspective? Please provide your reasons and answer."
}

# Prompts for the debator as well as the affirmative and negative prompts
DEBATER_META_PROMPT_REFLECTION = """You are a debater. Hello and welcome to the debate. It's not necessary to fully agree with each other's perspectives, as our objective is to find the correct answer.The debate topic is stated as follows: 
You will be given a previous reasoning trial in which an advanced reasoning agent was given access to an Docstore API environment and a question to answer. The agent was unsuccessful in answering the question either because it guessed the wrong answer with Finish[<answer>], or it used up its set number of reasoning steps. 
In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  
A few examples of such reflections are:
{examples}

""" 

DEBATOR_AFFIRMATIVE_PROMPT_REFLECTION = """The previous trial was:
Question: {question}{scratchpad}

Reflection:
"""

DEBATOR_NEGATIVE_PROMPT_REFLECTION = """The previous trial was:
Question: {question}{scratchpad}

The other debator came up with the answer {debator_response}. You disagree with this answer. Provide your answer and reasons.
"""

DEBATOR_REPLY = """The other debator responded with {debator_response}
Do you agree with my perspective? Please provide your reasons and answer."""

#Prompts to initialize the judge as well as the moderator prompts

JUDGE_META_PROMPT_REFLECTION = """You are a moderator. There will be two debaters involved in a debate. They will present their answers and discuss their perspectives on the following topic:
They will be given a previous reasoning trial in which an advanced reasoning agent was given access to an Docstore API environment and a question to answer. The agent was unsuccessful in answering the question either because it guessed the wrong answer with Finish[<answer>], or it used up its set number of reasoning steps. 
In a few sentences, they will diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure.   

At the end of each round, you will evaluate answers and decide which is correct.
"""

#TODO: Use the actual langchain library to enforce output format, fix this nightmare that I'm currently looking at
JUDGE_END_OF_ROUND_PROMPT_REFLECTION = """Now the ##round## round of debate for both sides has ended.
Affirmative side arguing:{affirmative_response}


Negative side arguing: {negative_response}
You, as the moderator, will evaluate both sides' answers and determine if there is a clear preference for an answer candidate. If so, please summarize your reasons for supporting affirmative/negative side and give the final answer that you think is correct, and the debate will conclude. 
If not, the debate will continue to the next round. Now please output your answer in json format, with the format as follows: 

{\"Whether there is a preference\": \"Yes or No\", \"Supported Side\": \"Affirmative or Negative\", \"Reason\": \"\", \"debate_answer\": \"\"}. Please strictly output in JSON format, do not output irrelevant content."""
 


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

## Debate prompt formats ##
debate_initial_prompt = PromptTemplate(
                        input_variables=["agent_id", "context", "question"],
                        template=DEBATE_INITIAL_INSTRUCTION,
                        )

debate_followup_prompt = PromptTemplate(
                        input_variables=["agent_id", "peer_responses", "context", "question"],
                        template=DEBATE_FOLLOWUP_INSTRUCTION,
                        )

# Debate, reflection prompts
debate_meta_reflection_prompt = PromptTemplate(
                                input_variables=["examples"],
                                template=DEBATER_META_PROMPT_REFLECTION
                                )

debate_affirmative_reflection_prompt = PromptTemplate(
                                input_variables= ["question", "scratchpad"],
                                template=DEBATOR_AFFIRMATIVE_PROMPT_REFLECTION
                                )

debate_negative_reflection_prompt = PromptTemplate(
                                input_variables= ["question", "scratchpad", "debate_response"],
                                template=DEBATOR_NEGATIVE_PROMPT_REFLECTION
                                )

debator_response_prompt = PromptTemplate( 
                                input_variables=["opponent_reponse"],
                                template=DEBATOR_REPLY
                                )

judge_meta_reflection_prompt = PromptTemplate(template=JUDGE_META_PROMPT_REFLECTION)

judge_end_of_round_reflection_prompt = PromptTemplate(
                                        input_variables=["affirmative_response", "negative_response"],
                                        template=JUDGE_END_OF_ROUND_PROMPT_REFLECTION
                                        )