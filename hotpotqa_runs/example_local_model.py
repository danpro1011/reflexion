import os
from llm import LocalLLM
from agents import ReactReflectAgent, ReflexionStrategy
try:
    from langchain_community.docstore.wikipedia import Wikipedia
except ImportError:
    from langchain import Wikipedia

def main():
    question = "What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?"
    key = "1,800 to 7,000 ft"

    print("=" * 80)
    print(f"\nQuestion: {question}")
    print(f"Expected Answer: {key}\n")
    print("=" * 80)


    shared_llm = LocalLLM(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        temperature=0,
        max_tokens=250,
        device="auto",
        load_in_8bit=True,
        model_kwargs={"stop": "\n"}
    )

    react_llm = shared_llm
    reflect_llm = shared_llm


    agent = ReactReflectAgent(
        question=question,
        key=key,
        max_steps=15,
        docstore=Wikipedia(),
        react_llm=react_llm,
        reflect_llm=reflect_llm
    )

    max_trials = 3
    for trial in range(max_trials):
        print(f"\n{'=' * 80}")
        print(f"Trial {trial + 1}/{max_trials}")
        print('=' * 80 + "\n")

        agent.run(
            reset=(trial == 0),
            reflect_strategy=ReflexionStrategy.REFLEXION
        )

        if agent.is_correct():
            print(f"\n✓ Correct answer found in trial {trial + 1}!")
            break
        elif agent.is_halted():
            print(f"\n✗ Agent halted in trial {trial + 1}")
        else:
            print(f"\n✗ Incorrect answer in trial {trial + 1}")

    print("\n" + "=" * 80)
    print("Final Results")
    print("=" * 80)
    print(f"Agent's Answer: {agent.answer}")
    print(f"Expected Answer: {key}")
    print(f"Correct: {agent.is_correct()}")
    print()


def example_with_cot_agent():
    from agents import CoTAgent

    question = "What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?"
    context = "The Colorado orogeny was an episode of mountain building..."
    key = "1,800 to 7,000 ft"

    print("\n" + "=" * 80)
    print("CoT Agent with Local Model Example")
    print("=" * 80 + "\n")

    llm = LocalLLM(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        temperature=0,
        max_tokens=250,
        device="auto",
        load_in_8bit=False,
        model_kwargs={"stop": "\n"}
    )

    agent = CoTAgent(
        question=question,
        context=context,
        key=key,
        self_reflect_llm=llm,
        action_llm=llm
    )

    max_trials = 3
    for trial in range(max_trials):
        print(f"\nTrial {trial + 1}/{max_trials}")
        agent.run(reflexion_strategy=ReflexionStrategy.REFLEXION)

        if agent.is_correct():
            print(f"✓ Correct answer found in trial {trial + 1}!")
            break

    print(f"\nAgent's Answer: {agent.answer}")
    print(f"Correct: {agent.is_correct()}")


if __name__ == "__main__":
    main()

