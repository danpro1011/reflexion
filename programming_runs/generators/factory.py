from .py_generate import PyGenerator
from .rs_generate import RsGenerator
from .generator_types import Generator
from .model import CodeLlama, ModelBase, GPT4, GPT35, StarChat, GPTDavinci
from llm import LocalLLM

def generator_factory(lang: str) -> Generator:
    if lang == "py" or lang == "python":
        return PyGenerator()
    elif lang == "rs" or lang == "rust":
        return RsGenerator()
    else:
        raise ValueError(f"Invalid language for generator: {lang}")

def model_factory(model_name: str) -> ModelBase:
    if model_name == "gpt-4":
        return GPT4()
    elif model_name == "gpt-3.5-turbo":
        return GPT35()
    elif model_name == "starchat":
        return StarChat()
    elif model_name.startswith("codellama"):
        kwargs = {}
        if "-" in model_name:
            kwargs["version"] = model_name.split("-")[1]
        return CodeLlama(**kwargs)
    elif model_name == "meta-llama/Llama-3.2-3B-Instruct":
        return LocalLLM(
            model_name="meta-llama/Llama-3.2-3B-Instruct",
            temperature=0,
            max_tokens=512,
            device="cpu",  
            load_in_8bit=False,
            model_kwargs={"stop": "\n"}
        )
    else:
        raise ValueError(f"Invalid model name: {model_name}")