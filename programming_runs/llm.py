from typing import Union, Literal, Optional, Any
try:
    from langchain_openai import ChatOpenAI, OpenAI
except ImportError:
    from langchain.chat_models import ChatOpenAI
    from langchain.llms import OpenAI

try:
    from langchain.schema import HumanMessage
except ImportError:
    from langchain_core.messages import HumanMessage

from dotenv import load_dotenv
load_dotenv()

class AnyOpenAILLM:
    def __init__(self, *args, **kwargs):
        model_name = kwargs.get('model_name', 'gpt-3.5-turbo')
        if model_name.split('-')[0] == 'text':
            self.model = OpenAI(*args, **kwargs)
            self.model_type = 'completion'
        else:
            self.model = ChatOpenAI(*args, **kwargs)
            self.model_type = 'chat'

    def _extract_text_from_result(self, result: Any) -> str:
        # common result shapes from different langchain versions
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        # Chat message objects
        if hasattr(result, "content"):
            return getattr(result, "content")
        # LLMResult / ChatResult with .generations or .generations[0][0].text
        if hasattr(result, "generations"):
            gens = result.generations
            try:
                first = gens[0]
                # sometimes it's a list
                if isinstance(first, list) and len(first) > 0:
                    g = first[0]
                else:
                    g = first
                if hasattr(g, "text"):
                    return g.text
                if hasattr(g, "generation_text"):
                    return g.generation_text
            except Exception:
                pass
        # fallback to .text or __str__
        if hasattr(result, "text"):
            return getattr(result, "text")
        return str(result)

    def _call_underlying(self, prompt_obj: Any) -> str:
        m = self.model
        # 1) prefer existing 'invoke' if present
        if hasattr(m, "invoke"):
            try:
                return self._extract_text_from_result(m.invoke(prompt_obj))
            except Exception:
                pass
        # 2) try direct __call__
        if callable(m):
            try:
                out = m(prompt_obj)
                return self._extract_text_from_result(out)
            except Exception:
                pass
        # 3) try 'generate' (langchain chat generate)
        if hasattr(m, "generate"):
            try:
                out = m.generate(prompt_obj)
                return self._extract_text_from_result(out)
            except Exception:
                pass
        # 4) last resort: stringify
        raise RuntimeError("Underlying model has no known callable API (invoke/__call__/generate)")

    def __call__(self, prompt: str):
        if self.model_type == 'completion':
            # completion models typically accept a raw prompt
            return self._call_underlying(prompt)
        else:
            # chat models usually expect message objects
            messages = [HumanMessage(content=prompt)]
            return self._call_underlying(messages)

    def query(self, query):
        if self.model_type == 'completion':
            return self._call_underlying(query)
        else:
            # if caller passes a raw prompt, wrap as message; if they pass messages, forward
            if isinstance(query, (list, tuple)):
                return self._call_underlying(query)
            return self._call_underlying([HumanMessage(content=query)])