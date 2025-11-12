from typing import Union, Literal, Optional
try:
    from langchain_openai import ChatOpenAI, OpenAI
except ImportError:
    from langchain.chat_models import ChatOpenAI
    from langchain.llms import OpenAI

try:
    from langchain.schema import HumanMessage
except ImportError:
    from langchain_core.messages import HumanMessage

class AnyOpenAILLM:
    def __init__(self, *args, **kwargs):
        # Determine model type from the kwargs
        model_name = kwargs.get('model_name', 'gpt-3.5-turbo')
        if model_name.split('-')[0] == 'text':
            self.model = OpenAI(*args, **kwargs)
            self.model_type = 'completion'
        else:
            self.model = ChatOpenAI(*args, **kwargs)
            self.model_type = 'chat'

    def __call__(self, prompt: str):
        if self.model_type == 'completion':
            return self.model(prompt)
        else:
            return self.model(
                [
                    HumanMessage(
                        content=prompt,
                    )
                ]
            ).content


class LocalLLM:
    """Local HuggingFace model wrapper"""
    def __init__(self,
                 model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
                 temperature: float = 0.0,
                 max_tokens: int = 100,
                 device: str = "auto",
                 load_in_8bit: bool = False,
                 model_kwargs: Optional[dict] = None,
                 **kwargs):
        """
        Initialize a local HuggingFace model.
        Args:
            model_name: HuggingFace model ID (default: Llama-3-8B-Instruct)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            device: Device to use ("auto", "cuda", "cpu")
            load_in_8bit: Use 8-bit quantization to reduce memory usage
            model_kwargs: Additional model kwargs (e.g., {"stop": "\n"})
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.temperature = max(temperature, 0.0001)
        self.max_tokens = max_tokens
        self.stop_sequences = model_kwargs.get("stop", []) if model_kwargs else []
        if isinstance(self.stop_sequences, str):
            self.stop_sequences = [self.stop_sequences]

        print(f"Loading local model: {model_name}...")

        model_kwargs = {
            "device_map": device,
            "torch_dtype": torch.bfloat16,
        }

        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
            print("Loading model in 8-bit mode to save memory...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side='left'
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Model loaded successfully on {device}!")

    def __call__(self, prompt: str, stop: Optional[list] = None) -> str:
        """Generate text from prompt"""
        import torch

        effective_stop = stop if stop is not None else self.stop_sequences

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": self.max_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        if self.temperature > 0.0001:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = 0.95

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs
            )

        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from output
        if full_text.startswith(prompt):
            generated_text = full_text[len(prompt):].strip()
        else:
            generated_text = full_text.strip()

        if effective_stop:
            for stop_seq in effective_stop:
                if stop_seq in generated_text:
                    generated_text = generated_text[:generated_text.index(stop_seq)]
                    break

        generated_text = generated_text.strip()

        return generated_text


class ClaudeLLM:
    def __init__(self,
                 model_name: str = "claude-3-haiku-20240307",
                 temperature: float = 0.0,
                 max_tokens: int = 250,
                 api_key: Optional[str] = None,
                 **kwargs):
        """
        Initialize Claude API client.
        """
        import os
        import re
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package not installed. "
                "Install with: pip install anthropic"
            )

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found in environment. "
                "Set with: export ANTHROPIC_API_KEY=<your-key>"
            )

        self.api_key = api_key  
        self.client = anthropic.Anthropic(api_key=api_key)
        print(f"✓ Claude model initialized: {model_name}")

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['client']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Recreate the Anthropic client
        import anthropic
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def __call__(self, prompt: str, stop: Optional[list] = None) -> str:
        import re

        # Add default stop sequences for ReAct-style prompts
        # Stop at next step markers to generate one piece at a time
        if stop is None:
            stop = ["\nThought", "\nAction", "\nObservation"]

        # System message with explicit examples
        system_message = (
            "You are following the ReAct framework. When prompted with 'Action N:', respond with ONLY one of:\n"
            "- Search[entity] - to search Wikipedia\n"
            "- Lookup[keyword] - to find text on current page\n"
            "- Finish[answer] - to give final answer\n\n"
            "Output ONLY the action command, nothing else.\n\n"
            "Example 1:\n"
            "Question: Who is the president of France?\n"
            "Thought 1: I need to find information about France's president\n"
            "Action 1: Search[President of France]\n\n"
            "Example 2:\n"
            "Observation 1: France is a country. Emmanuel Macron is the current president.\n"
            "Thought 2: I found the president's name\n"
            "Action 2: Finish[Emmanuel Macron]\n\n"
            "Example 3:\n"
            "Question: What company did VIVA Media become?\n"
            "Thought 1: I should search for VIVA Media\n"
            "Action 1: Search[VIVA Media]\n\n"
            "WRONG examples (NEVER do this):\n"
            "Action 1: I need to search for...\n"
            "Action 1: The search results do not...\n"
            "Action 1: Let me try searching for...\n\n"
            "When you see 'Action N:', output ONE command only: Search[X], Lookup[X], or Finish[X]"
        )

        kwargs = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": system_message,
            "messages": [{"role": "user", "content": prompt}]
        }

        if stop:
            kwargs["stop_sequences"] = stop

        try:
            response = self.client.messages.create(**kwargs)
            text = response.content[0].text.strip()

            text = re.sub(r'^(Thought|Action|Observation)\s+\d+:\s*', '', text)

            if prompt.rstrip().endswith(('Action 1:', 'Action 2:', 'Action 3:',
                                          'Action 4:', 'Action 5:', 'Action 6:')):
                # Check if the response is a valid action format
                action_pattern = r'^(Search|Lookup|Finish)\[.+\]'
                if not re.match(action_pattern, text):
                    if len(text) < 200: 
                        search_for_match = re.search(r'[Ss]earch\s+for\s+["\']?([A-Z][^"\'.!?,]{2,50})["\']?', text)
                        search_bracket_match = re.search(r'[Ss]earch\s*\[([^\]]{2,50})\]', text)
                        lookup_match = re.search(r'[Ll]ookup\s*\[([^\]]{2,50})\]', text)

                        if search_bracket_match:
                            entity = search_bracket_match.group(1).strip()
                            if len(entity) < 50 and not any(word in entity.lower() for word in ['results', 'information', 'details', 'search', 'find', 'try', 'will', 'should', 'cannot']):
                                text = f"Search[{entity}]"
                                print(f"[ClaudeLLM] Extracted Search: {text}")
                            else:
                                text = "Finish[Unable to determine from available information]"
                                print(f"[ClaudeLLM] Invalid entity, using Finish")
                        elif search_for_match:
                            entity = search_for_match.group(1).strip()
                            if len(entity) < 50 and not any(word in entity.lower() for word in ['results', 'information', 'details', 'search', 'find', 'try', 'will', 'should']):
                                text = f"Search[{entity}]"
                                print(f"[ClaudeLLM] Extracted Search: {text}")
                            else:
                                text = "Finish[Unable to determine from available information]"
                                print(f"[ClaudeLLM] Invalid entity, using Finish")
                        elif lookup_match:
                            keyword = lookup_match.group(1).strip()
                            if len(keyword) < 30:
                                text = f"Lookup[{keyword}]"
                                print(f"[ClaudeLLM] Extracted Lookup: {text}")
                            else:
                                text = "Finish[Unable to determine from available information]"
                        else:
                            print(f"[ClaudeLLM] No valid action found in: {text[:80]}...")
                            text = "Finish[Unable to determine from available information]"
                    else:
                        print(f"[ClaudeLLM] Response too long ({len(text)} chars), using Finish")
                        text = "Finish[Unable to determine from available information]"

            return text
        except Exception as e:
            print(f"Error calling Claude API: {e}")
            raise