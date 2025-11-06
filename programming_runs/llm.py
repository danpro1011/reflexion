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
                 model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
                 temperature: float = 0.0,
                 max_tokens: int = 100,
                 device: str = "auto",
                 load_in_8bit: bool = False,
                 model_kwargs: Optional[dict] = None,
                 **kwargs):
        """
        Initialize a local HuggingFace model.
        Args:
            model_name: HuggingFace model ID (default: meta-llama/Llama-3.2-3B-Instruct)
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
        self.is_chat = False
        if isinstance(self.stop_sequences, str):
            self.stop_sequences = [self.stop_sequences]

        print(f"Loading local model: {model_name}...")

        model_kwargs = {
            "device_map": "cpu",
            "dtype": torch.bfloat16,
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
            **model_kwargs,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Model loaded successfully on {device}!")

    def __call__(self, prompt: str, stop: Optional[list] = None) -> str:
        import torch
        try:
            print(f"Prompt to model: {repr(prompt)}")
            effective_stop = stop if stop is not None else self.stop_sequences
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
            print(f"Raw model output: {repr(full_text)}")
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
            print(f"Final generated text: {repr(generated_text)}")
            return generated_text
        except Exception as e:
            print(f"Error during generation: {e}")
            return ""
    
    def generate(self, prompt: str, stop: Optional[list] = None, **kwargs) -> str:
        """Alias for __call__, ignores extra kwargs for compatibility."""
        return self.__call__(prompt, stop)