"""In-process Hugging Face generation with the pipeline's completion interface."""

import json
import os
from functools import lru_cache
from types import SimpleNamespace


class LocalClient:
    def __init__(self, model_name):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("Local SEA-LION inference requires CUDA PyTorch in LLM_PYTHON")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        ).eval()
        self.chat = SimpleNamespace(completions=self)
        self.beta = SimpleNamespace(chat=self.chat)

    def create(self, *, messages, temperature=0.7, max_tokens=None, response_format=None, **kwargs):
        import torch

        messages = [dict(message) for message in messages]
        if response_format:
            schema = (
                response_format.model_json_schema()
                if hasattr(response_format, "model_json_schema")
                else None
            )
            messages.append(
                {
                    "role": "user",
                    "content": "Return only valid JSON."
                    + (" Schema: " + json.dumps(schema) if schema else ""),
                }
            )
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        options = dict(
            max_new_tokens=max_tokens or int(os.getenv("LLM_MAX_NEW_TOKENS", "2048")),
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if temperature > 0:
            options["temperature"] = temperature
        with torch.inference_mode():
            output = self.model.generate(inputs, **options)
        content = self.tokenizer.decode(output[0, inputs.shape[-1] :], skip_special_tokens=True)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    def parse(self, *, response_format, **kwargs):
        result = self.create(response_format=response_format, **kwargs)
        message = result.choices[0].message
        text = message.content.strip()
        start, end = text.find("{"), text.rfind("}")
        message.parsed = response_format.model_validate_json(text[start : end + 1])
        return result


@lru_cache(maxsize=1)
def local_client(model_name):
    return LocalClient(model_name)
