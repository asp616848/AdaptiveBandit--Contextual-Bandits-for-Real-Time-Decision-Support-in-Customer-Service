from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from threading import Lock
from typing import Any


_CLIENTS: dict[tuple[str, str], "LocalQwenChatClient"] = {}
_LOCK = Lock()


class LocalQwenChatClient:
    """Small shared chat wrapper for a local Hugging Face causal LM."""

    def __init__(
        self,
        model_path: str,
        device_map: str = "auto",
        max_input_tokens: int = 3072,
    ):
        self.model_path = str(Path(model_path).expanduser())
        self.device_map = str(device_map)
        self.max_input_tokens = int(max_input_tokens)
        self._load()

    def _load(self) -> None:
        try:
            import torch
            from transformers import logging as transformers_logging
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "Local Qwen backend requires torch and transformers in the active environment."
            ) from exc

        transformers_logging.set_verbosity_error()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        dtype = torch.float32
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=dtype,
            device_map=self.device_map if torch.cuda.is_available() else None,
        )
        self.model.eval()

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 150,
        temperature: float = 0.7,
    ) -> str:
        import torch

        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        input_device = next(self.model.parameters()).device
        inputs = {k: v.to(input_device) for k, v in inputs.items()}

        do_sample = float(temperature) > 0.0
        generation_config = deepcopy(self.model.generation_config)
        generation_config.do_sample = do_sample
        if do_sample:
            generation_config.temperature = float(temperature)
            generation_config.top_p = 0.9
            generation_config.top_k = 20
        else:
            generation_config.temperature = None
            generation_config.top_p = None
            generation_config.top_k = None

        generation_kwargs = {
            **inputs,
            "max_new_tokens": int(max_tokens),
            "generation_config": generation_config,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        with torch.inference_mode():
            output = self.model.generate(**generation_kwargs)

        generated = output[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def get_local_qwen_client(
    model_path: str | None = None,
    device_map: str | None = None,
    max_input_tokens: int | None = None,
) -> LocalQwenChatClient:
    path = model_path or os.getenv("SUPPORT_SIM_LOCAL_MODEL_PATH", "Qwen2.5-7B-Instruct-merged")
    dev_map = device_map or os.getenv("SUPPORT_SIM_LOCAL_DEVICE_MAP", "auto")
    max_tokens = int(max_input_tokens or os.getenv("SUPPORT_SIM_LOCAL_MAX_INPUT_TOKENS", "3072"))
    key = (str(Path(path).expanduser()), dev_map)
    with _LOCK:
        client = _CLIENTS.get(key)
        if client is None:
            client = LocalQwenChatClient(
                model_path=path,
                device_map=dev_map,
                max_input_tokens=max_tokens,
            )
            _CLIENTS[key] = client
        return client
