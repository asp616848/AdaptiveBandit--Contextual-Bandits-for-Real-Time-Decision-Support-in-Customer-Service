from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol


class ChatBackend(Protocol):
    def is_available(self) -> bool: ...

    def chat(self, *, model: str, messages: list[dict[str, str]], max_tokens: int, temperature: float) -> str: ...


@dataclass
class OllamaOpenAIBackend:
    endpoint: str

    def __post_init__(self) -> None:
        try:
            from openai import OpenAI  # type: ignore

            self._client = OpenAI(base_url=self.endpoint, api_key="ollama")
        except Exception:
            self._client = None

    def is_available(self) -> bool:
        return self._client is not None

    def chat(self, *, model: str, messages: list[dict[str, str]], max_tokens: int, temperature: float) -> str:
        if self._client is None:
            raise RuntimeError("Ollama/OpenAI backend unavailable")
        resp = self._client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
        )
        return (resp.choices[0].message.content or "").strip()


@dataclass
class HFTransformersBackend:
    model_id_or_path: str
    device: str | None = None

    def __post_init__(self) -> None:
        self._pipeline = None
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
            from transformers import pipeline  # type: ignore

            tok = AutoTokenizer.from_pretrained(self.model_id_or_path, trust_remote_code=True)
            mdl = AutoModelForCausalLM.from_pretrained(
                self.model_id_or_path,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="auto" if self.device is None else None,
            )
            self._pipeline = pipeline(
                "text-generation",
                model=mdl,
                tokenizer=tok,
                device_map=getattr(mdl, "hf_device_map", None),
            )
        except Exception:
            self._pipeline = None

    def is_available(self) -> bool:
        return self._pipeline is not None

    def chat(self, *, model: str, messages: list[dict[str, str]], max_tokens: int, temperature: float) -> str:
        _ = model
        if self._pipeline is None:
            raise RuntimeError("HF transformers backend unavailable")

        # Very simple chat formatting fallback. If your tokenizer has a chat template,
        # consider upgrading this to tok.apply_chat_template(...).
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + "\nassistant:"
        out = self._pipeline(
            prompt,
            max_new_tokens=int(max_tokens),
            do_sample=float(temperature) > 0.0,
            temperature=float(max(temperature, 1e-6)),
            return_full_text=False,
        )
        if not out:
            return ""
        text = out[0].get("generated_text", "")
        return str(text).strip()


def make_backend() -> ChatBackend | None:
    backend = os.getenv("SUPPORT_SIM_LLM_BACKEND", "ollama").strip().lower()
    if backend in {"ollama", "openai-ollama"}:
        endpoint = os.getenv("SUPPORT_SIM_LLM_ENDPOINT", "http://localhost:11434/v1")
        b = OllamaOpenAIBackend(endpoint=endpoint)
        return b if b.is_available() else None

    if backend in {"hf", "hf-qwen", "huggingface"}:
        model_id = os.getenv(
            "SUPPORT_SIM_HF_MODEL_ID",
            "abhi6168/ABCD_CustomerAgent_Qwen_2.5_7b",
        )
        local_path = os.getenv("SUPPORT_SIM_HF_MODEL_PATH", "").strip()
        b = HFTransformersBackend(model_id_or_path=(local_path or model_id))
        return b if b.is_available() else None

    return None
