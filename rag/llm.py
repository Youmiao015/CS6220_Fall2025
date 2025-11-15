from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .config import LLMConfig


class TextGenerator(Protocol):
    def __call__(self, prompt: str) -> str: ...


class HuggingFaceGenerator:
    """Thin wrapper around huggingface_hub.InferenceClient."""

    def __init__(self, config: LLMConfig):
        try:
            from huggingface_hub import InferenceClient  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover
            raise ImportError("huggingface_hub is required for remote LLM inference.") from exc

        if not config.model_id:
            raise ValueError("LLMConfig.model_id must be provided for HuggingFaceGenerator.")

        self._client = InferenceClient(model=config.model_id, token=config.hf_token, timeout=120)
        self._max_new_tokens = config.max_new_tokens
        self._temperature = config.temperature
        self._top_p = config.top_p

    def __call__(self, prompt: str) -> str:
        completion = self._client.text_generation(
            prompt,
            max_new_tokens=self._max_new_tokens,
            temperature=self._temperature,
            top_p=self._top_p,
            stream=False,
        )
        return completion.strip()


class TransformersGenerator:
    """Local text generator backed by transformers.pipeline."""

    def __init__(self, config: LLMConfig):
        try:
            from transformers import pipeline  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover
            raise ImportError("transformers is required for local LLM inference.") from exc

        if not config.model_id:
            raise ValueError("LLMConfig.model_id must be provided for TransformersGenerator.")

        task = config.task or "text-generation"
        self._pipeline = pipeline(task, model=config.model_id)
        self._task = task
        self._max_new_tokens = config.max_new_tokens
        self._temperature = config.temperature
        self._top_p = config.top_p

    def __call__(self, prompt: str) -> str:
        outputs = self._pipeline(
            prompt,
            max_new_tokens=self._max_new_tokens,
            temperature=self._temperature,
            top_p=self._top_p,
            do_sample=True,
        )
        if not outputs:
            return ""
        result = outputs[0]
        if isinstance(result, dict):
            return result.get("generated_text") or result.get("summary_text", "").strip()
        return str(result).strip()


def build_generator(config: LLMConfig | None) -> TextGenerator | None:
    if config is None:
        return None
    provider = config.provider.lower()
    if provider == "hf":
        return HuggingFaceGenerator(config)
    if provider == "transformers":
        return TransformersGenerator(config)
    raise ValueError(f"Unsupported LLM provider: {config.provider}")

