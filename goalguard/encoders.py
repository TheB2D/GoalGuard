from __future__ import annotations

import os
from collections.abc import Callable

import numpy as np

from demo_agent import encode_text


def make_demo_encoder() -> Callable[[str], np.ndarray]:
    return encode_text


def make_gemini_encoder(model_name: str = "models/text-embedding-004") -> Callable[[str], np.ndarray]:
    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise RuntimeError(
            "google-generativeai is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY in environment.")

    genai.configure(api_key=api_key)

    def _encode(text: str) -> np.ndarray:
        result = genai.embed_content(model=model_name, content=text)
        embedding = result["embedding"]
        return np.array(embedding, dtype=float)

    return _encode


def get_encoder(provider: str, gemini_model: str = "models/text-embedding-004") -> Callable[[str], np.ndarray]:
    provider_normalized = provider.strip().lower()
    if provider_normalized == "gemini":
        return make_gemini_encoder(model_name=gemini_model)
    return make_demo_encoder()
