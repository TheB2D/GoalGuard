from __future__ import annotations

import os
from collections.abc import Callable


CorrectionFn = Callable[[str, str, float, list[str]], str]


def correct_step(raw_step: str, goal: str, score: float) -> str:
    """
    Preserve continuity so the correction looks causal, not like a hard reset.
    """
    return f"Refocus: {goal} (continuing from: {raw_step})"


def demo_corrector(raw_step: str, goal: str, score: float, history: list[str]) -> str:
    return correct_step(raw_step, goal, score)


class GeminiCorrector:
    def __init__(self, model_name: str = "gemini-2.0-flash-lite") -> None:
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
        self._model = genai.GenerativeModel(model_name)

    def __call__(self, raw_step: str, goal: str, score: float, history: list[str]) -> str:
        history_text = "\n".join(f"- {item}" for item in history[-5:]) or "- (none)"
        if score < 0.45:
            instruction = (
                "The response is severely off-track. Rewrite it to directly and only address the goal."
            )
        else:
            instruction = (
                "The response has slightly drifted. Gently steer it back toward the goal while preserving useful content."
            )

        prompt = (
            f"The following response drifted from the intended goal (alignment score: {score:.2f}).\n\n"
            f"Goal: {goal}\n\n"
            f"Recent context:\n{history_text}\n\n"
            f"Drifted response:\n{raw_step}\n\n"
            f"{instruction}\n"
            "Do not explain what you are doing. Return only the corrected response text."
        )

        response = self._model.generate_content(
            prompt,
            generation_config={"temperature": 0.2},
        )
        corrected = (response.text or "").strip()
        if corrected:
            return corrected
        return demo_corrector(raw_step, goal, score, history)


def get_corrector(provider: str, gemini_model: str = "gemini-2.0-flash-lite") -> CorrectionFn:
    provider_normalized = provider.strip().lower()
    if provider_normalized == "gemini":
        return GeminiCorrector(model_name=gemini_model)
    return demo_corrector
