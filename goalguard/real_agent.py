from __future__ import annotations

import json
import os
import re
from typing import Any


class GeminiStructuredAgent:
    def __init__(self, task: str, goal: str, model_name: str = "gemini-2.0-flash", max_steps: int = 10) -> None:
        self.task = task
        self.goal = goal
        self.model_name = model_name
        self.max_steps = max_steps
        self.history: list[str] = []
        self._model = self._build_model()

    def _build_model(self):
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
        return genai.GenerativeModel(self.model_name)

    def _build_prompt(self, step_index: int, history: list[str]) -> str:
        history_text = "\n".join(f"- {item}" for item in history[-5:]) or "- (none)"
        return (
            "You are an agent producing one next step toward a goal.\n"
            f"Task: {self.task}\n"
            f"Goal: {self.goal}\n"
            f"Current step index: {step_index}\n"
            "Recent history:\n"
            f"{history_text}\n\n"
            "Return ONLY JSON with schema:\n"
            '{"step_text": string, "step_type": "on_task"|"conceptual_drift"|"overfocus_drift", "confidence": number}'
        )

    def _extract_step_text(self, response_text: str) -> str:
        cleaned = response_text.strip()

        # Handle markdown code fences that some model responses include.
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        # Fallback: extract first JSON object-shaped substring.
        if not cleaned.startswith("{"):
            obj_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if obj_match:
                cleaned = obj_match.group(0).strip()

        try:
            payload: Any = json.loads(cleaned)
            step_text = payload.get("step_text")
            if isinstance(step_text, str) and step_text.strip():
                return step_text.strip()
        except json.JSONDecodeError:
            pass
        return cleaned

    def generate_step(self, step_index: int, history: list[str]) -> str:
        prompt = self._build_prompt(step_index, history)
        response = self._model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.4,
            },
        )
        response_text = (response.text or "").strip()
        if not response_text:
            response_text = '{"step_text": "Continue with the task.", "step_type": "on_task", "confidence": 0.5}'
        return self._extract_step_text(response_text)

    def run(self):
        for step_index in range(1, self.max_steps + 1):
            step_text = self.generate_step(step_index=step_index, history=self.history)
            self.history.append(step_text)
            yield step_text
