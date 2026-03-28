from __future__ import annotations

import hashlib
import random
import re

import numpy as np


def encode_text(text: str, dim: int = 96) -> np.ndarray:
    """
    Stable placeholder embeddings using token hashing.
    """
    vector = np.zeros(dim, dtype=float)
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    if not tokens:
        return vector

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        magnitude = 1.0 + (digest[5] / 255.0)
        vector[idx] += sign * magnitude

    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


class DemoAgent:
    def __init__(self, task: str, seed: int = 42, max_steps: int = 10) -> None:
        self.task = task
        self.seed = seed
        self.max_steps = max_steps
        self._rng = random.Random(seed)
        self.steps = self._generate_steps()

    def _drift_step(self) -> str:
        mode = self._rng.random()
        if mode < 0.3:
            return "Define an unrelated historical concept in depth."
        if mode < 0.6:
            return "Expand a minor side detail far beyond what is needed."
        return "Insert a broad background tangent not required for the task."

    def _generate_steps(self) -> list[str]:
        steps = [
            f"Read the source and identify the core objective: {self.task}.",
            "Extract the key claims and supporting evidence.",
        ]

        while len(steps) < self.max_steps - 2:
            roll = self._rng.random()
            if roll < 0.4:
                steps.append(self._drift_step())
            elif roll < 0.7:
                steps.append("Continue concise summarization of the main argument.")
            else:
                steps.append("Condense the most important findings into short bullets.")

        steps.extend(
            [
                "Return to a concise summary of the central contribution.",
                "Write a short conclusion focused on actionable takeaways.",
            ]
        )
        return steps[: self.max_steps]

    def run(self):
        for step in self.steps:
            yield step
