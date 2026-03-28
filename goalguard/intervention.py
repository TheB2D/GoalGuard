from __future__ import annotations


def correct_step(raw_step: str, goal: str, score: float) -> str:
    """
    Preserve continuity so the correction looks causal, not like a hard reset.
    """
    return f"Refocus: {goal} (continuing from: {raw_step})"
