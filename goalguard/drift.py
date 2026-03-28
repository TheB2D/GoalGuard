from __future__ import annotations

from goalguard.types import StepStatus


ALIGNED_THRESHOLD = 0.7
DRIFTING_THRESHOLD = 0.4


def classify_status(
    score: float,
    aligned_threshold: float = ALIGNED_THRESHOLD,
    drifting_threshold: float = DRIFTING_THRESHOLD,
) -> StepStatus:
    if score >= aligned_threshold:
        return "aligned"
    if score >= drifting_threshold:
        return "drifting"
    return "off_goal"


def is_drifting(score: float, threshold: float = 0.6) -> bool:
    return score < threshold
