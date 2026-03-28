from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class PseudoPRMConfig:
    n_candidates: int = 4
    rollouts_per_step: int = 24
    rollout_depth: int = 4
    success_threshold: float = 0.65
    accept_quality_threshold: float = 0.62
    weak_quality_threshold: float = 0.52
    advantage_rewrite_threshold: float = -0.04


def score_to_unit_interval(score: float) -> float:
    return max(0.0, min(1.0, (score + 1.0) / 2.0))


def estimate_mc_value(
    score: float,
    *,
    rollouts_per_step: int,
    rollout_depth: int,
    success_threshold: float,
    rng: random.Random,
) -> tuple[float, float, list[float], list[bool]]:
    """
    Estimate expected final success by sampling stochastic continuations
    from a single-step alignment score.
    """
    start = score_to_unit_interval(score)
    terminal_values: list[float] = []
    success_mask: list[bool] = []
    for _ in range(max(1, rollouts_per_step)):
        current = start
        for _depth in range(max(1, rollout_depth)):
            drift_chance = 0.18 + (1.0 - current) * 0.55
            if rng.random() < drift_chance:
                current -= rng.uniform(0.05, 0.20)
            else:
                current += rng.uniform(0.03, 0.17)
            current = max(0.0, min(1.0, current))
        success = current >= success_threshold
        terminal_values.append(current)
        success_mask.append(success)

    n = max(1, rollouts_per_step)
    successes = sum(1 for item in success_mask if item)
    value_estimate = successes / n
    uncertainty = math.sqrt(value_estimate * (1.0 - value_estimate) / n)
    return value_estimate, uncertainty, terminal_values, success_mask


def compute_step_quality(score: float, value_estimate: float) -> float:
    semantic_progress = score_to_unit_interval(score)
    return 0.45 * semantic_progress + 0.55 * value_estimate


def choose_prm_action(
    quality: float,
    advantage: float,
    *,
    accept_quality_threshold: float,
    weak_quality_threshold: float,
    advantage_rewrite_threshold: float,
) -> str:
    if quality >= accept_quality_threshold and advantage >= advantage_rewrite_threshold:
        return "accept"
    if quality >= weak_quality_threshold or advantage > -0.12:
        return "rewrite"
    return "resample"
