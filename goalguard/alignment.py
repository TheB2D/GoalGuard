from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def _safe_unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return np.zeros_like(vec)
    return vec / norm


def compute_similarity(goal_vec: np.ndarray, step_vec: np.ndarray) -> float:
    similarity = cosine_similarity([goal_vec], [step_vec])[0][0]
    return float(similarity)


def compute_semantic_coordinates(goal_vec: np.ndarray, step_vec: np.ndarray) -> tuple[float, float]:
    """
    Return bounded semantic coordinates:
    - y_toward_goal in [-1, 1]
    - x_offtopic in [0, 1]
    """
    goal_unit = _safe_unit(goal_vec)
    step_unit = _safe_unit(step_vec)
    y_toward_goal = float(np.clip(np.dot(step_unit, goal_unit), -1.0, 1.0))
    x_offtopic = float(np.sqrt(max(0.0, 1.0 - y_toward_goal**2)))
    return x_offtopic, y_toward_goal
