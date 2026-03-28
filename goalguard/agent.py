from __future__ import annotations

from collections.abc import Callable

import numpy as np

from goalguard.alignment import compute_semantic_coordinates, compute_similarity
from goalguard.drift import classify_status, is_drifting
from goalguard.intervention import correct_step
from goalguard.types import AgentEvent


class GoalGuard:
    def __init__(
        self,
        agent: object,
        goal: str,
        encoder: Callable[[str], np.ndarray],
        threshold: float = 0.6,
    ) -> None:
        self.agent = agent
        self.goal = goal
        self.encoder = encoder
        self.threshold = threshold
        self.goal_vec = self.encoder(goal)

    def run(self):
        for step_index, raw_step in enumerate(self.agent.run(), start=1):
            raw_vec = self.encoder(raw_step)
            raw_score = compute_similarity(self.goal_vec, raw_vec)
            raw_x, raw_y = compute_semantic_coordinates(self.goal_vec, raw_vec)
            drift = is_drifting(raw_score, self.threshold)

            display_step = raw_step
            final_score = raw_score
            final_x = raw_x
            final_y = raw_y
            corrected = False

            event: AgentEvent = {
                "step_index": step_index,
                "raw_step": raw_step,
                "display_step": display_step,
                "score": final_score,
                "status": classify_status(final_score),
                "drift": drift,
                "corrected": corrected,
                "x_offtopic": final_x,
                "y_toward_goal": final_y,
            }

            if drift:
                corrected = True
                display_step = correct_step(raw_step, self.goal, raw_score)
                corrected_vec = self.encoder(display_step)
                final_score = compute_similarity(self.goal_vec, corrected_vec)
                final_x, final_y = compute_semantic_coordinates(self.goal_vec, corrected_vec)
                event = {
                    "step_index": step_index,
                    "raw_step": raw_step,
                    "display_step": display_step,
                    "score": final_score,
                    "status": classify_status(final_score),
                    "drift": True,
                    "corrected": corrected,
                    "x_offtopic": final_x,
                    "y_toward_goal": final_y,
                    "correction_from_x": raw_x,
                    "correction_from_y": raw_y,
                }

            yield event
