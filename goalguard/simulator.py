from __future__ import annotations

from collections import deque
from statistics import mean

import numpy as np

from demo_agent import DemoAgent
from goalguard.agent import GoalGuard
from goalguard.alignment import compute_semantic_coordinates, compute_similarity
from goalguard.drift import classify_status, is_drifting
from goalguard.types import AgentEvent, SimulationEvent


def _unguarded_event(
    step_index: int,
    step_text: str,
    goal_vec: np.ndarray,
    encoder,
    threshold: float,
) -> AgentEvent:
    step_vec = encoder(step_text)
    score = compute_similarity(goal_vec, step_vec)
    x_offtopic, y_toward_goal = compute_semantic_coordinates(goal_vec, step_vec)
    return {
        "step_index": step_index,
        "raw_step": step_text,
        "display_step": step_text,
        "score": score,
        "status": classify_status(score),
        "drift": is_drifting(score, threshold),
        "corrected": False,
        "x_offtopic": x_offtopic,
        "y_toward_goal": y_toward_goal,
    }


def run_simulation(
    task: str,
    goal: str,
    encoder,
    threshold: float = 0.6,
    seed: int = 42,
    max_steps: int = 10,
) -> list[SimulationEvent]:
    return list(
        run_simulation_stream(
            task=task,
            goal=goal,
            encoder=encoder,
            threshold=threshold,
            seed=seed,
            max_steps=max_steps,
        )
    )


def run_simulation_stream(
    task: str,
    goal: str,
    encoder,
    threshold: float = 0.6,
    seed: int = 42,
    max_steps: int = 10,
):
    """
    Run two agents with synchronized seeds:
    - unguarded baseline
    - guarded baseline with GoalGuard intervention
    """
    agent1 = DemoAgent(task=task, seed=seed, max_steps=max_steps)
    agent2 = DemoAgent(task=task, seed=seed, max_steps=max_steps)
    guarded_agent = GoalGuard(agent=agent2, goal=goal, encoder=encoder, threshold=threshold)

    goal_vec = encoder(goal)
    guarded_events = guarded_agent.run()
    delta_window: deque[float] = deque(maxlen=3)

    for step_index, (unguarded_step, guarded_event) in enumerate(
        zip(agent1.run(), guarded_events), start=1
    ):
        unguarded_event = _unguarded_event(
            step_index=step_index,
            step_text=unguarded_step,
            goal_vec=goal_vec,
            encoder=encoder,
            threshold=threshold,
        )

        alignment_delta = guarded_event["score"] - unguarded_event["score"]
        delta_window.append(alignment_delta)
        alignment_delta_smooth = float(mean(delta_window))

        yield {
            "step_index": step_index,
            "unguarded": unguarded_event,
            "guarded": guarded_event,
            "alignment_delta": float(alignment_delta),
            "alignment_delta_smooth": alignment_delta_smooth,
        }
