from __future__ import annotations

from collections import deque
from statistics import mean

import numpy as np

from demo_agent import DemoAgent
from goalguard.agent import GoalGuard
from goalguard.alignment import compute_semantic_coordinates, compute_similarity
from goalguard.drift import classify_status, is_drifting
from goalguard.intervention import CorrectionFn
from goalguard.real_agent import GeminiStructuredAgent
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
    agent_provider: str = "demo",
    agent_model: str = "gemini-2.0-flash",
    simulation_mode: str = "fair_replay",
    corrector: CorrectionFn | None = None,
) -> list[SimulationEvent]:
    return list(
        run_simulation_stream(
            task=task,
            goal=goal,
            encoder=encoder,
            threshold=threshold,
            seed=seed,
            max_steps=max_steps,
            agent_provider=agent_provider,
            agent_model=agent_model,
            simulation_mode=simulation_mode,
            corrector=corrector,
        )
    )


def run_simulation_stream(
    task: str,
    goal: str,
    encoder,
    threshold: float = 0.6,
    seed: int = 42,
    max_steps: int = 10,
    agent_provider: str = "demo",
    agent_model: str = "gemini-2.0-flash",
    simulation_mode: str = "fair_replay",
    corrector: CorrectionFn | None = None,
):
    """
    Run two agents with synchronized seeds:
    - unguarded baseline
    - guarded baseline with GoalGuard intervention
    """
    goal_vec = encoder(goal)
    delta_window: deque[float] = deque(maxlen=3)
    provider = agent_provider.strip().lower()
    mode = simulation_mode.strip().lower()

    if mode == "steering" and provider == "gemini":
        unguarded_agent = GeminiStructuredAgent(
            task=task, goal=goal, model_name=agent_model, max_steps=max_steps
        )
        guarded_generator_agent = GeminiStructuredAgent(
            task=task, goal=goal, model_name=agent_model, max_steps=max_steps
        )
        guarded_agent = GoalGuard(
            agent=_NoopAgent(),
            goal=goal,
            encoder=encoder,
            threshold=threshold,
            corrector=corrector,
        )

        unguarded_history: list[str] = []
        guarded_history: list[str] = []
        for step_index in range(1, max_steps + 1):
            unguarded_step = unguarded_agent.generate_step(
                step_index=step_index,
                history=unguarded_history,
            )
            unguarded_history.append(unguarded_step)
            unguarded_event = _unguarded_event(
                step_index=step_index,
                step_text=unguarded_step,
                goal_vec=goal_vec,
                encoder=encoder,
                threshold=threshold,
            )

            guarded_raw_step = guarded_generator_agent.generate_step(
                step_index=step_index,
                history=guarded_history,
            )
            guarded_event = guarded_agent.process_step(
                step_index=step_index,
                raw_step=guarded_raw_step,
                history=guarded_history,
            )
            guarded_history.append(guarded_event["display_step"])

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
        return

    if provider == "gemini":
        base_agent = GeminiStructuredAgent(task=task, goal=goal, model_name=agent_model, max_steps=max_steps)
    else:
        base_agent = DemoAgent(task=task, seed=seed, max_steps=max_steps)

    # Fair replay mode: both lanes start from identical baseline steps.
    shared_steps = list(base_agent.run())
    agent1 = _StepSequenceAgent(shared_steps)
    agent2 = _StepSequenceAgent(shared_steps)
    guarded_agent = GoalGuard(
        agent=agent2,
        goal=goal,
        encoder=encoder,
        threshold=threshold,
        corrector=corrector,
    )

    guarded_events = guarded_agent.run()
    for step_index, (unguarded_step, guarded_event) in enumerate(zip(agent1.run(), guarded_events), start=1):
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


class _StepSequenceAgent:
    def __init__(self, steps: list[str]) -> None:
        self._steps = steps

    def run(self):
        for step in self._steps:
            yield step


class _NoopAgent:
    def run(self):
        return iter(())
