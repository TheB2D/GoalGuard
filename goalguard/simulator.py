from __future__ import annotations

from collections import deque
import random
from statistics import mean

import numpy as np

from demo_agent import DemoAgent
from goalguard.agent import GoalGuard
from goalguard.alignment import compute_semantic_coordinates, compute_similarity
from goalguard.drift import classify_status, is_drifting
from goalguard.intervention import CorrectionFn
from goalguard.prm import (
    PseudoPRMConfig,
    choose_prm_action,
    compute_step_quality,
    estimate_mc_value,
)
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


def _with_prm_metrics(
    event: AgentEvent,
    *,
    previous_value: float,
    prm_cfg: PseudoPRMConfig,
    rng: random.Random,
    action_override: str | None = None,
    candidate_count: int = 1,
) -> tuple[AgentEvent, float]:
    value_estimate, uncertainty, terminal_values, success_mask = estimate_mc_value(
        event["score"],
        rollouts_per_step=prm_cfg.rollouts_per_step,
        rollout_depth=prm_cfg.rollout_depth,
        success_threshold=prm_cfg.success_threshold,
        rng=rng,
    )
    advantage = value_estimate - previous_value
    quality = compute_step_quality(event["score"], value_estimate)
    action = action_override or choose_prm_action(
        quality,
        advantage,
        accept_quality_threshold=prm_cfg.accept_quality_threshold,
        weak_quality_threshold=prm_cfg.weak_quality_threshold,
        advantage_rewrite_threshold=prm_cfg.advantage_rewrite_threshold,
    )
    enriched: AgentEvent = {
        **event,
        "prm_step_quality": float(quality),
        "prm_value_estimate": float(value_estimate),
        "prm_advantage": float(advantage),
        "prm_uncertainty": float(uncertainty),
        "prm_action": action,
        "prm_rollouts": prm_cfg.rollouts_per_step,
        "prm_candidate_count": candidate_count,
        "prm_rollout_success_count": int(sum(1 for item in success_mask if item)),
        "prm_rollout_terminal_values": terminal_values,
        "prm_rollout_success_mask": success_mask,
    }
    return enriched, float(value_estimate)


def _candidate_steps(
    *,
    raw_step: str,
    goal: str,
    history: list[str],
    corrector: CorrectionFn | None,
    score_hint: float,
    n_candidates: int,
) -> list[str]:
    # Build a compact candidate set with explicit "goal-centric" alternatives.
    candidates = [raw_step]
    if corrector is not None:
        corrected = corrector(raw_step, goal, score_hint, history)
        if corrected.strip():
            candidates.append(corrected.strip())

    candidates.extend(
        [
            f"Refocus tightly on goal: {goal}",
            "Summarize only the core objective and high-signal evidence.",
            "Produce a concise next step that advances the main goal directly.",
        ]
    )

    unique: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        key = item.strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(key)
        if len(unique) >= max(1, n_candidates):
            break
    return unique


def _best_prm_guarded_event(
    *,
    step_index: int,
    raw_step: str,
    goal: str,
    goal_vec: np.ndarray,
    encoder,
    threshold: float,
    history: list[str],
    corrector: CorrectionFn | None,
    previous_value: float,
    prm_cfg: PseudoPRMConfig,
    rng: random.Random,
) -> tuple[AgentEvent, float]:
    baseline = _unguarded_event(
        step_index=step_index,
        step_text=raw_step,
        goal_vec=goal_vec,
        encoder=encoder,
        threshold=threshold,
    )
    candidates = _candidate_steps(
        raw_step=raw_step,
        goal=goal,
        history=history,
        corrector=corrector,
        score_hint=baseline["score"],
        n_candidates=prm_cfg.n_candidates,
    )

    ranked: list[tuple[float, AgentEvent, float]] = []
    for candidate in candidates:
        candidate_event = _unguarded_event(
            step_index=step_index,
            step_text=candidate,
            goal_vec=goal_vec,
            encoder=encoder,
            threshold=threshold,
        )
        enriched, _value = _with_prm_metrics(
            candidate_event,
            previous_value=previous_value,
            prm_cfg=prm_cfg,
            rng=rng,
            action_override="accept",
            candidate_count=len(candidates),
        )
        ranked.append((enriched["prm_step_quality"], enriched, _value))

    ranked.sort(key=lambda item: item[0], reverse=True)
    best = ranked[0][1]
    final_value = ranked[0][2]
    best["corrected"] = best["display_step"] != raw_step
    if best["corrected"]:
        best["drift"] = True
        best["correction_from_x"] = baseline["x_offtopic"]
        best["correction_from_y"] = baseline["y_toward_goal"]
        best["raw_step"] = raw_step

    # If the winner is weak, request a rewrite instead of blind acceptance.
    if best["prm_step_quality"] < prm_cfg.accept_quality_threshold:
        best["prm_action"] = "rewrite"
    return best, final_value


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
    prm_n_candidates: int = 4,
    prm_rollouts_per_step: int = 24,
    prm_rollout_depth: int = 4,
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
            prm_n_candidates=prm_n_candidates,
            prm_rollouts_per_step=prm_rollouts_per_step,
            prm_rollout_depth=prm_rollout_depth,
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
    prm_n_candidates: int = 4,
    prm_rollouts_per_step: int = 24,
    prm_rollout_depth: int = 4,
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
    rng = random.Random(seed)
    prm_cfg = PseudoPRMConfig(
        n_candidates=max(1, prm_n_candidates),
        rollouts_per_step=max(10, prm_rollouts_per_step),
        rollout_depth=max(1, prm_rollout_depth),
    )
    unguarded_prev_value = 0.5
    guarded_prev_value = 0.5

    def _emit(step_index: int, unguarded_event: AgentEvent, guarded_event: AgentEvent):
        alignment_delta = guarded_event["score"] - unguarded_event["score"]
        delta_window.append(alignment_delta)
        alignment_delta_smooth = float(mean(delta_window))
        return {
            "step_index": step_index,
            "unguarded": unguarded_event,
            "guarded": guarded_event,
            "alignment_delta": float(alignment_delta),
            "alignment_delta_smooth": alignment_delta_smooth,
        }

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
            unguarded_event, unguarded_prev_value = _with_prm_metrics(
                unguarded_event,
                previous_value=unguarded_prev_value,
                prm_cfg=prm_cfg,
                rng=rng,
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
            guarded_event, guarded_prev_value = _with_prm_metrics(
                guarded_event,
                previous_value=guarded_prev_value,
                prm_cfg=prm_cfg,
                rng=rng,
            )
            guarded_history.append(guarded_event["display_step"])

            yield _emit(step_index, unguarded_event, guarded_event)
        return

    if provider == "gemini":
        base_agent = GeminiStructuredAgent(task=task, goal=goal, model_name=agent_model, max_steps=max_steps)
    else:
        base_agent = DemoAgent(task=task, seed=seed, max_steps=max_steps)

    # Fair replay mode: both lanes start from identical baseline steps.
    shared_steps = list(base_agent.run())
    agent1 = _StepSequenceAgent(shared_steps)
    agent2 = _StepSequenceAgent(shared_steps)
    if mode in {"prm_mc", "prm_best_of_n"}:
        guarded_history: list[str] = []
        for step_index, unguarded_step in enumerate(agent1.run(), start=1):
            unguarded_event = _unguarded_event(
                step_index=step_index,
                step_text=unguarded_step,
                goal_vec=goal_vec,
                encoder=encoder,
                threshold=threshold,
            )
            unguarded_event, unguarded_prev_value = _with_prm_metrics(
                unguarded_event,
                previous_value=unguarded_prev_value,
                prm_cfg=prm_cfg,
                rng=rng,
            )

            if mode == "prm_best_of_n":
                guarded_event, guarded_prev_value = _best_prm_guarded_event(
                    step_index=step_index,
                    raw_step=unguarded_step,
                    goal=goal,
                    goal_vec=goal_vec,
                    encoder=encoder,
                    threshold=threshold,
                    history=guarded_history,
                    corrector=corrector,
                    previous_value=guarded_prev_value,
                    prm_cfg=prm_cfg,
                    rng=rng,
                )
            else:
                base_guarded = _unguarded_event(
                    step_index=step_index,
                    step_text=unguarded_step,
                    goal_vec=goal_vec,
                    encoder=encoder,
                    threshold=threshold,
                )
                guarded_event, guarded_prev_value = _with_prm_metrics(
                    base_guarded,
                    previous_value=guarded_prev_value,
                    prm_cfg=prm_cfg,
                    rng=rng,
                    candidate_count=1,
                )
                if guarded_event["prm_action"] != "accept" and corrector is not None:
                    corrected = corrector(
                        unguarded_step,
                        goal,
                        guarded_event["score"],
                        guarded_history,
                    )
                    corrected_event = _unguarded_event(
                        step_index=step_index,
                        step_text=corrected,
                        goal_vec=goal_vec,
                        encoder=encoder,
                        threshold=threshold,
                    )
                    corrected_event, guarded_prev_value = _with_prm_metrics(
                        corrected_event,
                        previous_value=guarded_prev_value,
                        prm_cfg=prm_cfg,
                        rng=rng,
                        action_override="rewrite",
                        candidate_count=1,
                    )
                    corrected_event["corrected"] = True
                    corrected_event["drift"] = True
                    corrected_event["raw_step"] = unguarded_step
                    corrected_event["correction_from_x"] = base_guarded["x_offtopic"]
                    corrected_event["correction_from_y"] = base_guarded["y_toward_goal"]
                    guarded_event = corrected_event

            guarded_history.append(guarded_event["display_step"])
            yield _emit(step_index, unguarded_event, guarded_event)
        return

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
        unguarded_event, unguarded_prev_value = _with_prm_metrics(
            unguarded_event,
            previous_value=unguarded_prev_value,
            prm_cfg=prm_cfg,
            rng=rng,
        )
        guarded_event, guarded_prev_value = _with_prm_metrics(
            guarded_event,
            previous_value=guarded_prev_value,
            prm_cfg=prm_cfg,
            rng=rng,
        )

        yield _emit(step_index, unguarded_event, guarded_event)


class _StepSequenceAgent:
    def __init__(self, steps: list[str]) -> None:
        self._steps = steps

    def run(self):
        for step in self._steps:
            yield step


class _NoopAgent:
    def run(self):
        return iter(())
