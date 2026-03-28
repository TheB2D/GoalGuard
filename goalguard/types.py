from __future__ import annotations

from typing import Literal, TypedDict

from typing_extensions import NotRequired


StepStatus = Literal["aligned", "drifting", "off_goal"]


class AgentEvent(TypedDict):
    step_index: int
    raw_step: str
    display_step: str
    score: float
    status: StepStatus
    drift: bool
    corrected: bool
    x_offtopic: float
    y_toward_goal: float
    correction_from_x: NotRequired[float]
    correction_from_y: NotRequired[float]
    prm_step_quality: NotRequired[float]
    prm_value_estimate: NotRequired[float]
    prm_advantage: NotRequired[float]
    prm_uncertainty: NotRequired[float]
    prm_action: NotRequired[str]
    prm_rollouts: NotRequired[int]
    prm_candidate_count: NotRequired[int]
    prm_rollout_success_count: NotRequired[int]
    prm_rollout_terminal_values: NotRequired[list[float]]
    prm_rollout_success_mask: NotRequired[list[bool]]


class SimulationEvent(TypedDict):
    step_index: int
    unguarded: AgentEvent
    guarded: AgentEvent
    alignment_delta: float
    alignment_delta_smooth: float
