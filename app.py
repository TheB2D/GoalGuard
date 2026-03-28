from __future__ import annotations

import time

import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv

from goalguard.config import load_config
from goalguard.encoders import get_encoder
from goalguard.intervention import get_corrector
from goalguard.simulator import run_simulation_stream
from goalguard.types import SimulationEvent, StepStatus


STATUS_LABEL = {
    "aligned": "✅ ALIGNED",
    "drifting": "⚠️ DRIFTING",
    "off_goal": "❌ OFF-GOAL",
}


def _init_state() -> None:
    defaults = {
        "running": False,
        "simulation_stream": None,
        "unguarded_logs": [],
        "guarded_logs": [],
        "unguarded_scores": [],
        "guarded_scores": [],
        "unguarded_x": [],
        "unguarded_y": [],
        "guarded_x": [],
        "guarded_y": [],
        "delta_values": [],
        "delta_smooth_values": [],
        "correction_segments": [],
        "last_event": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _status_indicator(status: StepStatus) -> str:
    return STATUS_LABEL.get(status, "⚪ UNKNOWN")


def _append_event(event: SimulationEvent) -> None:
    unguarded = event["unguarded"]
    guarded = event["guarded"]
    idx = event["step_index"]

    st.session_state.unguarded_logs.append(f"{idx}. {unguarded['display_step']}")
    st.session_state.guarded_logs.append(f"{idx}. {guarded['display_step']}")
    st.session_state.unguarded_scores.append(unguarded["score"])
    st.session_state.guarded_scores.append(guarded["score"])
    st.session_state.unguarded_x.append(unguarded["x_offtopic"])
    st.session_state.unguarded_y.append(unguarded["y_toward_goal"])
    st.session_state.guarded_x.append(guarded["x_offtopic"])
    st.session_state.guarded_y.append(guarded["y_toward_goal"])
    st.session_state.delta_values.append(event["alignment_delta"])
    st.session_state.delta_smooth_values.append(event["alignment_delta_smooth"])

    if guarded.get("corrected", False) and "correction_from_x" in guarded and "correction_from_y" in guarded:
        st.session_state.correction_segments.append(
            (
                (guarded["correction_from_x"], guarded["correction_from_y"]),
                (guarded["x_offtopic"], guarded["y_toward_goal"]),
            )
        )

    st.session_state.last_event = event


def _plot_alignment_scores():
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.axhspan(0.7, 1.0, alpha=0.1)
    ax.axhspan(0.4, 0.7, alpha=0.1)
    ax.axhspan(-1.0, 0.4, alpha=0.1)

    if st.session_state.unguarded_scores:
        steps = list(range(1, len(st.session_state.unguarded_scores) + 1))
        ax.plot(steps, st.session_state.unguarded_scores, marker="o", label="Unguarded")
        ax.plot(steps, st.session_state.guarded_scores, marker="o", label="Guarded")

    ax.set_ylim(-1.0, 1.05)
    ax.set_xlabel("Step")
    ax.set_ylabel("Alignment Score")
    ax.set_title("Alignment Over Time")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    return fig


def _plot_trajectory():
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.scatter([0.0], [1.0], marker="*", s=200, label="Goal Anchor")

    if st.session_state.unguarded_x:
        ax.plot(
            st.session_state.unguarded_x,
            st.session_state.unguarded_y,
            marker="o",
            label="Unguarded Path",
        )
        ax.plot(
            st.session_state.guarded_x,
            st.session_state.guarded_y,
            marker="o",
            label="Guarded Path",
        )

    for start, end in st.session_state.correction_segments:
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            linestyle="--",
            linewidth=1.5,
        )
        ax.scatter([end[0]], [end[1]], s=80, marker="D")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-1.0, 1.05)
    ax.set_xlabel("Off-topic-ness (x_offtopic)")
    ax.set_ylabel("Toward-goal Progress (y_toward_goal)")
    ax.set_title("Goal-Anchor Semantic Trajectory")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    return fig


def _plot_delta():
    fig, ax = plt.subplots(figsize=(8, 3.0))
    if st.session_state.delta_values:
        steps = list(range(1, len(st.session_state.delta_values) + 1))
        ax.plot(steps, st.session_state.delta_values, marker="o", label="Delta")
        ax.plot(steps, st.session_state.delta_smooth_values, linewidth=2, label="Delta (Smoothed)")
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Step")
    ax.set_ylabel("Guarded - Unguarded")
    ax.set_title("Alignment Delta")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    return fig


def _reset_run_state() -> None:
    st.session_state.unguarded_logs = []
    st.session_state.guarded_logs = []
    st.session_state.unguarded_scores = []
    st.session_state.guarded_scores = []
    st.session_state.unguarded_x = []
    st.session_state.unguarded_y = []
    st.session_state.guarded_x = []
    st.session_state.guarded_y = []
    st.session_state.delta_values = []
    st.session_state.delta_smooth_values = []
    st.session_state.correction_segments = []
    st.session_state.last_event = None


def _render_dashboard() -> None:
    current_unguarded_status: StepStatus = "aligned"
    current_guarded_status: StepStatus = "aligned"
    if st.session_state.last_event:
        current_unguarded_status = st.session_state.last_event["unguarded"]["status"]
        current_guarded_status = st.session_state.last_event["guarded"]["status"]

    left, right = st.columns(2)
    with left:
        st.subheader("Unguarded Agent")
        st.markdown(_status_indicator(current_unguarded_status))
        st.code("\n".join(st.session_state.unguarded_logs[-12:]) or "No steps yet.")
    with right:
        st.subheader("Guarded Agent")
        st.markdown(_status_indicator(current_guarded_status))
        st.code("\n".join(st.session_state.guarded_logs[-12:]) or "No steps yet.")

    if st.session_state.last_event:
        if st.session_state.last_event["unguarded"]["drift"]:
            st.warning("⚠️ Drift detected in unguarded lane.")
        if st.session_state.last_event["guarded"]["corrected"]:
            st.success("✅ Correction applied: returning to goal trajectory.")

    kpi_a, kpi_b = st.columns(2)
    current_delta = st.session_state.delta_values[-1] if st.session_state.delta_values else 0.0
    current_delta_smooth = (
        st.session_state.delta_smooth_values[-1] if st.session_state.delta_smooth_values else 0.0
    )
    with kpi_a:
        st.metric("Alignment Delta", f"{current_delta:+.3f}")
    with kpi_b:
        st.metric("Alignment Delta (Smoothed)", f"{current_delta_smooth:+.3f}")

    st.pyplot(_plot_alignment_scores())
    st.pyplot(_plot_trajectory())
    st.pyplot(_plot_delta())


def main() -> None:
    st.set_page_config(page_title="GoalGuard Race Demo", layout="wide")
    _init_state()
    load_dotenv(dotenv_path=".env")
    config = load_config()

    st.title("GoalGuard Race Demo")
    st.caption("Side-by-side race: unguarded agent vs goal-guarded agent")

    default_goal = config["defaults"]["goal"]
    default_task = config["defaults"]["task"]
    goal = st.text_input("Goal", value=default_goal)
    task = st.text_input("Task", value=default_task)

    controls_a, controls_b, controls_c = st.columns(3)
    with controls_a:
        threshold = st.slider(
            "Drift Threshold",
            min_value=0.3,
            max_value=0.9,
            value=float(config["simulation"]["threshold"]),
            step=0.05,
        )
    with controls_b:
        speed = st.slider("Speed (seconds per step)", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
    with controls_c:
        max_steps = st.slider(
            "Max Steps",
            min_value=6,
            max_value=20,
            value=int(config["simulation"]["max_steps"]),
            step=1,
        )

    backend_a, backend_b = st.columns(2)
    with backend_a:
        embedding_provider = st.selectbox(
            "Embedding Backend",
            options=["demo", "gemini"],
            index=0 if config["embedding"]["provider"] == "demo" else 1,
        )
    with backend_b:
        agent_provider = st.selectbox(
            "Agent Backend",
            options=["demo", "gemini"],
            index=0 if config["agent"]["provider"] == "demo" else 1,
        )
    simulation_mode = st.selectbox(
        "Simulation Mode",
        options=["fair_replay", "steering"],
        index=0 if config["simulation"].get("mode", "fair_replay") == "fair_replay" else 1,
        help="fair_replay keeps identical base steps for both lanes. steering lets guarded corrections influence future steps.",
    )

    btn_a, btn_b, btn_c = st.columns(3)
    with btn_a:
        start_clicked = st.button("Start")
    with btn_b:
        stop_clicked = st.button("Stop")
    with btn_c:
        reset_clicked = st.button("Reset")

    if start_clicked:
        _reset_run_state()
        try:
            encoder = get_encoder(
                provider=embedding_provider,
                gemini_model=config["embedding"]["gemini_model"],
            )
            corrector = get_corrector(
                provider=config["intervention"]["provider"],
                gemini_model=config["intervention"]["gemini_model"],
            )
            st.session_state.simulation_stream = run_simulation_stream(
                task=task,
                goal=goal,
                encoder=encoder,
                threshold=threshold,
                seed=int(config["simulation"]["seed"]),
                max_steps=max_steps,
                agent_provider=agent_provider,
                agent_model=config["agent"]["gemini_model"],
                simulation_mode=simulation_mode,
                corrector=corrector,
            )
        except Exception as exc:
            st.error(f"Failed to start simulation: {exc}")
            st.session_state.running = False
            st.session_state.simulation_stream = None
            _render_dashboard()
            return
        st.session_state.running = True

    if stop_clicked:
        st.session_state.running = False

    if reset_clicked:
        st.session_state.running = False
        st.session_state.simulation_stream = None
        _reset_run_state()

    if st.session_state.running and st.session_state.simulation_stream is not None:
        placeholder = st.empty()
        for event in st.session_state.simulation_stream:
            if not st.session_state.running:
                break
            _append_event(event)
            with placeholder.container():
                _render_dashboard()
            time.sleep(speed)
        st.session_state.running = False
        st.session_state.simulation_stream = None
        return

    _render_dashboard()


if __name__ == "__main__":
    main()
