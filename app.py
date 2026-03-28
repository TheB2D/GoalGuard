from __future__ import annotations

import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
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
        "unguarded_prm_quality": [],
        "guarded_prm_quality": [],
        "unguarded_prm_value": [],
        "guarded_prm_value": [],
        "unguarded_prm_advantage": [],
        "guarded_prm_advantage": [],
        "unguarded_prm_uncertainty": [],
        "guarded_prm_uncertainty": [],
        "guarded_prm_actions": [],
        "guarded_prm_rollouts": [],
        "guarded_prm_candidates": [],
        "guarded_rollout_values_history": [],
        "guarded_rollout_success_history": [],
        "mc_spin_grid": None,
        "mc_spin_rng": np.random.default_rng(42),
        "mc_last_temp": 0.0,
        "mc_last_field": 0.0,
        "mc_lattice_size": 64,
        "mc_sweeps_per_step": 3,
        "mc_base_temp": 1.6,
        "mc_temp_scale": 2.2,
        "mc_field_gain": 1.3,
        "mc_coupling": 1.0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _status_indicator(status: StepStatus) -> str:
    return STATUS_LABEL.get(status, "⚪ UNKNOWN")


def _metropolis_sweep(
    grid: np.ndarray,
    *,
    temperature: float,
    field: float,
    coupling: float,
    rng: np.random.Generator,
) -> None:
    rows, cols = grid.shape
    inv_temp = 1.0 / max(temperature, 1e-6)
    for _ in range(rows * cols):
        i = int(rng.integers(0, rows))
        j = int(rng.integers(0, cols))
        s = grid[i, j]
        nn = (
            grid[(i - 1) % rows, j]
            + grid[(i + 1) % rows, j]
            + grid[i, (j - 1) % cols]
            + grid[i, (j + 1) % cols]
        )
        d_e = 2.0 * s * (coupling * nn + field)
        if d_e <= 0.0 or float(rng.random()) < float(np.exp(-d_e * inv_temp)):
            grid[i, j] = -s


def _evolve_spin_lattice(guarded_event: dict) -> None:
    size = int(st.session_state.mc_lattice_size)
    size = max(8, size)
    rng: np.random.Generator = st.session_state.mc_spin_rng
    grid = st.session_state.mc_spin_grid
    if grid is None or grid.shape != (size, size):
        grid = rng.choice(np.array([-1.0, 1.0], dtype=float), size=(size, size))

    value_estimate = float(guarded_event.get("prm_value_estimate", 0.5))
    uncertainty = float(guarded_event.get("prm_uncertainty", 0.2))
    field = (2.0 * value_estimate - 1.0) * float(st.session_state.mc_field_gain)
    temperature = float(st.session_state.mc_base_temp) + float(st.session_state.mc_temp_scale) * uncertainty
    coupling = float(st.session_state.mc_coupling)

    # Inject rollout outcomes as a weak observation field, then relax with Metropolis sweeps.
    rollout_mask = guarded_event.get("prm_rollout_success_mask", [])
    if rollout_mask:
        m = min(len(rollout_mask), size * size)
        idx = rng.permutation(size * size)[:m]
        spin_obs = np.array([1.0 if item else -1.0 for item in rollout_mask[:m]], dtype=float)
        flat = grid.ravel()
        flat[idx] = 0.75 * flat[idx] + 0.25 * spin_obs
        grid = np.where(flat.reshape(size, size) >= 0.0, 1.0, -1.0)

    for _ in range(max(1, int(st.session_state.mc_sweeps_per_step))):
        _metropolis_sweep(
            grid,
            temperature=temperature,
            field=field,
            coupling=coupling,
            rng=rng,
        )

    st.session_state.mc_spin_grid = grid
    st.session_state.mc_last_temp = temperature
    st.session_state.mc_last_field = field


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
    st.session_state.unguarded_prm_quality.append(unguarded.get("prm_step_quality", 0.0))
    st.session_state.guarded_prm_quality.append(guarded.get("prm_step_quality", 0.0))
    st.session_state.unguarded_prm_value.append(unguarded.get("prm_value_estimate", 0.0))
    st.session_state.guarded_prm_value.append(guarded.get("prm_value_estimate", 0.0))
    st.session_state.unguarded_prm_advantage.append(unguarded.get("prm_advantage", 0.0))
    st.session_state.guarded_prm_advantage.append(guarded.get("prm_advantage", 0.0))
    st.session_state.unguarded_prm_uncertainty.append(unguarded.get("prm_uncertainty", 0.0))
    st.session_state.guarded_prm_uncertainty.append(guarded.get("prm_uncertainty", 0.0))
    st.session_state.guarded_prm_actions.append(guarded.get("prm_action", "n/a"))
    st.session_state.guarded_prm_rollouts.append(guarded.get("prm_rollouts", 0))
    st.session_state.guarded_prm_candidates.append(guarded.get("prm_candidate_count", 1))
    st.session_state.guarded_rollout_values_history.append(guarded.get("prm_rollout_terminal_values", []))
    st.session_state.guarded_rollout_success_history.append(guarded.get("prm_rollout_success_mask", []))
    _evolve_spin_lattice(guarded)

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
        ax.legend(loc="lower right")

    ax.set_ylim(-1.0, 1.05)
    ax.set_xlabel("Step")
    ax.set_ylabel("Alignment Score")
    ax.set_title("Alignment Over Time")
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
        ax.legend(loc="lower right")

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
    ax.grid(alpha=0.2)
    return fig


def _plot_delta():
    fig, ax = plt.subplots(figsize=(8, 3.0))
    if st.session_state.delta_values:
        steps = list(range(1, len(st.session_state.delta_values) + 1))
        ax.plot(steps, st.session_state.delta_values, marker="o", label="Delta")
        ax.plot(steps, st.session_state.delta_smooth_values, linewidth=2, label="Delta (Smoothed)")
        ax.legend(loc="lower right")
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Step")
    ax.set_ylabel("Guarded - Unguarded")
    ax.set_title("Alignment Delta")
    ax.grid(alpha=0.2)
    return fig


def _plot_prm_value_quality():
    fig, ax = plt.subplots(figsize=(8, 3.2))
    if st.session_state.guarded_prm_value:
        steps = list(range(1, len(st.session_state.guarded_prm_value) + 1))
        guarded_value = st.session_state.guarded_prm_value
        unguarded_value = st.session_state.unguarded_prm_value
        quality = st.session_state.guarded_prm_quality
        uncertainty = st.session_state.guarded_prm_uncertainty
        low = [max(0.0, v - u) for v, u in zip(guarded_value, uncertainty)]
        high = [min(1.0, v + u) for v, u in zip(guarded_value, uncertainty)]
        ax.fill_between(steps, low, high, color="#4ba6ff", alpha=0.18, label="MC ± uncertainty")
        ax.plot(steps, unguarded_value, linewidth=2.0, color="#ff9aa2", label="Unguarded V(s)")
        ax.plot(steps, guarded_value, linewidth=2.4, color="#62daff", label="Guarded V(s)")
        ax.plot(steps, quality, linewidth=2.2, color="#9ef5bf", label="Guarded Step Quality")
        ax.legend(loc="lower right")

    ax.set_ylim(0.0, 1.02)
    ax.set_title("Pseudo-PRM Value and Quality")
    ax.set_xlabel("Step")
    ax.set_ylabel("Score (0-1)")
    ax.grid(alpha=0.2)
    return fig


def _plot_prm_advantage():
    fig, ax = plt.subplots(figsize=(8, 2.8))
    if st.session_state.guarded_prm_advantage:
        steps = list(range(1, len(st.session_state.guarded_prm_advantage) + 1))
        values = st.session_state.guarded_prm_advantage
        colors = ["#5ef3c3" if v >= 0 else "#ff8f8f" for v in values]
        ax.bar(steps, values, width=0.64, color=colors, alpha=0.8, label="Guarded Advantage")
        ax.plot(steps, st.session_state.unguarded_prm_advantage, linewidth=1.7, color="#f2da88", label="Unguarded Advantage")
        ax.legend(loc="upper right")
    ax.axhline(0.0, linestyle="--", linewidth=1.1)
    ax.set_title("Progress Advantage (Delta-to-goal)")
    ax.set_xlabel("Step")
    ax.set_ylabel("A(s, a)")
    ax.grid(alpha=0.2)
    return fig


def _plot_mc_rain_of_dots():
    fig, ax = plt.subplots(figsize=(8, 3.0))
    if not st.session_state.guarded_rollout_values_history:
        ax.set_title("Monte Carlo Rain of Dots")
        ax.text(0.5, 0.5, "No rollout samples yet.", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return fig

    values = st.session_state.guarded_rollout_values_history[-1]
    success_mask = st.session_state.guarded_rollout_success_history[-1]
    n = len(values)
    cols = 12
    xs = []
    ys = []
    colors = []
    for i, (value, success) in enumerate(zip(values, success_mask)):
        row = i // cols
        col = i % cols
        jitter = ((i * 37) % 17) / 60.0
        xs.append(col + jitter)
        ys.append(max(0.0, 1.0 - row * 0.12))
        colors.append("#2ca02c" if success else "#d62728")

    ax.scatter(xs, ys, s=58, c=colors, alpha=0.75, edgecolors="white", linewidth=0.6)
    estimate = st.session_state.guarded_prm_value[-1] if st.session_state.guarded_prm_value else 0.0
    ax.set_title(f"Monte Carlo Rain of Dots (latest step): success={estimate:.3f} over n={n}")
    ax.set_xlabel("Sample batch position")
    ax.set_ylabel("Drop level")
    ax.set_yticks([])
    ax.grid(alpha=0.2)
    return fig


def _plot_mc_density_wave():
    fig, ax = plt.subplots(figsize=(8, 3.0))
    if not st.session_state.guarded_rollout_values_history:
        ax.set_title("Monte Carlo Density Wave")
        ax.text(0.5, 0.5, "No rollout samples yet.", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return fig

    values = st.session_state.guarded_rollout_values_history[-1]
    ax.hist(values, bins=12, color="#1f77b4", alpha=0.65, edgecolor="white")
    estimate = st.session_state.guarded_prm_value[-1] if st.session_state.guarded_prm_value else 0.0
    ax.axvline(estimate, linestyle="--", linewidth=2.0, color="#ff7f0e", label=f"mean={estimate:.3f}")
    ax.set_xlim(0.0, 1.0)
    ax.set_title("Monte Carlo Density Wave (latest rollout outcomes)")
    ax.set_xlabel("Terminal rollout value")
    ax.set_ylabel("Count")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2)
    return fig


def _plot_mc_spin_lattice():
    fig, ax = plt.subplots(figsize=(8, 3.3))
    if st.session_state.mc_spin_grid is None:
        ax.set_title("Monte Carlo Spin Lattice")
        ax.text(0.5, 0.5, "No rollout spins yet.", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return fig

    lattice = st.session_state.mc_spin_grid

    im = ax.imshow(
        lattice,
        cmap="viridis",
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="equal",
    )
    magnetization = float(lattice.mean()) if lattice.size else 0.0
    success_rate = float(st.session_state.guarded_prm_value[-1]) if st.session_state.guarded_prm_value else 0.0
    temperature = float(st.session_state.mc_last_temp)
    field = float(st.session_state.mc_last_field)
    ax.set_title(
        "Monte Carlo Spin Lattice "
        f"(success={success_rate:.3f}, m={magnetization:+.3f}, T={temperature:.2f}, h={field:+.2f})"
    )
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="spin")
    return fig


def _plot_mc_phase_map():
    fig, ax = plt.subplots(figsize=(8, 3.0))
    history = st.session_state.guarded_rollout_success_history
    if not history:
        ax.set_title("Rollout Phase Map")
        ax.text(0.5, 0.5, "No rollout history yet.", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return fig

    recent = history[-20:]
    max_len = max(len(row) for row in recent)
    grid = np.zeros((len(recent), max_len), dtype=float)
    for i, row in enumerate(recent):
        spins = np.array([1.0 if item else -1.0 for item in row], dtype=float)
        if spins.size:
            grid[i, : spins.size] = spins

    im = ax.imshow(
        grid,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_title("Rollout Phase Map (latest 20 steps)")
    ax.set_xlabel("Rollout sample index")
    ax.set_ylabel("Simulation step window")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="spin")
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
    st.session_state.unguarded_prm_quality = []
    st.session_state.guarded_prm_quality = []
    st.session_state.unguarded_prm_value = []
    st.session_state.guarded_prm_value = []
    st.session_state.unguarded_prm_advantage = []
    st.session_state.guarded_prm_advantage = []
    st.session_state.unguarded_prm_uncertainty = []
    st.session_state.guarded_prm_uncertainty = []
    st.session_state.guarded_prm_actions = []
    st.session_state.guarded_prm_rollouts = []
    st.session_state.guarded_prm_candidates = []
    st.session_state.guarded_rollout_values_history = []
    st.session_state.guarded_rollout_success_history = []
    st.session_state.mc_spin_grid = None
    st.session_state.mc_last_temp = 0.0
    st.session_state.mc_last_field = 0.0


def _render_dashboard() -> None:
    current_unguarded_status: StepStatus = "aligned"
    current_guarded_status: StepStatus = "aligned"
    if st.session_state.last_event:
        current_unguarded_status = st.session_state.last_event["unguarded"]["status"]
        current_guarded_status = st.session_state.last_event["guarded"]["status"]

    kpi_a, kpi_b, kpi_c, kpi_d = st.columns(4)
    current_delta = st.session_state.delta_values[-1] if st.session_state.delta_values else 0.0
    current_delta_smooth = (
        st.session_state.delta_smooth_values[-1] if st.session_state.delta_smooth_values else 0.0
    )
    current_value = st.session_state.guarded_prm_value[-1] if st.session_state.guarded_prm_value else 0.0
    current_advantage = (
        st.session_state.guarded_prm_advantage[-1] if st.session_state.guarded_prm_advantage else 0.0
    )
    with kpi_a:
        st.metric("Alignment Delta", f"{current_delta:+.3f}")
    with kpi_b:
        st.metric("Alignment Delta (Smoothed)", f"{current_delta_smooth:+.3f}")
    with kpi_c:
        st.metric("Guarded MC Value", f"{current_value:.3f}")
    with kpi_d:
        st.metric("Guarded Advantage", f"{current_advantage:+.3f}")

    left_panel, right_panel = st.columns([1.65, 1.0], gap="large")
    with left_panel:
        status_a, status_b = st.columns(2)
        with status_a:
            st.markdown("**Unguarded Lane**")
            st.markdown(_status_indicator(current_unguarded_status))
        with status_b:
            st.markdown("**Guarded Lane**")
            st.markdown(_status_indicator(current_guarded_status))

        if st.session_state.last_event:
            if st.session_state.last_event["unguarded"]["drift"]:
                st.warning("⚠️ Drift pressure rising in unguarded lane.")
            if st.session_state.last_event["guarded"]["corrected"]:
                st.success("✅ Guarded lane corrected and re-centered toward goal.")

        st.pyplot(_plot_alignment_scores(), use_container_width=True, clear_figure=True)
        st.pyplot(_plot_trajectory(), use_container_width=True, clear_figure=True)
        st.pyplot(_plot_delta(), use_container_width=True, clear_figure=True)

    with right_panel:
        st.subheader("Pseudo-PRM Live Panel")
        st.caption(
            "Monte Carlo continuation estimates step value. Advantage tracks delta-to-goal and drives accept/rewrite/resample."
        )

        action_counts = Counter(st.session_state.guarded_prm_actions)
        latest_action = st.session_state.guarded_prm_actions[-1] if st.session_state.guarded_prm_actions else "n/a"
        latest_rollouts = st.session_state.guarded_prm_rollouts[-1] if st.session_state.guarded_prm_rollouts else 0
        latest_candidates = (
            st.session_state.guarded_prm_candidates[-1] if st.session_state.guarded_prm_candidates else 1
        )

        decision_a, decision_b = st.columns(2)
        with decision_a:
            st.metric("Latest PRM Action", latest_action)
            st.metric("Candidates Scored", str(latest_candidates))
        with decision_b:
            st.metric("MC Rollouts", str(latest_rollouts))
            st.metric("Rewrite Count", str(action_counts.get("rewrite", 0)))

        st.pyplot(_plot_mc_rain_of_dots(), use_container_width=True, clear_figure=True)
        st.pyplot(_plot_mc_density_wave(), use_container_width=True, clear_figure=True)
        st.pyplot(_plot_mc_spin_lattice(), use_container_width=True, clear_figure=True)
        st.pyplot(_plot_mc_phase_map(), use_container_width=True, clear_figure=True)
        st.pyplot(_plot_prm_value_quality(), use_container_width=True, clear_figure=True)
        st.pyplot(_plot_prm_advantage(), use_container_width=True, clear_figure=True)

        with st.expander("Latest Unguarded Steps", expanded=True):
            st.code("\n".join(st.session_state.unguarded_logs[-10:]) or "No steps yet.")
        with st.expander("Latest Guarded Steps", expanded=True):
            st.code("\n".join(st.session_state.guarded_logs[-10:]) or "No steps yet.")


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
        options=["fair_replay", "steering", "prm_mc", "prm_best_of_n"],
        index=["fair_replay", "steering", "prm_mc", "prm_best_of_n"].index(
            config["simulation"].get("mode", "fair_replay")
            if config["simulation"].get("mode", "fair_replay") in {"fair_replay", "steering", "prm_mc", "prm_best_of_n"}
            else "fair_replay"
        ),
        help=(
            "fair_replay: identical base steps. steering: guarded corrections feed future context. "
            "prm_mc: MC value+advantage gate. prm_best_of_n: score multiple candidates each step."
        ),
    )
    prm_a, prm_b, prm_c = st.columns(3)
    with prm_a:
        prm_n_candidates = st.slider(
            "PRM Candidates (N)",
            min_value=1,
            max_value=8,
            value=int(config.get("prm", {}).get("n_candidates", 4)),
            step=1,
            disabled=simulation_mode != "prm_best_of_n",
        )
    with prm_b:
        prm_rollouts = st.slider(
            "MC Rollouts/Step",
            min_value=4,
            max_value=60,
            value=int(config.get("prm", {}).get("rollouts_per_step", 24)),
            step=2,
        )
    with prm_c:
        prm_depth = st.slider(
            "MC Rollout Depth",
            min_value=1,
            max_value=8,
            value=int(config.get("prm", {}).get("rollout_depth", 4)),
            step=1,
        )
    ising_a, ising_b, ising_c = st.columns(3)
    with ising_a:
        st.session_state.mc_lattice_size = st.slider(
            "Ising Lattice Size",
            min_value=16,
            max_value=128,
            value=int(st.session_state.mc_lattice_size),
            step=8,
        )
    with ising_b:
        st.session_state.mc_sweeps_per_step = st.slider(
            "Metropolis Sweeps/Step",
            min_value=1,
            max_value=10,
            value=int(st.session_state.mc_sweeps_per_step),
            step=1,
        )
    with ising_c:
        st.session_state.mc_base_temp = st.slider(
            "Base Temperature",
            min_value=0.2,
            max_value=3.0,
            value=float(st.session_state.mc_base_temp),
            step=0.1,
        )
    ising_d, ising_e = st.columns(2)
    with ising_d:
        st.session_state.mc_temp_scale = st.slider(
            "Uncertainty -> Temperature Scale",
            min_value=0.0,
            max_value=4.0,
            value=float(st.session_state.mc_temp_scale),
            step=0.1,
        )
    with ising_e:
        st.session_state.mc_field_gain = st.slider(
            "Value -> Field Gain",
            min_value=0.0,
            max_value=3.0,
            value=float(st.session_state.mc_field_gain),
            step=0.1,
        )
    st.session_state.mc_coupling = st.slider(
        "Spin Coupling (J)",
        min_value=0.2,
        max_value=2.5,
        value=float(st.session_state.mc_coupling),
        step=0.1,
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
        st.session_state.mc_spin_rng = np.random.default_rng(int(config["simulation"]["seed"]))
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
                prm_n_candidates=prm_n_candidates,
                prm_rollouts_per_step=prm_rollouts,
                prm_rollout_depth=prm_depth,
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
