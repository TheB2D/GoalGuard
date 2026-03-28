# GoalGuard Race Demo

GoalGuard is a small Python SDK + Streamlit app that demonstrates a side-by-side race between:

- an unguarded agent (can drift),
- a goal-guarded agent (detects drift and corrects).

The demo is intentionally simulated (no live LLM calls) so it is reproducible and easy to present.

## Project Layout

- `goalguard/agent.py` - guarded wrapper (`GoalGuard`)
- `goalguard/types.py` - shared event schema used by SDK, simulator, and UI
- `goalguard/alignment.py` - similarity + semantic trajectory coordinates
- `goalguard/drift.py` - drift detection and status classification
- `goalguard/intervention.py` - correction policy
- `goalguard/simulator.py` - synchronized two-lane simulation + alignment delta
- `demo_agent.py` - deterministic baseline agent + placeholder encoder
- `app.py` - Streamlit live GUI

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Demo Features

- Side-by-side step logs for unguarded and guarded lanes
- Live lane status (`ALIGNED`, `DRIFTING`, `OFF-GOAL`)
- Alignment score chart with green/yellow/red status bands
- Goal-anchor semantic trajectory chart
  - fixed goal marker
  - unguarded and guarded paths
  - correction highlights and dashed causal correction connectors
- Alignment delta:
  - instantaneous `guarded - unguarded`
  - rolling smoothed trend

## What To Look For

- The unguarded path drifts laterally (`x_offtopic` increases).
- The guarded lane emits correction events and re-centers toward the goal.
- Alignment delta improves after interventions during drift-heavy segments.

## Notes

- Embeddings are deterministic placeholder vectors created from token hashing.
- Two `DemoAgent` instances run with identical seeds for fair comparison.
