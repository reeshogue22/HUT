# Unified Simple Field Theory (Particle Universe)

This repository contains a self‑contained exploratory simulation (`universe_sim.py`) that explores a compact, explicit closure for certainty‑gradient interactions, spin‑driven torsion, two‑sheet coupling, and horizon drift. It is **not** a physical GR or Einstein–Cartan implementation; it is a configurable sandbox for intuition and experimentation.

## What’s inside

- **`universe_sim.py`** — the simulation and visualization pipeline.
- **`unified_field_report.tex`** — a LaTeX report summarizing the model and equations.
- **`hut_sim.py`** — a minimal HUT-aligned lattice + kernel-field prototype.

## Quick start

### Install dependencies

```bash
python -m pip install numpy matplotlib
```

### Run (interactive animation)

```bash
python universe_sim.py
```

### Run headless and save a snapshot

```bash
ANIMATE=0 SHOW_PLOTS=0 UNIVERSE_OUTPUT=universe_sim.png N_STEPS=200 MPLBACKEND=Agg python universe_sim.py
```

## Configuration (environment variables)

- `ANIMATE` — set to `0`/`false`/`no` to disable animation.
- `SHOW_PLOTS` — set to `0`/`false`/`no` to suppress windowed output.
- `UNIVERSE_OUTPUT` — path to save the final figure (PNG recommended).
- `N_STEPS` — number of steps for non‑animated runs.
- `N_FRAMES`, `SUBSTEPS_PER_FRAME`, `FRAME_INTERVAL_MS` — animation controls.
- `PRINT_DIAGNOSTICS` — set to `1` to log summary metrics during a run.

## Model summary (high level)

- **Certainty (HUT canonical)**: \(C_i = 1 / \sqrt{\det(\Sigma_i)}\)
- **Heat (HUT canonical)**: \(H_i = 1 / C_i\)
- **Gravity**: driven by *certainty gradients* (delta‑C), modulated by a buoyancy term.
- **Torsion**: spin differences generate transverse forces in the projected plane.
- **Two‑sheet fabric**: matter/backsheet coupling with sign‑flipped interactions.
- **Horizon drift**: outward terminal flow near a boundary radius.

For full details, see the LaTeX report.

## HUT prototype

To run the HUT-aligned prototype implementation:

```bash
python hut_sim.py
```

## Git preflight (avoid conflict churn)

Before pushing or opening a PR, run:

```bash
git fetch --all --prune
git status --short --branch
rg -n "^(<<<<<<<|=======|>>>>>>>)" -S .
```

This catches unresolved conflict markers early and keeps PR branches clean.

## Notes

This code intentionally keeps the closure compact and explicit so each term can be modified independently. It is intended as a playground for experimentation rather than a statement of physical theory.
