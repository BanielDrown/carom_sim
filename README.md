# Carom Simulator

An event-driven three-ball carom billiards simulator for exploring cue-ball
shots, validating assignment success conditions, and exporting plots, tables,
and animations.

## Project Summary

The simulator models three translating rigid balls on a rectangular carom
table. The core workflow is:

1. Define an initial state for balls `A`, `B`, and cue ball `C`.
2. Predict the next ball-ball or ball-wall collision analytically.
3. Advance the system exactly to that event.
4. Apply an impulse-based collision response.
5. Track assignment success:
   - cue ball `C` must contact both object balls `A` and `B`
   - each ball `A`, `B`, and `C` must hit at least one wall

Search utilities can also randomize object-ball layouts and scan cue speed and
launch angle combinations for direct, one-cushion, and two-cushion examples.

## Model Assumptions

- Motion between collisions is constant-velocity and frictionless.
- Collisions are instantaneous.
- Ball-ball and ball-wall impacts are currently modeled as elastic by default.
- The table is rectangular and uses SI units throughout.
- Cue ball `C` starts from a fixed modeled location unless you override it in a
  custom state.
- Shot classification (`direct`, `one_cushion`, `two_cushion`) is descriptive
  metadata; assignment success is determined only by the contact and wall-hit
  rules above.

## What Was Optimized

The recent cleanup prioritized **accuracy first**, then **speed**, then
**simplicity**, while staying non-destructive:

- Assignment-rule bookkeeping is now centralized so simulation, validation,
  plotting, and animation all interpret success the same way.
- Angle wrapping now uses a constant-time modulo formulation instead of
  repeated loop-based wrapping.
- Randomized object-ball placement now uses squared-distance checks inside the
  hottest rejection-sampling loop to avoid unnecessary square roots.
- Search refinement no longer clamps promising coarse-search candidates because
  the configured maximum cue speed now matches the search range.

## Repository Layout

- `src/carom/`
  - `simulator.py`: event-driven simulation loop
  - `collisions.py`: analytic collision timing
  - `physics.py`: collision response
  - `search.py`: coarse and refined shot search
  - `validation.py`: assignment success and shot classification metadata
  - `plotting.py` / `animation.py`: visualization helpers
- `tests/`: unit tests for core physics, simulation, validation, and search
- `run.py`: main entry point for generating/debugging cases
- `launch.py`: convenience wrapper that creates output folders and runs `run.py`

## Setup

Recommended Python version: **3.10+**.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## How to Use

Run the main workflow:

```bash
python launch.py
```

Or run the script directly:

```bash
PYTHONPATH=src python run.py
```

Run tests:

```bash
PYTHONPATH=src pytest -q
```

## Outputs

Generated artifacts are written to:

- `outputs/animations/`
- `outputs/plots/<case_name>/` for trajectory, position-time, velocity-time, and velocity-displacement graphs
- `outputs/tables/<case_name>/` for numbered CSV exports, including motion intervals and collision-force summaries

## Rollback / Non-Destructive Workflow

This repository is tracked in Git, so the safest rollback path is version
control:

- Inspect changes with `git diff`
- Return to the previous committed state with `git checkout -- <file>`
- Revert a commit with `git revert <commit_sha>`
- Move back locally with `git reset --hard <commit_sha>` if you explicitly want
  to discard later local history

In other words, the optimization work is intended to be **non-destructive** and
easy to roll back through normal Git history rather than by overwriting the
original design.
