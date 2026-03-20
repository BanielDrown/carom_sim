"""
Formatting and export helpers for the carom simulator.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from carom.constants import CONTACT_DURATION_S
from carom.state import SimulationResult


def format_scalar(value: float, precision: int = 4) -> str:
    """
    Format a scalar to a fixed decimal precision.
    """
    return f"{float(value):.{precision}f}"


def format_vector_xy(v: np.ndarray, precision: int = 4) -> str:
    """
    Format a vector in coordinate form: (x, y).
    """
    x = format_scalar(v[0], precision)
    y = format_scalar(v[1], precision)
    return f"({x}, {y})"


def format_vector_ij(v: np.ndarray, precision: int = 4) -> str:
    """
    Format a vector in i,j notation: ax i +/- by j.
    """
    x = round(float(v[0]), precision)
    y = round(float(v[1]), precision)

    if abs(x) < 10 ** (-precision):
        x = 0.0
    if abs(y) < 10 ** (-precision):
        y = 0.0

    if y == 0.0:
        return f"{x} i"

    sign = "+" if y >= 0 else "-"
    return f"{x} i {sign} {abs(y)} j"


def format_position_vector(v: np.ndarray, precision: int = 4) -> str:
    """
    Format a position vector in standard notation.
    """
    return f"r = {format_vector_ij(v, precision)}"


def format_velocity_vector(v: np.ndarray, precision: int = 4) -> str:
    """
    Format a velocity vector in standard notation.
    """
    return f"v = {format_vector_ij(v, precision)}"


def format_momentum_vector(v: np.ndarray, mass: float, precision: int = 4) -> str:
    """
    Format momentum p = m v in standard notation.
    """
    momentum = mass * v
    return f"p = {format_vector_ij(momentum, precision)}"


def format_impulse_vector(v: np.ndarray, precision: int = 4) -> str:
    """
    Format an impulse vector in standard notation.
    """
    return f"J = {format_vector_ij(v, precision)}"


def format_force_vector_from_impulse(
    impulse: np.ndarray,
    contact_duration_s: float = CONTACT_DURATION_S,
    precision: int = 4,
) -> str:
    """
    Format average force vector derived from impulse over assumed contact duration.
    """
    force = impulse / contact_duration_s
    return f"F_avg = {format_vector_ij(force, precision)}"


def serialize_vector_map_xy(
    vector_map: dict[str, np.ndarray],
    precision: int = 4,
) -> str:
    """
    Serialize a mapping of labels to vectors in (x, y) format.
    """
    parts = [
        f"{label}: {format_vector_xy(vec, precision)}"
        for label, vec in sorted(vector_map.items())
    ]
    return "; ".join(parts)


def serialize_vector_map_ij(
    vector_map: dict[str, np.ndarray],
    prefix: str,
    precision: int = 4,
) -> str:
    """
    Serialize a mapping of labels to vectors in i,j format.
    """
    parts = [
        f"{label}: {prefix} = {format_vector_ij(vec, precision)}"
        for label, vec in sorted(vector_map.items())
    ]
    return "; ".join(parts)


def serialize_force_map_ij(
    impulse_map: dict[str, np.ndarray],
    precision: int = 4,
) -> str:
    """
    Serialize average force vectors derived from impulse over assumed contact duration.
    """
    force_map = {
        label: impulse_vec / CONTACT_DURATION_S
        for label, impulse_vec in impulse_map.items()
    }
    return serialize_vector_map_ij(force_map, "F_avg", precision)


def export_event_table_csv(
    result: SimulationResult,
    filepath: str | Path,
    precision: int = 4,
) -> None:
    """
    Export event history to CSV.

    Includes:
    - event number
    - time
    - event type
    - actors
    - collision position
    - scalar impulse magnitude
    - pre/post velocities for all balls
    - per-ball impulse vectors
    - average force vectors derived from the assumed contact duration
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "event_number",
            "time_s",
            "event_type",
            "actors",
            "position_xy_m",
            "position_ij",
            "impulse_magnitude_Ns",
            "pre_velocities_xy_mps",
            "pre_velocities_ij_mps",
            "post_velocities_xy_mps",
            "post_velocities_ij_mps",
            "impulse_vectors_xy_Ns",
            "impulse_vectors_ij_Ns",
            "avg_force_vectors_ij_N",
        ])

        for idx, event in enumerate(result.events, start=1):
            writer.writerow([
                idx,
                format_scalar(event.time, 6),
                event.event_type,
                str(event.actors),
                format_vector_xy(event.position, precision),
                format_position_vector(event.position, precision),
                "" if event.impulse is None else format_scalar(event.impulse, 6),
                serialize_vector_map_xy(event.pre_velocities, precision),
                serialize_vector_map_ij(event.pre_velocities, "v", precision),
                serialize_vector_map_xy(event.post_velocities, precision),
                serialize_vector_map_ij(event.post_velocities, "v", precision),
                serialize_vector_map_xy(event.impulse_vectors, precision),
                serialize_vector_map_ij(event.impulse_vectors, "J", precision),
                serialize_force_map_ij(event.impulse_vectors, precision),
            ])


def export_state_summary_csv(
    result: SimulationResult,
    filepath: str | Path,
    precision: int = 4,
) -> None:
    """
    Export initial and final state summaries for each ball.

    Includes:
    - initial/final positions
    - initial/final velocities
    - initial/final momenta
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ball",
            "initial_position_xy_m",
            "initial_position_ij",
            "initial_velocity_xy_mps",
            "initial_velocity_ij",
            "initial_momentum_ij_kgmps",
            "final_position_xy_m",
            "final_position_ij",
            "final_velocity_xy_mps",
            "final_velocity_ij",
            "final_momentum_ij_kgmps",
            "mass_kg",
        ])

        for label in sorted(result.initial_state.balls.keys()):
            initial_ball = result.initial_state.balls[label]
            final_ball = result.final_state.balls[label]

            writer.writerow([
                label,
                format_vector_xy(initial_ball.position, precision),
                format_position_vector(initial_ball.position, precision),
                format_vector_xy(initial_ball.velocity, precision),
                format_velocity_vector(initial_ball.velocity, precision),
                format_momentum_vector(initial_ball.velocity, initial_ball.mass, precision),
                format_vector_xy(final_ball.position, precision),
                format_position_vector(final_ball.position, precision),
                format_vector_xy(final_ball.velocity, precision),
                format_velocity_vector(final_ball.velocity, precision),
                format_momentum_vector(final_ball.velocity, final_ball.mass, precision),
                format_scalar(initial_ball.mass, 6),
            ])


def export_initial_conditions_csv(
    result: SimulationResult,
    filepath: str | Path,
    precision: int = 4,
) -> None:
    """
    Export a compact table of only the initial conditions of all balls.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ball",
            "initial_position_xy_m",
            "initial_position_ij",
            "initial_velocity_xy_mps",
            "initial_velocity_ij",
            "initial_momentum_ij_kgmps",
            "mass_kg",
        ])

        for label in sorted(result.initial_state.balls.keys()):
            ball = result.initial_state.balls[label]
            writer.writerow([
                label,
                format_vector_xy(ball.position, precision),
                format_position_vector(ball.position, precision),
                format_vector_xy(ball.velocity, precision),
                format_velocity_vector(ball.velocity, precision),
                format_momentum_vector(ball.velocity, ball.mass, precision),
                format_scalar(ball.mass, 6),
            ])


def export_assignment_summary_csv(
    result: SimulationResult,
    filepath: str | Path,
) -> None:
    """
    Export assignment success tracking summary.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    status = result.assignment_status

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "success",
            "classification",
            "termination_reason",
            "cue_contacts",
            "cue_hit_both",
            "A_wall_hits",
            "B_wall_hits",
            "C_wall_hits",
            "all_balls_hit_wall",
            "event_count",
            "final_time_s",
        ])
        writer.writerow([
            result.success,
            result.classification,
            result.termination_reason,
            ",".join(sorted(status.cue_contacts)),
            status.cue_hit_both,
            status.wall_hits["A"],
            status.wall_hits["B"],
            status.wall_hits["C"],
            status.all_balls_hit_wall,
            len(result.events),
            result.final_state.time,
        ])


def export_case_bundle(
    result: SimulationResult,
    stem: str | Path,
    precision: int = 4,
) -> None:
    """
    Export all CSV outputs for one case.

    Example
    -------
    stem='outputs/tables/direct'
    produces:
    - outputs/tables/direct_initial_conditions.csv
    - outputs/tables/direct_events.csv
    - outputs/tables/direct_states.csv
    - outputs/tables/direct_summary.csv
    """
    stem_path = Path(stem)
    export_initial_conditions_csv(
        result,
        stem_path.with_name(f"{stem_path.name}_initial_conditions.csv"),
        precision,
    )
    export_event_table_csv(
        result,
        stem_path.with_name(f"{stem_path.name}_events.csv"),
        precision,
    )
    export_state_summary_csv(
        result,
        stem_path.with_name(f"{stem_path.name}_states.csv"),
        precision,
    )
    export_assignment_summary_csv(
        result,
        stem_path.with_name(f"{stem_path.name}_summary.csv"),
    )