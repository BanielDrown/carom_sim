"""
Formatting and export helpers for the carom simulator.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from carom.constants import CONTACT_DURATION_S
from carom.state import SimulationResult, TrajectorySample


@dataclass(frozen=True)
class MotionInterval:
    """
    One piecewise-constant-velocity interval for the exported analysis.
    """

    interval_index: int
    start_time: float
    end_time: float
    start_positions: dict[str, np.ndarray]
    end_positions: dict[str, np.ndarray]
    velocities: dict[str, np.ndarray]
    displacement_start: dict[str, float]
    displacement_end: dict[str, float]

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


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


def format_component(value: float, basis: str, precision: int = 4) -> str:
    """
    Format one scalar basis-vector component such as 1.25 i or -0.75 j.
    """
    component = round(float(value), precision)
    if abs(component) < 10 ** (-precision):
        component = 0.0
    return f"{component} {basis}"


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


def format_unit_vector_nt(
    v: np.ndarray | None,
    prefix: str,
    precision: int = 4,
) -> str:
    """
    Format a collision-frame unit vector in n,t notation.
    """
    if v is None:
        return ""

    n_component = round(float(v[0]), precision)
    t_component = round(float(v[1]), precision)

    if abs(n_component) < 10 ** (-precision):
        n_component = 0.0
    if abs(t_component) < 10 ** (-precision):
        t_component = 0.0

    if t_component == 0.0:
        return f"{prefix} = {n_component} n"

    sign = "+" if t_component >= 0 else "-"
    return f"{prefix} = {n_component} n {sign} {abs(t_component)} t"


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


def _speed(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _position_equation(position: np.ndarray, velocity: np.ndarray, t0: float, precision: int) -> str:
    return (
        "r(t) = "
        f"({format_scalar(position[0], precision)} + "
        f"{format_scalar(velocity[0], precision)}(t - {format_scalar(t0, precision)})) i + "
        f"({format_scalar(position[1], precision)} + "
        f"{format_scalar(velocity[1], precision)}(t - {format_scalar(t0, precision)})) j"
    )


def _coordinate_equation(
    component_name: str,
    p0: float,
    v0: float,
    t0: float,
    precision: int,
) -> str:
    return (
        f"{component_name}(t) = {format_scalar(p0, precision)} + "
        f"{format_scalar(v0, precision)}(t - {format_scalar(t0, precision)})"
    )


def _velocity_equation(component_name: str, value: float, precision: int) -> str:
    return f"{component_name}(t) = {format_scalar(value, precision)}"


def _velocity_vector_equation(velocity: np.ndarray, precision: int) -> str:
    return (
        "v(t) = "
        f"{format_component(velocity[0], 'i', precision)} + "
        f"{format_component(velocity[1], 'j', precision)}"
    )


def _speed_equation(speed: float, precision: int) -> str:
    return f"|v|(t) = {format_scalar(speed, precision)}"


def _velocity_displacement_equation(speed: float, precision: int) -> str:
    return f"|v|(s) = {format_scalar(speed, precision)}"


def build_motion_intervals(
    result: SimulationResult,
    trajectory: list[TrajectorySample],
) -> list[MotionInterval]:
    """
    Reconstruct piecewise motion intervals from trajectory samples and event history.
    """
    if len(trajectory) < 2:
        return []

    labels = sorted(result.initial_state.balls.keys())
    cumulative_displacement = {label: 0.0 for label in labels}
    intervals: list[MotionInterval] = []

    for idx in range(len(trajectory) - 1):
        start_sample = trajectory[idx]
        end_sample = trajectory[idx + 1]

        if idx == 0:
            velocities = {
                label: result.initial_state.balls[label].velocity.copy()
                for label in labels
            }
        else:
            velocities = {
                label: result.events[idx - 1].post_velocities[label].copy()
                for label in labels
            }

        displacement_start = dict(cumulative_displacement)
        displacement_end: dict[str, float] = {}
        start_positions = {
            label: start_sample.positions[label].copy()
            for label in labels
        }
        end_positions = {
            label: end_sample.positions[label].copy()
            for label in labels
        }

        for label in labels:
            delta = end_positions[label] - start_positions[label]
            cumulative_displacement[label] += float(np.linalg.norm(delta))
            displacement_end[label] = cumulative_displacement[label]

        intervals.append(
            MotionInterval(
                interval_index=idx + 1,
                start_time=float(start_sample.time),
                end_time=float(end_sample.time),
                start_positions=start_positions,
                end_positions=end_positions,
                velocities=velocities,
                displacement_start=displacement_start,
                displacement_end=displacement_end,
            )
        )

    return intervals


def export_initial_conditions_csv(
    result: SimulationResult,
    filepath: str | Path,
    precision: int = 4,
) -> None:
    """
    Export a clearly labeled table of initial conditions for all balls.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ball_label",
            "mass_kg",
            "initial_position_i_m",
            "initial_position_j_m",
            "initial_position_vector_ij",
            "initial_velocity_i_mps",
            "initial_velocity_j_mps",
            "initial_velocity_vector_ij",
            "initial_speed_mps",
            "initial_momentum_vector_ij_kgmps",
        ])

        for label in sorted(result.initial_state.balls.keys()):
            ball = result.initial_state.balls[label]
            writer.writerow([
                label,
                format_scalar(ball.mass, 6),
                format_scalar(ball.position[0], precision),
                format_scalar(ball.position[1], precision),
                format_position_vector(ball.position, precision),
                format_scalar(ball.velocity[0], precision),
                format_scalar(ball.velocity[1], precision),
                format_velocity_vector(ball.velocity, precision),
                format_scalar(_speed(ball.velocity), precision),
                format_momentum_vector(ball.velocity, ball.mass, precision),
            ])


def export_event_table_csv(
    result: SimulationResult,
    filepath: str | Path,
    precision: int = 4,
) -> None:
    """
    Export event history with clearer labels and explicit numeric columns.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "event_number",
            "event_time_s",
            "event_type",
            "actor_1",
            "actor_2",
            "collision_position_i_m",
            "collision_position_j_m",
            "collision_position_vector_ij",
            "collision_normal_i",
            "collision_normal_j",
            "collision_normal_vector_nt",
            "collision_tangent_i",
            "collision_tangent_j",
            "collision_tangent_vector_nt",
            "impulse_magnitude_Ns",
            "pre_collision_velocities_ij_mps",
            "post_collision_velocities_ij_mps",
            "impulse_vectors_ij_Ns",
            "average_force_vectors_ij_N",
        ])

        for idx, event in enumerate(result.events, start=1):
            writer.writerow([
                idx,
                format_scalar(event.time, 6),
                event.event_type,
                event.actors[0],
                event.actors[1],
                format_scalar(event.position[0], precision),
                format_scalar(event.position[1], precision),
                format_position_vector(event.position, precision),
                "" if event.collision_normal is None else format_scalar(event.collision_normal[0], precision),
                "" if event.collision_normal is None else format_scalar(event.collision_normal[1], precision),
                format_unit_vector_nt(event.collision_normal, "n_hat", precision),
                "" if event.collision_tangent is None else format_scalar(event.collision_tangent[0], precision),
                "" if event.collision_tangent is None else format_scalar(event.collision_tangent[1], precision),
                format_unit_vector_nt(event.collision_tangent, "t_hat", precision),
                "" if event.impulse is None else format_scalar(event.impulse, 6),
                serialize_vector_map_ij(event.pre_velocities, "v", precision),
                serialize_vector_map_ij(event.post_velocities, "v", precision),
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
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ball_label",
            "mass_kg",
            "initial_position_i_m",
            "initial_position_j_m",
            "initial_velocity_i_mps",
            "initial_velocity_j_mps",
            "initial_speed_mps",
            "final_position_i_m",
            "final_position_j_m",
            "final_velocity_i_mps",
            "final_velocity_j_mps",
            "final_speed_mps",
            "initial_momentum_vector_ij_kgmps",
            "final_momentum_vector_ij_kgmps",
        ])

        for label in sorted(result.initial_state.balls.keys()):
            initial_ball = result.initial_state.balls[label]
            final_ball = result.final_state.balls[label]

            writer.writerow([
                label,
                format_scalar(initial_ball.mass, 6),
                format_scalar(initial_ball.position[0], precision),
                format_scalar(initial_ball.position[1], precision),
                format_scalar(initial_ball.velocity[0], precision),
                format_scalar(initial_ball.velocity[1], precision),
                format_scalar(_speed(initial_ball.velocity), precision),
                format_scalar(final_ball.position[0], precision),
                format_scalar(final_ball.position[1], precision),
                format_scalar(final_ball.velocity[0], precision),
                format_scalar(final_ball.velocity[1], precision),
                format_scalar(_speed(final_ball.velocity), precision),
                format_momentum_vector(initial_ball.velocity, initial_ball.mass, precision),
                format_momentum_vector(final_ball.velocity, final_ball.mass, precision),
            ])


def export_motion_intervals_csv(
    result: SimulationResult,
    trajectory: list[TrajectorySample],
    filepath: str | Path,
    precision: int = 4,
) -> None:
    """
    Export one row per ball per interval with piecewise equations.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    intervals = build_motion_intervals(result, trajectory)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "interval_number",
            "ball_label",
            "time_start_s",
            "time_end_s",
            "duration_s",
            "start_position_i_m",
            "start_position_j_m",
            "end_position_i_m",
            "end_position_j_m",
            "velocity_i_mps",
            "velocity_j_mps",
            "speed_mps",
            "path_displacement_start_m",
            "path_displacement_end_m",
            "r_i_of_t_equation",
            "r_j_of_t_equation",
            "position_vector_equation_ij",
            "v_i_of_t_equation",
            "v_j_of_t_equation",
            "velocity_vector_equation_ij",
            "speed_of_t_equation",
            "speed_of_displacement_equation",
        ])

        for interval in intervals:
            for label in sorted(interval.start_positions.keys()):
                start_position = interval.start_positions[label]
                end_position = interval.end_positions[label]
                velocity = interval.velocities[label]
                speed = _speed(velocity)
                writer.writerow([
                    interval.interval_index,
                    label,
                    format_scalar(interval.start_time, 6),
                    format_scalar(interval.end_time, 6),
                    format_scalar(interval.duration, 6),
                    format_scalar(start_position[0], precision),
                    format_scalar(start_position[1], precision),
                    format_scalar(end_position[0], precision),
                    format_scalar(end_position[1], precision),
                    format_scalar(velocity[0], precision),
                    format_scalar(velocity[1], precision),
                    format_scalar(speed, precision),
                    format_scalar(interval.displacement_start[label], precision),
                    format_scalar(interval.displacement_end[label], precision),
                    _coordinate_equation(
                        "r_i",
                        start_position[0],
                        velocity[0],
                        interval.start_time,
                        precision,
                    ),
                    _coordinate_equation(
                        "r_j",
                        start_position[1],
                        velocity[1],
                        interval.start_time,
                        precision,
                    ),
                    _position_equation(
                        start_position,
                        velocity,
                        interval.start_time,
                        precision,
                    ),
                    _velocity_equation("v_i", velocity[0], precision),
                    _velocity_equation("v_j", velocity[1], precision),
                    _velocity_vector_equation(velocity, precision),
                    _speed_equation(speed, precision),
                    _velocity_displacement_equation(speed, precision),
                ])


def export_collision_forces_csv(
    result: SimulationResult,
    filepath: str | Path,
    precision: int = 4,
) -> None:
    """
    Export the impulse and average-force list for each body involved in each collision.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "event_number",
            "event_time_s",
            "event_type",
            "actor_1",
            "actor_2",
            "body_label",
            "impulse_i_Ns",
            "impulse_j_Ns",
            "impulse_vector_ij_Ns",
            "impulse_magnitude_Ns",
            "assumed_contact_duration_s",
            "average_force_i_N",
            "average_force_j_N",
            "average_force_vector_ij_N",
            "average_force_magnitude_N",
        ])

        for idx, event in enumerate(result.events, start=1):
            for body_label, impulse_vector in sorted(event.impulse_vectors.items()):
                average_force = impulse_vector / CONTACT_DURATION_S
                writer.writerow([
                    idx,
                    format_scalar(event.time, 6),
                    event.event_type,
                    event.actors[0],
                    event.actors[1],
                    body_label,
                    format_scalar(impulse_vector[0], precision),
                    format_scalar(impulse_vector[1], precision),
                    format_impulse_vector(impulse_vector, precision),
                    format_scalar(np.linalg.norm(impulse_vector), 6),
                    format_scalar(CONTACT_DURATION_S, 6),
                    format_scalar(average_force[0], precision),
                    format_scalar(average_force[1], precision),
                    format_force_vector_from_impulse(impulse_vector, CONTACT_DURATION_S, precision),
                    format_scalar(np.linalg.norm(average_force), 6),
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
            "success_time_s",
            "final_time_s",
            "display_end_time_s",
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
            "" if result.success_time is None else result.success_time,
            result.final_state.time,
            "" if result.display_end_time is None else result.display_end_time,
        ])


def export_case_bundle(
    result: SimulationResult,
    trajectory: list[TrajectorySample],
    output_dir: str | Path,
    precision: int = 4,
) -> None:
    """
    Export all CSV outputs for one case into a dedicated directory.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    export_initial_conditions_csv(
        result,
        output_path / "01_initial_conditions.csv",
        precision,
    )
    export_motion_intervals_csv(
        result,
        trajectory,
        output_path / "02_motion_intervals.csv",
        precision,
    )
    export_event_table_csv(
        result,
        output_path / "03_collision_events.csv",
        precision,
    )
    export_collision_forces_csv(
        result,
        output_path / "04_collision_forces.csv",
        precision,
    )
    export_state_summary_csv(
        result,
        output_path / "05_state_summary.csv",
        precision,
    )
    export_assignment_summary_csv(
        result,
        output_path / "06_assignment_summary.csv",
    )
