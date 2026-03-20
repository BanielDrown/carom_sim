"""
Plotting utilities for the carom simulator.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, Rectangle

from carom.io_utils import build_motion_intervals, format_impulse_vector, format_scalar
from carom.physics import wall_normal_from_name
from carom.state import CollisionEvent, SimulationResult, Table, TrajectorySample
from carom.validation import first_success_event_index


BALL_COLORS = {
    "A": "red",
    "B": "blue",
    "C": "white",
}

BALL_NAMES = {
    "A": "Ball A",
    "B": "Ball B",
    "C": "Cue Ball C",
}

VECTOR_EPS = 1e-12
STANDARD_ARROW_FRACTION = 0.09


def _high_visibility_path_effects(
    outer_width: float = 5.0,
    inner_width: float = 3.0,
) -> list:
    """
    Return a stacked white/black outline effect for visibility on any background.
    """
    return [
        pe.Stroke(linewidth=outer_width, foreground="white"),
        pe.Stroke(linewidth=inner_width, foreground="black"),
        pe.Normal(),
    ]


def _arrow_visibility_path_effects() -> list:
    """
    White + black outline for vector arrows.
    """
    return _high_visibility_path_effects(outer_width=5.4, inner_width=3.6)


def _text_visibility_path_effects() -> list:
    """
    White + black outline for annotation text.
    """
    return _high_visibility_path_effects(outer_width=4.0, inner_width=2.2)


def _ball_text_color(facecolor: str) -> str:
    """
    Choose a contrasting label color for text placed inside a ball.
    """
    return "black" if facecolor == "white" else "white"


def _relevant_event_cutoff(result: SimulationResult) -> int:
    """
    Return the event index where the assignment success condition is first met.
    If never met, return the full event count.
    """
    return first_success_event_index(result.events) or len(result.events)


def _relevant_end_time(
    result: SimulationResult,
    relevant_only: bool = True,
    max_events_to_plot: int | None = None,
    post_end_fraction: float = 0.10,
) -> float:
    """
    Return the physical time through which plots should be displayed.

    Priority
    --------
    1. Use result.display_end_time if available.
    2. Otherwise reconstruct from event history.
    """
    if result.display_end_time is not None:
        return float(result.display_end_time)

    if not result.events:
        return float(result.final_state.time)

    cutoff = len(result.events)

    if relevant_only:
        cutoff = min(cutoff, _relevant_event_cutoff(result))

    if max_events_to_plot is not None:
        cutoff = min(cutoff, max_events_to_plot)

    if cutoff <= 0:
        return float(result.initial_state.time)

    end_time = float(result.events[cutoff - 1].time)
    return end_time * (1.0 + post_end_fraction)


def _trim_trajectory_by_time(
    trajectory: list[TrajectorySample],
    t_limit: float,
) -> list[TrajectorySample]:
    """
    Keep trajectory samples up to and including t_limit.
    """
    if not trajectory:
        return trajectory

    if t_limit <= trajectory[0].time:
        return trajectory[:1]

    trimmed = [sample for sample in trajectory if sample.time <= t_limit]

    if not trimmed:
        return [trajectory[0]]

    if trimmed[-1].time < min(t_limit, trajectory[-1].time):
        for sample in reversed(trajectory):
            if sample.time <= t_limit:
                if sample.time > trimmed[-1].time:
                    trimmed.append(sample)
                break

    if trajectory[-1].time <= t_limit and trimmed[-1].time < trajectory[-1].time:
        trimmed.append(trajectory[-1])

    return trimmed


def _events_up_to_time(
    result: SimulationResult,
    t_limit: float,
    max_events_to_plot: int | None = None,
) -> list:
    """
    Return events whose event time lies within the plotting window.
    """
    events = [event for event in result.events if event.time <= t_limit]

    if max_events_to_plot is not None:
        events = events[:max_events_to_plot]

    return events


def _build_ball_paths(trajectory: list[TrajectorySample]) -> dict[str, np.ndarray]:
    """
    Convert trajectory samples into one path array per ball.
    """
    ball_paths: dict[str, list[np.ndarray]] = {}

    for sample in trajectory:
        for label, pos in sample.positions.items():
            ball_paths.setdefault(label, []).append(pos)

    return {label: np.array(path) for label, path in ball_paths.items()}


def _event_anchor_position(
    result: SimulationResult,
    event: CollisionEvent,
) -> np.ndarray:
    """
    Return the physical point where a collision vector should be drawn.

    Ball-ball events already store the contact point. Ball-wall events store
    the ball center, so shift by one radius along the inward wall normal to
    recover the cushion contact point.
    """
    if event.event_type != "ball-wall":
        return event.position

    ball_label, wall_name = event.actors
    radius = result.initial_state.balls[ball_label].radius
    return event.position - radius * wall_normal_from_name(wall_name)


def _representative_impulse_vector(
    event: CollisionEvent,
) -> tuple[str, np.ndarray] | None:
    """
    Pick one authoritative impulse vector per event for visualization.

    Ball-ball events store equal-and-opposite vectors for both balls, so one is
    sufficient and avoids drawing the same pair twice. Ball-wall events only
    have a single ball-side impulse vector.
    """
    if not event.impulse_vectors:
        return None

    if event.event_type == "ball-wall":
        ball_label, _wall_name = event.actors
        impulse_vec = event.impulse_vectors.get(ball_label)
        if impulse_vec is None:
            return None
        return ball_label, impulse_vec

    label = sorted(event.impulse_vectors.keys())[0]
    return label, event.impulse_vectors[label]


def _standard_arrow_length(
    table: Table,
    max_fraction_of_table: float = STANDARD_ARROW_FRACTION,
) -> float:
    """
    Use one standardized arrow length for all reaction arrows.
    """
    return max_fraction_of_table * min(table.length, table.width)


def _draw_impulse_pair(
    ax,
    position: np.ndarray,
    impulse_vec: np.ndarray,
    color: str,
    arrow_length: float,
) -> None:
    """
    Draw equal and opposite standardized impulse arrows centered at the collision point.
    """
    mag = float(np.linalg.norm(impulse_vec))
    if mag <= VECTOR_EPS or arrow_length <= 0.0:
        return

    direction = impulse_vec / mag
    half = 0.5 * arrow_length * direction

    start1 = position - 0.5 * half
    start2 = position + 0.5 * half

    for tail, delta in ((start1, half), (start2, -half)):
        arrow = FancyArrowPatch(
            posA=(float(tail[0]), float(tail[1])),
            posB=(float(tail[0] + delta[0]), float(tail[1] + delta[1])),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.8,
            color=color,
            alpha=0.90,
            zorder=7,
        )
        arrow.set_path_effects(_arrow_visibility_path_effects())
        ax.add_patch(arrow)


def _draw_single_impulse_arrow(
    ax,
    position: np.ndarray,
    impulse_vec: np.ndarray,
    color: str,
    arrow_length: float,
) -> None:
    """
    Draw one standardized arrow for a ball-wall reaction.
    """
    mag = float(np.linalg.norm(impulse_vec))
    if mag <= VECTOR_EPS or arrow_length <= 0.0:
        return

    direction = impulse_vec / mag
    tail = position - 0.5 * arrow_length * direction
    head = position + 0.5 * arrow_length * direction

    arrow = FancyArrowPatch(
        posA=(float(tail[0]), float(tail[1])),
        posB=(float(head[0]), float(head[1])),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.8,
        color=color,
        alpha=0.90,
        zorder=7,
    )
    arrow.set_path_effects(_arrow_visibility_path_effects())
    ax.add_patch(arrow)


def _draw_impulse_label(
    ax,
    position: np.ndarray,
    impulse_vec: np.ndarray,
) -> None:
    """
    Display the impulse vector beside the collision arrows in i, j notation.
    """
    label = ax.text(
        float(position[0] + 0.03),
        float(position[1] + 0.03),
        format_impulse_vector(impulse_vec),
        fontsize=8.5,
        color="black",
        zorder=9,
    )
    label.set_path_effects(_text_visibility_path_effects())


def _draw_ball_marker(
    ax,
    center: np.ndarray,
    radius: float,
    label_text: str,
    facecolor: str,
    alpha: float = 1.0,
) -> None:
    """
    Draw a labeled ball with the label centered inside the circle.
    """
    ball = plt.Circle(
        (float(center[0]), float(center[1])),
        radius=radius,
        facecolor=facecolor,
        edgecolor="black",
        linewidth=2.0,
        alpha=alpha,
        zorder=4,
    )
    ball.set_path_effects(_high_visibility_path_effects(outer_width=4.6, inner_width=2.8))
    ax.add_patch(ball)

    text = ax.text(
        float(center[0]),
        float(center[1]),
        label_text,
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
        color=_ball_text_color(facecolor),
        zorder=5,
    )
    text.set_path_effects(_text_visibility_path_effects())


def plot_trajectories(
    result: SimulationResult,
    trajectory: list[TrajectorySample],
    table: Table,
    save_path: str | None = None,
    relevant_only: bool = True,
    max_events_to_plot: int | None = None,
    show_collisions: bool = True,
    show_velocity_vectors: bool = False,
    show_impulse_vectors: bool = False,
    debug: bool = False,
    post_end_fraction: float = 0.10,
) -> None:
    """
    Plot trajectories in a clean report-focused mode.

    Notes
    -----
    - Plotting is trimmed by physical time, not just by event count.
    - The preferred time limit is result.display_end_time when available.
    - Velocity arrows are intentionally omitted from the clean plot for clarity.
      Momentum arrows should be shown in the animation layer instead.
    - Ball-ball collisions are annotated using impulse-arrow pairs instead of
      green dots.
    """
    del show_velocity_vectors
    del show_impulse_vectors

    fig, ax = plt.subplots(figsize=(12, 6))
    radius = result.initial_state.balls["C"].radius

    ax.set_xlim(0.0, table.length)
    ax.set_ylim(0.0, table.width)
    ax.set_aspect("equal")
    ax.set_title("Carom Simulation Trajectories")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_facecolor("#fbf8ef")

    table_patch = Rectangle(
        (0.0, 0.0),
        table.length,
        table.width,
        linewidth=2.0,
        edgecolor="#444444",
        facecolor="none",
        zorder=0,
    )
    table_patch.set_path_effects(_high_visibility_path_effects(outer_width=5.0, inner_width=3.0))
    ax.add_patch(table_patch)

    t_limit = _relevant_end_time(
        result=result,
        relevant_only=relevant_only,
        max_events_to_plot=max_events_to_plot,
        post_end_fraction=post_end_fraction,
    )

    plotted_samples = _trim_trajectory_by_time(trajectory, t_limit)
    plotted_events = _events_up_to_time(
        result=result,
        t_limit=t_limit,
        max_events_to_plot=max_events_to_plot,
    )

    if not plotted_samples:
        raise ValueError("Cannot plot an empty trajectory.")

    ball_paths = _build_ball_paths(plotted_samples)

    # Trajectories and start/end balls
    for label, path_array in ball_paths.items():
        if len(path_array) == 0:
            continue

        color = BALL_COLORS.get(label, "gray")
        line_width = 2.8 if label == "C" else 2.0
        line_style = "-" if label == "C" else "--"

        ax.plot(
            path_array[:, 0],
            path_array[:, 1],
            linestyle=line_style,
            linewidth=line_width,
            color=color,
            alpha=0.92,
            label=BALL_NAMES.get(label, label),
            zorder=1,
            path_effects=_high_visibility_path_effects(outer_width=3.8, inner_width=2.0),
        )

        start = path_array[0]
        end = path_array[-1]

        _draw_ball_marker(ax, start, radius, label, color, alpha=1.0)
        _draw_ball_marker(ax, end, radius, f"{label}′", color, alpha=0.78)

    # Collision annotations
    if show_collisions:
        collision_index = 0
        arrow_length = _standard_arrow_length(table=table)

        for event in plotted_events:
            anchor_position = _event_anchor_position(result, event)
            representative_impulse = _representative_impulse_vector(event)

            if representative_impulse is None:
                continue

            collision_index += 1
            x, y = anchor_position
            first_label, impulse_vec = representative_impulse
            ax.scatter(
                x,
                y,
                color="white",
                edgecolors="black",
                linewidths=0.8,
                s=28,
                zorder=6,
            )
            if event.event_type == "ball-ball":
                _draw_impulse_pair(
                    ax=ax,
                    position=anchor_position,
                    impulse_vec=impulse_vec,
                    color=BALL_COLORS.get(first_label, "green"),
                    arrow_length=arrow_length,
                )
            else:
                _draw_single_impulse_arrow(
                    ax=ax,
                    position=anchor_position,
                    impulse_vec=impulse_vec,
                    color=BALL_COLORS.get(first_label, "#ffb347"),
                    arrow_length=arrow_length,
                )
            _draw_impulse_label(ax, anchor_position, impulse_vec)

            if debug:
                debug_text = ax.text(
                    float(x + 0.015),
                    float(y + 0.015),
                    f"C{collision_index}",
                    fontsize=9,
                    color="green",
                    weight="bold",
                    zorder=8,
                )
                debug_text.set_path_effects(_text_visibility_path_effects())

    class_label = result.classification if result.classification is not None else "unclassified"
    result_label = "success" if result.success else "failed"

    header = f"classification: {class_label} | result: {result_label}"
    if result.success_time is not None:
        header += f" | success_t: {result.success_time:.3f} s"
    if result.display_end_time is not None:
        header += f" | display_end_t: {result.display_end_time:.3f} s"

    header_text = ax.text(
        0.01,
        1.02,
        header,
        transform=ax.transAxes,
        fontsize=10,
    )
    header_text.set_path_effects(_text_visibility_path_effects())

    if debug:
        status = result.assignment_status
        status_text = ax.text(
            0.01,
            0.98,
            (
                f"cue_contacts={sorted(status.cue_contacts)} | "
                f"wall_hits=A:{status.wall_hits['A']} "
                f"B:{status.wall_hits['B']} "
                f"C:{status.wall_hits['C']} | "
                f"t_limit={t_limit:.3f} s | "
                f"n_events={len(plotted_events)}"
            ),
            transform=ax.transAxes,
            fontsize=9,
            va="top",
        )
        status_text.set_path_effects(_text_visibility_path_effects())

    ax.grid(True, alpha=0.18, linestyle=":")
    ax.legend(loc="upper left", framealpha=0.95)

    _save_or_show(fig, save_path)


def _save_or_show(fig, save_path: str | None) -> None:
    """
    Save a figure if requested, otherwise show it interactively.
    """
    if save_path:
        save_target = str(save_path)
        Path(save_target).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_target, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_piecewise_position_time_graphs(
    result: SimulationResult,
    trajectory: list[TrajectorySample],
    output_dir: str | None = None,
) -> None:
    """
    Plot x(t) and y(t) for each ball, labeled by interval equations.
    """
    intervals = build_motion_intervals(result, trajectory)
    if not intervals:
        return

    labels = sorted(result.initial_state.balls.keys())
    for label in labels:
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        color = BALL_COLORS.get(label, "gray")

        for axis_index, (ax, component_index, component_name) in enumerate(
            zip(axes, (0, 1), ("x", "y"))
        ):
            for interval in intervals:
                start_time = interval.start_time
                end_time = interval.end_time
                start_value = interval.start_positions[label][component_index]
                end_value = interval.end_positions[label][component_index]
                velocity_value = interval.velocities[label][component_index]

                ax.plot(
                    [start_time, end_time],
                    [start_value, end_value],
                    color=color,
                    linewidth=2.4,
                )

                midpoint_time = 0.5 * (start_time + end_time)
                midpoint_value = 0.5 * (start_value + end_value)
                equation = (
                    f"{component_name}(t) = {format_scalar(start_value)} + "
                    f"{format_scalar(velocity_value)}(t - {format_scalar(start_time)})"
                )
                vertical_offset = 0.02 * max(1.0, result.initial_state.balls[label].radius)
                ax.text(
                    midpoint_time,
                    midpoint_value + vertical_offset,
                    equation,
                    fontsize=8,
                    color=color,
                    ha="center",
                    va="bottom",
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
                )

            ax.set_ylabel(f"{component_name}(t) [m]")
            ax.grid(True, alpha=0.22, linestyle=":")
            if axis_index == 0:
                ax.set_title(f"{BALL_NAMES.get(label, label)} Position-Time Graph")

        axes[-1].set_xlabel("time [s]")
        fig.tight_layout()

        save_path = None
        if output_dir is not None:
            save_path = str(Path(output_dir) / f"ball_{label}_position_time.png")
        _save_or_show(fig, save_path)


def plot_velocity_time_graph(
    result: SimulationResult,
    trajectory: list[TrajectorySample],
    save_path: str | None = None,
) -> None:
    """
    Plot piecewise-constant speed versus time for each ball.
    """
    intervals = build_motion_intervals(result, trajectory)
    if not intervals:
        return

    labels = sorted(result.initial_state.balls.keys())
    fig, axes = plt.subplots(len(labels), 1, figsize=(11, 8), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, label in zip(axes, labels):
        color = BALL_COLORS.get(label, "gray")
        for interval in intervals:
            speed = float(np.linalg.norm(interval.velocities[label]))
            ax.plot(
                [interval.start_time, interval.end_time],
                [speed, speed],
                color=color,
                linewidth=2.6,
            )
            ax.vlines(interval.end_time, 0.0, speed, colors=color, alpha=0.20, linewidth=1.0)
            midpoint_time = 0.5 * (interval.start_time + interval.end_time)
            ax.text(
                midpoint_time,
                speed,
                f"|v|(t) = {format_scalar(speed)}",
                fontsize=8,
                color=color,
                ha="center",
                va="bottom",
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )

        ax.set_ylabel(f"{label} speed\n[m/s]")
        ax.grid(True, alpha=0.22, linestyle=":")

    axes[0].set_title("Velocity-Time Graph (Piecewise Constant Speed)")
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_velocity_displacement_graph(
    result: SimulationResult,
    trajectory: list[TrajectorySample],
    save_path: str | None = None,
) -> None:
    """
    Plot piecewise-constant speed versus cumulative path displacement for each ball.
    """
    intervals = build_motion_intervals(result, trajectory)
    if not intervals:
        return

    labels = sorted(result.initial_state.balls.keys())
    fig, axes = plt.subplots(len(labels), 1, figsize=(11, 8), sharex=False)
    axes = np.atleast_1d(axes)

    for ax, label in zip(axes, labels):
        color = BALL_COLORS.get(label, "gray")
        for interval in intervals:
            speed = float(np.linalg.norm(interval.velocities[label]))
            s0 = interval.displacement_start[label]
            s1 = interval.displacement_end[label]
            ax.plot([s0, s1], [speed, speed], color=color, linewidth=2.6)
            midpoint_s = 0.5 * (s0 + s1)
            ax.text(
                midpoint_s,
                speed,
                f"|v|(s) = {format_scalar(speed)}",
                fontsize=8,
                color=color,
                ha="center",
                va="bottom",
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )

        ax.set_ylabel(f"{label} speed\n[m/s]")
        ax.grid(True, alpha=0.22, linestyle=":")

    axes[0].set_title("Velocity-Displacement Graph (Cumulative Path Length)")
    axes[-1].set_xlabel("cumulative displacement [m]")
    fig.tight_layout()
    _save_or_show(fig, save_path)
