"""
Plotting utilities for the carom simulator.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from carom.state import SimulationResult, Table, TrajectorySample


BALL_COLORS = {
    "A": "red",
    "B": "blue",
    "C": "black",
}

BALL_NAMES = {
    "A": "Ball A",
    "B": "Ball B",
    "C": "Cue Ball C",
}


def _relevant_event_cutoff(result: SimulationResult) -> int:
    """
    Return the event index where the assignment success condition is first met.
    If never met, return the full event count.
    """
    cue_contacts: set[str] = set()
    wall_hits = {"A": 0, "B": 0, "C": 0}

    for i, event in enumerate(result.events, start=1):
        if event.event_type == "ball-ball":
            a, b = event.actors
            if "C" in (a, b):
                other = b if a == "C" else a
                if other in ("A", "B"):
                    cue_contacts.add(other)

        elif event.event_type == "ball-wall":
            ball_label, _wall = event.actors
            if ball_label in wall_hits:
                wall_hits[ball_label] += 1

        if cue_contacts == {"A", "B"} and all(wall_hits[k] >= 1 for k in ("A", "B", "C")):
            return i

    return len(result.events)


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


def _impulse_scale(
    events: list,
    table: Table,
    max_fraction_of_table: float = 0.10,
) -> float:
    """
    Compute one global impulse-vector scale for static plotting.
    """
    impulse_magnitudes = [
        float(np.linalg.norm(vec))
        for event in events
        for vec in event.impulse_vectors.values()
    ]

    if not impulse_magnitudes:
        return 0.0

    jmax = max(impulse_magnitudes)
    if jmax <= 1e-12:
        return 0.0

    max_arrow_length = max_fraction_of_table * min(table.length, table.width)
    return max_arrow_length / jmax


def _draw_impulse_pair(
    ax,
    position: np.ndarray,
    impulse_vec: np.ndarray,
    color: str,
    scale: float,
) -> None:
    """
    Draw equal and opposite impulse arrows centered at the collision point.
    """
    mag = float(np.linalg.norm(impulse_vec))
    if mag <= 1e-12 or scale <= 0.0:
        return

    half = scale * impulse_vec

    start1 = position - 0.5 * half
    start2 = position + 0.5 * half

    ax.arrow(
        float(start1[0]),
        float(start1[1]),
        float(half[0]),
        float(half[1]),
        length_includes_head=True,
        head_width=0.02,
        head_length=0.03,
        linewidth=1.4,
        color=color,
        alpha=0.85,
        zorder=7,
    )

    ax.arrow(
        float(start2[0]),
        float(start2[1]),
        float(-half[0]),
        float(-half[1]),
        length_includes_head=True,
        head_width=0.02,
        head_length=0.03,
        linewidth=1.4,
        color=color,
        alpha=0.85,
        zorder=7,
    )


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

    ax.set_xlim(0.0, table.length)
    ax.set_ylim(0.0, table.width)
    ax.set_aspect("equal")
    ax.set_title("Carom Simulation Trajectories")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

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
        line_width = 2.5 if label == "C" else 1.8

        ax.plot(
            path_array[:, 0],
            path_array[:, 1],
            linestyle="--",
            linewidth=line_width,
            color=color,
            label=BALL_NAMES.get(label, label),
            zorder=1,
        )

        start = path_array[0]
        end = path_array[-1]

        ax.scatter(
            start[0],
            start[1],
            s=220,
            color=color,
            edgecolors="black",
            linewidths=0.9,
            marker="o",
            zorder=4,
        )
        ax.text(
            float(start[0] + 0.02),
            float(start[1] + 0.02),
            label,
            color=color,
            fontsize=10,
            weight="bold",
            zorder=5,
        )

        ax.scatter(
            end[0],
            end[1],
            s=220,
            color=color,
            edgecolors="black",
            linewidths=0.9,
            marker="o",
            alpha=0.85,
            zorder=4,
        )
        ax.text(
            float(end[0] + 0.02),
            float(end[1] + 0.02),
            f"{label}'",
            color=color,
            fontsize=10,
            weight="bold",
            zorder=5,
        )

    # Collision annotations
    if show_collisions:
        collision_index = 0
        impulse_scale = _impulse_scale(
            plotted_events,
            table=table,
            max_fraction_of_table=0.10,
        )

        for event in plotted_events:
            if event.event_type != "ball-ball":
                if debug:
                    ax.scatter(
                        event.position[0],
                        event.position[1],
                        color="orange",
                        s=18,
                        zorder=3,
                    )
                continue

            collision_index += 1
            x, y = event.position

            # Draw one impulse-pair per collision to avoid duplicate overlays.
            if event.impulse_vectors and impulse_scale > 0.0:
                first_label = sorted(event.impulse_vectors.keys())[0]
                impulse_vec = event.impulse_vectors[first_label]

                _draw_impulse_pair(
                    ax=ax,
                    position=event.position,
                    impulse_vec=impulse_vec,
                    color=BALL_COLORS.get(first_label, "green"),
                    scale=impulse_scale,
                )
            else:
                ax.scatter(
                    x,
                    y,
                    color="limegreen",
                    edgecolors="black",
                    linewidths=0.6,
                    s=35,
                    zorder=5,
                )

            if debug:
                ax.text(
                    float(x + 0.015),
                    float(y + 0.015),
                    f"C{collision_index}",
                    fontsize=9,
                    color="green",
                    weight="bold",
                    zorder=8,
                )

    class_label = result.classification if result.classification is not None else "unclassified"
    result_label = "success" if result.success else "failed"

    header = f"classification: {class_label} | result: {result_label}"
    if result.success_time is not None:
        header += f" | success_t: {result.success_time:.3f} s"
    if result.display_end_time is not None:
        header += f" | display_end_t: {result.display_end_time:.3f} s"

    ax.text(
        0.01,
        1.02,
        header,
        transform=ax.transAxes,
        fontsize=10,
    )

    if debug:
        status = result.assignment_status
        ax.text(
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

    ax.grid(True, alpha=0.30)
    ax.legend(loc="upper left")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()