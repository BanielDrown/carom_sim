"""
Animation utilities for the carom simulator.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.animation as mpl_animation
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

from carom.io_utils import format_scalar, format_vector_sum
from carom.physics import wall_normal_from_name
from carom.state import CollisionEvent, SimulationResult, Table, TrajectorySample
from carom.validation import first_success_event_index


BALL_COLORS = {
    "A": "red",
    "B": "blue",
    "C": "white",
}

BALL_TRACE_COLORS = {
    "A": "red",
    "B": "blue",
    "C": "black",
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
    return _high_visibility_path_effects(outer_width=5.4, inner_width=3.6)


def _text_visibility_path_effects() -> list:
    return _high_visibility_path_effects(outer_width=4.0, inner_width=2.2)


def _ball_text_color(facecolor: str) -> str:
    return "black" if facecolor == "white" else "white"


def _line_equation_from_point_direction(
    point: np.ndarray,
    direction: np.ndarray | None,
    precision: int = 4,
) -> str:
    """
    Format the collision line in a compact analytic form.
    """
    if direction is None:
        return ""

    dx = float(direction[0])
    dy = float(direction[1])
    if abs(dx) <= VECTOR_EPS and abs(dy) <= VECTOR_EPS:
        return ""

    x0 = float(point[0])
    y0 = float(point[1])

    if abs(dx) <= VECTOR_EPS:
        return f"x = {format_scalar(x0, precision)}"

    slope = dy / dx
    intercept = y0 - slope * x0
    return f"y = {format_scalar(slope, precision)} x + {format_scalar(intercept, precision)}"


def _build_ball_paths(trajectory: list[TrajectorySample]) -> dict[str, np.ndarray]:
    """
    Convert trajectory samples into one path array per ball.
    """
    ball_paths: dict[str, list[np.ndarray]] = {}
    for sample in trajectory:
        for label, pos in sample.positions.items():
            ball_paths.setdefault(label, []).append(pos)
    return {label: np.array(path) for label, path in ball_paths.items()}


def _assignment_cutoff_index(result: SimulationResult) -> int:
    """
    Return the event index at which the assignment success condition is first met.
    If never met, return the full event count.
    """
    return first_success_event_index(result.events) or len(result.events)


def _relevant_end_time(
    result: SimulationResult,
    relevant_only: bool = True,
    max_events_to_include: int | None = None,
    post_end_fraction: float = 0.10,
) -> float:
    """
    Return the physical time through which the animation should display.

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
        cutoff = min(cutoff, _assignment_cutoff_index(result))

    if max_events_to_include is not None:
        cutoff = min(cutoff, max_events_to_include)

    if cutoff <= 0:
        return float(result.initial_state.time)

    end_time = float(result.events[cutoff - 1].time)
    return end_time * (1.0 + post_end_fraction)


def trim_trajectory_to_relevant_portion(
    result: SimulationResult,
    trajectory: list[TrajectorySample],
    relevant_only: bool = True,
    max_events_to_include: int | None = None,
    post_end_fraction: float = 0.10,
) -> list[TrajectorySample]:
    """
    Trim trajectory samples to the display-relevant portion.

    The target end time comes from result.display_end_time when available.
    """
    if not trajectory:
        return trajectory

    t_limit = _relevant_end_time(
        result=result,
        relevant_only=relevant_only,
        max_events_to_include=max_events_to_include,
        post_end_fraction=post_end_fraction,
    )

    if t_limit <= trajectory[0].time:
        return trajectory[:1]

    trimmed = [sample for sample in trajectory if sample.time <= t_limit]

    if not trimmed:
        return [trajectory[0]]

    # Ensure the latest available sample at or before the limit is included.
    if trimmed[-1].time < min(t_limit, trajectory[-1].time):
        for sample in reversed(trajectory):
            if sample.time <= t_limit:
                if sample.time > trimmed[-1].time:
                    trimmed.append(sample)
                break

    # If the raw trajectory itself ends before the requested display time,
    # keep the true final sample.
    if trajectory[-1].time <= t_limit and trimmed[-1].time < trajectory[-1].time:
        trimmed.append(trajectory[-1])

    return trimmed


def resample_uniform_in_time(
    trajectory: list[TrajectorySample],
    fps: int = 30,
    duration_s: float = 3.0,
) -> list[TrajectorySample]:
    """
    Resample trajectory so that physical time increments between animation frames
    are constant.
    """
    if len(trajectory) <= 1:
        return trajectory

    t_start = float(trajectory[0].time)
    t_end = float(trajectory[-1].time)

    if t_end <= t_start:
        return trajectory

    n_frames = max(2, int(round(duration_s * fps)))
    target_times = np.linspace(t_start, t_end, n_frames)

    labels = sorted(trajectory[0].positions.keys())
    resampled: list[TrajectorySample] = []

    seg_idx = 0
    for t in target_times:
        while seg_idx < len(trajectory) - 2 and trajectory[seg_idx + 1].time < t:
            seg_idx += 1

        s0 = trajectory[seg_idx]
        s1 = trajectory[seg_idx + 1]

        t0, t1 = float(s0.time), float(s1.time)
        alpha = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        alpha = max(0.0, min(1.0, alpha))

        positions = {
            label: (1.0 - alpha) * s0.positions[label] + alpha * s1.positions[label]
            for label in labels
        }

        resampled.append(TrajectorySample(time=float(t), positions=positions))

    return resampled


def _momentum_scale(
    result: SimulationResult,
    max_fraction_of_table: float,
    table: Table,
) -> float:
    """
    Compute one global momentum-arrow scale for the whole animation.
    """
    momenta: list[float] = []

    for ball in result.initial_state.balls.values():
        momenta.append(float(np.linalg.norm(ball.mass * ball.velocity)))

    for event in result.events:
        for label, velocity in event.post_velocities.items():
            mass = result.initial_state.balls[label].mass
            momenta.append(float(np.linalg.norm(mass * velocity)))

    for ball in result.final_state.balls.values():
        momenta.append(float(np.linalg.norm(ball.mass * ball.velocity)))

    pmax = max(momenta) if momenta else 0.0
    if pmax <= 1e-12:
        return 0.0

    max_arrow_length = max_fraction_of_table * min(table.length, table.width)
    return max_arrow_length / pmax


def _standard_arrow_length(
    table: Table,
    max_fraction_of_table: float = STANDARD_ARROW_FRACTION,
) -> float:
    """
    Use one standardized arrow length for all collision reaction arrows.
    """
    return max_fraction_of_table * min(table.length, table.width)


def _event_anchor_position(
    result: SimulationResult,
    event: CollisionEvent,
) -> np.ndarray:
    """
    Return the physical point where a collision vector should be drawn.
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
    Pick one authoritative impulse vector per collision for visualization.
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


def _velocity_at_time(
    result: SimulationResult,
    label: str,
    t: float,
) -> np.ndarray:
    """
    Recover a ball velocity at time t using event snapshots.

    Since motion is piecewise constant between events, the velocity at a frame
    is taken from:
    - initial state if before the first event
    - the post-collision state of the latest event whose time <= t
    """
    velocity = result.initial_state.balls[label].velocity.copy()

    for event in result.events:
        if event.time <= t and label in event.post_velocities:
            velocity = event.post_velocities[label].copy()
        elif event.time > t:
            break

    return velocity


def _draw_momentum_arrow(
    ax,
    origin: np.ndarray,
    velocity: np.ndarray,
    mass: float,
    radius: float,
    color: str,
    scale: float,
):
    """
    Draw a momentum arrow p = m v, with the tail beginning slightly inside the ball.
    """
    momentum = mass * velocity
    mag = float(np.linalg.norm(momentum))
    if mag <= VECTOR_EPS or scale <= 0.0:
        return None

    direction = momentum / mag
    tail = origin - 0.45 * radius * direction

    dx = scale * float(momentum[0])
    dy = scale * float(momentum[1])

    arrow = FancyArrowPatch(
        posA=(float(tail[0]), float(tail[1])),
        posB=(float(tail[0] + dx), float(tail[1] + dy)),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=2.0,
        color=color,
        alpha=0.92,
        zorder=5,
    )
    arrow.set_path_effects(_arrow_visibility_path_effects())
    ax.add_patch(arrow)
    return arrow


def _draw_impulse_pair(
    ax,
    position: np.ndarray,
    impulse_vec: np.ndarray,
    color: str,
    arrow_length: float,
):
    """
    Draw equal and opposite standardized impulse arrows centered at the collision point.
    """
    mag = float(np.linalg.norm(impulse_vec))
    if mag <= VECTOR_EPS or arrow_length <= 0.0:
        return []

    direction = impulse_vec / mag
    half = 0.5 * arrow_length * direction

    start1 = position - 0.5 * half
    start2 = position + 0.5 * half

    arrows = []
    for tail, delta in ((start1, half), (start2, -half)):
        arrow = FancyArrowPatch(
            posA=(float(tail[0]), float(tail[1])),
            posB=(float(tail[0] + delta[0]), float(tail[1] + delta[1])),
            arrowstyle="-|>",
            mutation_scale=12,
            color=color,
            linewidth=2.0,
            alpha=0.88,
            zorder=8,
        )
        arrow.set_path_effects(_arrow_visibility_path_effects())
        ax.add_patch(arrow)
        arrows.append(arrow)

    return arrows


def _draw_collision_line(
    ax,
    point: np.ndarray,
    direction: np.ndarray | None,
    table: Table,
    color: str = "dimgray",
):
    """
    Draw the line of collision across the visible table.
    """
    if direction is None:
        return None

    norm = float(np.linalg.norm(direction))
    if norm <= VECTOR_EPS:
        return None

    unit = direction / norm
    span = 1.4 * np.hypot(table.length, table.width)
    start = point - span * unit
    end = point + span * unit
    (line,) = ax.plot(
        [float(start[0]), float(end[0])],
        [float(start[1]), float(end[1])],
        linestyle=":",
        linewidth=1.5,
        color=color,
        alpha=0.55,
        zorder=2,
    )
    line.set_path_effects(_high_visibility_path_effects(4.0, 2.0))
    return line


def animate_trajectory(
    result: SimulationResult,
    trajectory: list[TrajectorySample],
    table: Table,
    save_path: str,
    relevant_only: bool = True,
    max_events_to_include: int | None = None,
    fps: int = 30,
    duration_s: float = 3.0,
    show_momentum_arrows: bool = True,
    show_impulse_pairs: bool = True,
    post_end_fraction: float = 0.10,
) -> None:
    """
    Export an animation of the ball motion to GIF or MP4.

    Notes
    -----
    - The physical simulation time is uniformly normalized to duration_s seconds.
    - Each animation frame represents a constant physical timestep.
    - The display window uses result.display_end_time when available.
    - Impulse arrows remain visible after the corresponding collision occurs.
    """
    trimmed = trim_trajectory_to_relevant_portion(
        result=result,
        trajectory=trajectory,
        relevant_only=relevant_only,
        max_events_to_include=max_events_to_include,
        post_end_fraction=post_end_fraction,
    )

    frames = resample_uniform_in_time(
        trajectory=trimmed,
        fps=fps,
        duration_s=duration_s,
    )

    if not frames:
        raise ValueError("Cannot animate an empty trajectory.")

    physical_dt = 0.0 if len(frames) <= 1 else (float(frames[-1].time) - float(frames[0].time)) / (len(frames) - 1)

    display_end_time = _relevant_end_time(
        result=result,
        relevant_only=relevant_only,
        max_events_to_include=max_events_to_include,
        post_end_fraction=post_end_fraction,
    )

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0.0, table.length)
    ax.set_ylim(0.0, table.width)
    ax.set_aspect("equal")
    ax.set_title("Carom Simulation Animation", fontsize=16, weight="normal", pad=14)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_facecolor("#fbf8ef")

    table_patch = Rectangle(
        (0.0, 0.0),
        table.length,
        table.width,
        fill=False,
        linewidth=2.0,
        edgecolor="black",
    )
    table_patch.set_path_effects(_high_visibility_path_effects(outer_width=5.0, inner_width=3.0))
    ax.add_patch(table_patch)
    ax.grid(True, alpha=0.18, linestyle=":")

    radius = result.initial_state.balls["C"].radius
    pscale = _momentum_scale(result, max_fraction_of_table=0.14, table=table)
    arrow_length = _standard_arrow_length(table=table)

    ball_patches: dict[str, Circle] = {}
    ball_labels: dict[str, any] = {}
    momentum_artists: dict[str, any] = {}
    trace_lines: dict[str, any] = {}
    sampled_paths = _build_ball_paths(frames)

    first_sample = frames[0]
    for label in sorted(first_sample.positions.keys()):
        pos = first_sample.positions[label]
        color = BALL_TRACE_COLORS.get(label, "gray")
        line_style = "-" if label == "C" else "--"
        line_width = 2.8 if label == "C" else 2.0

        (trace_line,) = ax.plot(
            [float(pos[0])],
            [float(pos[1])],
            linestyle=line_style,
            linewidth=line_width,
            color=color,
            alpha=0.92,
            label=BALL_NAMES.get(label, label),
            zorder=1,
        )
        trace_line.set_path_effects(_high_visibility_path_effects(4.2, 2.6))
        trace_lines[label] = trace_line

        patch = Circle(
            (float(pos[0]), float(pos[1])),
            radius=radius,
            facecolor=BALL_COLORS.get(label, "gray"),
            edgecolor="black",
            linewidth=2.0,
            zorder=4,
        )
        patch.set_path_effects(_high_visibility_path_effects(outer_width=4.6, inner_width=2.8))
        ax.add_patch(patch)
        ball_patches[label] = patch

        text = ax.text(
            float(pos[0]),
            float(pos[1]),
            label,
            fontsize=10,
            weight="bold",
            color=_ball_text_color(BALL_COLORS.get(label, "gray")),
            ha="center",
            va="center",
            zorder=5,
        )
        text.set_path_effects(_text_visibility_path_effects())
        ball_labels[label] = text

        momentum_artists[label] = None

    impulse_artists_by_event: list[list] = []
    impulse_events: list[CollisionEvent] = []

    if show_impulse_pairs:
        for event in result.events:
            if event.time > display_end_time:
                continue

            representative_impulse = _representative_impulse_vector(event)
            if representative_impulse is None:
                continue

            label, impulse_vec = representative_impulse
            event_artists: list = []
            color = BALL_TRACE_COLORS.get(label, "gray")
            anchor = _event_anchor_position(result, event)

            collision_line = _draw_collision_line(
                ax=ax,
                point=anchor,
                direction=event.collision_normal,
                table=table,
                color="dimgray",
            )
            if collision_line is not None:
                event_artists.append(collision_line)

            event_artists.extend(
                _draw_impulse_pair(
                    ax=ax,
                    position=anchor,
                    impulse_vec=impulse_vec,
                    color=color,
                    arrow_length=arrow_length,
                )
            )

            label_artist = ax.text(
                float(anchor[0] + 0.03),
                float(anchor[1] + 0.03),
                (
                    f"|J| = {format_scalar(float(np.linalg.norm(impulse_vec)), 4)} N·s\n"
                    f"J = {format_vector_sum(impulse_vec, 'i', 'j')}\n"
                    f"line: {_line_equation_from_point_direction(anchor, event.collision_normal)}"
                ),
                fontsize=8.5,
                color="black",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "black",
                    "linewidth": 0.8,
                    "alpha": 0.90,
                },
                zorder=9,
            )
            label_artist.set_path_effects(_text_visibility_path_effects())
            event_artists.append(label_artist)

            for artist in event_artists:
                artist.set_visible(False)

            impulse_events.append(event)
            impulse_artists_by_event.append(event_artists)

    legend = ax.legend(loc="upper left", framealpha=0.95)
    legend.set_zorder(12)

    status_text = ax.text(
        0.99,
        0.99,
        "",
        transform=ax.transAxes,
        fontsize=9.5,
        ha="right",
        va="top",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "black",
            "linewidth": 0.8,
            "alpha": 0.90,
        },
        zorder=12,
    )
    status_text.set_path_effects(_text_visibility_path_effects())

    def update(frame_idx: int):
        sample = frames[frame_idx]
        artists = []

        for label, patch in ball_patches.items():
            pos = sample.positions[label]
            patch.center = (float(pos[0]), float(pos[1]))
            ball_labels[label].set_position((float(pos[0]), float(pos[1])))
            history = sampled_paths[label][: frame_idx + 1]
            trace_lines[label].set_data(history[:, 0], history[:, 1])

            artists.append(trace_lines[label])
            artists.append(patch)
            artists.append(ball_labels[label])

            if show_momentum_arrows:
                old_arrow = momentum_artists[label]
                if old_arrow is not None:
                    try:
                        old_arrow.remove()
                    except Exception:
                        pass

                velocity = _velocity_at_time(result, label, sample.time)
                mass = result.initial_state.balls[label].mass

                momentum_artists[label] = _draw_momentum_arrow(
                    ax=ax,
                    origin=pos,
                    velocity=velocity,
                    mass=mass,
                    radius=radius,
                    color=BALL_COLORS.get(label, "gray"),
                    scale=pscale,
                )

                if momentum_artists[label] is not None:
                    artists.append(momentum_artists[label])

        if show_impulse_pairs:
            for event_idx, event in enumerate(impulse_events):
                visible = event.time <= sample.time
                for artist in impulse_artists_by_event[event_idx]:
                    artist.set_visible(visible)
                    if visible:
                        artists.append(artist)

        status_text.set_text(
            (
                f"t = {sample.time:.3f} s | Δt = {physical_dt:.4f} s"
                + (
                    f" | success_t = {result.success_time:.3f} s"
                    if result.success_time is not None
                    else ""
                )
                + f"\nclassification: {result.classification or 'unclassified'}"
                + f" | result: {'success' if result.success else 'failed'}"
            )
        )

        artists.append(status_text)
        return artists

    anim = mpl_animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000 / fps,
        blit=False,
    )

    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        anim.save(str(output_path), writer="pillow", fps=fps)
    else:
        anim.save(str(output_path), writer="ffmpeg", fps=fps)

    plt.close(fig)


def export_animation_frame_snapshots(
    result: SimulationResult,
    trajectory: list[TrajectorySample],
    table: Table,
    save_path: str,
    relevant_only: bool = True,
    max_events_to_include: int | None = None,
    fps: int = 30,
    duration_s: float = 3.0,
    post_end_fraction: float = 0.10,
) -> None:
    """
    Export trajectory-style plots generated from representative animation frames.

    Frames include the beginning, every collision event, and the ending state.
    Each panel shows the animation styling plus the traced history accumulated up
    to that frame.
    """
    trimmed = trim_trajectory_to_relevant_portion(
        result=result,
        trajectory=trajectory,
        relevant_only=relevant_only,
        max_events_to_include=max_events_to_include,
        post_end_fraction=post_end_fraction,
    )
    frames = resample_uniform_in_time(trimmed, fps=fps, duration_s=duration_s)
    if not frames:
        raise ValueError("Cannot export frame snapshots from an empty trajectory.")

    display_end_time = _relevant_end_time(
        result=result,
        relevant_only=relevant_only,
        max_events_to_include=max_events_to_include,
        post_end_fraction=post_end_fraction,
    )
    event_times = [
        event.time
        for event in result.events[:max_events_to_include]
        if event.time <= display_end_time
    ]
    target_times = [frames[0].time, *event_times, frames[-1].time]

    frame_indices: list[int] = []
    for target_time in target_times:
        frame_idx = min(
            range(len(frames)),
            key=lambda idx: abs(float(frames[idx].time) - float(target_time)),
        )
        if frame_idx not in frame_indices:
            frame_indices.append(frame_idx)

    n_panels = len(frame_indices)
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(12, max(3.8, 3.2 * n_panels)),
        layout="constrained",
    )
    axes = np.atleast_1d(axes)
    sampled_paths = _build_ball_paths(frames)
    radius = result.initial_state.balls["C"].radius
    arrow_length = _standard_arrow_length(table=table)

    visible_events = [event for event in result.events if event.time <= display_end_time]

    for panel_idx, frame_idx in enumerate(frame_indices):
        ax = axes[panel_idx]
        sample = frames[frame_idx]
        ax.set_xlim(0.0, table.length)
        ax.set_ylim(0.0, table.width)
        ax.set_aspect("equal")
        ax.set_facecolor("#fbf8ef")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(True, alpha=0.18, linestyle=":")
        ax.set_title(f"Animation frame at t = {sample.time:.3f} s", fontsize=12, loc="left")

        table_patch = Rectangle(
            (0.0, 0.0),
            table.length,
            table.width,
            fill=False,
            linewidth=2.0,
            edgecolor="black",
        )
        table_patch.set_path_effects(_high_visibility_path_effects(outer_width=5.0, inner_width=3.0))
        ax.add_patch(table_patch)

        for label, path in sampled_paths.items():
            history = path[: frame_idx + 1]
            color = BALL_TRACE_COLORS.get(label, "gray")
            style = "-" if label == "C" else "--"
            width = 2.8 if label == "C" else 2.0
            (line,) = ax.plot(
                history[:, 0],
                history[:, 1],
                linestyle=style,
                linewidth=width,
                color=color,
                alpha=0.92,
                label=BALL_NAMES.get(label, label),
                zorder=1,
            )
            line.set_path_effects(_high_visibility_path_effects(4.2, 2.6))

            pos = sample.positions[label]
            ball = Circle(
                (float(pos[0]), float(pos[1])),
                radius=radius,
                facecolor=BALL_COLORS.get(label, "gray"),
                edgecolor="black",
                linewidth=2.0,
                zorder=4,
            )
            ball.set_path_effects(_high_visibility_path_effects(outer_width=4.6, inner_width=2.8))
            ax.add_patch(ball)

            text = ax.text(
                float(pos[0]),
                float(pos[1]),
                label,
                fontsize=10,
                weight="bold",
                color=_ball_text_color(BALL_COLORS.get(label, "gray")),
                ha="center",
                va="center",
                zorder=5,
            )
            text.set_path_effects(_text_visibility_path_effects())

        for event in visible_events:
            if event.time > sample.time:
                continue
            representative_impulse = _representative_impulse_vector(event)
            if representative_impulse is None:
                continue
            first_label, impulse_vec = representative_impulse
            anchor = _event_anchor_position(result, event)
            _draw_collision_line(ax, anchor, event.collision_normal, table)
            _draw_impulse_pair(
                ax,
                anchor,
                impulse_vec,
                BALL_TRACE_COLORS.get(first_label, "black"),
                arrow_length,
            )
            note = ax.text(
                float(anchor[0] + 0.03),
                float(anchor[1] + 0.03),
                (
                    f"|J| = {format_scalar(float(np.linalg.norm(impulse_vec)), 4)} N·s\n"
                    f"J = {format_vector_sum(impulse_vec, 'i', 'j')}\n"
                    f"line: {_line_equation_from_point_direction(anchor, event.collision_normal)}"
                ),
                fontsize=8,
                color="black",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "black",
                    "linewidth": 0.8,
                    "alpha": 0.90,
                },
                zorder=9,
            )
            note.set_path_effects(_text_visibility_path_effects())

        legend = ax.legend(loc="upper left", framealpha=0.95)
        legend.set_zorder(12)

    save_target = Path(save_path)
    save_target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_target), dpi=300, bbox_inches="tight")
    plt.close(fig)
