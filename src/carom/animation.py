"""
Animation utilities for the carom simulator.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.animation as mpl_animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

from carom.state import SimulationResult, Table, TrajectorySample
from carom.validation import first_success_event_index


BALL_COLORS = {
    "A": "red",
    "B": "blue",
    "C": "black",
}


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


def _impulse_scale(
    result: SimulationResult,
    table: Table,
    max_fraction_of_table: float = 0.12,
) -> float:
    """
    Compute one global impulse-arrow scale for the whole animation.
    """
    impulse_magnitudes = [
        float(np.linalg.norm(vec))
        for event in result.events
        for vec in event.impulse_vectors.values()
    ]

    if not impulse_magnitudes:
        return 0.0

    jmax = max(impulse_magnitudes)
    if jmax <= 1e-12:
        return 0.0

    max_arrow_length = max_fraction_of_table * min(table.length, table.width)
    return max_arrow_length / jmax


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
    if mag <= 1e-12 or scale <= 0.0:
        return None

    direction = momentum / mag
    tail = origin - 0.35 * radius * direction

    dx = scale * float(momentum[0])
    dy = scale * float(momentum[1])

    return ax.arrow(
        float(tail[0]),
        float(tail[1]),
        dx,
        dy,
        length_includes_head=True,
        head_width=0.025,
        head_length=0.04,
        linewidth=1.8,
        color=color,
        alpha=0.9,
        zorder=5,
    )


def _draw_impulse_pair(
    ax,
    position: np.ndarray,
    impulse_vec: np.ndarray,
    color: str,
    scale: float,
):
    """
    Draw equal and opposite impulse arrows centered at the collision point.
    """
    mag = float(np.linalg.norm(impulse_vec))
    if mag <= 1e-12 or scale <= 0.0:
        return []

    half = scale * impulse_vec

    start1 = position - 0.5 * half
    start2 = position + 0.5 * half

    a1 = ax.arrow(
        float(start1[0]),
        float(start1[1]),
        float(half[0]),
        float(half[1]),
        length_includes_head=True,
        head_width=0.02,
        head_length=0.03,
        color=color,
        linewidth=1.8,
        alpha=0.85,
        zorder=8,
    )

    a2 = ax.arrow(
        float(start2[0]),
        float(start2[1]),
        float(-half[0]),
        float(-half[1]),
        length_includes_head=True,
        head_width=0.02,
        head_length=0.03,
        color=color,
        linewidth=1.8,
        alpha=0.85,
        zorder=8,
    )

    return [a1, a2]


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

    physical_start_time = float(frames[0].time)
    physical_end_time = float(frames[-1].time)
    physical_dt = 0.0 if len(frames) <= 1 else (physical_end_time - physical_start_time) / (len(frames) - 1)

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
    ax.set_title("Carom Simulation Animation")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    table_patch = Rectangle(
        (0.0, 0.0),
        table.length,
        table.width,
        fill=False,
        linewidth=2.0,
        edgecolor="black",
    )
    ax.add_patch(table_patch)

    radius = result.initial_state.balls["C"].radius
    pscale = _momentum_scale(result, max_fraction_of_table=0.18, table=table)
    jscale = _impulse_scale(result, table=table, max_fraction_of_table=0.12)

    ball_patches: dict[str, Circle] = {}
    ball_labels: dict[str, any] = {}
    momentum_artists: dict[str, any] = {}

    first_sample = frames[0]
    for label in sorted(first_sample.positions.keys()):
        pos = first_sample.positions[label]

        patch = Circle(
            (float(pos[0]), float(pos[1])),
            radius=radius,
            color=BALL_COLORS.get(label, "gray"),
            zorder=3,
        )
        ax.add_patch(patch)
        ball_patches[label] = patch

        text = ax.text(
            float(pos[0] + 0.02),
            float(pos[1] + 0.02),
            label,
            fontsize=10,
            weight="bold",
            color=BALL_COLORS.get(label, "gray"),
            zorder=4,
        )
        ball_labels[label] = text

        momentum_artists[label] = None

    impulse_artists_by_event: list[list] = []
    impulse_events: list = []

    if show_impulse_pairs:
        for event in result.events:
            if event.time > display_end_time:
                continue

            event_artists = []
            for label, impulse_vec in event.impulse_vectors.items():
                color = BALL_COLORS.get(label, "gray")
                new_artists = _draw_impulse_pair(
                    ax=ax,
                    position=event.position,
                    impulse_vec=impulse_vec,
                    color=color,
                    scale=jscale,
                )
                for artist in new_artists:
                    artist.set_visible(False)
                event_artists.extend(new_artists)

            impulse_events.append(event)
            impulse_artists_by_event.append(event_artists)

    status_text = ax.text(
        0.01,
        1.02,
        "",
        transform=ax.transAxes,
        fontsize=10,
    )

    def update(frame_idx: int):
        sample = frames[frame_idx]
        artists = []

        for label, patch in ball_patches.items():
            pos = sample.positions[label]
            patch.center = (float(pos[0]), float(pos[1]))
            ball_labels[label].set_position((float(pos[0] + 0.02), float(pos[1] + 0.02)))

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
            f"t = {sample.time:.3f} s | "
            f"Δt_frame = {physical_dt:.4f} s | "
            f"success_t = {result.success_time:.3f} s | " if result.success_time is not None else
            f"t = {sample.time:.3f} s | "
            f"Δt_frame = {physical_dt:.4f} s | "
        )

        # Append classification/result in a second call to keep the conditional readable.
        current_status = status_text.get_text()
        current_status += (
            f"classification: {result.classification or 'unclassified'} | "
            f"result: {'success' if result.success else 'failed'}"
        )
        status_text.set_text(current_status)

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
