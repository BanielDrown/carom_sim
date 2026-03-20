"""
Event-driven simulator for the three-ball carom system.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .collisions import find_next_event
from .constants import (
    BALL_BALL_RESTITUTION,
    BALL_WALL_RESTITUTION,
    POST_CONTACT_EVENT_BUFFER,
)
from .physics import (
    ball_ball_collision_response,
    wall_collision_response,
    wall_normal_from_name,
)
from .state import (
    AssignmentStatus,
    CollisionEvent,
    SimulationResult,
    SimulationState,
    Table,
    TrajectorySample,
)
from .validation import advance_assignment_status, validate_result


def advance_state(state: SimulationState, dt: float) -> None:
    """
    Advance all balls by dt using constant-velocity motion.
    """
    if dt < 0.0:
        raise ValueError(f"dt must be nonnegative, got {dt}")

    for ball in state.balls.values():
        ball.position = ball.position + ball.velocity * dt
    state.time += dt


def snapshot_positions(state: SimulationState) -> dict[str, np.ndarray]:
    """
    Copy the current ball positions.
    """
    return {label: ball.position.copy() for label, ball in state.balls.items()}


def _make_collision_position(
    state: SimulationState,
    event_type: str,
    actors: tuple[str, str],
) -> np.ndarray:
    """
    Construct a representative collision position.
    """
    if event_type == "ball-ball":
        a, b = actors
        return 0.5 * (state.balls[a].position + state.balls[b].position)

    if event_type == "ball-wall":
        ball_label, _wall = actors
        return state.balls[ball_label].position.copy()

    raise ValueError(f"Unknown event type: {event_type}")


def _append_trajectory_sample(
    trajectory: list[TrajectorySample],
    state: SimulationState,
) -> None:
    """
    Append a trajectory sample at the current state time.
    """
    trajectory.append(
        TrajectorySample(
            time=state.time,
            positions=snapshot_positions(state),
        )
    )


def _collision_frame(
    state: SimulationState,
    event_type: str,
    actors: tuple[str, str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct orthonormal collision-frame unit vectors (n, t).

    n points along the collision normal and t is the +90 degree rotation of n.
    """
    if event_type == "ball-ball":
        a, b = actors
        delta = state.balls[b].position - state.balls[a].position
        norm = float(np.linalg.norm(delta))
        if norm <= 0.0:
            raise ValueError("Cannot build a collision frame for coincident ball centers.")
        normal = delta / norm
    elif event_type == "ball-wall":
        _ball_label, wall_name = actors
        normal = wall_normal_from_name(wall_name)
    else:
        raise ValueError(f"Unknown event type: {event_type}")

    tangent = np.array([-normal[1], normal[0]], dtype=float)
    return normal, tangent


def simulate(
    initial_state: SimulationState,
    table: Optional[Table] = None,
    max_events: int = 50,
    max_time: float = 20.0,
    ball_ball_restitution: float = BALL_BALL_RESTITUTION,
    ball_wall_restitution: float = BALL_WALL_RESTITUTION,
    post_success_fraction: float = 0.10,
) -> tuple[SimulationResult, list[TrajectorySample]]:
    """
    Run an event-driven simulation.

    Success condition
    -----------------
    - cue ball C contacts both A and B
    - each ball A, B, and C hits at least one wall

    Notes
    -----
    - The simulation does not stop immediately upon assignment success.
    - When success is first achieved at time t_success, the simulation continues
      until t_stop = (1 + post_success_fraction) * t_success so that downstream
      plots and animations can show the final motion clearly.
    - If there are no more collisions before t_stop, the state is advanced by
      free motion to t_stop and a final trajectory sample is stored.
    """
    if table is None:
        table = Table()

    if post_success_fraction < 0.0:
        raise ValueError(
            f"post_success_fraction must be nonnegative, got {post_success_fraction}"
        )

    state = initial_state.copy()
    events: list[CollisionEvent] = []
    trajectory: list[TrajectorySample] = []

    assignment_status = AssignmentStatus()
    first_all_contacts_event_index: int | None = None

    success_time: float | None = None
    stop_time_after_success: float | None = None

    _append_trajectory_sample(trajectory, state)

    termination_reason = "max_events"

    for _ in range(max_events):
        # If success has already been reached and the current time has met or
        # exceeded the extended stop time, the run is complete.
        if stop_time_after_success is not None and state.time >= stop_time_after_success:
            termination_reason = "post_success_extension_complete"
            break

        next_ev = find_next_event(state, table)

        # No more collisions exist.
        if next_ev is None:
            if stop_time_after_success is not None:
                # Continue by free motion to the planned stop time, provided the
                # global max_time constraint allows it.
                final_target_time = min(stop_time_after_success, max_time)

                if final_target_time > state.time:
                    advance_state(state, final_target_time - state.time)
                    _append_trajectory_sample(trajectory, state)

                termination_reason = (
                    "post_success_extension_complete"
                    if stop_time_after_success <= max_time
                    else "max_time"
                )
            else:
                termination_reason = "no_future_event"
            break

        dt = next_ev.time_to_event
        event_time = state.time + dt
        event_type = next_ev.event_type
        actors = next_ev.actors

        # If success has already occurred and the next event would happen after
        # the extended stop time, advance only to the stop time and finish.
        if stop_time_after_success is not None and event_time > stop_time_after_success:
            final_target_time = min(stop_time_after_success, max_time)

            if final_target_time > state.time:
                advance_state(state, final_target_time - state.time)
                _append_trajectory_sample(trajectory, state)

            termination_reason = (
                "post_success_extension_complete"
                if stop_time_after_success <= max_time
                else "max_time"
            )
            break

        # Global simulation time cap.
        if event_time > max_time:
            if stop_time_after_success is not None:
                final_target_time = max_time
                if final_target_time > state.time:
                    advance_state(state, final_target_time - state.time)
                    _append_trajectory_sample(trajectory, state)
            termination_reason = "max_time"
            break

        advance_state(state, dt)

        pre_velocities = {label: ball.velocity.copy() for label, ball in state.balls.items()}
        impulse_magnitude = 0.0
        impulse_vectors: dict[str, np.ndarray] = {}

        if event_type == "ball-ball":
            a, b = actors
            ball1 = state.balls[a]
            ball2 = state.balls[b]

            (
                v1_after,
                v2_after,
                impulse_magnitude,
                impulse_on_ball1,
                impulse_on_ball2,
            ) = ball_ball_collision_response(
                ball1,
                ball2,
                restitution=ball_ball_restitution,
            )

            ball1.velocity = v1_after
            ball2.velocity = v2_after

            impulse_vectors[a] = impulse_on_ball1
            impulse_vectors[b] = impulse_on_ball2

        elif event_type == "ball-wall":
            ball_label, wall_name = actors
            ball = state.balls[ball_label]
            wall_normal = wall_normal_from_name(wall_name)

            v_after, impulse_magnitude, impulse_on_ball = wall_collision_response(
                ball,
                wall_normal,
                restitution=ball_wall_restitution,
            )

            ball.velocity = v_after
            impulse_vectors[ball_label] = impulse_on_ball

        else:
            raise ValueError(f"Unknown event type: {event_type}")

        post_velocities = {label: ball.velocity.copy() for label, ball in state.balls.items()}

        collision_position = _make_collision_position(
            state=state,
            event_type=event_type,
            actors=actors,
        )
        collision_normal, collision_tangent = _collision_frame(
            state=state,
            event_type=event_type,
            actors=actors,
        )

        event = CollisionEvent(
            time=state.time,
            event_type=event_type,
            actors=actors,
            position=collision_position,
            impulse=float(impulse_magnitude),
            pre_velocities=pre_velocities,
            post_velocities=post_velocities,
            impulse_vectors=impulse_vectors,
            collision_normal=collision_normal,
            collision_tangent=collision_tangent,
        )
        events.append(event)

        advance_assignment_status(assignment_status, event)
        _append_trajectory_sample(trajectory, state)

        if assignment_status.cue_hit_both and first_all_contacts_event_index is None:
            first_all_contacts_event_index = len(events)

        # First time the full assignment success condition is met:
        # record the success time and continue 10% longer in physical time.
        if assignment_status.success and success_time is None:
            success_time = state.time
            stop_time_after_success = (1.0 + post_success_fraction) * success_time

            # Edge case: success_time == 0. In practice this should not happen for
            # a collision-driven success condition, but handle it defensively.
            if stop_time_after_success <= state.time:
                termination_reason = "post_success_extension_complete"
                break

        # Pre-success buffer logic remains active only until actual success occurs.
        if (
            success_time is None
            and first_all_contacts_event_index is not None
            and len(events) - first_all_contacts_event_index >= POST_CONTACT_EVENT_BUFFER
            and not assignment_status.success
        ):
            termination_reason = "post_contact_buffer_exhausted"
            break

    result = SimulationResult(
        initial_state=initial_state.copy(),
        final_state=state.copy(),
        events=events,
        success=assignment_status.success,
        classification=None,
        termination_reason=termination_reason,
        assignment_status=assignment_status,
        success_time=success_time,
        display_end_time=state.time,
    )

    result = validate_result(result)

    return result, trajectory
