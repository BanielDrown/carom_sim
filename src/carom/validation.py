"""
Validation utilities for assignment-specific success conditions and
supplementary shot classification.
"""

from __future__ import annotations

from carom.classifier import classify_shot
from carom.state import AssignmentStatus, CollisionEvent, SimulationResult


def advance_assignment_status(
    status: AssignmentStatus,
    event: CollisionEvent,
) -> None:
    """
    Update assignment progress from a single collision event.

    This centralizes the assignment rule interpretation so simulation,
    validation, plotting, and animation all agree on what counts as:
    - a cue-ball contact with A or B
    - a wall hit for A, B, or C
    """
    if event.event_type == "ball-ball":
        a, b = event.actors

        if "C" in (a, b):
            other = b if a == "C" else a
            if other in ("A", "B"):
                status.cue_contacts.add(other)

        return

    if event.event_type == "ball-wall":
        ball_label, _wall = event.actors
        if ball_label in status.wall_hits:
            status.wall_hits[ball_label] += 1


def first_success_event_index(events: list[CollisionEvent]) -> int | None:
    """
    Return the 1-based event index where assignment success is first achieved.
    """
    status = AssignmentStatus()

    for index, event in enumerate(events, start=1):
        advance_assignment_status(status, event)
        if status.success:
            return index

    return None


def build_assignment_status(result: SimulationResult) -> AssignmentStatus:
    """
    Reconstruct final assignment progress from the full event history.

    Success criteria:
    - Cue ball C must contact both A and B
    - Each ball A, B, and C must hit at least one wall
    """
    status = AssignmentStatus()

    for event in result.events:
        advance_assignment_status(status, event)

    return status


def first_success_time(result: SimulationResult) -> float | None:
    """
    Reconstruct the first physical time at which the assignment success
    condition becomes true.

    Returns
    -------
    float | None
        The first success time if success is achieved, otherwise None.
    """
    status = AssignmentStatus()

    for event in result.events:
        advance_assignment_status(status, event)
        if status.success:
            return float(event.time)

    return None


def validate_result(result: SimulationResult) -> SimulationResult:
    """
    Update SimulationResult with assignment-based success and supplementary
    shot classification.

    Notes
    -----
    - success is determined by the assignment requirements
    - classification is retained as descriptive metadata only
    - success_time is reconstructed if missing
    - display_end_time defaults to final_state.time if missing
    """
    assignment_status = build_assignment_status(result)
    classification = classify_shot(result.events)

    result.assignment_status = assignment_status
    result.classification = classification
    result.success = assignment_status.success

    if result.success_time is None:
        result.success_time = first_success_time(result)

    if result.display_end_time is None:
        result.display_end_time = float(result.final_state.time)

    return result
