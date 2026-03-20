import numpy as np

from carom.simulator import simulate
from carom.state import (
    AssignmentStatus,
    Ball,
    CollisionEvent,
    SimulationResult,
    SimulationState,
    Table,
    vec2,
)
from carom.validation import build_assignment_status, validate_result


def make_event(event_type, actors):
    return CollisionEvent(
        time=0.0,
        event_type=event_type,
        actors=actors,
        position=np.array([0.0, 0.0]),
        impulse=0.0,
        pre_velocities={},
        post_velocities={},
        impulse_vectors={},
    )


def test_simulate_result_has_validation_fields():
    initial_state = SimulationState(
        time=0.0,
        balls={
            "A": Ball(label="A", position=vec2(2.0, 0.5), velocity=vec2(0.0, 0.0)),
            "B": Ball(label="B", position=vec2(2.5, 0.5), velocity=vec2(0.0, 0.0)),
            "C": Ball(label="C", position=vec2(1.0, 0.5), velocity=vec2(1.0, 0.0)),
        },
    )

    result, _ = simulate(
        initial_state,
        table=Table(),
        max_events=10,
        max_time=10.0,
    )

    assert hasattr(result, "success")
    assert hasattr(result, "classification")
    assert hasattr(result, "assignment_status")
    assert isinstance(result.assignment_status, AssignmentStatus)


def test_build_assignment_status_success_case():
    dummy_state = SimulationState(time=0.0, balls={})

    result = SimulationResult(
        initial_state=dummy_state,
        final_state=dummy_state,
        events=[
            make_event("ball-ball", ("C", "A")),
            make_event("ball-wall", ("C", "left")),
            make_event("ball-wall", ("A", "top")),
            make_event("ball-wall", ("B", "right")),
            make_event("ball-ball", ("C", "B")),
        ],
        success=False,
        classification=None,
        termination_reason="test",
    )

    status = build_assignment_status(result)

    assert status.cue_hit_a is True
    assert status.cue_hit_b is True
    assert status.cue_hit_both is True
    assert status.wall_hits["A"] == 1
    assert status.wall_hits["B"] == 1
    assert status.wall_hits["C"] == 1
    assert status.all_balls_hit_wall is True
    assert status.success is True


def test_build_assignment_status_failure_case_missing_b_wall():
    dummy_state = SimulationState(time=0.0, balls={})

    result = SimulationResult(
        initial_state=dummy_state,
        final_state=dummy_state,
        events=[
            make_event("ball-ball", ("C", "A")),
            make_event("ball-wall", ("C", "left")),
            make_event("ball-wall", ("A", "top")),
            make_event("ball-ball", ("C", "B")),
        ],
        success=False,
        classification=None,
        termination_reason="test",
    )

    status = build_assignment_status(result)

    assert status.cue_hit_both is True
    assert status.wall_hits["A"] == 1
    assert status.wall_hits["B"] == 0
    assert status.wall_hits["C"] == 1
    assert status.all_balls_hit_wall is False
    assert status.success is False


def test_validate_result_sets_assignment_success_and_classification():
    dummy_state = SimulationState(time=0.0, balls={})

    result = SimulationResult(
        initial_state=dummy_state,
        final_state=dummy_state,
        events=[
            make_event("ball-ball", ("C", "A")),
            make_event("ball-wall", ("C", "left")),
            make_event("ball-wall", ("A", "top")),
            make_event("ball-wall", ("B", "right")),
            make_event("ball-ball", ("C", "B")),
        ],
        success=False,
        classification=None,
        termination_reason="test",
    )

    validated = validate_result(result)

    assert validated.success is True
    assert validated.classification == "one_cushion"
    assert validated.assignment_status.success is True


def test_validate_result_failure_even_if_classified():
    dummy_state = SimulationState(time=0.0, balls={})

    result = SimulationResult(
        initial_state=dummy_state,
        final_state=dummy_state,
        events=[
            make_event("ball-ball", ("C", "A")),
            make_event("ball-wall", ("C", "left")),
            make_event("ball-ball", ("C", "B")),
        ],
        success=False,
        classification=None,
        termination_reason="test",
    )

    validated = validate_result(result)

    assert validated.classification == "one_cushion"
    assert validated.assignment_status.success is False
    assert validated.success is False