import numpy as np

from carom.search import (
    SearchCandidate,
    candidate_score,
    deduplicate_angles,
    make_state_with_cue_velocity,
    randomize_object_positions,
)
from carom.state import AssignmentStatus, Ball, SimulationResult, SimulationState, Table, vec2


def test_make_state_with_cue_velocity_sets_c_velocity_only():
    base_state = SimulationState(
        time=0.0,
        balls={
            "A": Ball("A", vec2(2.0, 0.8), vec2(0.0, 0.0)),
            "B": Ball("B", vec2(2.5, 0.3), vec2(0.0, 0.0)),
            "C": Ball("C", vec2(1.0, 0.5), vec2(0.0, 0.0)),
        },
    )

    state = make_state_with_cue_velocity(base_state, speed=2.0, angle_rad=0.0)

    assert np.allclose(state.balls["C"].velocity, vec2(2.0, 0.0))
    assert np.allclose(state.balls["A"].velocity, vec2(0.0, 0.0))
    assert np.allclose(state.balls["B"].velocity, vec2(0.0, 0.0))


def test_randomize_object_positions_returns_separated_positions():
    table = Table()
    cue_position = vec2(1.0, 0.5)
    ball_radius = 0.03075

    a_pos, b_pos = randomize_object_positions(
        table=table,
        cue_position=cue_position,
        ball_radius=ball_radius,
    )

    assert 0.0 <= a_pos[0] <= table.length
    assert 0.0 <= a_pos[1] <= table.width
    assert 0.0 <= b_pos[0] <= table.length
    assert 0.0 <= b_pos[1] <= table.width

    assert np.linalg.norm(a_pos - b_pos) > 0.0
    assert np.linalg.norm(a_pos - cue_position) > 0.0
    assert np.linalg.norm(b_pos - cue_position) > 0.0


def test_deduplicate_angles_removes_near_duplicates():
    angles = [0.0, 1e-8, np.pi / 2, np.pi / 2 + 1e-8]
    deduped = deduplicate_angles(angles, tol=1e-6)

    assert len(deduped) == 2


def test_candidate_score_prefers_fewer_events():
    dummy_state = SimulationState(time=0.0, balls={})

    result_short = SimulationResult(
        initial_state=dummy_state,
        final_state=dummy_state,
        events=[1, 2],
        success=True,
        assignment_status=AssignmentStatus(),
    )
    result_long = SimulationResult(
        initial_state=dummy_state,
        final_state=dummy_state,
        events=[1, 2, 3],
        success=True,
        assignment_status=AssignmentStatus(),
    )

    c1 = SearchCandidate(
        angle_rad=0.0,
        speed_mps=1.0,
        result=result_short,
        trajectory=[],
        layout_id=1,
    )
    c2 = SearchCandidate(
        angle_rad=0.0,
        speed_mps=7.5,
        result=result_long,
        trajectory=[],
        layout_id=1,
    )

    assert candidate_score(c1) < candidate_score(c2)


def test_candidate_score_prefers_higher_speed_when_other_terms_equal():
    dummy_state = SimulationState(time=0.0, balls={})
    status = AssignmentStatus()

    result_a = SimulationResult(
        initial_state=dummy_state,
        final_state=dummy_state,
        events=[1, 2],
        success=True,
        assignment_status=status,
    )
    result_b = SimulationResult(
        initial_state=dummy_state,
        final_state=dummy_state,
        events=[1, 2],
        success=True,
        assignment_status=AssignmentStatus(),
    )

    slower = SearchCandidate(
        angle_rad=0.0,
        speed_mps=3.0,
        result=result_a,
        trajectory=[],
        layout_id=1,
    )
    faster = SearchCandidate(
        angle_rad=0.0,
        speed_mps=7.5,
        result=result_b,
        trajectory=[],
        layout_id=1,
    )

    assert candidate_score(faster) < candidate_score(slower)