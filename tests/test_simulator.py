import numpy as np

from carom.simulator import advance_state, simulate
from carom.state import Ball, SimulationState, Table, vec2


def assert_vec_close(a: np.ndarray, b: np.ndarray, tol: float = 1e-9) -> None:
    assert np.allclose(a, b, atol=tol), f"{a} != {b}"


def test_advance_state_updates_position_and_time():
    balls = {
        "C": Ball(label="C", position=vec2(1.0, 0.5), velocity=vec2(2.0, -1.0)),
    }
    state = SimulationState(time=0.0, balls=balls)

    advance_state(state, 0.5)

    assert abs(state.time - 0.5) < 1e-9
    assert_vec_close(state.balls["C"].position, vec2(2.0, 0.0))


def test_simulate_single_wall_collision():
    table = Table()
    initial_state = SimulationState(
        time=0.0,
        balls={
            "C": Ball(label="C", position=vec2(1.0, 0.5), velocity=vec2(1.0, 0.0)),
        },
    )

    result, trajectory = simulate(
        initial_state,
        table=table,
        max_events=1,
        max_time=10.0,
    )

    assert len(result.events) == 1
    event = result.events[0]

    assert event.event_type == "ball-wall"
    assert event.actors == ("C", "right")
    assert result.final_state.balls["C"].velocity[0] < 0.0
    assert len(trajectory) == 2

    assert "C" in event.impulse_vectors
    assert event.impulse is not None
    assert event.impulse > 0.0


def test_simulate_single_ball_ball_collision():
    initial_state = SimulationState(
        time=0.0,
        balls={
            "B": Ball(label="B", position=vec2(1.3, 0.5), velocity=vec2(0.0, 0.0)),
            "C": Ball(label="C", position=vec2(1.0, 0.5), velocity=vec2(1.0, 0.0)),
        },
    )

    result, _trajectory = simulate(
        initial_state,
        table=Table(),
        max_events=1,
        max_time=10.0,
    )

    assert len(result.events) == 1
    event = result.events[0]

    assert event.event_type == "ball-ball"
    assert event.actors == ("B", "C")

    vB = result.final_state.balls["B"].velocity
    vC = result.final_state.balls["C"].velocity

    assert abs(vB[0] - 1.0) < 1e-9
    assert abs(vC[0] - 0.0) < 1e-9

    assert "B" in event.impulse_vectors
    assert "C" in event.impulse_vectors
    assert event.impulse is not None
    assert event.impulse > 0.0

    assert_vec_close(
        event.impulse_vectors["B"] + event.impulse_vectors["C"],
        vec2(0.0, 0.0),
    )


def test_assignment_status_updates_on_wall_hit():
    table = Table()
    initial_state = SimulationState(
        time=0.0,
        balls={
            "C": Ball(label="C", position=vec2(1.0, 0.5), velocity=vec2(1.0, 0.0)),
        },
    )

    result, _trajectory = simulate(
        initial_state,
        table=table,
        max_events=1,
        max_time=10.0,
    )

    assert result.assignment_status.wall_hits["C"] == 1
    assert result.assignment_status.wall_hits["A"] == 0
    assert result.assignment_status.wall_hits["B"] == 0
    assert result.success is False


def test_assignment_status_updates_on_cue_ball_contact():
    initial_state = SimulationState(
        time=0.0,
        balls={
            "A": Ball(label="A", position=vec2(1.3, 0.5), velocity=vec2(0.0, 0.0)),
            "C": Ball(label="C", position=vec2(1.0, 0.5), velocity=vec2(1.0, 0.0)),
        },
    )

    result, _trajectory = simulate(
        initial_state,
        table=Table(),
        max_events=1,
        max_time=10.0,
    )

    assert "A" in result.assignment_status.cue_contacts
    assert "B" not in result.assignment_status.cue_contacts
    assert result.assignment_status.cue_hit_a is True
    assert result.assignment_status.cue_hit_b is False
    assert result.assignment_status.cue_hit_both is False