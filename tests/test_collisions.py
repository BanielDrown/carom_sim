from carom.collisions import (
    CandidateEvent,
    find_next_event,
    time_to_ball_ball_collision,
    time_to_horizontal_wall_collision,
    time_to_vertical_wall_collision,
)
from carom.state import Ball, SimulationState, Table, vec2


def test_time_to_right_wall():
    table = Table()
    ball = Ball(label="C", position=vec2(1.0, 0.5), velocity=vec2(2.0, 0.0))

    result = time_to_vertical_wall_collision(ball, table)

    assert result is not None
    t, wall = result
    expected = (table.x_max - ball.radius - 1.0) / 2.0

    assert abs(t - expected) < 1e-9
    assert wall == "right"


def test_time_to_top_wall():
    table = Table()
    ball = Ball(label="C", position=vec2(1.0, 0.5), velocity=vec2(0.0, 1.0))

    result = time_to_horizontal_wall_collision(ball, table)

    assert result is not None
    t, wall = result
    expected = table.y_max - ball.radius - 0.5

    assert abs(t - expected) < 1e-9
    assert wall == "top"


def test_no_wall_collision_if_stationary_in_axis():
    table = Table()
    ball = Ball(label="C", position=vec2(1.0, 0.5), velocity=vec2(0.0, 2.0))

    assert time_to_vertical_wall_collision(ball, table) is None


def test_ball_ball_collision_head_on():
    b1 = Ball(label="A", position=vec2(0.0, 0.0), velocity=vec2(1.0, 0.0))
    b2 = Ball(label="B", position=vec2(1.0, 0.0), velocity=vec2(0.0, 0.0))

    t = time_to_ball_ball_collision(b1, b2)
    assert t is not None

    sigma = b1.radius + b2.radius
    expected = (1.0 - sigma) / 1.0
    assert abs(t - expected) < 1e-9


def test_no_ball_ball_collision_when_moving_apart():
    b1 = Ball(label="A", position=vec2(0.0, 0.0), velocity=vec2(-1.0, 0.0))
    b2 = Ball(label="B", position=vec2(1.0, 0.0), velocity=vec2(0.0, 0.0))

    assert time_to_ball_ball_collision(b1, b2) is None


def test_find_next_event_prefers_earliest():
    table = Table()
    balls = {
        "A": Ball(label="A", position=vec2(0.5, 1.2), velocity=vec2(0.0, 0.0)),
        "B": Ball(label="B", position=vec2(1.3, 0.5), velocity=vec2(0.0, 0.0)),
        "C": Ball(label="C", position=vec2(1.0, 0.5), velocity=vec2(1.0, 0.0)),
    }
    state = SimulationState(time=0.0, balls=balls)

    event = find_next_event(state, table)

    assert event is not None
    assert isinstance(event, CandidateEvent)
    assert event.event_type == "ball-ball"
    assert event.actors == ("B", "C")
    assert event.time_to_event > 0.0


def test_find_next_event_prefers_ball_ball_over_wall_when_times_match():
    table = Table()

    # C will hit B before any wall if configured like this.
    balls = {
        "B": Ball(label="B", position=vec2(1.2, 0.5), velocity=vec2(0.0, 0.0)),
        "C": Ball(label="C", position=vec2(1.0, 0.5), velocity=vec2(1.0, 0.0)),
    }
    state = SimulationState(time=0.0, balls=balls)

    event = find_next_event(state, table)

    assert event is not None
    assert event.event_type == "ball-ball"
    assert event.actors == ("B", "C")