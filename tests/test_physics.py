import numpy as np

from src.carom.physics import (
    reflect_velocity_from_wall,
    ball_ball_collision_response,
    wall_normal_from_name,
)
from src.carom.state import Ball, vec2


def assert_vec_close(a: np.ndarray, b: np.ndarray, tol: float = 1e-9) -> None:
    assert np.allclose(a, b, atol=tol), f"{a} != {b}"


def test_reflect_left_wall_head_on():
    v = vec2(-2.0, 0.0)
    n = wall_normal_from_name("left")
    v_after = reflect_velocity_from_wall(v, n, restitution=1.0)
    assert_vec_close(v_after, vec2(2.0, 0.0))


def test_reflect_top_wall_oblique():
    v = vec2(1.5, 3.0)
    n = wall_normal_from_name("top")
    v_after = reflect_velocity_from_wall(v, n, restitution=1.0)
    assert_vec_close(v_after, vec2(1.5, -3.0))


def test_reflect_inelastic_wall():
    v = vec2(-4.0, 1.0)
    n = wall_normal_from_name("left")
    v_after = reflect_velocity_from_wall(v, n, restitution=0.5)
    assert_vec_close(v_after, vec2(2.0, 1.0))


def test_head_on_equal_mass_velocity_swap():
    b1 = Ball(label="A", position=vec2(0.0, 0.0), velocity=vec2(1.0, 0.0))
    b2 = Ball(label="B", position=vec2(1.0, 0.0), velocity=vec2(0.0, 0.0))

    v1_after, v2_after, impulse = ball_ball_collision_response(b1, b2, restitution=1.0)

    assert_vec_close(v1_after, vec2(0.0, 0.0))
    assert_vec_close(v2_after, vec2(1.0, 0.0))
    assert impulse > 0.0


def test_head_on_equal_mass_inelastic():
    b1 = Ball(label="A", position=vec2(0.0, 0.0), velocity=vec2(2.0, 0.0))
    b2 = Ball(label="B", position=vec2(1.0, 0.0), velocity=vec2(0.0, 0.0))

    v1_after, v2_after, impulse = ball_ball_collision_response(b1, b2, restitution=0.5)

    assert_vec_close(v1_after, vec2(0.5, 0.0))
    assert_vec_close(v2_after, vec2(1.5, 0.0))
    assert impulse > 0.0


def test_glancing_collision_preserves_tangential_component():
    b1 = Ball(label="A", position=vec2(0.0, 0.0), velocity=vec2(1.0, 1.0))
    b2 = Ball(label="B", position=vec2(1.0, 0.0), velocity=vec2(0.0, 0.0))

    v1_after, v2_after, _ = ball_ball_collision_response(b1, b2, restitution=1.0)

    # Normal is x-direction, so y-component of ball1 should remain unchanged.
    assert abs(v1_after[1] - 1.0) < 1e-9
    assert abs(v2_after[1] - 0.0) < 1e-9


def test_no_impulse_when_separating():
    b1 = Ball(label="A", position=vec2(0.0, 0.0), velocity=vec2(-1.0, 0.0))
    b2 = Ball(label="B", position=vec2(1.0, 0.0), velocity=vec2(0.0, 0.0))

    v1_after, v2_after, impulse = ball_ball_collision_response(b1, b2, restitution=1.0)

    assert_vec_close(v1_after, b1.velocity)
    assert_vec_close(v2_after, b2.velocity)
    assert impulse == 0.0