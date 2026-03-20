"""
Collision response functions for the three-ball carom simulator.
"""

from __future__ import annotations

import numpy as np

from .constants import BALL_BALL_RESTITUTION, BALL_WALL_RESTITUTION, EPS
from .geometry import dot, norm_squared, unit
from .state import Ball


def reflect_velocity_from_wall(
    velocity: np.ndarray,
    wall_normal: np.ndarray,
    restitution: float = BALL_WALL_RESTITUTION,
) -> np.ndarray:
    """
    Reflect a velocity vector from a rigid wall.

    The tangential component is unchanged.
    The normal component is reversed and scaled by restitution.
    """
    n = unit(wall_normal)
    v_n = dot(velocity, n) * n
    v_t = velocity - v_n
    return v_t - restitution * v_n


def wall_collision_response(
    ball: Ball,
    wall_normal: np.ndarray,
    restitution: float = BALL_WALL_RESTITUTION,
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Compute the post-collision velocity for a ball-wall impact.

    Parameters
    ----------
    ball : Ball
        Ball at the instant of impact.
    wall_normal : np.ndarray
        Unit normal pointing inward from the wall into the table.
    restitution : float
        Ball-wall coefficient of restitution.

    Returns
    -------
    tuple[np.ndarray, float, np.ndarray]
        (v_after, impulse_magnitude, impulse_vector_on_ball)

    Notes
    -----
    The impulse vector is computed from momentum change:
        J = m (v_after - v_before)
    """
    v_before = ball.velocity.copy()
    v_after = reflect_velocity_from_wall(
        velocity=v_before,
        wall_normal=wall_normal,
        restitution=restitution,
    )

    impulse_vector = ball.mass * (v_after - v_before)
    impulse_magnitude = float(np.linalg.norm(impulse_vector))

    return v_after, impulse_magnitude, impulse_vector


def ball_ball_collision_response(
    ball1: Ball,
    ball2: Ball,
    restitution: float = BALL_BALL_RESTITUTION,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Compute the post-collision velocities for a 2D collision between two balls.

    Parameters
    ----------
    ball1, ball2 : Ball
        Balls at the instant of impact.
    restitution : float
        Ball-ball coefficient of restitution.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]
        (
            v1_after,
            v2_after,
            impulse_magnitude,
            impulse_on_ball1,
            impulse_on_ball2,
        )

    Notes
    -----
    This function assumes the balls are in contact at the moment of collision.
    Only the normal component along the line of centers is changed.

    If the balls are not approaching, no impulse is applied and zero impulse
    vectors are returned.
    """
    r12 = ball2.position - ball1.position
    if norm_squared(r12) < EPS:
        raise ValueError("Ball centers are coincident or too close to define collision normal.")

    n = unit(r12)
    v_rel = ball1.velocity - ball2.velocity
    v_rel_n = dot(v_rel, n)

    if v_rel_n <= 0:
        zero = np.zeros(2, dtype=float)
        return (
            ball1.velocity.copy(),
            ball2.velocity.copy(),
            0.0,
            zero.copy(),
            zero.copy(),
        )

    m1 = ball1.mass
    m2 = ball2.mass

    impulse_mag = (1.0 + restitution) * v_rel_n / ((1.0 / m1) + (1.0 / m2))

    impulse_on_ball1 = -impulse_mag * n
    impulse_on_ball2 = +impulse_mag * n

    v1_after = ball1.velocity + impulse_on_ball1 / m1
    v2_after = ball2.velocity + impulse_on_ball2 / m2

    return (
        v1_after,
        v2_after,
        float(impulse_mag),
        impulse_on_ball1,
        impulse_on_ball2,
    )


def wall_normal_from_name(wall_name: str) -> np.ndarray:
    """
    Return a consistent unit normal vector for a named wall.

    Convention:
    normals point inward from the cushion into the playable table area.
    """
    if wall_name == "left":
        return np.array([1.0, 0.0], dtype=float)
    if wall_name == "right":
        return np.array([-1.0, 0.0], dtype=float)
    if wall_name == "bottom":
        return np.array([0.0, 1.0], dtype=float)
    if wall_name == "top":
        return np.array([0.0, -1.0], dtype=float)

    raise ValueError(f"Unknown wall name: {wall_name}")