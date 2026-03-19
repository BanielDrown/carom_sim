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

    Parameters
    ----------
    velocity : np.ndarray
        Incoming velocity vector.
    wall_normal : np.ndarray
        Unit outward/inward normal direction of the wall.
        Only the direction matters; it will be normalized internally.
    restitution : float
        Coefficient of restitution for ball-wall collision.

    Returns
    -------
    np.ndarray
        Reflected velocity vector.

    Notes
    -----
    The tangential component is unchanged.
    The normal component is reversed and scaled by restitution.
    """
    n = unit(wall_normal)
    v_n = dot(velocity, n) * n
    v_t = velocity - v_n
    return v_t - restitution * v_n


def ball_ball_collision_response(
    ball1: Ball,
    ball2: Ball,
    restitution: float = BALL_BALL_RESTITUTION,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute the post-collision velocities for a 2D collision
    between two balls.

    Parameters
    ----------
    ball1, ball2 : Ball
        Balls at the instant of impact.
    restitution : float
        Ball-ball coefficient of restitution.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        (v1_after, v2_after, impulse_magnitude)

    Notes
    -----
    This function assumes the balls are in contact at the moment of collision.
    Only the normal component along the line of centers is changed.
    """
    r12 = ball2.position - ball1.position
    if norm_squared(r12) < EPS:
        raise ValueError("Ball centers are coincident or too close to define collision normal.")

    n = unit(r12)
    v_rel = ball1.velocity - ball2.velocity
    v_rel_n = dot(v_rel, n)

    # If relative normal velocity is not directed into collision, no impulse.
    if v_rel_n <= 0:
        return ball1.velocity.copy(), ball2.velocity.copy(), 0.0

    m1 = ball1.mass
    m2 = ball2.mass

    impulse_mag = (1.0 + restitution) * v_rel_n / ((1.0 / m1) + (1.0 / m2))

    v1_after = ball1.velocity - (impulse_mag / m1) * n
    v2_after = ball2.velocity + (impulse_mag / m2) * n

    return v1_after, v2_after, float(impulse_mag)


def wall_normal_from_name(wall_name: str) -> np.ndarray:
    """
    Return a consistent unit normal vector for a named wall.
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