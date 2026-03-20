"""
Collision timing and event detection for the three-ball carom simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import sqrt
from typing import Optional

from carom.constants import EPS, TIME_EPS
from carom.geometry import dot, norm_squared
from carom.state import Ball, SimulationState, Table


@dataclass(frozen=True)
class CandidateEvent:
    """
    Represent one possible future collision event.
    """

    time_to_event: float
    event_type: str          # "ball-ball" or "ball-wall"
    actors: tuple[str, str]  # e.g. ("A", "B") or ("C", "left")


def time_to_vertical_wall_collision(ball: Ball, table: Table) -> Optional[tuple[float, str]]:
    """
    Return time until the ball hits the left or right wall, if any.
    """
    x = ball.position[0]
    vx = ball.velocity[0]
    r = ball.radius

    if abs(vx) < EPS:
        return None

    if vx > 0.0:
        t = (table.x_max - r - x) / vx
        wall = "right"
    else:
        t = (table.x_min + r - x) / vx
        wall = "left"

    if t <= TIME_EPS:
        return None

    return float(t), wall


def time_to_horizontal_wall_collision(ball: Ball, table: Table) -> Optional[tuple[float, str]]:
    """
    Return time until the ball hits the bottom or top wall, if any.
    """
    y = ball.position[1]
    vy = ball.velocity[1]
    r = ball.radius

    if abs(vy) < EPS:
        return None

    if vy > 0.0:
        t = (table.y_max - r - y) / vy
        wall = "top"
    else:
        t = (table.y_min + r - y) / vy
        wall = "bottom"

    if t <= TIME_EPS:
        return None

    return float(t), wall


def time_to_ball_ball_collision(ball1: Ball, ball2: Ball) -> Optional[float]:
    """
    Return time until two balls collide, if they will collide in the future.

    Collision condition:
        ||(r2 - r1) + (v2 - v1)t|| = r1 + r2
    """
    dr = ball2.position - ball1.position
    dv = ball2.velocity - ball1.velocity
    sigma = ball1.radius + ball2.radius

    a = norm_squared(dv)
    b = 2.0 * dot(dr, dv)
    c = norm_squared(dr) - sigma**2

    if a < EPS:
        return None

    discriminant = b**2 - 4.0 * a * c
    if discriminant < 0.0:
        return None

    sqrt_disc = sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    candidates = [t for t in (t1, t2) if t > TIME_EPS]
    if not candidates:
        return None

    t_collision = min(candidates)

    # Guard: ensure balls are still approaching at the selected collision time.
    dr_collision = dr + dv * t_collision
    if dot(dr_collision, dv) >= 0.0:
        return None

    return float(t_collision)


def _event_priority(event: CandidateEvent) -> tuple[float, int, tuple[str, str]]:
    """
    Deterministic ordering for simultaneous or near-simultaneous events.

    Priority:
    1. earliest time
    2. ball-ball before ball-wall
    3. lexical actor ordering
    """
    type_priority = 0 if event.event_type == "ball-ball" else 1
    return (event.time_to_event, type_priority, event.actors)


def find_next_event(state: SimulationState, table: Table) -> Optional[CandidateEvent]:
    """
    Search all possible ball-wall and ball-ball events and return the earliest one.
    """
    candidates: list[CandidateEvent] = []

    for label, ball in state.balls.items():
        vertical = time_to_vertical_wall_collision(ball, table)
        if vertical is not None:
            t, wall = vertical
            candidates.append(
                CandidateEvent(
                    time_to_event=t,
                    event_type="ball-wall",
                    actors=(label, wall),
                )
            )

        horizontal = time_to_horizontal_wall_collision(ball, table)
        if horizontal is not None:
            t, wall = horizontal
            candidates.append(
                CandidateEvent(
                    time_to_event=t,
                    event_type="ball-wall",
                    actors=(label, wall),
                )
            )

    for label1, label2 in combinations(state.balls.keys(), 2):
        ball1 = state.balls[label1]
        ball2 = state.balls[label2]

        t = time_to_ball_ball_collision(ball1, ball2)
        if t is not None:
            actors = tuple(sorted((label1, label2)))
            candidates.append(
                CandidateEvent(
                    time_to_event=t,
                    event_type="ball-ball",
                    actors=actors,
                )
            )

    if not candidates:
        return None

    return min(candidates, key=_event_priority)