"""
State and event data structures for the three-ball carom simulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .constants import (
    BALL_MASS_KG,
    BALL_RADIUS_M,
    TABLE_LENGTH_M,
    TABLE_WIDTH_M,
)


def vec2(x: float, y: float) -> np.ndarray:
    """
    Create a 2D NumPy vector with float dtype.
    """
    return np.array([x, y], dtype=float)


@dataclass
class Ball:
    """
    Represents one billiard ball as a 2D translating rigid disk.

    Attributes
    ----------
    label : str
        Ball identifier, typically 'A', 'B', or 'C'.
    position : np.ndarray
        Ball center position [x, y] in meters.
    velocity : np.ndarray
        Ball velocity [vx, vy] in m/s.
    radius : float
        Ball radius in meters.
    mass : float
        Ball mass in kilograms.
    """

    label: str
    position: np.ndarray
    velocity: np.ndarray
    radius: float = BALL_RADIUS_M
    mass: float = BALL_MASS_KG

    def copy(self) -> "Ball":
        """
        Return a deep-enough copy for simulation state updates.
        """
        return Ball(
            label=self.label,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            radius=self.radius,
            mass=self.mass,
        )


@dataclass
class Table:
    """
    Represents a rectangular carom table.

    Coordinates:
    - x ranges from 0 to length
    - y ranges from 0 to width
    """

    length: float = TABLE_LENGTH_M
    width: float = TABLE_WIDTH_M

    @property
    def x_min(self) -> float:
        return 0.0

    @property
    def x_max(self) -> float:
        return self.length

    @property
    def y_min(self) -> float:
        return 0.0

    @property
    def y_max(self) -> float:
        return self.width


@dataclass
class CollisionEvent:
    """
    Records a single collision event in the simulation history.

    event_type:
        'ball-ball' or 'ball-wall'
    actors:
        For ball-ball: ('C', 'A')
        For ball-wall: ('C', 'left')
    time:
        Simulation time at which the event occurs.
    position:
        Approximate event position in meters.
    impulse:
        Collision impulse magnitude in N*s, if computed.
    """

    time: float
    event_type: str
    actors: tuple[str, str]
    position: np.ndarray
    impulse: Optional[float] = None
    pre_velocities: dict[str, np.ndarray] = field(default_factory=dict)
    post_velocities: dict[str, np.ndarray] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """
        Convert the event to a JSON/CSV-friendly dictionary.
        """
        def serialize_velocity_map(velocity_map: dict[str, np.ndarray]) -> dict[str, list[float]]:
            return {key: value.tolist() for key, value in velocity_map.items()}

        return {
            "time": self.time,
            "event_type": self.event_type,
            "actors": list(self.actors),
            "position": self.position.tolist(),
            "impulse": self.impulse,
            "pre_velocities": serialize_velocity_map(self.pre_velocities),
            "post_velocities": serialize_velocity_map(self.post_velocities),
        }


@dataclass
class SimulationState:
    """
    Stores the full dynamic state of the system at a single time.
    """

    time: float
    balls: dict[str, Ball]

    def copy(self) -> "SimulationState":
        return SimulationState(
            time=self.time,
            balls={label: ball.copy() for label, ball in self.balls.items()},
        )


@dataclass
class SimulationResult:
    """
    Final output of a simulation run.
    """

    initial_state: SimulationState
    final_state: SimulationState
    events: list[CollisionEvent]
    success: bool = False
    classification: Optional[str] = None
    termination_reason: str = ""


@dataclass
class TrajectorySample:
    """
    Stores the positions of all balls at one sampled time.
    Useful later for plotting and animation.
    """

    time: float
    positions: dict[str, np.ndarray]