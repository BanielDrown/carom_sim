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


def _serialize_vector_map(vector_map: dict[str, np.ndarray]) -> dict[str, list[float]]:
    """
    Convert a mapping of label -> NumPy vector into JSON-friendly lists.
    """
    return {key: value.tolist() for key, value in vector_map.items()}


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
        Return a deep copy suitable for simulation state updates.
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

    Coordinates
    -----------
    x ranges from 0 to length
    y ranges from 0 to width
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
    Record one collision event in the simulation history.

    event_type
    ----------
    'ball-ball' or 'ball-wall'

    actors
    ------
    For ball-ball: ('A', 'C')
    For ball-wall: ('C', 'left')

    time
    ----
    Simulation time at which the event occurs.

    position
    --------
    Representative collision position in meters.

    impulse
    -------
    Scalar impulse magnitude in N*s.

    impulse_vectors
    ---------------
    Per-body impulse vectors applied during the collision.
    For a ball-ball collision this contains two equal-and-opposite vectors.
    For a ball-wall collision this contains one vector for the ball.
    """

    time: float
    event_type: str
    actors: tuple[str, str]
    position: np.ndarray
    impulse: Optional[float] = None
    pre_velocities: dict[str, np.ndarray] = field(default_factory=dict)
    post_velocities: dict[str, np.ndarray] = field(default_factory=dict)
    impulse_vectors: dict[str, np.ndarray] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """
        Convert the event to a JSON/CSV-friendly dictionary.
        """
        return {
            "time": self.time,
            "event_type": self.event_type,
            "actors": list(self.actors),
            "position": self.position.tolist(),
            "impulse": self.impulse,
            "pre_velocities": _serialize_vector_map(self.pre_velocities),
            "post_velocities": _serialize_vector_map(self.post_velocities),
            "impulse_vectors": _serialize_vector_map(self.impulse_vectors),
        }


@dataclass
class SimulationState:
    """
    Store the full dynamic state of the system at one time.
    """

    time: float
    balls: dict[str, Ball]

    def copy(self) -> "SimulationState":
        return SimulationState(
            time=self.time,
            balls={label: ball.copy() for label, ball in self.balls.items()},
        )


@dataclass
class AssignmentStatus:
    """
    Track assignment-specific progress conditions.

    cue_contacts
    ------------
    Distinct object balls contacted by cue ball C.

    wall_hits
    ---------
    Number of wall collisions recorded for each ball.
    """

    cue_contacts: set[str] = field(default_factory=set)
    wall_hits: dict[str, int] = field(
        default_factory=lambda: {"A": 0, "B": 0, "C": 0}
    )

    def copy(self) -> "AssignmentStatus":
        return AssignmentStatus(
            cue_contacts=set(self.cue_contacts),
            wall_hits=dict(self.wall_hits),
        )

    @property
    def cue_hit_a(self) -> bool:
        return "A" in self.cue_contacts

    @property
    def cue_hit_b(self) -> bool:
        return "B" in self.cue_contacts

    @property
    def cue_hit_both(self) -> bool:
        return self.cue_hit_a and self.cue_hit_b

    @property
    def all_balls_hit_wall(self) -> bool:
        return all(count >= 1 for count in self.wall_hits.values())

    @property
    def success(self) -> bool:
        return self.cue_hit_both and self.all_balls_hit_wall

    def to_dict(self) -> dict:
        return {
            "cue_contacts": sorted(self.cue_contacts),
            "wall_hits": dict(self.wall_hits),
            "cue_hit_a": self.cue_hit_a,
            "cue_hit_b": self.cue_hit_b,
            "cue_hit_both": self.cue_hit_both,
            "all_balls_hit_wall": self.all_balls_hit_wall,
            "success": self.success,
        }


@dataclass
class SimulationResult:
    """
    Final output of a simulation run.

    Attributes
    ----------
    success_time
        First physical time at which the assignment success condition is satisfied.
        None if success was never achieved.

    display_end_time
        Physical time through which plots/animations should display the run.
        Typically this is equal to final_state.time and, for successful runs,
        includes the extra post-success extension.
    """

    initial_state: SimulationState
    final_state: SimulationState
    events: list[CollisionEvent]
    success: bool = False
    classification: Optional[str] = None
    termination_reason: str = ""
    assignment_status: AssignmentStatus = field(default_factory=AssignmentStatus)
    success_time: Optional[float] = None
    display_end_time: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "classification": self.classification,
            "termination_reason": self.termination_reason,
            "assignment_status": self.assignment_status.to_dict(),
            "initial_state_time": self.initial_state.time,
            "final_state_time": self.final_state.time,
            "success_time": self.success_time,
            "display_end_time": self.display_end_time,
            "events": [event.to_dict() for event in self.events],
        }


@dataclass
class TrajectorySample:
    """
    Store ball positions at one sampled time for plotting and animation.
    """

    time: float
    positions: dict[str, np.ndarray]

    def to_dict(self) -> dict:
        return {
            "time": self.time,
            "positions": _serialize_vector_map(self.positions),
        }