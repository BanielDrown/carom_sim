"""
Geometric helper functions for 2D billiards calculations.
"""

from __future__ import annotations

import numpy as np

from .constants import EPS


def norm(v: np.ndarray) -> float:
    """
    Euclidean norm of a 2D vector.
    """
    return float(np.linalg.norm(v))


def norm_squared(v: np.ndarray) -> float:
    """
    Squared Euclidean norm of a 2D vector.
    """
    return float(np.dot(v, v))


def unit(v: np.ndarray) -> np.ndarray:
    """
    Return the unit vector in the direction of v.

    Raises
    ------
    ValueError
        If the input vector magnitude is too small.
    """
    magnitude = norm(v)
    if magnitude < EPS:
        raise ValueError("Cannot normalize a near-zero vector.")
    return v / magnitude


def dot(a: np.ndarray, b: np.ndarray) -> float:
    """
    Dot product of two 2D vectors.
    """
    return float(np.dot(a, b))


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Euclidean distance between two 2D points.
    """
    return norm(a - b)


def nearly_equal(a: float, b: float, tol: float = EPS) -> bool:
    """
    Floating-point comparison helper.
    """
    return abs(a - b) <= tol