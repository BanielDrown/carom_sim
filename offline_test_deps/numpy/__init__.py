"""
Minimal NumPy compatibility shim for offline test execution.

This is intentionally tiny and only implements the subset of the NumPy API that
this repository's runtime logic and unit tests require when real NumPy is not
available.
"""

from __future__ import annotations

import math
from copy import deepcopy

pi = math.pi


class ndarray:
    def __init__(self, values):
        self._data = _normalize(values)

    def copy(self) -> "ndarray":
        return ndarray(deepcopy(self._data))

    def tolist(self):
        return deepcopy(self._data)

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __repr__(self) -> str:
        return repr(self._data)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            if self.ndim != 2:
                raise TypeError("tuple indexing is only supported for 2D arrays")
            row_sel, col_sel = key
            rows = self._data[row_sel] if isinstance(row_sel, slice) else [self._data[row_sel]]
            values = [row[col_sel] for row in rows]
            return values if isinstance(row_sel, slice) else values[0]

        value = self._data[key]
        return ndarray(value) if isinstance(value, list) else value

    @property
    def ndim(self) -> int:
        return 2 if self._data and isinstance(self._data[0], list) else 1

    def _binary_op(self, other, op):
        return ndarray(_elementwise(self._data, _unwrap(other), op))

    def __add__(self, other):
        return self._binary_op(other, lambda a, b: a + b)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._binary_op(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return ndarray(_elementwise(_unwrap(other), self._data, lambda a, b: a - b))

    def __mul__(self, other):
        return self._binary_op(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._binary_op(other, lambda a, b: a / b)

    def __neg__(self):
        return ndarray(_elementwise(self._data, -1.0, lambda a, b: a * b))


class _LinalgModule:
    @staticmethod
    def norm(value) -> float:
        flat = _flatten(_unwrap(value))
        return math.sqrt(sum(component * component for component in flat))


linalg = _LinalgModule()


def _unwrap(value):
    if isinstance(value, ndarray):
        return value._data
    return value


def _normalize(value):
    if isinstance(value, ndarray):
        return deepcopy(value._data)
    if isinstance(value, (list, tuple)):
        return [_normalize(item) if isinstance(item, (list, tuple, ndarray)) else float(item) for item in value]
    return float(value)


def _flatten(value) -> list[float]:
    if isinstance(value, list):
        items: list[float] = []
        for item in value:
            items.extend(_flatten(item))
        return items
    return [float(value)]


def _elementwise(left, right, op):
    if isinstance(left, list) and isinstance(right, list):
        return [_elementwise(a, b, op) for a, b in zip(left, right)]
    if isinstance(left, list):
        return [_elementwise(item, right, op) for item in left]
    if isinstance(right, list):
        return [_elementwise(left, item, op) for item in right]
    return op(float(left), float(right))


def array(values, dtype=float) -> ndarray:
    del dtype
    return ndarray(values)


def zeros(shape, dtype=float) -> ndarray:
    del dtype
    if isinstance(shape, int):
        return ndarray([0.0 for _ in range(shape)])
    rows, cols = shape
    return ndarray([[0.0 for _ in range(cols)] for _ in range(rows)])


def dot(a, b) -> float:
    left = _flatten(_unwrap(a))
    right = _flatten(_unwrap(b))
    return float(sum(x * y for x, y in zip(left, right)))


def allclose(a, b, atol=1e-8) -> bool:
    left = _flatten(_unwrap(a))
    right = _flatten(_unwrap(b))
    if len(left) != len(right):
        return False
    return all(abs(x - y) <= atol for x, y in zip(left, right))


def degrees(value: float) -> float:
    return math.degrees(value)


def radians(value: float) -> float:
    return math.radians(value)


def arctan2(y: float, x: float) -> float:
    return math.atan2(y, x)


def linspace(start: float, stop: float, num: int) -> list[float]:
    if num <= 1:
        return [float(stop)]
    step = (stop - start) / (num - 1)
    return [float(start + step * idx) for idx in range(num)]


def atleast_1d(value):
    if isinstance(value, (list, tuple)):
        return value
    return [value]
