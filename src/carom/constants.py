"""
Project-wide physical constants and numerical tolerances for the
three-ball carom simulator.
"""

from __future__ import annotations

# ============================================================
# TABLE GEOMETRY
# Standard carom billiards table: 152 cm x 305 cm
# Internally, SI units are used throughout.
# ============================================================

TABLE_LENGTH_M: float = 3.05   # x-direction
TABLE_WIDTH_M: float = 1.52    # y-direction

# Convenient wall names
LEFT_WALL: str = "left"
RIGHT_WALL: str = "right"
BOTTOM_WALL: str = "bottom"
TOP_WALL: str = "top"

WALL_NAMES: tuple[str, str, str, str] = (
    LEFT_WALL,
    RIGHT_WALL,
    BOTTOM_WALL,
    TOP_WALL,
)

# ============================================================
# BALL PROPERTIES
# Typical carom ball diameter ~ 61.5 mm, mass ~ 210 g
# ============================================================

BALL_DIAMETER_M: float = 0.0615
BALL_RADIUS_M: float = BALL_DIAMETER_M / 2.0
BALL_MASS_KG: float = 0.210

BALL_LABELS: tuple[str, str, str] = ("A", "B", "C")

# ============================================================
# COLLISION PROPERTIES
# Start with perfectly elastic collisions unless later adjusted.
# ============================================================

BALL_BALL_RESTITUTION: float = 1.0
BALL_WALL_RESTITUTION: float = 1.0

# ============================================================
# NUMERICAL CONTROL
# ============================================================

EPS: float = 1e-9
TIME_EPS: float = 1e-10
POSITION_EPS: float = 1e-9
VELOCITY_EPS: float = 1e-10

# To prevent runaway simulations
DEFAULT_MAX_EVENTS: int = 100
DEFAULT_MAX_SIM_TIME_S: float = 30.0

# ============================================================
# SEARCH / INITIALIZATION DEFAULTS
# These can be overridden later by YAML config files.
# ============================================================

DEFAULT_CUE_SPEED_MPS: float = 1.5
MIN_CUE_SPEED_MPS: float = 0.05
MAX_CUE_SPEED_MPS: float = 5.0