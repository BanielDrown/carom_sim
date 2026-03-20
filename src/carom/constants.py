"""
Project-wide physical constants and numerical tolerances for the
three-ball carom simulator.
"""

from __future__ import annotations

# ============================================================
# TABLE GEOMETRY
# Standard carom billiards table: 1.52 m x 3.05 m
# Internally, SI units are used throughout.
# x-direction: table length
# y-direction: table width
# ============================================================

TABLE_LENGTH_M: float = 3.05
TABLE_WIDTH_M: float = 1.52

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
# DEFAULT STARTING POSITIONS
# ------------------------------------------------------------
# Cue ball C is placed at a fixed modeled starting position.
# This is an assignment-oriented assumption intended to provide
# a repeatable legal-style starting location for the cue ball.
# A and B may be randomized subject to non-overlap constraints.
# ============================================================

C_START_X_M: float = TABLE_LENGTH_M * 0.25
C_START_Y_M: float = TABLE_WIDTH_M * 0.50

# Optional nominal object-ball seeds for deterministic tests
A_DEFAULT_X_M: float = TABLE_LENGTH_M * 0.65
A_DEFAULT_Y_M: float = TABLE_WIDTH_M * 0.70

B_DEFAULT_X_M: float = TABLE_LENGTH_M * 0.82
B_DEFAULT_Y_M: float = TABLE_WIDTH_M * 0.30

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

# ============================================================
# SIMULATION CONTROL
# ============================================================

DEFAULT_MAX_EVENTS: int = 100
DEFAULT_MAX_SIM_TIME_S: float = 30.0

# After cue ball C has contacted both object balls, allow a limited
# number of additional events so the remaining balls may still reach
# a wall before declaring the run finished or ineffective.
POST_CONTACT_EVENT_BUFFER: int = 12

# Optional contact duration for average-force reporting:
# F_avg = J / CONTACT_DURATION_S
# Keep as a modeling assumption only; impulse is the primary quantity.
CONTACT_DURATION_S: float = 1.0e-3

# ============================================================
# SEARCH / INITIALIZATION DEFAULTS
# These can be overridden later by YAML config files.
# ============================================================

DEFAULT_CUE_SPEED_MPS: float = 1.5
MIN_CUE_SPEED_MPS: float = 0.05
MAX_CUE_SPEED_MPS: float = 1.0

DEFAULT_SPEED_SAMPLES: int = 14
DEFAULT_ANGLE_SAMPLES: int = 181

# Randomized object-ball placement controls
SEARCH_LAYOUT_TRIALS: int = 50
OBJECT_BALL_MARGIN_M: float = 0.20
OBJECT_BALL_MIN_CLEARANCE_M: float = 0.12