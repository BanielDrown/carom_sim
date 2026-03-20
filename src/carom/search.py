"""
Search utilities for finding valid carom shots.

Search objective
----------------
- Cue ball C starts from a fixed modeled starting position.
- Object balls A and B are randomized on the table subject to
  non-overlap and clearance constraints.
- A successful result satisfies the assignment success condition:
    * cue ball C contacts both A and B
    * balls A, B, and C each hit at least one wall
- Shot classification ("direct", "one_cushion", "two_cushion") is
  retained as descriptive metadata only.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin
from typing import Optional
import random
import time

import numpy as np

from carom.constants import (
    DEFAULT_ANGLE_SAMPLES,
    DEFAULT_SPEED_SAMPLES,
    MAX_CUE_SPEED_MPS,
    MIN_CUE_SPEED_MPS,
    OBJECT_BALL_MARGIN_M,
    OBJECT_BALL_MIN_CLEARANCE_M,
    SEARCH_LAYOUT_TRIALS,
)
from carom.simulator import simulate
from carom.state import SimulationResult, SimulationState, Table, TrajectorySample, vec2


@dataclass
class SearchCandidate:
    angle_rad: float
    speed_mps: float
    result: SimulationResult
    trajectory: list[TrajectorySample]
    layout_id: int = 0


def angle_deg(angle_rad: float) -> float:
    """
    Convert radians to degrees.
    """
    return float(np.degrees(angle_rad))


def candidate_score(candidate: SearchCandidate) -> tuple[int, int, float]:
    """
    Lower score is better.

    Priority:
    1. fewer total events
    2. fewer additional wall hits beyond the minimum required
    3. higher cue speed
    """
    wall_hits = candidate.result.assignment_status.wall_hits
    excess_wall_hits = sum(max(0, wall_hits[label] - 1) for label in ("A", "B", "C"))

    return (
        len(candidate.result.events),
        excess_wall_hits,
        -candidate.speed_mps,
    )


def candidate_signature(
    candidate: SearchCandidate,
    angle_tol_deg: float = 2.0,
) -> tuple[int, str | None, int]:
    """
    Signature for deduplicating near-identical shots.
    """
    angle_bucket = int(round(angle_deg(candidate.angle_rad) / angle_tol_deg))
    return (
        candidate.layout_id,
        candidate.result.classification,
        angle_bucket,
    )


def deduplicate_candidates(
    candidates: list[SearchCandidate],
    angle_tol_deg: float = 2.0,
) -> list[SearchCandidate]:
    """
    Keep only the best representative of each near-duplicate candidate group.
    """
    best_by_signature: dict[tuple[int, str | None, int], SearchCandidate] = {}

    for candidate in sorted(candidates, key=candidate_score):
        signature = candidate_signature(candidate, angle_tol_deg=angle_tol_deg)
        if signature not in best_by_signature:
            best_by_signature[signature] = candidate

    deduped = list(best_by_signature.values())
    deduped.sort(key=candidate_score)
    return deduped


def wrap_angle(angle: float) -> float:
    """
    Wrap angle to (-pi, pi].
    """
    wrapped = (angle + np.pi) % (2.0 * np.pi) - np.pi
    if wrapped <= -np.pi:
        return float(np.pi)
    return float(wrapped)


def angle_to_target(from_pos: np.ndarray, to_pos: np.ndarray) -> float:
    """
    Return the heading angle from from_pos to to_pos.
    """
    delta = to_pos - from_pos
    return float(np.arctan2(delta[1], delta[0]))


def angle_difference(a: float, b: float) -> float:
    """
    Smallest wrapped angular difference.
    """
    return abs(wrap_angle(a - b))


def make_state_with_cue_velocity(
    base_state: SimulationState,
    speed: float,
    angle_rad: float,
) -> SimulationState:
    """
    Copy a base state and assign cue-ball C velocity from polar parameters.
    """
    state = base_state.copy()
    state.balls["C"].velocity = vec2(speed * cos(angle_rad), speed * sin(angle_rad))
    return state


def layout_signature(
    a_pos: np.ndarray,
    b_pos: np.ndarray,
    grid_m: float = 0.05,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Coarse signature for deduplicating near-identical randomized layouts.
    """
    a_key = (int(round(a_pos[0] / grid_m)), int(round(a_pos[1] / grid_m)))
    b_key = (int(round(b_pos[0] / grid_m)), int(round(b_pos[1] / grid_m)))
    return tuple(sorted((a_key, b_key)))


def randomize_object_positions(
    table: Table,
    cue_position: np.ndarray,
    ball_radius: float,
    margin: float = OBJECT_BALL_MARGIN_M,
    min_clearance: float = OBJECT_BALL_MIN_CLEARANCE_M,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate random positions for A and B with separation constraints.
    """
    effective_margin = max(margin, ball_radius + 0.02)
    min_clearance_sq = min_clearance**2

    def rand_pos() -> np.ndarray:
        return vec2(
            random.uniform(effective_margin, table.length - effective_margin),
            random.uniform(effective_margin, table.width - effective_margin),
        )

    while True:
        a_pos = rand_pos()
        b_pos = rand_pos()

        # Use squared distances here to avoid repeated square roots in the
        # highest-frequency rejection-sampling loop of the search path.
        if float(np.dot(a_pos - b_pos, a_pos - b_pos)) < min_clearance_sq:
            continue
        if float(np.dot(a_pos - cue_position, a_pos - cue_position)) < min_clearance_sq:
            continue
        if float(np.dot(b_pos - cue_position, b_pos - cue_position)) < min_clearance_sq:
            continue

        return a_pos, b_pos


def biased_angle_values(
    base_state: SimulationState,
    spread: float = 0.6,
    steps_per_center: int = 17,
) -> list[float]:
    """
    Generate angle samples concentrated around likely useful directions.
    """
    c_pos = base_state.balls["C"].position
    a_pos = base_state.balls["A"].position
    b_pos = base_state.balls["B"].position
    mid_pos = 0.5 * (a_pos + b_pos)

    centers = [
        angle_to_target(c_pos, a_pos),
        angle_to_target(c_pos, b_pos),
        angle_to_target(c_pos, mid_pos),
    ]

    values: list[float] = []
    for center in centers:
        local_angles = np.linspace(center - spread, center + spread, steps_per_center)
        values.extend(wrap_angle(float(angle)) for angle in local_angles)

    return deduplicate_angles(values)


def mirror_point_across_wall(point: np.ndarray, table: Table, wall: str) -> np.ndarray:
    """
    Mirror a point across a named wall of the rectangular table.
    """
    x, y = point

    if wall == "left":
        return vec2(-x, y)
    if wall == "right":
        return vec2(2.0 * table.length - x, y)
    if wall == "bottom":
        return vec2(x, -y)
    if wall == "top":
        return vec2(x, 2.0 * table.width - y)

    raise ValueError(f"Unknown wall: {wall}")


def mirrored_target_angles(
    base_state: SimulationState,
    table: Table,
    classification: str,
) -> list[float]:
    """
    Generate analytical aiming angles using mirrored target positions.
    """
    c_pos = base_state.balls["C"].position
    a_pos = base_state.balls["A"].position
    b_pos = base_state.balls["B"].position

    walls = ["left", "right", "bottom", "top"]
    angles: list[float] = []

    if classification == "one_cushion":
        for obj_pos in (a_pos, b_pos):
            for wall in walls:
                mirrored = mirror_point_across_wall(obj_pos, table, wall)
                angles.append(angle_to_target(c_pos, mirrored))

    elif classification == "two_cushion":
        for obj_pos in (a_pos, b_pos):
            for wall_1 in walls:
                first_mirror = mirror_point_across_wall(obj_pos, table, wall_1)
                for wall_2 in walls:
                    second_mirror = mirror_point_across_wall(first_mirror, table, wall_2)
                    angles.append(angle_to_target(c_pos, second_mirror))

    return deduplicate_angles([wrap_angle(angle) for angle in angles])


def deduplicate_angles(angle_values: list[float], tol: float = 1e-6) -> list[float]:
    """
    Remove near-duplicate wrapped angles while preserving order.
    """
    unique: list[float] = []

    for angle in angle_values:
        wrapped = wrap_angle(angle)
        if not any(abs(wrap_angle(wrapped - existing)) < tol for existing in unique):
            unique.append(wrapped)

    return unique


def is_promising_angle(
    base_state: SimulationState,
    angle_rad: float,
    desired_classification: Optional[str],
    max_direct_angle_error_deg: float = 45.0,
) -> bool:
    """
    Cheap geometric prefilter to skip obviously poor launch angles.

    Applied only to direct shots. Cushion-shot searches rely heavily on mirrored
    angles, so this filter is intentionally conservative there.
    """
    if desired_classification not in (None, "direct"):
        return True

    c_pos = base_state.balls["C"].position
    a_pos = base_state.balls["A"].position
    b_pos = base_state.balls["B"].position

    a_dir = angle_to_target(c_pos, a_pos)
    b_dir = angle_to_target(c_pos, b_pos)
    limit = np.radians(max_direct_angle_error_deg)

    return (
        angle_difference(angle_rad, a_dir) <= limit
        or angle_difference(angle_rad, b_dir) <= limit
    )


def _format_elapsed(seconds: float) -> str:
    """
    Format elapsed time as mm:ss.s
    """
    minutes = int(seconds // 60)
    remainder = seconds - 60 * minutes
    return f"{minutes:02d}:{remainder:04.1f}"


def grid_search_cue_shots(
    base_state: SimulationState,
    table: Optional[Table] = None,
    speed_values: Optional[list[float]] = None,
    angle_values: Optional[list[float]] = None,
    desired_classification: Optional[str] = None,
    max_results: int = 10,
    max_events: int = 50,
    max_time: float = 20.0,
) -> list[SearchCandidate]:
    """
    Search over cue-ball speed and angle while keeping positions fixed.
    """
    if table is None:
        table = Table()

    if speed_values is None:
        speed_values = list(np.linspace(MIN_CUE_SPEED_MPS, MAX_CUE_SPEED_MPS, DEFAULT_SPEED_SAMPLES))

    if angle_values is None:
        angle_values = list(np.linspace(-np.pi, np.pi, DEFAULT_ANGLE_SAMPLES))

    matches: list[SearchCandidate] = []

    for speed in speed_values:
        for angle in angle_values:
            wrapped_angle = wrap_angle(angle)

            if not is_promising_angle(base_state, wrapped_angle, desired_classification):
                continue

            state = make_state_with_cue_velocity(base_state, speed, wrapped_angle)

            result, trajectory = simulate(
                state,
                table=table,
                max_events=max_events,
                max_time=max_time,
            )

            if not result.success:
                continue

            if desired_classification is not None and result.classification != desired_classification:
                continue

            matches.append(
                SearchCandidate(
                    angle_rad=wrapped_angle,
                    speed_mps=speed,
                    result=result,
                    trajectory=trajectory,
                )
            )

    matches = deduplicate_candidates(matches, angle_tol_deg=2.0)
    return matches[:max_results]


def refine_search_around_candidate(
    base_state: SimulationState,
    candidate: SearchCandidate,
    table: Table,
    angle_window: float = 0.2,
    speed_window: float = 0.4,
    angle_steps: int = 21,
    speed_steps: int = 7,
    desired_classification: Optional[str] = None,
    max_events: int = 50,
    max_time: float = 20.0,
) -> list[SearchCandidate]:
    """
    Perform local search around a promising candidate.
    """
    angles = np.linspace(
        candidate.angle_rad - angle_window,
        candidate.angle_rad + angle_window,
        angle_steps,
    )

    speeds = np.linspace(
        max(MIN_CUE_SPEED_MPS, candidate.speed_mps - speed_window),
        min(MAX_CUE_SPEED_MPS, candidate.speed_mps + speed_window),
        speed_steps,
    )

    matches: list[SearchCandidate] = []

    for speed in speeds:
        for angle in angles:
            wrapped_angle = wrap_angle(float(angle))
            state = make_state_with_cue_velocity(base_state, speed, wrapped_angle)

            result, trajectory = simulate(
                state,
                table=table,
                max_events=max_events,
                max_time=max_time,
            )

            if not result.success:
                continue

            if desired_classification is not None and result.classification != desired_classification:
                continue

            matches.append(
                SearchCandidate(
                    angle_rad=wrapped_angle,
                    speed_mps=float(speed),
                    result=result,
                    trajectory=trajectory,
                    layout_id=candidate.layout_id,
                )
            )

    matches = deduplicate_candidates(matches, angle_tol_deg=2.0)
    return matches


def search_shots_with_random_layouts(
    base_state: SimulationState,
    desired_classification: Optional[str] = None,
    table: Optional[Table] = None,
    layout_trials: int = SEARCH_LAYOUT_TRIALS,
    max_results: int = 3,
    max_events: int = 50,
    max_time: float = 20.0,
    max_coarse_refine_per_layout: int = 2,
    show_progress: bool = True,
) -> list[SearchCandidate]:
    """
    Search for valid shots while randomizing A and B positions.

    Parameters
    ----------
    show_progress
        If True, print a live progress line with layout count, elapsed time,
        and number of matches found so far.
    """
    if table is None:
        table = Table()

    cue_position = base_state.balls["C"].position.copy()
    ball_radius = base_state.balls["C"].radius

    all_matches: list[SearchCandidate] = []
    seen_layouts: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    t_start = time.perf_counter()
    unique_layout_count = 0

    for layout_id in range(1, layout_trials + 1):
        state = base_state.copy()

        a_pos, b_pos = randomize_object_positions(
            table=table,
            cue_position=cue_position,
            ball_radius=ball_radius,
        )

        sig = layout_signature(a_pos, b_pos)
        if sig in seen_layouts:
            if show_progress:
                elapsed = _format_elapsed(time.perf_counter() - t_start)
                print(
                    f"\r[{desired_classification or 'any'}] "
                    f"layout {layout_id}/{layout_trials} | "
                    f"unique {unique_layout_count} | "
                    f"matches {len(all_matches)} | "
                    f"elapsed {elapsed}",
                    end="",
                    flush=True,
                )
            continue

        seen_layouts.add(sig)
        unique_layout_count += 1

        state.balls["A"].position = a_pos
        state.balls["B"].position = b_pos

        coarse_angles = biased_angle_values(state, spread=0.6, steps_per_center=17)

        if desired_classification in {"one_cushion", "two_cushion"}:
            coarse_angles.extend(mirrored_target_angles(state, table, desired_classification))

        coarse_angles = deduplicate_angles(coarse_angles)

        coarse_matches = grid_search_cue_shots(
            base_state=state,
            table=table,
            speed_values=[1.5, 3.0, 4.5, 6.0, 7.5],
            angle_values=coarse_angles,
            desired_classification=desired_classification,
            max_results=max_coarse_refine_per_layout,
            max_events=max_events,
            max_time=max_time,
        )

        for coarse in coarse_matches[:max_coarse_refine_per_layout]:
            coarse.layout_id = layout_id

            refined_matches = refine_search_around_candidate(
                base_state=state,
                candidate=coarse,
                table=table,
                desired_classification=desired_classification,
                max_events=max_events,
                max_time=max_time,
            )

            for refined in refined_matches:
                refined.layout_id = layout_id
                all_matches.append(refined)

        all_matches = deduplicate_candidates(all_matches, angle_tol_deg=2.0)

        if show_progress:
            elapsed = _format_elapsed(time.perf_counter() - t_start)
            print(
                f"\r[{desired_classification or 'any'}] "
                f"layout {layout_id}/{layout_trials} | "
                f"unique {unique_layout_count} | "
                f"matches {len(all_matches)} | "
                f"elapsed {elapsed}",
                end="",
                flush=True,
            )

    if show_progress:
        elapsed = _format_elapsed(time.perf_counter() - t_start)
        print(
            f"\r[{desired_classification or 'any'}] "
            f"layout {layout_trials}/{layout_trials} | "
            f"unique {unique_layout_count} | "
            f"matches {len(all_matches)} | "
            f"elapsed {elapsed}"
        )

    all_matches = deduplicate_candidates(all_matches, angle_tol_deg=2.0)
    return all_matches[:max_results]
