from __future__ import annotations

import math

from carom.animation import animate_trajectory
from carom.constants import C_START_X_M, C_START_Y_M
from carom.io_utils import (
    export_case_bundle,
    format_position_vector,
    format_velocity_vector,
)
from carom.plotting import plot_trajectories
from carom.search import SearchCandidate, angle_deg, search_shots_with_random_layouts
from carom.simulator import simulate
from carom.state import Ball, SimulationState, Table, vec2


POST_SUCCESS_FRACTION = 0.10
PLOT_RELEVANT_ONLY = True
ANIMATION_DURATION_S = 3.0
ANIMATION_FPS = 20


def print_events(result) -> None:
    """
    Print a compact event log for one simulation result.
    """
    print(
        f"classification={result.classification}, "
        f"success={result.success}, "
        f"termination={result.termination_reason}, "
        f"success_time={result.success_time}, "
        f"display_end_time={result.display_end_time}"
    )
    print(
        f"cue_contacts={sorted(result.assignment_status.cue_contacts)}, "
        f"wall_hits={result.assignment_status.wall_hits}"
    )

    for i, event in enumerate(result.events, start=1):
        print(
            f"{i}: t={event.time:.6f}, "
            f"type={event.event_type}, "
            f"actors={event.actors}, "
            f"pos={event.position}"
        )


def make_base_state() -> SimulationState:
    """
    Create the base simulation state.
    Cue ball C starts from the modeled fixed starting position.
    """
    return SimulationState(
        time=0.0,
        balls={
            "A": Ball("A", vec2(2.0, 0.8), vec2(0.0, 0.0)),
            "B": Ball("B", vec2(2.5, 0.3), vec2(0.0, 0.0)),
            "C": Ball("C", vec2(C_START_X_M, C_START_Y_M), vec2(0.0, 0.0)),
        },
    )


def make_debug_state(
    a_pos: tuple[float, float],
    b_pos: tuple[float, float],
    cue_speed_mps: float,
    cue_angle_rad: float,
) -> SimulationState:
    """
    Build one explicit manual test case for fast debugging.
    """
    vx = cue_speed_mps * math.cos(cue_angle_rad)
    vy = cue_speed_mps * math.sin(cue_angle_rad)

    return SimulationState(
        time=0.0,
        balls={
            "A": Ball("A", vec2(*a_pos), vec2(0.0, 0.0)),
            "B": Ball("B", vec2(*b_pos), vec2(0.0, 0.0)),
            "C": Ball("C", vec2(C_START_X_M, C_START_Y_M), vec2(vx, vy)),
        },
    )


def print_case_summary_from_result(case_name: str, result) -> None:
    """
    Print a concise human-readable summary for a finished simulation result.
    """
    print(f"{case_name}:")
    print(
        f"A: {format_position_vector(result.initial_state.balls['A'].position)} | "
        f"B: {format_position_vector(result.initial_state.balls['B'].position)} | "
        f"C: {format_position_vector(result.initial_state.balls['C'].position)}"
    )
    print(
        f"Initial cue velocity: "
        f"{format_velocity_vector(result.initial_state.balls['C'].velocity)}"
    )
    print(
        f"Final cue velocity:   "
        f"{format_velocity_vector(result.final_state.balls['C'].velocity)}"
    )

    if result.success_time is not None:
        print(f"Success time:        {result.success_time:.4f} s")
    else:
        print("Success time:        None")

    if result.display_end_time is not None:
        print(f"Display end time:    {result.display_end_time:.4f} s")
    else:
        print(f"Display end time:    {result.final_state.time:.4f} s")


def print_case_summary(case_name: str, best: SearchCandidate) -> None:
    """
    Print a concise human-readable summary of the selected search result.
    """
    result = best.result
    degrees = angle_deg(best.angle_rad)

    print(
        f"Best {case_name}: "
        f"layout={best.layout_id}, "
        f"speed={best.speed_mps:.3f} m/s, "
        f"angle={best.angle_rad:.3f} rad ({degrees:.1f} deg)"
    )
    print_case_summary_from_result(case_name, result)


def export_case(case_name: str, result, trajectory, table: Table) -> None:
    """
    Export plot, tables, animation, and console event log for one case.
    """
    plot_trajectories(
        result=result,
        trajectory=trajectory,
        table=table,
        save_path=f"outputs/plots/{case_name}.png",
        relevant_only=PLOT_RELEVANT_ONLY,
        max_events_to_plot=None,
        show_collisions=True,
        show_velocity_vectors=False,
        show_impulse_vectors=False,
        debug=False,
        post_end_fraction=POST_SUCCESS_FRACTION,
    )

    export_case_bundle(
        result=result,
        stem=f"outputs/tables/{case_name}",
        precision=4,
    )

    try:
        animate_trajectory(
            result=result,
            trajectory=trajectory,
            table=table,
            save_path=f"outputs/animations/{case_name}.gif",
            relevant_only=PLOT_RELEVANT_ONLY,
            max_events_to_include=None,
            fps=ANIMATION_FPS,
            duration_s=ANIMATION_DURATION_S,
            show_momentum_arrows=True,
            show_impulse_pairs=True,
            post_end_fraction=POST_SUCCESS_FRACTION,
        )
    except Exception as exc:
        print(f"Animation export failed for {case_name}: {exc}")

    print_events(result)


def run_search_mode() -> None:
    """
    Search for direct, one-cushion, and two-cushion cases.
    """
    table = Table()
    base_state = make_base_state()

    for target in ("direct", "one_cushion", "two_cushion"):
        print(f"\nSearching for {target} shots...")

        matches = search_shots_with_random_layouts(
            base_state=base_state,
            desired_classification=target,
            table=table,
            layout_trials=30,
            max_results=3,
            max_events=100,
            max_time=30.0,
            max_coarse_refine_per_layout=2,
        )

        print(f"Found {len(matches)} {target} shot(s)")

        if not matches:
            continue

        best = matches[0]
        print_case_summary(target, best)
        export_case(target, best.result, best.trajectory, table)


def run_debug_mode() -> None:
    """
    Run one manually specified case without search for fast debugging.

    Edit the values below directly when debugging.
    """
    table = Table()

    debug_state = make_debug_state(
        a_pos=(2.4363, 0.8050),
        b_pos=(2.6586, 0.6863),
        cue_speed_mps=7.1,
        cue_angle_rad=-0.008,
    )

    result, trajectory = simulate(
        debug_state,
        table=table,
        max_events=100,
        max_time=30.0,
        post_success_fraction=POST_SUCCESS_FRACTION,
    )

    print_case_summary_from_result("debug_case", result)
    export_case("debug_case", result, trajectory, table)


def main() -> None:
    """
    Main driver.

    Set mode to:
    - "search" for automatic case finding
    - "debug" for one manual fast simulation
    """
    mode = "search"

    if mode == "debug":
        run_debug_mode()
    else:
        run_search_mode()


if __name__ == "__main__":
    main()