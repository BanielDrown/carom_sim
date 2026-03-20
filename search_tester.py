from __future__ import annotations

from collections import Counter

from carom.constants import C_START_X_M, C_START_Y_M
from carom.search import angle_deg, search_shots_with_random_layouts
from carom.state import Ball, SimulationState, Table, vec2


def make_base_state() -> SimulationState:
    return SimulationState(
        time=0.0,
        balls={
            "A": Ball("A", vec2(2.0, 0.8), vec2(0.0, 0.0)),
            "B": Ball("B", vec2(2.5, 0.3), vec2(0.0, 0.0)),
            "C": Ball("C", vec2(C_START_X_M, C_START_Y_M), vec2(0.0, 0.0)),
        },
    )


def main() -> None:
    table = Table()
    base_state = make_base_state()

    layout_trials = 30
    max_results = 5
    summary = Counter()

    for target in ("direct", "one_cushion", "two_cushion"):
        print(f"\nSearching for {target} shots...")

        matches = search_shots_with_random_layouts(
            base_state=base_state,
            desired_classification=target,
            table=table,
            layout_trials=layout_trials,
            max_results=max_results,
            max_events=100,
            max_time=30.0,
            max_coarse_refine_per_layout=2,
            show_progress=True,
        )

        summary[target] = len(matches)

        print(f"\n{target}: {len(matches)} match(es)")
        for i, match in enumerate(matches, start=1):
            print(
                f"  {i}. "
                f"layout={match.layout_id}, "
                f"speed={match.speed_mps:.3f} m/s, "
                f"angle={match.angle_rad:.3f} rad ({angle_deg(match.angle_rad):.1f} deg), "
                f"events={len(match.result.events)}, "
                f"termination={match.result.termination_reason}"
            )

    print("\nSummary")
    for key in ("direct", "one_cushion", "two_cushion"):
        print(f"  {key}: {summary[key]}")


if __name__ == "__main__":
    main()