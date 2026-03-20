from pathlib import Path

from carom.io_utils import build_motion_intervals, export_case_bundle
from carom.simulator import simulate
from carom.state import Ball, SimulationState, Table, vec2


def test_build_motion_intervals_uses_piecewise_post_event_velocities():
    initial_state = SimulationState(
        time=0.0,
        balls={
            "B": Ball(label="B", position=vec2(1.3, 0.5), velocity=vec2(0.0, 0.0)),
            "C": Ball(label="C", position=vec2(1.0, 0.5), velocity=vec2(1.0, 0.0)),
        },
    )

    result, trajectory = simulate(
        initial_state,
        table=Table(),
        max_events=2,
        max_time=10.0,
    )

    intervals = build_motion_intervals(result, trajectory)

    assert len(intervals) == len(trajectory) - 1
    assert intervals[0].velocities["C"][0] == 1.0
    assert intervals[0].velocities["B"][0] == 0.0

    if len(intervals) > 1:
        assert intervals[1].velocities["B"][0] > 0.0
        assert intervals[1].displacement_end["B"] >= intervals[1].displacement_start["B"]


def test_export_case_bundle_writes_reorganized_csv_files(tmp_path: Path):
    initial_state = SimulationState(
        time=0.0,
        balls={
            "C": Ball(label="C", position=vec2(1.0, 0.5), velocity=vec2(1.0, 0.0)),
        },
    )

    result, trajectory = simulate(
        initial_state,
        table=Table(),
        max_events=1,
        max_time=10.0,
    )

    export_case_bundle(result=result, trajectory=trajectory, output_dir=tmp_path, precision=4)

    expected_files = {
        "01_initial_conditions.csv",
        "02_motion_intervals.csv",
        "03_collision_events.csv",
        "04_collision_forces.csv",
        "05_state_summary.csv",
        "06_assignment_summary.csv",
    }
    assert expected_files.issubset({path.name for path in tmp_path.iterdir()})

    motion_csv = (tmp_path / "02_motion_intervals.csv").read_text()
    assert "position_vector_equation_ij" in motion_csv
    assert "velocity_vector_equation_ij" in motion_csv
    assert "speed_of_displacement_equation" in motion_csv
    assert "r_i_of_t_equation" not in motion_csv
    assert "velocity_i_mps" not in motion_csv

    initial_csv = (tmp_path / "01_initial_conditions.csv").read_text()
    assert "initial_position_vector_ij" in initial_csv
    assert "initial_position_i_m" not in initial_csv
    assert "initial_velocity_j_mps" not in initial_csv

    events_csv = (tmp_path / "03_collision_events.csv").read_text()
    assert "impulse_vectors_ij_Ns" in events_csv
    assert "collision_normal_vector_nt" in events_csv
    assert "collision_tangent_vector_nt" in events_csv
    assert "impulse_vectors_xy_Ns" not in events_csv
    assert "collision_normal_i" not in events_csv
    assert "collision_position_i_m" not in events_csv

    forces_csv = (tmp_path / "04_collision_forces.csv").read_text()
    assert "average_force_vector_ij_N" in forces_csv
    assert "impulse_vector_xy_Ns" not in forces_csv
    assert "body_label" in forces_csv
    assert "impulse_i_Ns" not in forces_csv


def test_collision_event_export_includes_collision_frame_vectors(tmp_path: Path):
    initial_state = SimulationState(
        time=0.0,
        balls={
            "A": Ball(label="A", position=vec2(1.4, 0.5), velocity=vec2(0.0, 0.0)),
            "C": Ball(label="C", position=vec2(1.0, 0.5), velocity=vec2(1.0, 0.0)),
        },
    )

    result, trajectory = simulate(
        initial_state,
        table=Table(),
        max_events=1,
        max_time=10.0,
    )
    export_case_bundle(result=result, trajectory=trajectory, output_dir=tmp_path, precision=4)

    events_csv = (tmp_path / "03_collision_events.csv").read_text()
    assert "n_hat = -1.0 n" in events_csv
    assert "t_hat = 0.0 n - 1.0 t" in events_csv
    assert "A: 0.0 i" in events_csv or "C: 0.0 i" in events_csv
