from carom.search import wrap_angle
from carom.validation import advance_assignment_status, first_success_event_index
from carom.state import AssignmentStatus, CollisionEvent, vec2


def test_wrap_angle_maps_negative_pi_to_positive_pi():
    assert wrap_angle(-3.141592653589793) == 3.141592653589793


def test_advance_assignment_status_tracks_success_rules():
    status = AssignmentStatus()

    events = [
        CollisionEvent(0.1, "ball-ball", ("A", "C"), vec2(0.0, 0.0)),
        CollisionEvent(0.2, "ball-wall", ("A", "left"), vec2(0.0, 0.0)),
        CollisionEvent(0.3, "ball-wall", ("C", "top"), vec2(0.0, 0.0)),
        CollisionEvent(0.4, "ball-ball", ("B", "C"), vec2(0.0, 0.0)),
        CollisionEvent(0.5, "ball-wall", ("B", "right"), vec2(0.0, 0.0)),
    ]

    for event in events:
        advance_assignment_status(status, event)

    assert status.cue_hit_both is True
    assert status.all_balls_hit_wall is True
    assert status.success is True


def test_first_success_event_index_returns_first_matching_event():
    events = [
        CollisionEvent(0.1, "ball-ball", ("A", "C"), vec2(0.0, 0.0)),
        CollisionEvent(0.2, "ball-wall", ("A", "left"), vec2(0.0, 0.0)),
        CollisionEvent(0.3, "ball-wall", ("C", "top"), vec2(0.0, 0.0)),
        CollisionEvent(0.4, "ball-ball", ("B", "C"), vec2(0.0, 0.0)),
        CollisionEvent(0.5, "ball-wall", ("B", "right"), vec2(0.0, 0.0)),
        CollisionEvent(0.6, "ball-wall", ("C", "bottom"), vec2(0.0, 0.0)),
    ]

    assert first_success_event_index(events) == 5
