import numpy as np

from carom.classifier import should_stop_early
from carom.state import CollisionEvent


def make_event(event_type, actors):
    return CollisionEvent(
        time=0.0,
        event_type=event_type,
        actors=actors,
        position=np.array([0.0, 0.0]),
    )


def test_should_stop_when_valid_direct_found():
    events = [
        make_event("ball-ball", ("C", "A")),
        make_event("ball-ball", ("C", "B")),
    ]
    assert should_stop_early(events) is True


def test_should_stop_after_too_many_c_walls_after_first_hit():
    events = [
        make_event("ball-ball", ("C", "A")),
        make_event("ball-wall", ("C", "left")),
        make_event("ball-wall", ("C", "top")),
        make_event("ball-wall", ("C", "right")),
    ]
    assert should_stop_early(events) is True


def test_should_not_stop_too_early():
    events = [
        make_event("ball-ball", ("C", "A")),
        make_event("ball-wall", ("C", "left")),
    ]
    assert should_stop_early(events) is False