from carom.classifier import classify_shot
from carom.state import CollisionEvent
import numpy as np


def make_event(event_type, actors):
    return CollisionEvent(
        time=0.0,
        event_type=event_type,
        actors=actors,
        position=np.array([0.0, 0.0]),
    )


def test_direct():
    events = [
        make_event("ball-ball", ("C", "A")),
        make_event("ball-ball", ("C", "B")),
    ]
    assert classify_shot(events) == "direct"


def test_one_cushion():
    events = [
        make_event("ball-ball", ("C", "A")),
        make_event("ball-wall", ("C", "left")),
        make_event("ball-ball", ("C", "B")),
    ]
    assert classify_shot(events) == "one_cushion"


def test_two_cushion():
    events = [
        make_event("ball-ball", ("C", "A")),
        make_event("ball-wall", ("C", "left")),
        make_event("ball-wall", ("C", "top")),
        make_event("ball-ball", ("C", "B")),
    ]
    assert classify_shot(events) == "two_cushion"


def test_invalid_no_second_ball():
    events = [
        make_event("ball-ball", ("C", "A")),
        make_event("ball-wall", ("C", "left")),
    ]
    assert classify_shot(events) is None


def test_ignore_other_ball_collisions():
    events = [
        make_event("ball-ball", ("A", "B")),  # should be ignored
        make_event("ball-ball", ("C", "A")),
        make_event("ball-ball", ("C", "B")),
    ]
    assert classify_shot(events) == "direct"