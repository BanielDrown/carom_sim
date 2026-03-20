"""
Shot classification helpers for carom billiards simulation.

Notes
-----
These functions provide descriptive metadata only.
They do not determine assignment success.
Assignment success is handled in validation.py via:
- cue ball C contacting both A and B
- each ball A, B, and C hitting at least one wall
"""

from __future__ import annotations

from typing import Optional

from carom.state import CollisionEvent


def first_two_object_ball_contacts(events: list[CollisionEvent]) -> list[str]:
    """
    Return the ordered distinct object-ball contacts made by cue ball C.

    Examples
    --------
    ["A"]
    ["A", "B"]
    """
    contacts: list[str] = []

    for event in events:
        if event.event_type != "ball-ball":
            continue

        a, b = event.actors
        if "C" not in (a, b):
            continue

        other = b if a == "C" else a
        if other not in ("A", "B"):
            continue

        if other not in contacts:
            contacts.append(other)

        if len(contacts) == 2:
            break

    return contacts


def cue_ball_wall_count_between_object_contacts(events: list[CollisionEvent]) -> int:
    """
    Count cue-ball wall contacts after its first distinct object-ball contact
    and before its second distinct object-ball contact.
    """
    distinct_contacts: list[str] = []
    wall_count = 0

    for event in events:
        if event.event_type == "ball-ball":
            a, b = event.actors
            if "C" not in (a, b):
                continue

            other = b if a == "C" else a
            if other not in ("A", "B"):
                continue

            if other not in distinct_contacts:
                distinct_contacts.append(other)

            if len(distinct_contacts) >= 2:
                break

        elif event.event_type == "ball-wall":
            ball, _wall = event.actors
            if ball == "C" and len(distinct_contacts) == 1:
                wall_count += 1

    return wall_count


def classify_shot(events: list[CollisionEvent]) -> Optional[str]:
    """
    Classify the cue-ball sequence as descriptive metadata.

    Returns
    -------
    str or None
        "direct", "one_cushion", "two_cushion", or None

    Meaning
    -------
    - direct: cue ball C contacts both object balls with no cue-ball wall
      contact between those two distinct contacts
    - one_cushion: exactly one cue-ball wall contact between the two
      distinct object-ball contacts
    - two_cushion: exactly two cue-ball wall contacts between the two
      distinct object-ball contacts
    """
    contacts = first_two_object_ball_contacts(events)
    if len(contacts) < 2:
        return None

    wall_count = cue_ball_wall_count_between_object_contacts(events)

    if wall_count == 0:
        return "direct"
    if wall_count == 1:
        return "one_cushion"
    if wall_count == 2:
        return "two_cushion"

    return None


def cue_ball_contacted_both(events: list[CollisionEvent]) -> bool:
    """
    Return True if cue ball C contacted both A and B at least once.
    """
    contacts = first_two_object_ball_contacts(events)
    return "A" in contacts and "B" in contacts


def wall_hits_by_ball(events: list[CollisionEvent]) -> dict[str, int]:
    """
    Return the number of wall hits for each ball.
    """
    counts = {"A": 0, "B": 0, "C": 0}

    for event in events:
        if event.event_type != "ball-wall":
            continue

        ball_label, _wall = event.actors
        if ball_label in counts:
            counts[ball_label] += 1

    return counts


def all_balls_hit_wall(events: list[CollisionEvent]) -> bool:
    """
    Return True if each ball A, B, and C hit at least one wall.
    """
    counts = wall_hits_by_ball(events)
    return all(counts[label] >= 1 for label in ("A", "B", "C"))