import math
from typing import Iterable

from vln_core.config.models import SafetySummary


def summarize_scan(
    ranges: Iterable[float],
    angle_min: float,
    angle_increment: float,
    corridor_half_width_m: float = 0.5,
    stop_distance_m: float = 1.0,
) -> SafetySummary:
    min_clearance = float("inf")
    blocked = False

    for index, distance in enumerate(ranges):
        if distance is None or distance <= 0.0 or math.isinf(distance):
            continue
        angle = angle_min + index * angle_increment
        x = distance * math.cos(angle)
        y = distance * math.sin(angle)
        if x < 0.0 or abs(y) > corridor_half_width_m:
            continue
        min_clearance = min(min_clearance, distance)
        if distance <= stop_distance_m:
            blocked = True

    if min_clearance == float("inf"):
        return SafetySummary(blocked=False, clearance_m=999.0, reason="no_front_returns")
    if blocked:
        return SafetySummary(blocked=True, clearance_m=min_clearance, reason="front_corridor_blocked")
    return SafetySummary(blocked=False, clearance_m=min_clearance, reason="clear")

