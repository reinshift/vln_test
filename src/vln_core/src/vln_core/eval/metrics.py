from typing import Iterable, Optional

from vln_core.config.models import Pose2D, Waypoint


def path_length(waypoints: Iterable[Waypoint], start_pose: Optional[Pose2D] = None) -> float:
    total = 0.0
    prev = start_pose
    for waypoint in waypoints:
        if prev is not None:
            dx = waypoint.x - prev.x
            dy = waypoint.y - prev.y
            total += (dx * dx + dy * dy) ** 0.5
        prev = waypoint
    return total
