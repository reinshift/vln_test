from typing import Dict, Iterable, List

from vln_core.config.models import Pose2D


def group_tasks_by_world(tasks: Iterable[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {}
    for task in tasks:
        world_name = str(task["world_name"])
        grouped.setdefault(world_name, []).append(task)
    return grouped


def goal_error_xy(goal: dict, pose: Pose2D) -> float:
    dx = float(goal.get("x", 0.0)) - float(pose.x)
    dy = float(goal.get("y", 0.0)) - float(pose.y)
    return (dx * dx + dy * dy) ** 0.5
