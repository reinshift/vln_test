import json
from pathlib import Path


REQUIRED_TASK_KEYS = {"name", "instruction", "spawn", "timeout_sec", "world_name"}
REQUIRED_SPAWN_KEYS = {"x", "y", "z", "yaw"}


def _require_numeric(mapping, key, task_name):
    if not isinstance(mapping.get(key), (int, float)):
        raise ValueError(f"Task '{task_name}' field '{key}' must be numeric")


def validate_tasks(tasks):
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Task file must contain a non-empty list")

    seen_names = set()
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError(f"Task at index {index} must be an object")
        missing = sorted(REQUIRED_TASK_KEYS.difference(task.keys()))
        if missing:
            raise ValueError(f"Task at index {index} missing keys: {', '.join(missing)}")

        task_name = str(task["name"]).strip()
        if not task_name:
            raise ValueError(f"Task at index {index} has an empty name")
        if task_name in seen_names:
            raise ValueError(f"Duplicate task name: {task_name}")
        seen_names.add(task_name)

        if not str(task["instruction"]).strip():
            raise ValueError(f"Task '{task_name}' instruction must be non-empty")

        world_name = str(task["world_name"]).strip()
        if not world_name:
            raise ValueError(f"Task '{task_name}' world_name must be non-empty")

        timeout_sec = task["timeout_sec"]
        if not isinstance(timeout_sec, (int, float)) or float(timeout_sec) <= 0:
            raise ValueError(f"Task '{task_name}' timeout_sec must be > 0")

        spawn = task["spawn"]
        if not isinstance(spawn, dict):
            raise ValueError(f"Task '{task_name}' spawn must be an object")
        missing_spawn = sorted(REQUIRED_SPAWN_KEYS.difference(spawn.keys()))
        if missing_spawn:
            raise ValueError(f"Task '{task_name}' spawn missing keys: {', '.join(missing_spawn)}")
        for key in REQUIRED_SPAWN_KEYS:
            _require_numeric(spawn, key, task_name)

        if "goal" in task:
            goal = task["goal"]
            if not isinstance(goal, dict):
                raise ValueError(f"Task '{task_name}' goal must be an object")
            for key in ("x", "y", "z"):
                _require_numeric(goal, key, task_name)

        if "goal_tolerance_m" in task:
            tolerance = task["goal_tolerance_m"]
            if not isinstance(tolerance, (int, float)) or tolerance <= 0:
                raise ValueError(f"Task '{task_name}' goal_tolerance_m must be > 0")

    return tasks


def load_and_validate_tasks(path):
    return validate_tasks(json.loads(Path(path).read_text(encoding="utf-8")))
