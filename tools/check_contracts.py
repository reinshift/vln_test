#!/usr/bin/env python3

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _require(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"missing required path: {path}")


def main() -> None:
    required_paths = [
        ROOT / "src" / "vln_msgs" / "msg" / "Mission.msg",
        ROOT / "src" / "vln_runtime" / "scripts" / "planner_node.py",
        ROOT / "src" / "vln_bringup" / "launch" / "simulation.launch",
        ROOT / "src" / "vln_bringup" / "launch" / "px4_contract.launch",
        ROOT / "src" / "vln_eval" / "scripts" / "run_px4_eval.py",
        ROOT / "sim" / "manifests" / "robots.lock.yaml",
        ROOT / "tests" / "goldens" / "mock_uav_tasks.json",
    ]
    for path in required_paths:
        _require(path)

    robots = json.loads((ROOT / "sim" / "manifests" / "robots.lock.yaml").read_text(encoding="utf-8"))
    if "uav" not in robots:
        raise SystemExit("robots.lock.yaml must contain the `uav` entry")

    tasks = json.loads((ROOT / "tests" / "goldens" / "mock_uav_tasks.json").read_text(encoding="utf-8"))
    if any("world_name" not in task for task in tasks):
        raise SystemExit("every golden task must define `world_name` for the PX4 runner")

    print("[contracts] static contract checks passed")


if __name__ == "__main__":
    main()
