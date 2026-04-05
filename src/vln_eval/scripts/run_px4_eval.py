#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from vln_core.config.models import EpisodeOutcome
from vln_core.eval.reporter import write_report
from vln_core.mission.compiler import compile_instruction
from vln_eval.px4_eval import group_tasks_by_world
from vln_eval.task_schema import load_and_validate_tasks
from vln_sim import require_px4_env


ROOT = Path(__file__).resolve().parents[3]


def _bool_string(value: bool) -> str:
    return "true" if value else "false"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hard-relaunch PX4 SITL evaluation for the UAV-first VLN stack.")
    parser.add_argument("--tasks-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--px4-autopilot-dir", default="")
    parser.add_argument("--headless", default="true")
    parser.add_argument("--run-id", default="px4-e2e")
    parser.add_argument("--min-success-rate", type=float, default=1.0)
    args = parser.parse_args()

    tasks_file = Path(args.tasks_file).resolve()
    output_dir = Path(args.output_dir).resolve()
    tasks = load_and_validate_tasks(tasks_file)
    px4_root = require_px4_env(args.px4_autopilot_dir)
    headless = str(args.headless).strip().lower() not in {"false", "0", "no"}
    episode_script = Path(__file__).resolve().with_name("run_px4_episode.py")

    outcomes = []
    for world_name, world_tasks in group_tasks_by_world(tasks).items():
        for task in world_tasks:
            mission = compile_instruction(task["instruction"], mission_id=f"eval-{task['name']}")
            with tempfile.TemporaryDirectory(prefix=f"px4-episode-{world_name}-") as tmp:
                output_json = Path(tmp) / f"{task['name']}.json"
                subprocess.run(
                    [
                        sys.executable,
                        str(episode_script),
                        "--tasks-file",
                        str(tasks_file),
                        "--episode-name",
                        str(task["name"]),
                        "--output-json",
                        str(output_json),
                        "--px4-autopilot-dir",
                        str(px4_root),
                        "--headless",
                        _bool_string(headless),
                    ],
                    check=True,
                    cwd=str(ROOT),
                )
                payload = json.loads(output_json.read_text(encoding="utf-8"))

            metrics = dict(payload.get("metrics", {}))
            metrics["step_count"] = float(len(mission.steps))
            metrics["waypoint_count"] = float(len(mission.steps))
            outcomes.append(
                EpisodeOutcome(
                    run_id=args.run_id,
                    episode_name=str(task["name"]),
                    success=bool(payload["success"]),
                    score=float(payload["score"]),
                    summary_path="summary.md",
                    detail=str(payload["detail"]),
                    metrics=metrics,
                    mission_id=str(payload.get("mission_id") or mission.mission_id),
                )
            )

    write_report(output_dir, outcomes)
    success_rate = sum(1 for item in outcomes if item.success) / max(1, len(outcomes))
    if success_rate < args.min_success_rate:
        raise SystemExit(
            f"success rate {success_rate:.3f} below required minimum {args.min_success_rate:.3f}"
        )


if __name__ == "__main__":
    main()
