#!/usr/bin/env python3

import argparse
import json
import math
import sys
from pathlib import Path

from vln_core.config.models import EpisodeOutcome, Pose2D, SemanticLandmark
from vln_core.eval.metrics import path_length
from vln_core.eval.reporter import write_report
from vln_core.mission.compiler import compile_instruction
from vln_core.planning.scoring import build_trajectory
from vln_eval.task_schema import load_and_validate_tasks


def _load_landmarks(path: Path):
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        item["label"]: SemanticLandmark(
            label=item["label"],
            x=float(item["x"]),
            y=float(item["y"]),
            z=float(item.get("z", 0.0)),
            confidence=float(item.get("confidence", 1.0)),
            observation_count=int(item.get("observation_count", 1)),
            state=str(item.get("state", "tracked")),
        )
        for item in payload
    }


def _distance(goal, waypoint) -> float:
    return math.sqrt(
        ((goal.get("x", 0.0) - waypoint.x) ** 2) +
        ((goal.get("y", 0.0) - waypoint.y) ** 2)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic golden evaluation for the UAV-first VLN stack.")
    parser.add_argument("--tasks-file", required=True)
    parser.add_argument("--landmarks-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="golden-mock")
    parser.add_argument("--min-success-rate", type=float, default=0.0)
    args = parser.parse_args()

    tasks = load_and_validate_tasks(Path(args.tasks_file))
    landmarks = _load_landmarks(Path(args.landmarks_file))
    outcomes = []

    for task in tasks:
        mission = compile_instruction(task["instruction"], mission_id=f"eval-{task['name']}")
        spawn_pose = Pose2D(
            x=float(task["spawn"]["x"]),
            y=float(task["spawn"]["y"]),
            z=float(task["spawn"]["z"]),
            yaw_rad=float(task["spawn"]["yaw"]),
        )
        trajectory = build_trajectory(
            mission,
            semantic_landmarks=landmarks.values(),
            start_pose=spawn_pose,
        )
        final_waypoint = trajectory.waypoints[-1] if trajectory.waypoints else None
        goal = task.get("goal", {})
        error_m = _distance(goal, final_waypoint) if final_waypoint is not None and goal else 0.0
        tolerance = float(task.get("goal_tolerance_m", 1.0))
        success = error_m <= tolerance
        score = round(1.0 / (1.0 + error_m), 4)
        path_length_m = round(path_length(trajectory.waypoints, start_pose=spawn_pose), 4)
        outcomes.append(
            EpisodeOutcome(
                run_id=args.run_id,
                episode_name=task["name"],
                success=success,
                score=score,
                summary_path="summary.md",
                detail=f"goal_error_m={error_m:.3f}",
                metrics={
                    "goal_error_m": round(error_m, 4),
                    "goal_tolerance_m": tolerance,
                    "path_length_m": path_length_m,
                    "waypoint_count": len(trajectory.waypoints),
                    "step_count": len(mission.steps),
                    "timeout_sec": float(task["timeout_sec"]),
                },
                mission_id=mission.mission_id,
            )
        )

    write_report(Path(args.output_dir), outcomes)
    success_rate = (sum(1 for item in outcomes if item.success) / max(1, len(outcomes))) if outcomes else 0.0
    if success_rate < args.min_success_rate:
        print(
            f"success rate {success_rate:.3f} below required minimum {args.min_success_rate:.3f}",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
