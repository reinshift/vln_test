#!/usr/bin/env python3

import argparse
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

import rosgraph
import rospy
from nav_msgs.msg import Odometry
from vln_core.config.models import Pose2D
from vln_eval.px4_eval import goal_error_xy
from vln_eval.task_schema import load_and_validate_tasks
from vln_msgs.msg import RuntimeState
from vln_msgs.srv import ResetEpisode
from vln_sim import require_px4_env

try:
    from mavros_msgs.msg import State as MavrosState
except ImportError as exc:  # pragma: no cover - only triggered on PX4-misconfigured machines
    raise RuntimeError("mavros_msgs is required for the real PX4 evaluation lane") from exc


ROOT = Path(__file__).resolve().parents[3]


def _bool_string(value: bool) -> str:
    return "true" if value else "false"


def _kill_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait(timeout=15)
    except Exception:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except Exception:
            pass


def _safe_px4_work_dir(episode_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", episode_name).strip("_") or "episode"
    return f"sitl_{slug}_{int(time.time() * 1000)}"


def _wait_for_master(timeout_sec: float) -> None:
    deadline = time.time() + timeout_sec
    master = rosgraph.Master("/vln_px4_episode")
    while time.time() < deadline:
        try:
            master.getPid()
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("timed out waiting for ROS master from PX4 launch stack")


class EpisodeMonitor:
    def __init__(self) -> None:
        self.latest_pose = None
        self.previous_pose = None
        self.path_length_m = 0.0
        self.connected = False
        self.mission_complete = False
        self.last_phase = ""
        self.last_detail = ""
        self.mission_id = ""

    def on_odom(self, msg: Odometry) -> None:
        pose = Pose2D(
            x=msg.pose.pose.position.x,
            y=msg.pose.pose.position.y,
            z=msg.pose.pose.position.z,
            yaw_rad=0.0,
        )
        if self.latest_pose is not None:
            dx = pose.x - self.latest_pose.x
            dy = pose.y - self.latest_pose.y
            self.path_length_m += (dx * dx + dy * dy) ** 0.5
        self.previous_pose = self.latest_pose
        self.latest_pose = pose

    def on_runtime_state(self, msg: RuntimeState) -> None:
        self.last_phase = msg.phase
        self.last_detail = msg.detail
        if msg.mission_id:
            self.mission_id = msg.mission_id
        if msg.mission_complete or msg.phase == "mission_complete":
            self.mission_complete = True

    def on_mavros_state(self, msg: MavrosState) -> None:
        self.connected = bool(msg.connected)


def _load_task(tasks_file: Path, episode_name: str) -> dict:
    tasks = load_and_validate_tasks(tasks_file)
    for task in tasks:
        if str(task["name"]) == episode_name:
            return task
    raise ValueError(f"episode '{episode_name}' not found in {tasks_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single PX4 SITL episode and emit a JSON outcome.")
    parser.add_argument("--tasks-file", required=True)
    parser.add_argument("--episode-name", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--px4-autopilot-dir", default="")
    parser.add_argument("--headless", default="true")
    parser.add_argument("--launch-timeout-sec", type=float, default=120.0)
    args = parser.parse_args()

    tasks_file = Path(args.tasks_file).resolve()
    task = _load_task(tasks_file, args.episode_name)
    px4_root = require_px4_env(args.px4_autopilot_dir)
    output_json = Path(args.output_json).resolve()
    headless = str(args.headless).strip().lower() not in {"false", "0", "no"}

    launch_cmd = [
        "roslaunch",
        "vln_bringup",
        "evaluation.launch",
        "sim_backend:=px4",
        f"px4_autopilot_dir:={px4_root}",
        f"world_name:={task['world_name']}",
        f"tasks_file:={tasks_file}",
        f"headless:={_bool_string(headless)}",
        f"px4_work_dir:={_safe_px4_work_dir(str(task['name']))}",
    ]

    launch_process = subprocess.Popen(
        launch_cmd,
        cwd=str(ROOT),
        preexec_fn=os.setsid,
    )

    monitor = EpisodeMonitor()
    try:
        _wait_for_master(args.launch_timeout_sec)
        rospy.init_node("vln_px4_episode_runner", anonymous=True, disable_signals=True)
        rospy.Subscriber("/sim/uav/odom", Odometry, monitor.on_odom, queue_size=50)
        rospy.Subscriber("/vln/runtime/state", RuntimeState, monitor.on_runtime_state, queue_size=50)
        rospy.Subscriber("/mavros/state", MavrosState, monitor.on_mavros_state, queue_size=50)

        rospy.wait_for_service("/vln/sim/reset_episode", timeout=args.launch_timeout_sec)
        deadline = time.time() + args.launch_timeout_sec
        while time.time() < deadline:
            if launch_process.poll() is not None:
                raise RuntimeError("PX4 launch process exited before the episode became ready")
            if monitor.connected and monitor.latest_pose is not None:
                break
            rospy.sleep(0.25)
        else:
            raise RuntimeError("timed out waiting for PX4 FCU connection and /sim/uav/odom")

        reset_proxy = rospy.ServiceProxy("/vln/sim/reset_episode", ResetEpisode)
        reset_response = reset_proxy(str(task["name"]), True)
        if not reset_response.accepted:
            raise RuntimeError(f"reset service rejected episode '{task['name']}': {reset_response.message}")

        episode_deadline = time.time() + float(task["timeout_sec"])
        while time.time() < episode_deadline:
            if launch_process.poll() is not None:
                raise RuntimeError("PX4 launch process exited during episode execution")
            if monitor.mission_complete:
                break
            rospy.sleep(0.1)

        final_pose = monitor.latest_pose or Pose2D(
            x=float(task["spawn"]["x"]),
            y=float(task["spawn"]["y"]),
            z=float(task["spawn"]["z"]),
            yaw_rad=float(task["spawn"]["yaw"]),
        )
        goal = task.get("goal", {})
        goal_error_m = goal_error_xy(goal, final_pose) if goal else 0.0
        tolerance = float(task.get("goal_tolerance_m", 1.0))
        success = bool(monitor.mission_complete) and goal_error_m <= tolerance
        score = round(1.0 / (1.0 + goal_error_m), 4)

        payload = {
            "episode_name": task["name"],
            "success": success,
            "score": score,
            "detail": (
                f"world={task['world_name']} goal_error_m={goal_error_m:.3f} "
                f"phase={monitor.last_phase or 'timeout'}"
            ),
            "metrics": {
                "goal_error_m": round(goal_error_m, 4),
                "goal_tolerance_m": tolerance,
                "path_length_m": round(monitor.path_length_m, 4),
                "waypoint_count": 0.0,  # filled by the outer runner from compiled mission/task context
                "step_count": 0.0,
                "timeout_sec": float(task["timeout_sec"]),
            },
            "mission_id": monitor.mission_id or f"episode-{task['name']}",
        }
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    finally:
        _kill_process_group(launch_process)


if __name__ == "__main__":
    main()
