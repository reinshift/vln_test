import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional


PX4_GIT_REF = "v1.14.0"
PX4_LAUNCH_RELATIVE_PATH = Path("launch") / "mavros_posix_sitl.launch"
PX4_REQUIRED_ROOT_MARKERS = (
    "ROMFS",
    "Tools",
    "Makefile",
)
PX4_REQUIRED_BUILD_MARKERS = (
    Path("build") / "px4_sitl_default" / "bin" / "px4",
    Path("build") / "px4_sitl_default" / "build_gazebo-classic" / "libgazebo_mavlink_interface.so",
)
PX4_REQUIRED_ROS_PACKAGES = (
    "gazebo_ros",
    "mavros",
    "topic_tools",
)


def resolve_px4_autopilot_dir(explicit_path: str = "") -> Optional[Path]:
    candidate = (explicit_path or os.environ.get("PX4_AUTOPILOT_DIR") or "").strip()
    return Path(candidate).expanduser().resolve() if candidate else None


def px4_launch_file(explicit_path: str = "") -> Path:
    root = resolve_px4_autopilot_dir(explicit_path)
    return root / PX4_LAUNCH_RELATIVE_PATH if root is not None else Path()


def world_file(repo_root: Path, world_name: str) -> Path:
    return (repo_root / "sim" / "worlds" / "generated" / f"{world_name}.world").resolve()


def task_groups_by_world(tasks: Iterable[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {}
    for task in tasks:
        world_name = str(task["world_name"])
        grouped.setdefault(world_name, []).append(task)
    return grouped


def px4_env_errors(explicit_path: str = "") -> List[str]:
    errors: List[str] = []
    root = resolve_px4_autopilot_dir(explicit_path)

    if root is None:
        return [
            "PX4_AUTOPILOT_DIR is not set.",
            f"Expected an external PX4 checkout pinned to {PX4_GIT_REF}.",
        ]

    if not root.exists():
        errors.append(f"PX4_AUTOPILOT_DIR does not exist: {root}")
        return errors

    if not root.is_dir():
        errors.append(f"PX4_AUTOPILOT_DIR is not a directory: {root}")
        return errors

    for marker in PX4_REQUIRED_ROOT_MARKERS:
        if not (root / marker).exists():
            errors.append(f"PX4 checkout is missing required path: {root / marker}")

    for marker in PX4_REQUIRED_BUILD_MARKERS:
        if not (root / marker).exists():
            errors.append(f"PX4 SITL build artifact is missing: {root / marker}")

    launch_file = px4_launch_file(explicit_path)
    if not launch_file.exists():
        errors.append(f"PX4 launch file is missing: {launch_file}")

    for command in ("roslaunch", "gzserver"):
        if shutil.which(command) is None:
            errors.append(f"Required command is not available on PATH: {command}")

    rospack = shutil.which("rospack")
    if rospack is None:
        errors.append("Required command is not available on PATH: rospack")
        return errors

    for package_name in PX4_REQUIRED_ROS_PACKAGES:
        completed = subprocess.run(
            [rospack, "find", package_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if completed.returncode != 0:
            errors.append(f"Required ROS package is missing from ROS_PACKAGE_PATH: {package_name}")

    return errors


def px4_env_ready(explicit_path: str = "") -> bool:
    return not px4_env_errors(explicit_path)


def require_px4_env(explicit_path: str = "") -> Path:
    errors = px4_env_errors(explicit_path)
    if errors:
        raise RuntimeError(px4_env_summary(errors))
    return resolve_px4_autopilot_dir(explicit_path)


def px4_env_summary(errors: Iterable[str]) -> str:
    lines = ["PX4 SITL environment is not ready:"]
    lines.extend(f"- {item}" for item in errors)
    lines.append(
        f"- Export PX4_AUTOPILOT_DIR to an external PX4-Autopilot checkout pinned to {PX4_GIT_REF}."
    )
    return "\n".join(lines)
