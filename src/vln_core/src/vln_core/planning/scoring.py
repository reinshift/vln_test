import math
from typing import Dict, Iterable, Optional

from vln_core.config.models import Mission, Pose2D, SemanticLandmark, TrajectoryPlan, Waypoint


def _normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _heading_for_action(current_yaw: float, action: str) -> float:
    if action == "left":
        return _normalize_angle(current_yaw + math.pi / 2.0)
    if action == "right":
        return _normalize_angle(current_yaw - math.pi / 2.0)
    if action == "backward":
        return _normalize_angle(current_yaw + math.pi)
    return _normalize_angle(current_yaw)


def build_trajectory(
    mission: Mission,
    semantic_landmarks: Optional[Iterable[SemanticLandmark]] = None,
    start_pose: Optional[Pose2D] = None,
    default_distance_m: float = 2.0,
) -> TrajectoryPlan:
    pose = start_pose or Pose2D(x=0.0, y=0.0, yaw_rad=0.0, z=1.5)
    lookup: Dict[str, SemanticLandmark] = {item.label: item for item in (semantic_landmarks or [])}
    waypoints = []

    for step in mission.steps:
        heading = _heading_for_action(pose.yaw_rad, step.action)
        target = lookup.get(step.target_label)
        if target is not None:
            waypoint = Waypoint(x=target.x, y=target.y, z=max(pose.z, target.z or pose.z), yaw_rad=heading)
            pose = Pose2D(x=target.x, y=target.y, yaw_rad=heading, z=max(pose.z, target.z or pose.z))
        else:
            distance = step.desired_distance_m or default_distance_m
            next_x = pose.x + math.cos(heading) * distance
            next_y = pose.y + math.sin(heading) * distance
            waypoint = Waypoint(x=next_x, y=next_y, z=pose.z, yaw_rad=heading)
            pose = Pose2D(x=next_x, y=next_y, yaw_rad=heading, z=pose.z)
        waypoints.append(waypoint)

    return TrajectoryPlan(mission_id=mission.mission_id, frame_id="map", waypoints=waypoints)


def compute_command(plan: TrajectoryPlan, current_pose: Optional[Pose2D] = None, max_speed: float = 1.0) -> Dict[str, float]:
    pose = current_pose or Pose2D(x=0.0, y=0.0, yaw_rad=0.0, z=1.5)
    if not plan.waypoints:
        return {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.0}

    target = plan.waypoints[0]
    dx = target.x - pose.x
    dy = target.y - pose.y
    distance = math.hypot(dx, dy)
    desired_heading = math.atan2(dy, dx) if distance > 1e-6 else pose.yaw_rad
    heading_error = _normalize_angle(desired_heading - pose.yaw_rad)

    linear_x = min(max_speed, distance)
    if abs(heading_error) > 0.6:
        linear_x = 0.0
    return {
        "linear_x": linear_x,
        "linear_y": 0.0,
        "angular_z": max(-1.0, min(1.0, heading_error)),
    }

