from typing import Dict, List, Optional

from vln_core.config.models import Pose2D, TrajectoryPlan, Waypoint
from vln_core.planning.scoring import compute_command


class TrajectoryFollower:
    def __init__(self, waypoint_tolerance_m: float = 0.5) -> None:
        self.waypoint_tolerance_m = waypoint_tolerance_m
        self.reset()

    def reset(self) -> None:
        self.plan: Optional[TrajectoryPlan] = None
        self.active_waypoint_index = 0
        self.mission_complete = False

    def set_plan(self, plan: TrajectoryPlan) -> None:
        self.plan = plan
        self.active_waypoint_index = 0
        self.mission_complete = len(plan.waypoints) == 0

    def _distance(self, pose: Pose2D, waypoint: Waypoint) -> float:
        dx = waypoint.x - pose.x
        dy = waypoint.y - pose.y
        return (dx * dx + dy * dy) ** 0.5

    def update_pose(self, pose: Pose2D) -> List[Dict[str, object]]:
        events: List[Dict[str, object]] = []
        if self.plan is None or self.mission_complete:
            return events

        while self.active_waypoint_index < len(self.plan.waypoints):
            waypoint = self.plan.waypoints[self.active_waypoint_index]
            if self._distance(pose, waypoint) > self.waypoint_tolerance_m:
                break
            events.append(
                {
                    "type": "waypoint_reached",
                    "waypoint_index": self.active_waypoint_index + 1,
                    "mission_id": self.plan.mission_id,
                }
            )
            self.active_waypoint_index += 1

        if self.active_waypoint_index >= len(self.plan.waypoints) and self.plan.waypoints:
            self.mission_complete = True
            events.append({"type": "mission_complete", "mission_id": self.plan.mission_id})
        return events

    def current_waypoint(self) -> Optional[Waypoint]:
        if self.plan is None or self.mission_complete:
            return None
        if self.active_waypoint_index >= len(self.plan.waypoints):
            return None
        return self.plan.waypoints[self.active_waypoint_index]

    def remaining_plan(self) -> Optional[TrajectoryPlan]:
        if self.plan is None or self.mission_complete:
            return None
        return TrajectoryPlan(
            mission_id=self.plan.mission_id,
            frame_id=self.plan.frame_id,
            waypoints=self.plan.waypoints[self.active_waypoint_index:],
            target_speed=self.plan.target_speed,
            planner_mode=self.plan.planner_mode,
        )

    def compute_command(self, pose: Pose2D, max_speed: float) -> Dict[str, float]:
        remaining = self.remaining_plan()
        if remaining is None:
            return {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.0}
        return compute_command(remaining, pose, max_speed=max_speed)

