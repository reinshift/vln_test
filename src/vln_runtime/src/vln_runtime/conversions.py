import math

import rospy
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from vln_core.config.models import Mission, MissionStep, SemanticLandmark, TrajectoryPlan
from vln_msgs.msg import Mission as MissionMsg
from vln_msgs.msg import MissionStep as MissionStepMsg
from vln_msgs.msg import RuntimeState, SemanticLandmark as SemanticLandmarkMsg, Trajectory


def runtime_state_msg(
    mission_id: str,
    step_index: int,
    phase: str,
    detail: str,
    status_code: int = 1,
    mission_complete: bool = False,
    elapsed_sec: float = 0.0,
):
    msg = RuntimeState()
    msg.header.stamp = rospy.Time.now()
    msg.mission_id = mission_id
    msg.active_step_index = int(step_index)
    msg.phase = phase
    msg.detail = detail
    msg.status_code = int(status_code)
    msg.elapsed_sec = float(elapsed_sec)
    msg.mission_complete = bool(mission_complete)
    return msg


def yaw_to_quaternion(yaw_rad: float) -> Quaternion:
    half = float(yaw_rad) / 2.0
    return Quaternion(x=0.0, y=0.0, z=math.sin(half), w=math.cos(half))


def yaw_from_quaternion(quaternion: Quaternion) -> float:
    siny_cosp = 2.0 * ((quaternion.w * quaternion.z) + (quaternion.x * quaternion.y))
    cosy_cosp = 1.0 - 2.0 * ((quaternion.y * quaternion.y) + (quaternion.z * quaternion.z))
    return math.atan2(siny_cosp, cosy_cosp)


def mission_to_msg(mission: Mission) -> MissionMsg:
    msg = MissionMsg()
    msg.header.stamp = rospy.Time.now()
    msg.mission_id = mission.mission_id
    msg.raw_instruction = mission.raw_instruction
    msg.steps = [
        MissionStepMsg(
            step_index=step.step_index,
            action=step.action,
            target_label=step.target_label,
            desired_yaw_rad=step.desired_yaw_rad,
            desired_distance_m=step.desired_distance_m,
            terminal=step.terminal,
        )
        for step in mission.steps
    ]
    return msg


def mission_from_msg(msg: MissionMsg) -> Mission:
    return Mission(
        mission_id=msg.mission_id,
        raw_instruction=msg.raw_instruction,
        steps=[
            MissionStep(
                step_index=step.step_index,
                action=step.action,
                target_label=step.target_label,
                desired_yaw_rad=step.desired_yaw_rad,
                desired_distance_m=step.desired_distance_m,
                terminal=step.terminal,
            )
            for step in msg.steps
        ],
    )


def trajectory_to_msg(plan: TrajectoryPlan) -> Trajectory:
    msg = Trajectory()
    msg.header.stamp = rospy.Time.now()
    msg.mission_id = plan.mission_id
    msg.frame_id = plan.frame_id
    msg.target_speed = plan.target_speed
    msg.planner_mode = plan.planner_mode
    for waypoint in plan.waypoints:
        pose = PoseStamped()
        pose.header.stamp = msg.header.stamp
        pose.header.frame_id = plan.frame_id
        pose.pose.position = Point(x=waypoint.x, y=waypoint.y, z=waypoint.z)
        pose.pose.orientation = yaw_to_quaternion(waypoint.yaw_rad)
        msg.waypoints.append(pose)
    return msg


def semantic_landmark_to_msg(item: SemanticLandmark) -> SemanticLandmarkMsg:
    msg = SemanticLandmarkMsg()
    msg.header.stamp = rospy.Time.now()
    msg.label = item.label
    msg.position = Point(x=item.x, y=item.y, z=item.z)
    msg.confidence = item.confidence
    msg.observation_count = item.observation_count
    msg.state = item.state
    return msg
