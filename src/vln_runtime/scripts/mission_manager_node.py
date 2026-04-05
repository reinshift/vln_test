#!/usr/bin/env python3

import json

import rospy
from std_msgs.msg import String
from vln_core.runtime.lifecycle import MissionTracker
from vln_core.mission.compiler import compile_instruction
from vln_runtime.conversions import mission_to_msg, runtime_state_msg
from vln_msgs.msg import Mission as MissionMsg, RuntimeState


class MissionManagerNode:
    def __init__(self) -> None:
        self.raw_instruction_topic = rospy.get_param("~raw_instruction_topic", "/vln/mission/raw_instruction")
        self.compiled_topic = rospy.get_param("~compiled_topic", "/vln/mission/compiled")
        self.runtime_state_topic = rospy.get_param("~runtime_state_topic", "/vln/runtime/state")
        self.feedback_topic = rospy.get_param("~feedback_topic", "/vln/runtime/controller_feedback")
        self.reset_topic = rospy.get_param("~reset_topic", "/sim/uav/reset_event")
        self.heartbeat_hz = float(rospy.get_param("~heartbeat_hz", 2.0))

        self.tracker = MissionTracker()
        self.active_mission = None
        self.mission_pub = rospy.Publisher(self.compiled_topic, MissionMsg, queue_size=10, latch=True)
        self.state_pub = rospy.Publisher(self.runtime_state_topic, RuntimeState, queue_size=20)
        self.sub = rospy.Subscriber(self.raw_instruction_topic, String, self._on_instruction, queue_size=10)
        self.feedback_sub = rospy.Subscriber(self.feedback_topic, RuntimeState, self._on_feedback, queue_size=20)
        self.reset_sub = rospy.Subscriber(self.reset_topic, String, self._on_reset, queue_size=20)
        self.timer = rospy.Timer(rospy.Duration(1.0 / max(1.0, self.heartbeat_hz)), self._heartbeat)

    def _on_instruction(self, msg: String) -> None:
        self._start_mission(msg.data)

    def _start_mission(self, instruction: str, mission_id: str = "") -> None:
        mission = compile_instruction(instruction, mission_id=mission_id)
        self.active_mission = mission
        progress = self.tracker.start(mission, rospy.get_time())
        self.mission_pub.publish(mission_to_msg(mission))
        self.state_pub.publish(
            runtime_state_msg(
                mission.mission_id,
                progress.active_step_index,
                "mission_compiled",
                f"compiled {len(mission.steps)} steps",
                elapsed_sec=progress.elapsed_sec,
            )
        )

    def _on_feedback(self, msg: RuntimeState) -> None:
        if self.active_mission is None or msg.mission_id != self.active_mission.mission_id:
            return

        progress = self.tracker.advance(
            active_step_index=msg.active_step_index,
            phase=msg.phase,
            detail=msg.detail,
            current_time_sec=rospy.get_time(),
            mission_complete=msg.mission_complete,
        )
        self.state_pub.publish(
            runtime_state_msg(
                progress.mission_id,
                progress.active_step_index,
                progress.phase,
                progress.detail,
                status_code=msg.status_code,
                mission_complete=progress.mission_complete,
                elapsed_sec=progress.elapsed_sec,
            )
        )
        if progress.mission_complete:
            self.active_mission = None
            self.tracker.reset()

    def _on_reset(self, msg: String) -> None:
        payload = {}
        try:
            payload = json.loads(msg.data) if msg.data else {}
        except json.JSONDecodeError:
            payload = {}

        detail = str(payload.get("episode_name", msg.data or "reset_event"))
        self.active_mission = None
        self.tracker.reset()
        self.state_pub.publish(runtime_state_msg("", 0, "mission_reset", detail, status_code=0))
        instruction = str(payload.get("instruction", "")).strip()
        mission_id = str(payload.get("episode_name", "")).strip()
        if instruction:
            self._start_mission(instruction, mission_id=f"episode-{mission_id}" if mission_id else "")

    def _heartbeat(self, _event) -> None:
        now_sec = rospy.get_time()
        if self.active_mission is None:
            progress = self.tracker.snapshot(now_sec)
            if progress.phase == "idle":
                return
        else:
            progress = self.tracker.snapshot(now_sec)
        self.state_pub.publish(
            runtime_state_msg(
                progress.mission_id,
                progress.active_step_index,
                progress.phase,
                progress.detail,
                mission_complete=progress.mission_complete,
                elapsed_sec=progress.elapsed_sec,
            )
        )


def main() -> None:
    rospy.init_node("mission_manager_node")
    MissionManagerNode()
    rospy.spin()


if __name__ == "__main__":
    main()
