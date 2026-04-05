#!/usr/bin/env python3

import json
from pathlib import Path

import rospy
from std_msgs.msg import String
from vln_msgs.srv import ResetEpisode, ResetEpisodeResponse


class EpisodeResetServiceNode:
    def __init__(self) -> None:
        self.service_name = rospy.get_param("~service_name", "/vln/sim/reset_episode")
        self.reset_topic = rospy.get_param("~reset_topic", "/sim/uav/reset_event")
        self.tasks_file = rospy.get_param("~tasks_file", "")
        self.task_lookup = self._load_tasks(self.tasks_file)
        self.pub = rospy.Publisher(self.reset_topic, String, queue_size=10)
        self.service = rospy.Service(self.service_name, ResetEpisode, self._handle_reset)

    def _load_tasks(self, tasks_file: str):
        if not tasks_file:
            return {}
        path = Path(tasks_file)
        if not path.exists():
            rospy.logwarn("reset tasks file not found: %s", tasks_file)
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        lookup = {}
        for item in payload:
            if not isinstance(item, dict) or "name" not in item:
                continue
            lookup[str(item["name"])] = item
        return lookup

    def _handle_reset(self, request) -> ResetEpisodeResponse:
        episode_name = request.episode_name or "unnamed_episode"
        task = self.task_lookup.get(episode_name, {})
        payload = {
            "episode_name": episode_name,
            "hard_reset": bool(request.hard_reset),
        }
        if isinstance(task.get("spawn"), dict):
            payload["spawn"] = task["spawn"]
        if "world_name" in task:
            payload["world_name"] = task["world_name"]
        if "instruction" in task:
            payload["instruction"] = task["instruction"]
        serialized = json.dumps(payload, sort_keys=True)
        self.pub.publish(String(data=serialized))
        return ResetEpisodeResponse(accepted=True, message=f"reset published for {episode_name}")


def main() -> None:
    rospy.init_node("episode_reset_service_node")
    EpisodeResetServiceNode()
    rospy.spin()


if __name__ == "__main__":
    main()
