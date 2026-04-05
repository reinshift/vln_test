#!/usr/bin/env python3

import math
from pathlib import Path

import rospy
from gazebo_msgs.srv import DeleteModel, GetWorldProperties, SpawnModel
from geometry_msgs.msg import Pose


def _quaternion_from_euler(roll: float, pitch: float, yaw: float):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


class VehicleSpawner:
    def __init__(self) -> None:
        self.model_file = Path(rospy.get_param("~model_file")).expanduser().resolve()
        self.model_name = str(rospy.get_param("~model_name"))
        self.robot_namespace = str(rospy.get_param("~robot_namespace", ""))
        self.reference_frame = str(rospy.get_param("~reference_frame", "world"))

        self.spawn_service_name = str(rospy.get_param("~spawn_service", "/gazebo/spawn_sdf_model"))
        self.delete_service_name = str(rospy.get_param("~delete_service", "/gazebo/delete_model"))
        self.world_properties_service_name = str(rospy.get_param("~world_properties_service", "/gazebo/get_world_properties"))
        self.wait_for_service_sec = float(rospy.get_param("~wait_for_service_sec", 60.0))
        self.startup_delay_sec = float(rospy.get_param("~startup_delay_sec", 2.0))
        self.retry_delay_sec = float(rospy.get_param("~retry_delay_sec", 2.0))
        self.max_attempts = int(rospy.get_param("~max_attempts", 4))
        self.delete_if_exists = bool(rospy.get_param("~delete_if_exists", True))

        self.pose = Pose()
        self.pose.position.x = float(rospy.get_param("~x", 0.0))
        self.pose.position.y = float(rospy.get_param("~y", 0.0))
        self.pose.position.z = float(rospy.get_param("~z", 0.35))
        roll = float(rospy.get_param("~roll", 0.0))
        pitch = float(rospy.get_param("~pitch", 0.0))
        yaw = float(rospy.get_param("~yaw", 0.0))
        qx, qy, qz, qw = _quaternion_from_euler(roll, pitch, yaw)
        self.pose.orientation.x = qx
        self.pose.orientation.y = qy
        self.pose.orientation.z = qz
        self.pose.orientation.w = qw

        if not self.model_file.exists():
            raise FileNotFoundError(f"vehicle model file does not exist: {self.model_file}")

        self.model_xml = self.model_file.read_text(encoding="utf-8")

    def _model_exists(self, world_proxy) -> bool:
        try:
            response = world_proxy()
        except rospy.ServiceException as exc:
            rospy.logdebug("get_world_properties service failed while checking %s: %s", self.model_name, exc)
            return False
        return self.model_name in set(getattr(response, "model_names", []))

    def _delete_existing(self, delete_proxy, world_proxy) -> None:
        if not self.delete_if_exists:
            return
        if not self._model_exists(world_proxy):
            return
        try:
            response = delete_proxy(self.model_name)
        except rospy.ServiceException as exc:
            rospy.logdebug("delete_model service failed for %s: %s", self.model_name, exc)
            return
        if getattr(response, "success", False):
            rospy.loginfo("Deleted stale Gazebo model '%s' before respawn.", self.model_name)

    def run(self) -> int:
        if self.startup_delay_sec > 0.0:
            rospy.loginfo("Waiting %.1fs before spawning '%s' so Gazebo can finish loading.", self.startup_delay_sec, self.model_name)
            rospy.sleep(self.startup_delay_sec)

        rospy.loginfo("Waiting for Gazebo spawn service: %s", self.spawn_service_name)
        rospy.wait_for_service(self.spawn_service_name, timeout=self.wait_for_service_sec)

        spawn_proxy = rospy.ServiceProxy(self.spawn_service_name, SpawnModel)
        delete_proxy = rospy.ServiceProxy(self.delete_service_name, DeleteModel)
        world_proxy = rospy.ServiceProxy(self.world_properties_service_name, GetWorldProperties)

        for attempt in range(1, max(1, self.max_attempts) + 1):
            self._delete_existing(delete_proxy, world_proxy)
            try:
                response = spawn_proxy(
                    self.model_name,
                    self.model_xml,
                    self.robot_namespace,
                    self.pose,
                    self.reference_frame,
                )
            except rospy.ServiceException as exc:
                rospy.logwarn("Spawn attempt %s/%s for '%s' raised a Gazebo service error: %s", attempt, self.max_attempts, self.model_name, exc)
            else:
                if response.success:
                    rospy.loginfo("Spawned Gazebo vehicle '%s' from %s", self.model_name, self.model_file)
                    return 0
                rospy.logwarn(
                    "Spawn attempt %s/%s for '%s' failed: %s",
                    attempt,
                    self.max_attempts,
                    self.model_name,
                    response.status_message,
                )

            if attempt < self.max_attempts:
                rospy.sleep(self.retry_delay_sec)

        rospy.logerr("Unable to spawn Gazebo vehicle '%s' after %s attempts.", self.model_name, self.max_attempts)
        return 1


def main() -> None:
    rospy.init_node("spawn_vehicle_node")
    try:
        spawner = VehicleSpawner()
    except Exception as exc:  # pragma: no cover - defensive node bootstrap
        rospy.logerr("Vehicle spawner initialization failed: %s", exc)
        raise SystemExit(1)

    raise SystemExit(spawner.run())


if __name__ == "__main__":
    main()
