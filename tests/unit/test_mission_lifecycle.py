import unittest

from vln_core.config.models import Mission, MissionStep, Pose2D, TrajectoryPlan, Waypoint
from vln_core.planning.follower import TrajectoryFollower
from vln_core.runtime.lifecycle import MissionTracker


class MissionLifecycleTest(unittest.TestCase):
    def test_mission_tracker_tracks_elapsed_progress(self):
        mission = Mission(
            mission_id="mission-1",
            raw_instruction="go to the bench",
            steps=[MissionStep(step_index=1, action="forward", target_label="bench", terminal=True)],
        )
        tracker = MissionTracker()
        started = tracker.start(mission, started_at_sec=10.0)
        self.assertEqual(started.phase, "mission_active")
        self.assertEqual(started.active_step_index, 1)

        progressed = tracker.advance(
            active_step_index=1,
            phase="waypoint_reached",
            detail="waypoint 1 reached",
            current_time_sec=12.5,
            mission_complete=True,
        )
        self.assertTrue(progressed.mission_complete)
        self.assertAlmostEqual(progressed.elapsed_sec, 2.5)

    def test_trajectory_follower_emits_waypoint_and_completion_events(self):
        follower = TrajectoryFollower(waypoint_tolerance_m=0.25)
        follower.set_plan(
            TrajectoryPlan(
                mission_id="mission-2",
                frame_id="map",
                waypoints=[
                    Waypoint(x=1.0, y=0.0, z=1.5, yaw_rad=0.0),
                    Waypoint(x=2.0, y=0.0, z=1.5, yaw_rad=0.0),
                ],
            )
        )

        first_events = follower.update_pose(Pose2D(x=1.02, y=0.0, yaw_rad=0.0))
        self.assertEqual(first_events[0]["type"], "waypoint_reached")
        self.assertEqual(first_events[0]["waypoint_index"], 1)

        second_events = follower.update_pose(Pose2D(x=2.0, y=0.0, yaw_rad=0.0))
        self.assertEqual(second_events[0]["type"], "waypoint_reached")
        self.assertEqual(second_events[1]["type"], "mission_complete")
        self.assertTrue(follower.mission_complete)
        self.assertIsNone(follower.current_waypoint())


if __name__ == "__main__":
    unittest.main()
