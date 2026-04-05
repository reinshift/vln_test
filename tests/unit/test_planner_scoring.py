import unittest

from vln_core.config.models import Mission, MissionStep, Pose2D, SemanticLandmark
from vln_core.planning.scoring import build_trajectory, compute_command


class PlannerScoringTest(unittest.TestCase):
    def test_build_trajectory_uses_semantic_target(self):
        mission = Mission(
            mission_id="m1",
            raw_instruction="go to the bench",
            steps=[MissionStep(step_index=1, action="forward", target_label="bench", terminal=True)],
        )
        plan = build_trajectory(
            mission,
            semantic_landmarks=[SemanticLandmark(label="bench", x=8.0, y=2.0)],
            start_pose=Pose2D(x=0.0, y=0.0, yaw_rad=0.0),
        )
        self.assertAlmostEqual(plan.waypoints[-1].x, 8.0)
        self.assertAlmostEqual(plan.waypoints[-1].y, 2.0)

    def test_compute_command_rotates_before_translating(self):
        mission = Mission(
            mission_id="m2",
            raw_instruction="turn right",
            steps=[MissionStep(step_index=1, action="right", terminal=True)],
        )
        plan = build_trajectory(mission)
        cmd = compute_command(plan, Pose2D(x=0.0, y=0.0, yaw_rad=1.57))
        self.assertEqual(cmd["linear_x"], 0.0)
        self.assertLess(cmd["angular_z"], 0.0)


if __name__ == "__main__":
    unittest.main()

