import unittest

from vln_core.config.models import Pose2D
from vln_eval.px4_eval import goal_error_xy, group_tasks_by_world


class Px4EvalHelpersTest(unittest.TestCase):
    def test_group_tasks_by_world_preserves_world_buckets(self):
        grouped = group_tasks_by_world(
            [
                {"name": "a", "world_name": "campus_stub"},
                {"name": "b", "world_name": "urban_courtyard"},
                {"name": "c", "world_name": "campus_stub"},
            ]
        )
        self.assertEqual([item["name"] for item in grouped["campus_stub"]], ["a", "c"])
        self.assertEqual([item["name"] for item in grouped["urban_courtyard"]], ["b"])

    def test_goal_error_xy_uses_planar_distance(self):
        error = goal_error_xy({"x": 3.0, "y": 4.0, "z": 7.0}, Pose2D(x=0.0, y=0.0, z=10.0, yaw_rad=0.0))
        self.assertAlmostEqual(error, 5.0)


if __name__ == "__main__":
    unittest.main()
