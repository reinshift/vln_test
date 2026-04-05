import unittest

from vln_core.config.models import Pose2D, Waypoint
from vln_core.eval.metrics import path_length


class EvalMetricsTest(unittest.TestCase):
    def test_path_length_includes_start_pose(self):
        total = path_length(
            [
                Waypoint(x=3.0, y=4.0, z=1.5, yaw_rad=0.0),
                Waypoint(x=6.0, y=8.0, z=1.5, yaw_rad=0.0),
            ],
            start_pose=Pose2D(x=0.0, y=0.0, yaw_rad=0.0),
        )
        self.assertAlmostEqual(total, 10.0)


if __name__ == "__main__":
    unittest.main()
