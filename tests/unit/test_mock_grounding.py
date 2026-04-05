import unittest

from vln_core.config.models import Mission, MissionStep
from vln_perception.plugins.mock_grounding import MockGroundingBackend


class MockGroundingDebugBoxesTest(unittest.TestCase):
    def test_debug_boxes_are_generated_for_unique_targets(self):
        backend = MockGroundingBackend()
        mission = Mission(
            mission_id="mission-1",
            raw_instruction="go to the tree then the bench",
            steps=[
                MissionStep(step_index=0, action="forward", target_label="tree"),
                MissionStep(step_index=1, action="forward", target_label="bench"),
                MissionStep(step_index=2, action="forward", target_label="tree"),
            ],
        )

        boxes = backend.debug_boxes_for_mission(mission)

        self.assertEqual([item["label"] for item in boxes], ["tree", "bench"])
        self.assertTrue(all(0.0 <= item["xmin"] < item["xmax"] <= 1.0 for item in boxes))
        self.assertTrue(all(0.0 <= item["ymin"] < item["ymax"] <= 1.0 for item in boxes))


if __name__ == "__main__":
    unittest.main()
