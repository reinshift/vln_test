import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GUI_PATH = ROOT / "src" / "vln_bringup" / "operator_gui.py"
SPEC = importlib.util.spec_from_file_location("vln_operator_gui_module", GUI_PATH)
GUI_MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(GUI_MODULE)


class OperatorGuiHelpersTest(unittest.TestCase):
    def test_parse_operator_command_detects_episode_shortcut(self):
        parsed = GUI_MODULE.parse_operator_command("/episode tree_then_finish")
        self.assertEqual(parsed["kind"], "episode")
        self.assertEqual(parsed["episode_name"], "tree_then_finish")

    def test_parse_operator_command_defaults_to_instruction(self):
        parsed = GUI_MODULE.parse_operator_command("fly to the bench and stop")
        self.assertEqual(parsed["kind"], "instruction")
        self.assertEqual(parsed["instruction"], "fly to the bench and stop")

    def test_build_value_map_biases_toward_waypoints_and_landmarks(self):
        grid = GUI_MODULE.GridSpec(width=24, height=24, resolution=1.0, origin_x=-12.0, origin_y=-12.0)
        pose = GUI_MODULE.PoseSnapshot(x=0.0, y=0.0, z=1.5, yaw_rad=0.0)

        heatmap = GUI_MODULE.build_value_map(
            grid=grid,
            pose=pose,
            waypoints=[(6.0, 4.0)],
            landmarks=[("tree", 6.0, 4.0, 0.95)],
            scan_points=[],
            active_labels=["tree"],
        )

        target_y = int(round((4.0 - grid.origin_y) / grid.resolution - 0.5))
        target_x = int(round((6.0 - grid.origin_x) / grid.resolution - 0.5))
        center_y = int(round((0.0 - grid.origin_y) / grid.resolution - 0.5))
        center_x = int(round((0.0 - grid.origin_x) / grid.resolution - 0.5))
        self.assertGreater(heatmap[target_y, target_x], heatmap[center_y, center_x])


if __name__ == "__main__":
    unittest.main()
