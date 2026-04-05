import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class LaunchContractTest(unittest.TestCase):
    def test_simulation_launch_defaults_to_px4(self):
        launch_text = (ROOT / "src" / "vln_bringup" / "launch" / "simulation.launch").read_text(encoding="utf-8")
        self.assertIn('name="sim_backend" default="px4"', launch_text)
        self.assertIn("px4_contract.launch", launch_text)
        self.assertIn("mission_manager_node", launch_text)
        self.assertIn("safety_supervisor_node", launch_text)
        self.assertIn('name="tasks_file"', launch_text)
        self.assertIn('name="world_name"', launch_text)
        self.assertIn('name="headless"', launch_text)

    def test_evaluation_launch_wires_px4_and_golden_tasks(self):
        launch_text = (ROOT / "src" / "vln_bringup" / "launch" / "evaluation.launch").read_text(encoding="utf-8")
        self.assertIn("mock_uav_tasks.json", launch_text)
        self.assertIn('name="sim_backend" default="px4"', launch_text)
        self.assertIn('name="headless" default="true"', launch_text)
        self.assertIn('name="px4_autopilot_dir"', launch_text)

    def test_operator_gui_launch_exists(self):
        launch_text = (ROOT / "src" / "vln_bringup" / "launch" / "operator_gui.launch").read_text(encoding="utf-8")
        self.assertIn("operator_gui.py", launch_text)
        self.assertIn('name="world_name"', launch_text)
        self.assertIn('name="tasks_file"', launch_text)
        self.assertIn('name="autostart_sim"', launch_text)

    def test_px4_contract_launch_mentions_mavros_and_bridge(self):
        launch_text = (ROOT / "src" / "vln_bringup" / "launch" / "px4_contract.launch").read_text(encoding="utf-8")
        self.assertIn('pkg="px4" type="px4"', launch_text)
        self.assertIn("iris_vln_rgb_lidar", launch_text)
        self.assertIn("px4.launch", launch_text)
        self.assertIn("px4_uav_bridge_node.py", launch_text)
        self.assertIn("spawn_vehicle_node.py", launch_text)


if __name__ == "__main__":
    unittest.main()
