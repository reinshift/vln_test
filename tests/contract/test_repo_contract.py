import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class RepoContractTest(unittest.TestCase):
    def test_expected_packages_exist(self):
        expected = {
            "vln_msgs",
            "vln_core",
            "vln_runtime",
            "vln_perception",
            "vln_sim",
            "vln_bringup",
            "vln_eval",
        }
        found = {path.name for path in (ROOT / "src").iterdir() if path.is_dir()}
        self.assertTrue(expected.issubset(found))

    def test_asset_manifests_are_parseable(self):
        robots = json.loads((ROOT / "sim" / "manifests" / "robots.lock.yaml").read_text(encoding="utf-8"))
        worlds = json.loads((ROOT / "sim" / "manifests" / "world_assets.lock.yaml").read_text(encoding="utf-8"))
        self.assertIn("uav", robots)
        self.assertEqual(len(worlds["worlds"]), 2)
        fuel_models = {item["name"] for item in worlds["fuel_models"]}
        self.assertTrue(
            {"Oak tree", "Pine Tree", "Lamp Post", "FoodCourtBenchShort", "Prius Hybrid"}.issubset(fuel_models)
        )

    def test_world_templates_reference_real_outdoor_assets(self):
        campus = (ROOT / "sim" / "worlds" / "templates" / "campus_stub.world.jinja").read_text(encoding="utf-8")
        courtyard = (ROOT / "sim" / "worlds" / "templates" / "urban_courtyard.world.jinja").read_text(encoding="utf-8")
        self.assertIn("model://Oak tree", campus)
        self.assertIn("model://FoodCourtBenchShort", campus)
        self.assertIn("model://Prius Hybrid", campus)
        self.assertIn("model://Lamp Post", courtyard)
        self.assertIn("model://vln_finish_zone", courtyard)

    def test_ci_workflows_reference_px4_checkout(self):
        pr_workflow = (ROOT / ".github" / "workflows" / "pr.yml").read_text(encoding="utf-8")
        nightly_workflow = (ROOT / ".github" / "workflows" / "nightly.yml").read_text(encoding="utf-8")
        self.assertIn("PX4_AUTOPILOT_DIR", pr_workflow)
        self.assertIn("PX4-Autopilot.git", pr_workflow)
        self.assertIn("PX4_AUTOPILOT_DIR", nightly_workflow)
        self.assertIn("PX4-Autopilot.git", nightly_workflow)

    def test_operator_gui_client_exists(self):
        gui_path = ROOT / "src" / "vln_bringup" / "operator_gui.py"
        self.assertTrue(gui_path.exists())
        gui_text = gui_path.read_text(encoding="utf-8")
        self.assertIn("VLN Operator Client", gui_text)
        self.assertIn("Tk", gui_text)


if __name__ == "__main__":
    unittest.main()
