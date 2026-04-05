import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class GoldenEvalTest(unittest.TestCase):
    def test_golden_eval_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            env = os.environ.copy()
            pythonpath_entries = [
                str(ROOT / "src" / "vln_core" / "src"),
                str(ROOT / "src" / "vln_eval" / "src"),
                env.get("PYTHONPATH", ""),
            ]
            env["PYTHONPATH"] = ":".join(item for item in pythonpath_entries if item)
            subprocess.run(
                [
                    "python3",
                    str(ROOT / "src" / "vln_eval" / "scripts" / "run_uav_eval.py"),
                    "--tasks-file",
                    str(ROOT / "tests" / "goldens" / "mock_uav_tasks.json"),
                    "--landmarks-file",
                    str(ROOT / "tests" / "fixtures" / "world_landmarks.json"),
                    "--output-dir",
                    str(output_dir),
                ],
                check=True,
                env=env,
            )
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["episode_count"], 2)
            self.assertEqual(summary["success_count"], 2)
            self.assertEqual(summary["episodes"][0]["run_id"], "golden-mock")
            self.assertIn("path_length_m", summary["episodes"][0]["metrics"])
            self.assertIn("waypoint_count", summary["episodes"][0]["metrics"])


if __name__ == "__main__":
    unittest.main()
