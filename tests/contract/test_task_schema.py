import json
import tempfile
import unittest
from pathlib import Path

from vln_eval.task_schema import load_and_validate_tasks


class TaskSchemaTest(unittest.TestCase):
    def test_load_and_validate_tasks(self):
        payload = [
            {
                "name": "demo",
                "world_name": "campus_stub",
                "instruction": "go to the bench",
                "spawn": {"x": 0.0, "y": 0.0, "z": 1.5, "yaw": 0.0},
                "timeout_sec": 10,
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tasks.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = load_and_validate_tasks(path)
            self.assertEqual(result[0]["name"], "demo")

    def test_world_name_is_required(self):
        payload = [
            {
                "name": "demo",
                "instruction": "go to the bench",
                "spawn": {"x": 0.0, "y": 0.0, "z": 1.5, "yaw": 0.0},
                "timeout_sec": 10,
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tasks.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_and_validate_tasks(path)


if __name__ == "__main__":
    unittest.main()
