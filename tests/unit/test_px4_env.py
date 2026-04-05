import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from vln_sim.uav_px4 import (
    PX4_GIT_REF,
    px4_env_errors,
    px4_env_ready,
    px4_launch_file,
    world_file,
)


class Px4EnvTest(unittest.TestCase):
    def test_missing_env_reports_fail_fast_message(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            errors = px4_env_errors("")
        self.assertIn("PX4_AUTOPILOT_DIR is not set.", errors[0])
        self.assertIn(PX4_GIT_REF, errors[1])

    def test_ready_env_passes_static_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "launch").mkdir()
            (root / "launch" / "mavros_posix_sitl.launch").write_text("<launch/>", encoding="utf-8")
            (root / "Tools").mkdir()
            (root / "ROMFS").mkdir()
            (root / "build" / "px4_sitl_default" / "bin").mkdir(parents=True)
            (root / "build" / "px4_sitl_default" / "bin" / "px4").write_text("", encoding="utf-8")
            (root / "build" / "px4_sitl_default" / "build_gazebo-classic").mkdir(parents=True)
            (root / "build" / "px4_sitl_default" / "build_gazebo-classic" / "libgazebo_mavlink_interface.so").write_text("", encoding="utf-8")
            (root / "Makefile").write_text("all:\n", encoding="utf-8")

            with mock.patch("shutil.which", side_effect=lambda name: f"/usr/bin/{name}"):
                with mock.patch("subprocess.run") as run_mock:
                    run_mock.return_value.returncode = 0
                    self.assertTrue(px4_env_ready(str(root)))
                    self.assertEqual(px4_launch_file(str(root)), root / "launch" / "mavros_posix_sitl.launch")

    def test_missing_build_artifacts_fail_fast(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "launch").mkdir()
            (root / "launch" / "mavros_posix_sitl.launch").write_text("<launch/>", encoding="utf-8")
            (root / "Tools").mkdir()
            (root / "ROMFS").mkdir()
            (root / "Makefile").write_text("all:\n", encoding="utf-8")

            with mock.patch("shutil.which", side_effect=lambda name: f"/usr/bin/{name}"):
                with mock.patch("subprocess.run") as run_mock:
                    run_mock.return_value.returncode = 0
                    errors = px4_env_errors(str(root))

        self.assertTrue(any("PX4 SITL build artifact is missing" in item for item in errors))

    def test_world_file_resolves_generated_world_path(self):
        repo_root = Path("/tmp/example")
        self.assertEqual(
            world_file(repo_root, "campus_stub"),
            repo_root / "sim" / "worlds" / "generated" / "campus_stub.world",
        )


if __name__ == "__main__":
    unittest.main()
