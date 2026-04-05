import tempfile
import unittest
from pathlib import Path

from vln_core.config.models import EpisodeOutcome
from vln_core.eval.reporter import write_report


class EvalReporterTest(unittest.TestCase):
    def test_write_report_creates_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            write_report(
                run_dir,
                [
                    EpisodeOutcome(
                        run_id="r1",
                        episode_name="episode_a",
                        success=True,
                        score=1.0,
                        summary_path="summary.md",
                        detail="ok",
                    )
                ],
            )
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "summary.md").exists())


if __name__ == "__main__":
    unittest.main()

