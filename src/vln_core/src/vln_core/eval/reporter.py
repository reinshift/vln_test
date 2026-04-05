import json
from pathlib import Path
from typing import Iterable

from vln_core.config.models import EpisodeOutcome


def write_report(run_dir: Path, outcomes: Iterable[EpisodeOutcome]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    items = list(outcomes)
    summary = {
        "episodes": [
            {
                "run_id": item.run_id,
                "episode_name": item.episode_name,
                "success": item.success,
                "score": item.score,
                "summary_path": item.summary_path,
                "detail": item.detail,
                "metrics": item.metrics,
                "mission_id": item.mission_id,
            }
            for item in items
        ],
        "success_count": sum(1 for item in items if item.success),
        "episode_count": len(items),
        "average_score": round(sum(item.score for item in items) / max(1, len(items)), 4),
    }

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# UAV Evaluation Summary",
        "",
        f"- Episodes: {summary['episode_count']}",
        f"- Successes: {summary['success_count']}",
        f"- Average score: {summary['average_score']}",
        "",
        "## Episodes",
        "",
        "| Episode | Success | Score | Detail |",
        "| --- | --- | ---: | --- |",
    ]
    for item in items:
        lines.append(f"| {item.episode_name} | {item.success} | {item.score:.3f} | {item.detail} |")
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
