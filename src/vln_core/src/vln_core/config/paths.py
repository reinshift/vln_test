from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def generated_world_dir() -> Path:
    return repo_root() / "sim" / "worlds" / "generated"


def system_report_dir() -> Path:
    return repo_root() / "tests" / "reports" / "system" / "latest"

