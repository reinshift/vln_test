#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "vln_sim" / "src"))

from vln_sim import px4_env_errors, px4_env_summary  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-fast PX4 SITL environment validation.")
    parser.add_argument("--px4-autopilot-dir", default="")
    args = parser.parse_args()

    errors = px4_env_errors(args.px4_autopilot_dir)
    if errors:
        raise SystemExit(px4_env_summary(errors))
    print("[px4-env] environment ready")


if __name__ == "__main__":
    main()
