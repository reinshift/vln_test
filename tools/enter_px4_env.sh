#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "[px4-env] run this with: source tools/enter_px4_env.sh" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source /opt/ros/noetic/setup.bash

if [[ -f "$ROOT/venv310/bin/activate" ]]; then
  # Prefer the repo-local toolchain so PX4 Python helpers stay reproducible.
  source "$ROOT/venv310/bin/activate"
fi

if [[ -f "$ROOT/devel_local/setup.bash" ]]; then
  source "$ROOT/devel_local/setup.bash"
fi

source "$ROOT/tools/load_local_env.sh"

echo "[px4-env] activated"
echo "[px4-env] PX4_AUTOPILOT_DIR=$PX4_AUTOPILOT_DIR"
