#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVEL_SETUP="$ROOT/devel_local/setup.bash"

if [[ ! -f "$DEVEL_SETUP" ]]; then
  echo "[sim-smoke] missing $DEVEL_SETUP. Run make rebuild first." >&2
  exit 1
fi

source /opt/ros/noetic/setup.bash
source "$DEVEL_SETUP"
# shellcheck disable=SC1091
source "$ROOT/tools/load_local_env.sh"
export PYTHONPATH="$ROOT/src/vln_sim/src:${PYTHONPATH:-}"

python3 "$ROOT/tools/check_px4_env.py" --px4-autopilot-dir "${PX4_AUTOPILOT_DIR:-}"
roslaunch --nodes vln_bringup simulation.launch sim_backend:=px4 enable_instruction_gateway:=false headless:=true

echo "[sim-smoke] launch graph OK"
