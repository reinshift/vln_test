#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT/tests/reports/system/latest"
DEVEL_SETUP="$ROOT/devel_local/setup.bash"

if [[ ! -f "$DEVEL_SETUP" ]]; then
  echo "[e2e] missing $DEVEL_SETUP. Run make rebuild first." >&2
  exit 1
fi

source /opt/ros/noetic/setup.bash
source "$DEVEL_SETUP"
# shellcheck disable=SC1091
source "$ROOT/tools/load_local_env.sh"
export PYTHONPATH="$ROOT/src/vln_core/src:$ROOT/src/vln_perception/src:$ROOT/src/vln_eval/src:$ROOT/src/vln_runtime/src:$ROOT/src/vln_sim/src:${PYTHONPATH:-}"

python3 "$ROOT/tools/check_px4_env.py" --px4-autopilot-dir "${PX4_AUTOPILOT_DIR:-}"
python3 "$ROOT/src/vln_eval/scripts/run_px4_eval.py" \
  --tasks-file "$ROOT/tests/goldens/mock_uav_tasks.json" \
  --output-dir "$OUT_DIR" \
  --run-id "local-px4-e2e" \
  --min-success-rate 1.0

python3 -m unittest discover -s "$ROOT/tests/system" -p 'test_*.py'

echo "[e2e] reports saved under $OUT_DIR"
