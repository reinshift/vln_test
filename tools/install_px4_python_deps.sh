#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT/venv310"
PX4_REQS="$ROOT/sim/cache/PX4-Autopilot/Tools/setup/requirements.txt"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[px4-prepare] missing venv310 at $VENV_DIR" >&2
  exit 1
fi

if [[ ! -f "$PX4_REQS" ]]; then
  echo "[px4-prepare] missing PX4 requirements file at $PX4_REQS" >&2
  echo "[px4-prepare] clone PX4 into $ROOT/sim/cache/PX4-Autopilot or export PX4_AUTOPILOT_DIR first" >&2
  exit 1
fi

source "$VENV_DIR/bin/activate"

# PX4 v1.14 ships a requirement string that newer pip rejects. Normalize it
# into a temporary file so installation remains reproducible without mutating
# the upstream checkout.
TMP_REQS="$(mktemp /tmp/px4_requirements.XXXXXX.txt)"
trap 'rm -f "$TMP_REQS"' EXIT
sed 's/^matplotlib>=3\.0\.\*$/matplotlib>=3.0/' "$PX4_REQS" > "$TMP_REQS"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$TMP_REQS"
python -m pip install "empy==3.3.4"

echo "[px4-prepare] PX4 Python build dependencies installed into $VENV_DIR"
