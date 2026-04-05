#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -d "$ROOT/venv310" ]]; then
  echo "[bootstrap] missing venv310 at $ROOT/venv310" >&2
  exit 1
fi

mkdir -p "$ROOT/tests/reports/contract" "$ROOT/tests/reports/integration" "$ROOT/tests/reports/system/latest"
mkdir -p "$ROOT/sim/cache" "$ROOT/sim/worlds/generated"

echo "[bootstrap] repo ready"

