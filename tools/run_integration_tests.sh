#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source "$ROOT/venv310/bin/activate"
export PYTHONPATH="$ROOT/src/vln_core/src:$ROOT/src/vln_perception/src:$ROOT/src/vln_eval/src:$ROOT/src/vln_runtime/src:$ROOT/src/vln_sim/src:${PYTHONPATH:-}"

python -m unittest discover -s "$ROOT/tests/integration" -p 'test_*.py'
