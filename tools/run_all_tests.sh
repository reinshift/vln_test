#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

bash "$ROOT/tools/run_unit_tests.sh"
bash "$ROOT/tools/run_contract_tests.sh"
bash "$ROOT/tools/run_integration_tests.sh"
bash "$ROOT/tools/run_sim_smoke.sh"
bash "$ROOT/tools/run_e2e_tests.sh"

