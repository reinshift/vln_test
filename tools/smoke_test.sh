#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVEL_DIR="${DEVEL_DIR:-devel_local}"
SETUP_BASH="${ROOT_DIR}/${DEVEL_DIR}/setup.bash"
LAUNCH_FILE="${LAUNCH_FILE:-vln_system.launch}"
LAUNCH_ARGS="${LAUNCH_ARGS:-}"

if [[ ! -f "${SETUP_BASH}" ]]; then
  echo "Missing setup file: ${SETUP_BASH}" >&2
  echo "Run tools/rebuild_workspace.sh first, or set DEVEL_DIR to an existing devel space." >&2
  exit 1
fi

echo "[smoke] byte-compiling Python nodes"
python3 -m py_compile \
  "${ROOT_DIR}"/src/vln_mock/scripts/*.py \
  "${ROOT_DIR}"/src/llm_model/scripts/*.py \
  "${ROOT_DIR}"/src/aruco_detector/scripts/*.py

echo "[smoke] sourcing ${SETUP_BASH}"
# shellcheck disable=SC1090
source "${SETUP_BASH}"

echo "[smoke] checking roslaunch node graph for ${LAUNCH_FILE} ${LAUNCH_ARGS}"
if [[ -n "${LAUNCH_ARGS}" ]]; then
  # shellcheck disable=SC2086
  roslaunch --nodes vln_mock "${LAUNCH_FILE}" ${LAUNCH_ARGS}
else
  roslaunch --nodes vln_mock "${LAUNCH_FILE}"
fi

if [[ "${RUN_MODEL_SMOKE_TEST:-0}" == "1" ]]; then
  echo "[smoke] running optional model smoke test"
  model_args=()
  if [[ -n "${QWEN_MODEL_PATH:-}" ]]; then
    model_args+=(--qwen-path "${QWEN_MODEL_PATH}")
  fi
  if [[ -n "${GROUNDING_MODEL_PATH:-}" ]]; then
    model_args+=(--grounding-path "${GROUNDING_MODEL_PATH}")
  fi
  if [[ "${SKIP_QWEN_MODEL_TEST:-0}" == "1" ]]; then
    model_args+=(--skip-qwen)
  fi
  if [[ "${SKIP_GROUNDING_MODEL_TEST:-0}" == "1" ]]; then
    model_args+=(--skip-grounding)
  fi
  if [[ "${SKIP_FLORENCE_MODEL_TEST:-0}" == "1" ]]; then
    model_args+=(--skip-florence)
  fi
  python3 "${ROOT_DIR}/tools/model_smoke_test.py" "${model_args[@]}"
fi

echo "[smoke] smoke test passed"
