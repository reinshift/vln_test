#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-build_local}"
DEVEL_DIR="${DEVEL_DIR:-devel_local}"
INSTALL_DIR="${INSTALL_DIR:-install_local}"
JOBS="${JOBS:-2}"
CC_BIN="${CC:-/usr/bin/gcc}"
CXX_BIN="${CXX:-/usr/bin/g++}"

if [[ ! -x "${CC_BIN}" ]]; then
  echo "C compiler not found: ${CC_BIN}" >&2
  exit 1
fi

if [[ ! -x "${CXX_BIN}" ]]; then
  echo "C++ compiler not found: ${CXX_BIN}" >&2
  exit 1
fi

echo "[rebuild] root=${ROOT_DIR}"
echo "[rebuild] build=${BUILD_DIR} devel=${DEVEL_DIR} install=${INSTALL_DIR}"
echo "[rebuild] compiler=${CC_BIN} ${CXX_BIN}"

cmake \
  -S "${ROOT_DIR}/src" \
  -B "${ROOT_DIR}/${BUILD_DIR}" \
  -DCATKIN_DEVEL_PREFIX="${ROOT_DIR}/${DEVEL_DIR}" \
  -DCMAKE_INSTALL_PREFIX="${ROOT_DIR}/${INSTALL_DIR}" \
  -DCMAKE_C_COMPILER="${CC_BIN}" \
  -DCMAKE_CXX_COMPILER="${CXX_BIN}"

cmake --build "${ROOT_DIR}/${BUILD_DIR}" -j"${JOBS}"

echo "[rebuild] build finished successfully"
