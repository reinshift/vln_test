#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build_local}"
DEVEL_DIR="${DEVEL_DIR:-$ROOT/devel_local}"
INSTALL_DIR="${INSTALL_DIR:-$ROOT/install_local}"
JOBS="${JOBS:-2}"
CC_BIN="${CC_BIN:-/usr/bin/gcc}"
CXX_BIN="${CXX_BIN:-/usr/bin/g++}"

source /opt/ros/noetic/setup.bash

cmake \
  -S "$ROOT/src" \
  -B "$BUILD_DIR" \
  -DCATKIN_DEVEL_PREFIX="$DEVEL_DIR" \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
  -DCMAKE_C_COMPILER="$CC_BIN" \
  -DCMAKE_CXX_COMPILER="$CXX_BIN"

cmake --build "$BUILD_DIR" -j"$JOBS"

echo "[rebuild] build finished"
