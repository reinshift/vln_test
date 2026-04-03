#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${ROOT_DIR}"

echo "[cleanup] removing build/devel/install artifacts from ${ROOT_DIR}"
rm -rf \
  build build_* \
  devel devel_* \
  install install_*

echo "[cleanup] done"
