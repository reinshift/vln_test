#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_GAZEBO_MODEL_DIR="$ROOT/sim/models"
REPO_FUEL_MODEL_DIR="$ROOT/sim/cache/assets/fuel"
mkdir -p "$ROOT/logs/ros"

prepend_unique_paths() {
  local current_var_name="$1"
  shift
  local current_value="${!current_var_name:-}"
  local merged=()
  local path
  local existing

  append_unique() {
    local candidate="$1"
    [[ -z "$candidate" ]] && return
    for existing in "${merged[@]:-}"; do
      [[ "$existing" == "$candidate" ]] && return
    done
    merged+=("$candidate")
  }

  for path in "$@"; do
    append_unique "$path"
  done

  IFS=':' read -r -a existing_paths <<< "$current_value"
  for path in "${existing_paths[@]:-}"; do
    append_unique "$path"
  done

  local joined=""
  for path in "${merged[@]:-}"; do
    if [[ -n "$joined" ]]; then
      joined="${joined}:"
    fi
    joined="${joined}${path}"
  done
  printf '%s' "$joined"
}

if [[ -z "${ROS_LOG_DIR:-}" ]]; then
  export ROS_LOG_DIR="$ROOT/logs/ros"
fi

if [[ -z "${PYTHONNOUSERSITE:-}" ]]; then
  export PYTHONNOUSERSITE=1
fi

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ROOT/.env"
  set +a
fi

if [[ -z "${PX4_AUTOPILOT_DIR:-}" && -d "$ROOT/sim/cache/PX4-Autopilot" ]]; then
  export PX4_AUTOPILOT_DIR="$ROOT/sim/cache/PX4-Autopilot"
fi

if [[ -n "${PX4_AUTOPILOT_DIR:-}" && -d "$PX4_AUTOPILOT_DIR" ]]; then
  PX4_GAZEBO_CLASSIC_DIR="$PX4_AUTOPILOT_DIR/Tools/simulation/gazebo-classic/sitl_gazebo-classic"
  PX4_GAZEBO_PLUGIN_DIR="$PX4_AUTOPILOT_DIR/build/px4_sitl_default/build_gazebo-classic"

  if [[ -d "$PX4_GAZEBO_CLASSIC_DIR" ]]; then
    export ROS_PACKAGE_PATH="$(prepend_unique_paths ROS_PACKAGE_PATH "$PX4_AUTOPILOT_DIR" "$PX4_GAZEBO_CLASSIC_DIR")"
    export GAZEBO_MODEL_PATH="$(prepend_unique_paths GAZEBO_MODEL_PATH "$REPO_GAZEBO_MODEL_DIR" "$REPO_FUEL_MODEL_DIR" "$PX4_GAZEBO_CLASSIC_DIR/models")"
    export GAZEBO_RESOURCE_PATH="$(prepend_unique_paths GAZEBO_RESOURCE_PATH "$PX4_GAZEBO_CLASSIC_DIR")"
    if [[ -d "$PX4_GAZEBO_PLUGIN_DIR" ]]; then
      export GAZEBO_PLUGIN_PATH="$(prepend_unique_paths GAZEBO_PLUGIN_PATH "$PX4_GAZEBO_PLUGIN_DIR")"
    fi
  else
    export ROS_PACKAGE_PATH="$(prepend_unique_paths ROS_PACKAGE_PATH "$PX4_AUTOPILOT_DIR")"
  fi
fi
