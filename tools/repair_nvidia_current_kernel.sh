#!/usr/bin/env bash

set -euo pipefail

KERNEL_REL="$(uname -r)"
COMMON_HEADERS_PKG="linux-headers-${KERNEL_REL%-generic}"
GENERIC_HEADERS_PKG="linux-headers-${KERNEL_REL}"
NVIDIA_DKMS_PKG="nvidia-dkms-570"
DKMS_LOG="/var/lib/dkms/nvidia/570.133.07/build/make.log"
EXTRACT_HEADERS_ONLY=0
CC_BIN="${CC_BIN:-}"
CXX_BIN="${CXX_BIN:-}"
HOSTCC_BIN="${HOSTCC_BIN:-}"
HOSTCXX_BIN="${HOSTCXX_BIN:-}"
APT_OPTS=()
WRAPPER_DIR="/usr/local/bin"
PATH_OVERRIDE="${PATH}"
PIN_SYSTEM_CC_ALTERNATIVE="${PIN_SYSTEM_CC_ALTERNATIVE:-0}"
KERNEL_SOURCE_DIR="${KERNEL_SOURCE_DIR:-}"

if [[ "${1:-}" == "--extract-headers" ]]; then
  EXTRACT_HEADERS_ONLY=1
  shift
fi

if [[ -n "${CC_BIN}" ]] && [[ -z "${CXX_BIN}" ]]; then
  CXX_BIN="${CC_BIN/gcc/g++}"
fi

if [[ -n "${CC_BIN}" ]] && [[ -z "${HOSTCC_BIN}" ]]; then
  HOSTCC_BIN="${CC_BIN}"
fi

if [[ -n "${CXX_BIN}" ]] && [[ -z "${HOSTCXX_BIN}" ]]; then
  HOSTCXX_BIN="${CXX_BIN}"
fi

if [[ "${APT_FORCE_IPV4:-0}" == "1" ]]; then
  APT_OPTS+=("-o" "Acquire::ForceIPv4=true")
fi

if [[ -n "${CC_BIN}" ]]; then
  mkdir -p "${WRAPPER_DIR}"
  ln -sfn "${CC_BIN}" "${WRAPPER_DIR}/cc"
  ln -sfn "${HOSTCC_BIN}" "${WRAPPER_DIR}/hostcc"
  if [[ -n "${CXX_BIN}" ]]; then
    ln -sfn "${CXX_BIN}" "${WRAPPER_DIR}/c++"
  fi
  if [[ -n "${HOSTCXX_BIN}" ]]; then
    ln -sfn "${HOSTCXX_BIN}" "${WRAPPER_DIR}/hostcxx"
  fi

  cc_base="$(basename "${CC_BIN}")"
  cxx_base="$(basename "${CXX_BIN}")"
  if [[ "${cc_base}" =~ ^gcc-([0-9]+)$ ]]; then
    ln -sfn "${CC_BIN}" "${WRAPPER_DIR}/x86_64-linux-gnu-gcc-${BASH_REMATCH[1]}"
  fi
  if [[ "${cxx_base}" =~ ^g\+\+-([0-9]+)$ ]]; then
    ln -sfn "${CXX_BIN}" "${WRAPPER_DIR}/x86_64-linux-gnu-g++-${BASH_REMATCH[1]}"
  fi
  PATH_OVERRIDE="${WRAPPER_DIR}:${PATH}"
fi

if [[ "${PIN_SYSTEM_CC_ALTERNATIVE}" == "1" ]] && [[ -n "${CC_BIN}" ]]; then
  echo "[info] pinning system cc/c++ alternatives to ${CC_BIN} and ${CXX_BIN}"
  update-alternatives --install /usr/bin/cc cc "${CC_BIN}" 130
  update-alternatives --set cc "${CC_BIN}"
  if [[ -n "${CXX_BIN}" ]]; then
    update-alternatives --install /usr/bin/c++ c++ "${CXX_BIN}" 130
    update-alternatives --set c++ "${CXX_BIN}"
  fi
fi

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this script with sudo:" >&2
  echo "  sudo bash tools/repair_nvidia_current_kernel.sh [path-to-generic-headers.deb]" >&2
  exit 1
fi

echo "[info] kernel: ${KERNEL_REL}"

if ! dpkg-query -W -f='${Status}' "${COMMON_HEADERS_PKG}" 2>/dev/null | grep -q "install ok installed"; then
  echo "[error] Missing ${COMMON_HEADERS_PKG}. Install the common headers first." >&2
  exit 1
fi

COMMON_HEADERS_VER="$(dpkg-query -W -f='${Version}' "${COMMON_HEADERS_PKG}")"
echo "[info] common headers version: ${COMMON_HEADERS_VER}"
if [[ -n "${CC_BIN}" ]]; then
  echo "[info] using compiler: ${CC_BIN}"
  [[ -n "${CXX_BIN}" ]] && echo "[info] using C++ compiler: ${CXX_BIN}"
  [[ -n "${HOSTCC_BIN}" ]] && echo "[info] using host compiler: ${HOSTCC_BIN}"
  [[ -n "${HOSTCXX_BIN}" ]] && echo "[info] using host C++ compiler: ${HOSTCXX_BIN}"
  echo "[info] wrapper dir prepended to PATH: ${WRAPPER_DIR}"
fi

GENERIC_STATUS="$(dpkg-query -W -f='${Status}' "${GENERIC_HEADERS_PKG}" 2>/dev/null || true)"
if [[ "${EXTRACT_HEADERS_ONLY}" -eq 1 ]] && [[ -n "${GENERIC_STATUS}" ]] && [[ "${GENERIC_STATUS}" != "install ok installed" ]]; then
  echo "[info] removing broken dpkg state for ${GENERIC_HEADERS_PKG}: ${GENERIC_STATUS}"
  dpkg --remove --force-remove-reinstreq "${GENERIC_HEADERS_PKG}" || true
fi

if ! dpkg-query -W -f='${Status}' "${GENERIC_HEADERS_PKG}" 2>/dev/null | grep -q "install ok installed"; then
  HEADER_DEB="${1:-}"
  if [[ -z "${HEADER_DEB}" ]]; then
    HEADER_DEB="$(find "$(pwd)" "$(dirname "$0")" -maxdepth 1 -type f -name "${GENERIC_HEADERS_PKG}_*.deb" | head -n 1 || true)"
  fi

  if [[ -z "${HEADER_DEB}" ]]; then
    echo "[error] Missing ${GENERIC_HEADERS_PKG}." >&2
    echo "[hint] Download the matching package and rerun this script:" >&2
    echo "  ${GENERIC_HEADERS_PKG}_${COMMON_HEADERS_VER}_amd64.deb" >&2
    exit 1
  fi

  HEADER_DEPS="$(dpkg-deb -f "${HEADER_DEB}" Depends 2>/dev/null || true)"
  if [[ "${EXTRACT_HEADERS_ONLY}" -eq 1 ]]; then
    echo "[info] extracting generic headers payload from ${HEADER_DEB}"
    dpkg-deb -x "${HEADER_DEB}" /
  else
    if grep -Eq 'libc6 \(>= 2\.38\)|libelf1t64|libssl3t64' <<<"${HEADER_DEPS}"; then
      echo "[error] ${HEADER_DEB} depends on newer 24.04-era runtime libraries:" >&2
      echo "        ${HEADER_DEPS}" >&2
      echo "[hint] On Ubuntu 20.04, do not try to upgrade libc just for this." >&2
      echo "[hint] Re-run this script with --extract-headers to unpack the header files without registering the package:" >&2
      echo "       sudo bash tools/repair_nvidia_current_kernel.sh --extract-headers ${HEADER_DEB}" >&2
      exit 1
    fi

    echo "[info] installing generic headers from ${HEADER_DEB}"
    dpkg -i "${HEADER_DEB}"
  fi
fi

if [[ ! -e "/lib/modules/${KERNEL_REL}/build/Makefile" ]]; then
  if [[ -d "/usr/src/${GENERIC_HEADERS_PKG}" ]]; then
    echo "[info] fixing /lib/modules/${KERNEL_REL}/build symlink"
    ln -sfn "/usr/src/${GENERIC_HEADERS_PKG}" "/lib/modules/${KERNEL_REL}/build"
  else
    echo "[error] ${GENERIC_HEADERS_PKG} still did not populate /usr/src" >&2
    exit 1
  fi
fi

if [[ ! -e "/lib/modules/${KERNEL_REL}/source" ]]; then
  if [[ -d "/usr/src/${COMMON_HEADERS_PKG}" ]]; then
    echo "[info] fixing /lib/modules/${KERNEL_REL}/source symlink"
    ln -sfn "/usr/src/${COMMON_HEADERS_PKG}" "/lib/modules/${KERNEL_REL}/source"
  fi
fi

if [[ -n "${KERNEL_SOURCE_DIR}" ]]; then
  if [[ ! -f "${KERNEL_SOURCE_DIR}/.config" ]] && [[ -f "/boot/config-${KERNEL_REL}" ]]; then
    echo "[info] seeding full kernel source config from /boot/config-${KERNEL_REL}"
    cp "/boot/config-${KERNEL_REL}" "${KERNEL_SOURCE_DIR}/.config"
  fi
fi

OBJTOOL_BIN="/usr/src/${GENERIC_HEADERS_PKG}/tools/objtool/objtool"
if [[ -x "${OBJTOOL_BIN}" ]]; then
  if ldd "${OBJTOOL_BIN}" 2>&1 | grep -q 'GLIBC_.*not found'; then
    echo "[info] objtool is linked against a newer glibc; rebuilding it locally"
    if ! dpkg-query -W -f='${Status}' libelf-dev 2>/dev/null | grep -q "install ok installed"; then
      apt-get "${APT_OPTS[@]}" update || true
      env PATH="${PATH_OVERRIDE}" DEBIAN_FRONTEND=noninteractive apt-get "${APT_OPTS[@]}" install -y libelf-dev || true
      if ! dpkg-query -W -f='${Status}' libelf-dev 2>/dev/null | grep -q "install ok installed"; then
        echo "[error] libelf-dev is required to rebuild objtool, but it is still not installed." >&2
        exit 1
      fi
    fi

    OBJTOOL_BUILD_ROOT="/usr/src/${GENERIC_HEADERS_PKG}"
    if [[ -n "${KERNEL_SOURCE_DIR}" ]]; then
      if [[ ! -f "${KERNEL_SOURCE_DIR}/tools/build/Build.include" ]]; then
        echo "[error] KERNEL_SOURCE_DIR is missing tools/build/Build.include: ${KERNEL_SOURCE_DIR}" >&2
        exit 1
      fi
      if ! find "${KERNEL_SOURCE_DIR}/tools/objtool" -maxdepth 1 -name '*.c' -print -quit 2>/dev/null | grep -q .; then
        echo "[error] KERNEL_SOURCE_DIR does not contain full objtool sources: ${KERNEL_SOURCE_DIR}" >&2
        exit 1
      fi
      OBJTOOL_BUILD_ROOT="${KERNEL_SOURCE_DIR}"
      echo "[info] rebuilding objtool from full kernel source: ${KERNEL_SOURCE_DIR}"
    elif [[ ! -f "/usr/src/${COMMON_HEADERS_PKG}/tools/build/Build.include" ]] || \
         ! find "/usr/src/${GENERIC_HEADERS_PKG}/tools/objtool" -maxdepth 1 -name '*.c' -print -quit 2>/dev/null | grep -q .; then
      echo "[error] Installed headers do not include a complete objtool source tree." >&2
      echo "[hint] Download the full Linux ${KERNEL_REL%-generic} source tree and rerun with:" >&2
      echo "       sudo env KERNEL_SOURCE_DIR=/path/to/linux-${KERNEL_REL%%-*} bash tools/repair_nvidia_current_kernel.sh ..." >&2
      exit 1
    fi

    env PATH="${PATH_OVERRIDE}" \
      make -C "${OBJTOOL_BUILD_ROOT}/tools/objtool" clean
    env PATH="${PATH_OVERRIDE}" \
      make -C "${OBJTOOL_BUILD_ROOT}/tools/objtool" \
        HOSTCC="${HOSTCC_BIN:-${CC_BIN:-cc}}" \
        HOSTLD=/usr/bin/ld.bfd \
        HOSTAR=/usr/bin/ar

    if [[ "${OBJTOOL_BUILD_ROOT}" != "/usr/src/${GENERIC_HEADERS_PKG}" ]]; then
      install -m 0755 "${OBJTOOL_BUILD_ROOT}/tools/objtool/objtool" "${OBJTOOL_BIN}"
    fi
    echo "[info] rebuilt objtool:"
    ldd "${OBJTOOL_BIN}" || true
  fi
fi

FIXDEP_BIN="/usr/src/${GENERIC_HEADERS_PKG}/scripts/basic/fixdep"
MODPOST_BIN="/usr/src/${GENERIC_HEADERS_PKG}/scripts/mod/modpost"
GENKSYMS_BIN="/usr/src/${GENERIC_HEADERS_PKG}/scripts/genksyms/genksyms"

if [[ -n "${KERNEL_SOURCE_DIR}" ]]; then
  NEED_REBUILD_SCRIPT_TOOLS=0
  for bin in "${FIXDEP_BIN}" "${MODPOST_BIN}" "${GENKSYMS_BIN}"; do
    if [[ -x "${bin}" ]] && ldd "${bin}" 2>&1 | grep -q 'GLIBC_.*not found'; then
      NEED_REBUILD_SCRIPT_TOOLS=1
      break
    fi
  done

  if [[ "${NEED_REBUILD_SCRIPT_TOOLS}" -eq 1 ]]; then
    echo "[info] rebuilding kernel host tools from full source: fixdep/modpost/genksyms"
    if ! command -v flex >/dev/null 2>&1 || ! command -v bison >/dev/null 2>&1 || \
       ! dpkg-query -W -f='${Status}' libelf-dev 2>/dev/null | grep -q "install ok installed"; then
      apt-get "${APT_OPTS[@]}" update || true
      env PATH="${PATH_OVERRIDE}" DEBIAN_FRONTEND=noninteractive \
        apt-get "${APT_OPTS[@]}" install -y flex bison libelf-dev || true
    fi

    if ! command -v flex >/dev/null 2>&1; then
      echo "[error] flex is required to rebuild scripts/genksyms, but it is still unavailable." >&2
      exit 1
    fi
    if ! command -v bison >/dev/null 2>&1; then
      echo "[error] bison is required to rebuild scripts/genksyms, but it is still unavailable." >&2
      exit 1
    fi

    env PATH="${PATH_OVERRIDE}" \
      make -C "${KERNEL_SOURCE_DIR}" olddefconfig prepare scripts/basic/fixdep scripts/mod/modpost scripts/genksyms/genksyms \
      HOSTCC="${HOSTCC_BIN:-${CC_BIN:-cc}}" \
      HOSTCXX="${HOSTCXX_BIN:-${CXX_BIN:-c++}}" \
      CC="${CC_BIN:-cc}" \
      LD=/usr/bin/ld.bfd

    install -m 0755 "${KERNEL_SOURCE_DIR}/scripts/basic/fixdep" "${FIXDEP_BIN}"
    install -m 0755 "${KERNEL_SOURCE_DIR}/scripts/mod/modpost" "${MODPOST_BIN}"
    install -m 0755 "${KERNEL_SOURCE_DIR}/scripts/genksyms/genksyms" "${GENKSYMS_BIN}"

    echo "[info] rebuilt fixdep/modpost/genksyms:"
    ldd "${FIXDEP_BIN}" || true
    ldd "${MODPOST_BIN}" || true
    ldd "${GENKSYMS_BIN}" || true
  fi
fi

echo "[info] installing DKMS + NVIDIA driver build package"
if ! dpkg-query -W -f='${Status}' dctrl-tools 2>/dev/null | grep -q "install ok installed" || \
   ! dpkg-query -W -f='${Status}' dkms 2>/dev/null | grep -q "install ok installed" || \
   ! dpkg-query -W -f='${Status}' "${NVIDIA_DKMS_PKG}" 2>/dev/null | grep -q "install ok installed"; then
  apt-get "${APT_OPTS[@]}" update || true
  if [[ -n "${CC_BIN}" ]]; then
    env PATH="${PATH_OVERRIDE}" CC="${CC_BIN}" CXX="${CXX_BIN}" HOSTCC="${HOSTCC_BIN}" HOSTCXX="${HOSTCXX_BIN}" DEBIAN_FRONTEND=noninteractive \
      apt-get "${APT_OPTS[@]}" install -y dctrl-tools dkms "${NVIDIA_DKMS_PKG}" || true
  else
    env PATH="${PATH_OVERRIDE}" DEBIAN_FRONTEND=noninteractive apt-get "${APT_OPTS[@]}" install -y dctrl-tools dkms "${NVIDIA_DKMS_PKG}" || true
  fi
fi

if [[ -n "${CC_BIN}" ]]; then
  env PATH="${PATH_OVERRIDE}" CC="${CC_BIN}" CXX="${CXX_BIN}" HOSTCC="${HOSTCC_BIN}" HOSTCXX="${HOSTCXX_BIN}" DEBIAN_FRONTEND=noninteractive dpkg --configure -a || true
else
  env PATH="${PATH_OVERRIDE}" DEBIAN_FRONTEND=noninteractive dpkg --configure -a || true
fi

echo "[info] building NVIDIA module for ${KERNEL_REL}"
if command -v dkms >/dev/null 2>&1; then
  dkms remove -m nvidia -v 570.133.07 --all >/dev/null 2>&1 || true
  dkms add -m nvidia -v 570.133.07 >/dev/null 2>&1 || true
  if [[ -n "${CC_BIN}" ]]; then
    env PATH="${PATH_OVERRIDE}" CC="${CC_BIN}" CXX="${CXX_BIN}" HOSTCC="${HOSTCC_BIN}" HOSTCXX="${HOSTCXX_BIN}" dkms autoinstall -k "${KERNEL_REL}"
  else
    env PATH="${PATH_OVERRIDE}" dkms autoinstall -k "${KERNEL_REL}"
  fi
else
  echo "[error] dkms is still unavailable after package installation." >&2
  exit 1
fi

echo "[info] loading nvidia module"
modprobe nvidia

echo "[info] validating nvidia-smi"
if nvidia-smi; then
  echo "[ok] NVIDIA driver is now communicating with the current kernel."
else
  echo "[error] nvidia-smi still failed." >&2
  if [[ -f "${DKMS_LOG}" ]]; then
    echo "[hint] inspect DKMS build log: ${DKMS_LOG}" >&2
  fi
  exit 1
fi
