#!/usr/bin/env bash
# Installs Blender 4.5 LTS (headless) plus the Python packages the cloak generator needs.
set -euo pipefail

BLENDER_VERSION="4.5.16"
BLENDER_ROOT="/tmp/blender"
BLENDER_DIR="${BLENDER_ROOT}/blender-${BLENDER_VERSION}-linux-x64"

if [ ! -x "${BLENDER_DIR}/blender" ]; then
  mkdir -p "${BLENDER_ROOT}"
  curl -sL -o "${BLENDER_ROOT}/blender.tar.xz" \
    "https://download.blender.org/release/Blender4.5/blender-${BLENDER_VERSION}-linux-x64.tar.xz"
  tar -xf "${BLENDER_ROOT}/blender.tar.xz" -C "${BLENDER_ROOT}"
  rm -f "${BLENDER_ROOT}/blender.tar.xz"
fi

"${BLENDER_DIR}/4.5/python/bin/python3.11" -m ensurepip || true
"${BLENDER_DIR}/4.5/python/bin/python3.11" -m pip install -q shapely triangle scipy pillow
"${BLENDER_DIR}/4.5/python/bin/python3.11" "$(dirname "$0")/check_deps.py"
