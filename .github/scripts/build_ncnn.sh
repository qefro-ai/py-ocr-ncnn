#!/usr/bin/env bash
set -euo pipefail

NCNN_TAG="${NCNN_TAG:-20241226}"
NCNN_DIR="${NCNN_DIR:-$HOME/.local/ncnn}"
SRC_DIR="${RUNNER_TEMP:-/tmp}/ncnn-src"
BUILD_DIR="${RUNNER_TEMP:-/tmp}/ncnn-build"

if [[ -f "${NCNN_DIR}/lib/cmake/ncnn/ncnnConfig.cmake" ]]; then
  echo "ncnn already installed at ${NCNN_DIR}"
  exit 0
fi

git clone --depth 1 --branch "${NCNN_TAG}" https://github.com/Tencent/ncnn.git "${SRC_DIR}"
cmake -S "${SRC_DIR}" -B "${BUILD_DIR}" \
  -DNCNN_VULKAN=OFF \
  -DNCNN_BUILD_TOOLS=OFF \
  -DNCNN_BUILD_EXAMPLES=OFF \
  -DCMAKE_INSTALL_PREFIX="${NCNN_DIR}"
cmake --build "${BUILD_DIR}" -j"$(nproc)"
cmake --install "${BUILD_DIR}"

echo "NCNN_DIR=${NCNN_DIR}" >> "${GITHUB_ENV:-/dev/null}"
