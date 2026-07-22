#!/usr/bin/env bash
# Build ncnn-ocr wheels for publishing.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p dist
rm -rf build

if [[ "$(uname -s)" == "Darwin" ]]; then
  export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-/opt/homebrew/opt/ncnn:/opt/homebrew/opt/opencv@4:/opt/homebrew/opt/libomp}"
  export OpenCV_DIR="${OpenCV_DIR:-/opt/homebrew/opt/opencv@4/lib/cmake/opencv4}"
  python3 -m pip install -q build "scikit-build-core>=0.10" "pybind11>=2.11"
  python3 -m build --wheel
  python3 -m build --sdist
  echo "Built macOS wheel + sdist in dist/"
  echo "For Linux x86_64 wheel, run on server:"
  echo "  NCNN_DIR=\$HOME/.local/ncnn CMAKE_PREFIX_PATH=\$NCNN_DIR python3 -m build --wheel"
else
  export NCNN_DIR="${NCNN_DIR:-$HOME/.local/ncnn}"
  export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-$NCNN_DIR}"
  python3 -m pip install -q build "scikit-build-core>=0.10" "pybind11>=2.11"
  python3 -m build --wheel
  echo "Built Linux wheel in dist/"
fi

python3 -m pip install -q twine
python3 -m twine check dist/*
