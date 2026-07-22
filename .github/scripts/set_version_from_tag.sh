#!/usr/bin/env bash
set -euo pipefail

TAG="${GITHUB_REF_NAME:-}"
if [[ -z "$TAG" ]]; then
  echo "GITHUB_REF_NAME is not set" >&2
  exit 1
fi

VERSION="${TAG#v}"
if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+([a-zA-Z0-9.-]*)?$ ]]; then
  echo "Invalid release tag: $TAG (expected vX.Y.Z)" >&2
  exit 1
fi

echo "Setting package version to $VERSION"

sed -i.bak "s/^version = .*/version = \"$VERSION\"/" pyproject.toml
sed -i.bak "s/^__version__ = .*/__version__ = \"$VERSION\"/" src/ncnn_ocr/__init__.py
rm -f pyproject.toml.bak src/ncnn_ocr/__init__.py.bak

echo "PROJECT_VERSION=$VERSION" >> "${GITHUB_ENV:-/dev/null}"
