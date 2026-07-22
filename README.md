# ncnn-ocr

Production-ready Python OCR built on [PaddleOCR PP-OCR Mobile](https://github.com/PaddlePaddle/PaddleOCR) models running on [Tencent NCNN](https://github.com/Tencent/ncnn).

Standalone, publishable to PyPI, and independent of Qefro.

## Features

- **Fast CPU inference** — NCNN-backed det + cls + rec pipeline (same engine as [PaddleOCR-ncnn-CPP](https://github.com/Avafly/PaddleOCR-ncnn-CPP))
- **PP-OCRv5 multilingual** — Tamil, Arabic, Devanagari, Latin, Korean, Cyrillic, and more
- **PP-OCRv6 profiles** — `v6_tiny`, `v6_small`, `v6_medium`
- **Simple API** — `recognize()`, `recognize_text()`, `batch()`
- **Auto model download** — pulls weights from [developerabu/pp-OCRv5-6-ncnn](https://huggingface.co/developerabu/pp-OCRv5-6-ncnn) on first use
- **Arabic post-processing** — optional visual-order → logical-order fix

## Install

### From source (requires native deps)

```bash
# macOS (Homebrew)
brew install cmake opencv ncnn

# Debian/Ubuntu
sudo apt install cmake libopencv-dev libomp-dev

# Build ncnn if not packaged (see https://github.com/Tencent/ncnn)
export NCNN_DIR=$HOME/.local/ncnn   # optional, CMake searches common paths

pip install .
```

### Python dependencies only (no extension)

Pure-Python modules import without the `_ncnn_ocr` extension; `OCR()` raises a clear error until the native module is built.

## Quick start

```python
from ncnn_ocr import OCR

ocr = OCR(lang="ta")
text = ocr.recognize_text("page.png")
print(text)

lines = ocr.recognize("page.png")
for line in lines:
    print(line.text, line.confidence, line.box)
```

One-shot helpers:

```python
from ncnn_ocr import recognize_text

print(recognize_text("scan.jpg", lang="en"))
```

## Supported languages (PP-OCRv5)

| Code | Script / language |
|------|---------------------|
| `en` | English |
| `ta`, `te` | Tamil, Telugu |
| `hi`, `mr`, `ne` | Devanagari |
| `ar`, `fa`, `ur` | Arabic |
| `ko`, `th`, `el` | Korean, Thai, Greek |
| `ru`, `uk`, `bg` | Cyrillic / Slavic |
| `de`, `fr`, `es`, `latin` | Latin |
| `ch`, `zh` | Chinese (generic v5 rec) |

```python
from ncnn_ocr import supported_langs
print(supported_langs())
```

## PP-OCRv6

```python
ocr = OCR(lang="en", profile="v6_tiny")
text = ocr.recognize_text("page.png")
```

Profiles: `v6_tiny`, `v6_small`, `v6_medium`.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NCNN_OCR_MODEL_DIR` | `~/.cache/ncnn-ocr/models` | Local model cache |
| `NCNN_OCR_HF_REPO` | `developerabu/pp-OCRv5-6-ncnn` | HuggingFace model repo |

## Build layout

```
py-ocr-ncnn/
├── src/ncnn_ocr/       # Python package
├── native/
│   ├── bindings.cpp    # pybind11 wrapper
│   └── paddleocr/      # vendored NCNN OCR engine (Apache-2.0)
├── CMakeLists.txt
└── pyproject.toml
```

## Models

On first run, recognition models and keys are downloaded from HuggingFace. Shared PP-OCRv5 detection and angle-classifier weights are fetched from HuggingFace when available, otherwise from the [PaddleOCR-ncnn-CPP v0.3.0 archive](https://github.com/Avafly/PaddleOCR-ncnn-CPP/releases/download/v0.3.0/archive.tar.gz).

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
PYTHONPATH=src pytest -q
```

## Release to PyPI

| Event | Workflow | What happens |
|-------|----------|--------------|
| Push to `main` / PR | `ci.yml` | Tests + builds wheels (artifacts only, **no PyPI**) |
| Push tag `v*` | `release.yml` | Tests + builds wheels + **publishes to PyPI** |

Pushing a version tag triggers `.github/workflows/release.yml`, which:

1. Runs unit tests
2. Builds Linux (`x86_64`) and macOS (`arm64`) wheels plus an sdist
3. Runs an integration OCR smoke test on each wheel
4. Publishes all artifacts to PyPI

### One-time PyPI setup

1. Create a PyPI API token at [pypi.org/manage/account/token](https://pypi.org/manage/account/token/) scoped to `ncnn-ocr`
2. Add it as a GitHub secret:
   - Go to [github.com/qefro-ai/py-ocr-ncnn/settings/secrets/actions](https://github.com/qefro-ai/py-ocr-ncnn/settings/secrets/actions)
   - **New repository secret**
   - Name: **`PYPI_API_TOKEN`**
   - Value: your `pypi-...` token
3. Create GitHub environment **`pypi`** (Settings → Environments) — used by the release workflow

### Cut a release

```bash
git tag v0.1.0
git push origin v0.1.0
```

The workflow reads the tag (`v0.1.0` → version `0.1.0`) and uploads:

- `ncnn_ocr-*-cp312-cp312-linux_x86_64.whl`
- `ncnn_ocr-*-cp312-cp312-macosx_*_arm64.whl`
- `ncnn_ocr-*.tar.gz`

### Manual publish (fallback)

```bash
bash scripts/build_wheel.sh
python -m twine upload dist/*
```

Requires `TWINE_USERNAME=__token__` and `TWINE_PASSWORD` set to a PyPI API token.

## License

Apache-2.0 — see [LICENSE](LICENSE). The vendored PaddleOCR-ncnn-CPP sources retain their original license.

## Credits

- [Avafly/PaddleOCR-ncnn-CPP](https://github.com/Avafly/PaddleOCR-ncnn-CPP) — NCNN OCR engine
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) — PP-OCR models
- [Tencent NCNN](https://github.com/Tencent/ncnn) — inference framework
