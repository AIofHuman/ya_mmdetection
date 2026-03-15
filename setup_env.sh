#!/usr/bin/env bash

set -euo pipefail

if [[ -n "${PYTHON_BIN:-}" ]]; then
  SELECTED_PYTHON="${PYTHON_BIN}"
else
  for candidate in python3.10 python3.11; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      SELECTED_PYTHON="${candidate}"
      break
    fi
  done
fi

if [[ -z "${SELECTED_PYTHON:-}" ]]; then
  cat <<'EOF'
No supported Python found in PATH.
Install Python 3.10 or 3.11, then rerun:
  PYTHON_BIN=python3.10 bash setup_env.sh

Examples on macOS:
  brew install python@3.10
  brew install python@3.11
EOF
  exit 1
fi

VENV_DIR="${VENV_DIR:-.venv}"

echo "Using python: ${SELECTED_PYTHON}"
"${SELECTED_PYTHON}" -m venv "${VENV_DIR}"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip wheel setuptools
python -m pip install "setuptools<81"

echo "Install PyTorch for your CUDA version before running this script fully."
echo "If CUDA 11.8 is installed, run:"
echo "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"

python -m pip install -r requirements-gpu.txt
python -m mim install "mmcv>=2.0.0,<2.2.0"

python - <<'PY'
from pathlib import Path
import os
import mmdet

project_root = Path.cwd()
tools_src = Path(mmdet.__file__).resolve().parent / ".mim" / "tools"
tools_dst = project_root / "tools"

if tools_dst.exists() or tools_dst.is_symlink():
    tools_dst.unlink()

os.symlink(tools_src, tools_dst)
print(f"Linked tools -> {tools_src}")
PY

mkdir -p artifacts/fcos artifacts/yolo artifacts/inference/fcos artifacts/inference/yolo artifacts/videos artifacts/metrics checkpoints

echo
echo "Environment is ready."
echo "Activate with: source ${VENV_DIR}/bin/activate"
echo "Then open Jupyter in this directory."
