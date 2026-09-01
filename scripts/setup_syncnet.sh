#!/usr/bin/env bash

# Download the official SyncNet implementation and pretrained weights.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SYNCNET_DIR="${SYNCNET_DIR:-${REPO_DIR}/models/syncnet}"
SYNCNET_CODE_DIR="${SYNCNET_DIR}/syncnet_python"
SYNCNET_MODEL="${SYNCNET_DIR}/syncnet_v2.model"

mkdir -p "$SYNCNET_DIR"

"$PYTHON_BIN" -m pip install python_speech_features

if [ ! -f "${SYNCNET_CODE_DIR}/SyncNetInstance.py" ]; then
    git clone --depth 1 https://github.com/joonson/syncnet_python.git "$SYNCNET_CODE_DIR"
fi

if [ ! -s "$SYNCNET_MODEL" ]; then
    curl --fail --location --retry 3 \
        https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model \
        --output "$SYNCNET_MODEL"
fi

"$PYTHON_BIN" - "$SYNCNET_CODE_DIR" "$SYNCNET_MODEL" <<'PY'
import sys

sys.path.insert(0, sys.argv[1])
from SyncNetInstance import SyncNetInstance

model = SyncNetInstance(device="cpu")
model.loadParameters(sys.argv[2])
print("SyncNet was installed and loaded successfully.")
PY
