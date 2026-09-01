#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="${1:-${REPO_DIR}/config.generated.yaml}"

cd "$REPO_DIR"
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

for stage in candidates approve-auto export validate; do
    "$PYTHON_BIN" -u src/TFM_pipeline_voxceleb_style.py \
        --config "$CONFIG" \
        --stage "$stage"
done

