#!/usr/bin/env bash
# Reference command for starting landmark training on the Teeth3DS+ annotations.
#
# DO NOT run this until:
#   1. the environment prerequisites in landmark_extension/README.md are done
#      (MSVC 14.44 toolset + pointops build), and
#   2. the user has explicitly authorised training.
#
# Usage:
#   bash landmark_extension/train_landmarks.sh upper
#   bash landmark_extension/train_landmarks.sh lower

set -euo pipefail

JAW="${1:-upper}"
PYTHON="${PYTHON:-${ENV_PREFIX:-$HOME/envs/3dteethland}/bin/python}"
if [ ! -x "$PYTHON" ]; then
    # fall back to the Windows development environment
    PYTHON='D:/WorkSpace/conda/envs/3dteethland/python.exe'
fi
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$REPO_ROOT"

"$PYTHON" landmark_extension/run_landmark.py \
    --config "landmark_extension/configs/landmark_${JAW}.yaml" \
    --devices "${DEVICES:-1}" \
    "${@:2}"
