#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Reproducible environment setup for the 3dteethland landmark pipeline on the
# Linux + RTX 5090 target machine.
#
# Target: Linux x86_64, NVIDIA RTX 5090 (compute capability sm_120),
#         driver CUDA 13.0, 128 GB RAM.
#
# There is NO conda requirement. The repository imports only pip packages, so
# a plain `python -m venv` is enough. conda is used here only as a convenient
# supplier of `nvcc` for building the pointops CUDA extension; if the machine
# already has a CUDA toolkit, or if you let pip install the nvcc wheel, no
# conda is involved at all.
#
# Everything the Windows laptop needed workarounds for (MSVC host-compiler
# version guard, Smart App Control blocking unsigned DLLs) does not exist
# here, so this script is a plain, linear install.
#
# Usage:
#   ENV_KIND=venv    bash landmark_extension/env/setup_linux.sh   # default: plain .venv
#   ENV_KIND=conda   bash landmark_extension/env/setup_linux.sh   # conda env instead
#   CUDA_INDEX=https://download.pytorch.org/whl/cu130 bash .../setup_linux.sh
#   SKIP_TOOLKIT=1 bash .../setup_linux.sh    # nvcc already on PATH / in the env
# ---------------------------------------------------------------------------

set -euo pipefail

ENV_KIND="${ENV_KIND:-venv}"
ENV_PREFIX="${ENV_PREFIX:-$HOME/envs/3dteethland}"
PY_VERSION="${PY_VERSION:-3.10}"
CUDA_INDEX="${CUDA_INDEX:-https://download.pytorch.org/whl/cu128}"
NVCC_WHEEL="${NVCC_WHEEL:-nvidia-cuda-nvcc-cu12==12.8.93}"
PYG_CUDA="${PYG_CUDA:-cu128}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"   # sm_120 = RTX 5090
SKIP_TOOLKIT="${SKIP_TOOLKIT:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$ENV_PREFIX/bin/python"

echo "==================================================================="
echo " 3dteethland landmark environment"
echo "   repo        : $REPO_ROOT"
echo "   env kind    : $ENV_KIND"
echo "   env prefix  : $ENV_PREFIX"
echo "   torch index : $CUDA_INDEX"
echo "   arch list   : $TORCH_CUDA_ARCH_LIST"
echo "==================================================================="

# --- 1. create the environment (venv by default, conda optional) ----------
if [ -x "$PY" ]; then
    echo "[1/7] reusing existing environment at $ENV_PREFIX"
elif [ "$ENV_KIND" = "conda" ]; then
    echo "[1/7] creating conda environment at $ENV_PREFIX"
    if command -v mamba >/dev/null 2>&1; then
        mamba create -y -p "$ENV_PREFIX" "python=$PY_VERSION"
    elif command -v conda >/dev/null 2>&1; then
        conda create -y -p "$ENV_PREFIX" "python=$PY_VERSION"
    else
        echo "ERROR: ENV_KIND=conda but neither mamba nor conda is on PATH" >&2
        exit 1
    fi
else
    echo "[1/7] creating virtualenv at $ENV_PREFIX"
    BASE_PY="${BASE_PYTHON:-python$PY_VERSION}"
    if ! command -v "$BASE_PY" >/dev/null 2>&1; then
        BASE_PY="$(command -v python3 || true)"
    fi
    if [ -z "$BASE_PY" ]; then
        echo "ERROR: no python3 on PATH; install python$PY_VERSION first" >&2
        exit 1
    fi
    echo "      base interpreter: $BASE_PY ($("$BASE_PY" -V 2>&1))"
    "$BASE_PY" -m venv "$ENV_PREFIX"
fi

"$PY" -m pip install --upgrade pip wheel setuptools

# --- 2. provide nvcc for the pointops extension ----------------------------
NVCC=""
if [ "$SKIP_TOOLKIT" = "1" ]; then
    echo "[2/7] skipping nvcc provisioning (SKIP_TOOLKIT=1)"
elif [ -x "$ENV_PREFIX/bin/nvcc" ]; then
    echo "[2/7] nvcc already present in the environment"
    NVCC="$ENV_PREFIX/bin/nvcc"
elif command -v nvcc >/dev/null 2>&1; then
    echo "[2/7] using the system nvcc: $(command -v nvcc)"
    NVCC="$(command -v nvcc)"
elif [ "$ENV_KIND" = "conda" ]; then
    echo "[2/7] installing CUDA toolkit 12.8 through conda"
    conda install -y -p "$ENV_PREFIX" -c nvidia cuda-toolkit=12.8
else
    echo "[2/7] installing $NVCC_WHEEL (nvcc via pip, no conda needed)"
    "$PY" -m pip install --progress-bar off "$NVCC_WHEEL"
fi
if [ -x "$ENV_PREFIX/bin/nvcc" ]; then
    NVCC="$ENV_PREFIX/bin/nvcc"
elif [ -z "$NVCC" ] && [ -x "$ENV_PREFIX/Library/bin/nvcc" ]; then
    NVCC="$ENV_PREFIX/Library/bin/nvcc"   # windows layout, harmless on linux
fi
if [ -n "$NVCC" ]; then
    echo "      nvcc: $("$NVCC" --version 2>/dev/null | tail -1 || echo 'NOT FOUND')"
fi

# gcc must be within the range CUDA 12.8 supports (<= 13)
if command -v gcc >/dev/null 2>&1; then
    echo "      gcc : $(gcc -dumpfullversion -dumpversion)  (CUDA 12.8 supports gcc <= 13)"
fi

# --- 3. torch built against a CUDA runtime that supports sm_120 ------------
echo "[3/7] installing torch/torchvision/torchaudio from $CUDA_INDEX"
"$PY" -m pip install --progress-bar off \
    torch torchvision torchaudio --index-url "$CUDA_INDEX"

# --- 4. the rest of the project requirements ------------------------------
# requirements.txt pins torch==2.3.0 and an --extra-index-url of cu121; both are
# intentionally overridden above.
echo "[4/7] installing project requirements"
"$PY" -m pip install --progress-bar off "numpy<2.0.0" lxml==5.2.2 \
    opencv-python==4.10.0.84 open3d==0.17.0 gco-wrapper==3.0.9 \
    pymeshlab==2023.12.post1 pytest==8.2.2 pytorch-lightning==2.3.3 \
    tensorboard==2.17.0 timm==1.0.7 torchtyping==0.1.4

# Imports the repository actually needs but does not declare in
# requirements.txt: sklearn (teethland/tensor.py, teethland/data/transforms.py)
# and pandas (the 3DTeethLand evaluation path).
echo "      + undeclared imports: scikit-learn, scikit-multilearn, pandas"
"$PY" -m pip install --progress-bar off scikit-learn pandas scikit-multilearn

# --- 5. torch-scatter -----------------------------------------------------
TORCH_VER="$("$PY" -c 'import torch; print(torch.__version__.split("+")[0])')"
echo "[5/7] installing torch-scatter for torch ${TORCH_VER}+${PYG_CUDA}"
if ! "$PY" -m pip install --progress-bar off torch-scatter \
        -f "https://data.pyg.org/whl/torch-${TORCH_VER}+${PYG_CUDA}.html"; then
    echo "      no wheel on the PyG index; building torch-scatter from source"
    "$PY" -m pip install --progress-bar off --no-build-isolation torch-scatter
fi

# pointops needs a working host CUDA compiler; fail early with a clear message
if [ ! -x "$NVCC" ]; then
    echo "ERROR: nvcc not found at $NVCC." >&2
    echo "       Install a CUDA toolkit (conda -c nvidia cuda-toolkit=12.8) or" >&2
    echo "       re-run with SKIP_TOOLKIT=1 only if nvcc really is on PATH." >&2
    exit 1
fi

# --- 6. pointops (kNN / ball query / FPS / stratified CRPE attention) ------
echo "[6/7] building the pointops CUDA extension (this takes a while)"
# For a pip-installed nvcc the CUDA headers land in <env>/include (or under
# nvidia/cuda_nvcc/include), for a toolkit install they live next to bin/.
if [ -z "$CUDA_HOME" ]; then
    if [ -f "$ENV_PREFIX/include/cuda_runtime.h" ]; then
        CUDA_HOME="$ENV_PREFIX"
    else
        CUDA_HOME="$(dirname "$(dirname "$NVCC")")"
    fi
fi
export CUDA_PATH="$CUDA_HOME"
export CPATH="$CUDA_HOME/include:${CPATH:-}"
export TORCH_CUDA_ARCH_LIST
export PATH="$CUDA_HOME/bin:$ENV_PREFIX/bin:$PATH"
echo "      CUDA_HOME=$CUDA_HOME"
cd "$REPO_ROOT"
if ! "$PY" setup.py build_ext --inplace; then
    echo "      retrying as a proper editable install"
    "$PY" -m pip install -v -e .
fi

# --- 7. verify ------------------------------------------------------------
echo "[7/7] verifying"
cd "$REPO_ROOT"
"$PY" - <<'PYCHECK'
import torch
print(f"torch            : {torch.__version__} (cuda runtime {torch.version.cuda})")
print(f"cuda available   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device           : {torch.cuda.get_device_name(0)}")
    print(f"capability       : {torch.cuda.get_device_capability(0)}")
    print(f"arch list        : {torch.cuda.get_arch_list()}")
    a = torch.randn(2048, 2048, device='cuda')
    b = torch.randn(2048, 2048, device='cuda')
    print(f"matmul ok        : {(a @ b).sum().item():.1f}")
    free, total = torch.cuda.mem_get_info()
    print(f"vram             : {total / 1024**3:.1f} GB total")

import numpy, open3d, pytorch_lightning, torch_scatter, pointops  # noqa: F401
print(f"numpy            : {numpy.__version__}")
print(f"pytorch-lightning: {pytorch_lightning.__version__}")
print(f"torch_scatter    : {torch_scatter.__version__}")
print(f"pointops         : {pointops.__file__}")

x = torch.randn(4096, 3, device='cuda')
idx, counts = pointops.farthestPointSampling(
    x, torch.tensor([4096], device='cuda'), 0.25, 1024,
)
print(f"pointops fps ok  : {tuple(idx.shape)} {idx.dtype} {idx.device}")
PYCHECK

cat <<EOF

===================================================================
 Environment ready: $PY

 Next steps (nothing is trained by this script):
   cd $REPO_ROOT
   "$PY" landmark_extension/validation/smoke_test_landmark_pipeline.py \\
       --jaw upper --batch-size 4

 Then, after you have authorised training:
   "$PY" landmark_extension/run_landmark.py \\
       --config landmark_extension/configs/landmark_upper.yaml --devices 1
===================================================================
EOF
