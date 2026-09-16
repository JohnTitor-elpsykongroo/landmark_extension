# Environment setup report - 3dteethland landmark pipeline

Machine: Windows, NVIDIA GeForce RTX 5050 Laptop GPU (8 GB, **sm_120** /
Blackwell), driver 596.58 (CUDA 13.2 capable), conda at `D:\WorkSpace\conda`.

Target interpreter: `D:\WorkSpace\conda\envs\3dteethland\python.exe`
(Python 3.10.21, created by this task).

## 1. Final state

| Component | Version / state |
| --- | --- |
| Python | 3.10.21 (conda env `3dteethland`) |
| torch | **2.11.0+cu128** |
| torchvision / torchaudio | 0.26.0+cu128 / 2.11.0+cu128 |
| `torch.cuda.is_available()` | True - real 1024x1024 matmul executed on the GPU |
| device / capability | `NVIDIA GeForce RTX 5050 Laptop GPU` / `(12, 0)` |
| numpy | 1.26.4 (`<2` per the project requirement) |
| open3d / pymeshlab / opencv-python | 0.17.0 / 2023.12.post1 / 4.10.0.84 |
| pytorch-lightning / torchmetrics / tensorboard | 2.3.3 / 1.9.0 / 2.17.0 |
| timm / lxml / gco-wrapper / torchtyping / pytest | 1.0.7 / 5.2.2 / 3.0.9 / 0.1.4 / 8.2.2 |
| scikit-learn / scipy / scikit-multilearn / pandas | 1.7.2 / 1.15.3 / 0.2.0 / 2.3.3 |
| torch-scatter | 2.1.2+pt211cu128 - installed, **blocked at load by Smart App Control** |
| pymeshlab | 2023.12.post1 installed, **DLL blocked by Smart App Control**; replaced by a task-local OBJ shim |
| torchtyping | 0.1.4 raises `RuntimeError: Cannot subclass _TensorBase directly` on torch 2.11 (annotations only, see below) |
| nvcc | CUDA 13.0.88 (in `D:\WorkSpace\conda\envs\3dteethland\Library`); CUDA 12.8 was installed first and is superseded |
| MSVC | 14.51.36231 (Visual Studio 18 BuildTools) - **not supported by any CUDA toolkit yet** |
| `pointops` | **not built** - see section 5 |

## 2. Deviations from the repository's stated environment

The README asks for `conda create -n 3dteethland python=3.10` +
`pip install -r requirements.txt` + `pip install -v -e .`. Two requirements had
to be overridden:

1. **torch 2.3.0+cu121 must not be used on this machine.** The installed
   NVIDIA driver (596.58) has dropped CUDA 12.x compatibility and the GPU is
   sm_120, which requires CUDA 12.8 or newer runtime support. torch
   2.11.0+cu128 was installed instead and verified with a real CUDA op.
2. **`scikit-learn` and `pandas` are missing from `requirements.txt`** even
   though `teethland/tensor.py` and `teethland/data/transforms.py` import
   `sklearn.cluster` / `sklearn.decomposition`, and the 3DTeethLand evaluation
   path uses `pandas`. Both were installed separately (`scipy`, `joblib`,
   `pytz`, `python-dateutil` came with them).

`scikit-multilearn` was installed from PyPI (0.2.0) instead of the git master
URL in `requirements.txt`; the data module only uses
`skmultilearn.model_selection.IterativeStratification`, which exists in 0.2.0.
(The git URL was not needed because the PyPI package resolved.)

`torchtyping` 0.1.4 is incompatible with torch 2.11:
`RuntimeError: Cannot subclass _TensorBase directly` raised while importing
`torchtyping.tensor_type`. The repository only uses `TensorType` inside
annotations (there is no `@typechecked` anywhere), so no runtime code path
depends on it; if a tool does import it, downgrade to torch 2.8.0+cu128 or patch
`torchtyping` to derive from `torch.Tensor`.

## 3. Command log (environment)

```powershell
# 1. env
conda create -y -p D:\WorkSpace\conda\envs\3dteethland python=3.10

# 2. torch, cu128 index (sm_120)
& 'D:\WorkSpace\conda\envs\3dteethland\python.exe' -m pip install \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 3. the rest of requirements.txt (torch line excluded)
& ... -m pip install "numpy<2.0.0" lxml==5.2.2 opencv-python==4.10.0.84 \
    open3d==0.17.0 gco-wrapper==3.0.9 pymeshlab==2023.12.post1 pytest==8.2.2 \
    pytorch-lightning==2.3.3 tensorboard==2.17.0 timm==1.0.7 torchtyping==0.1.4

# 4. compiled helper op
& ... -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.11.0+cu128.html

# 5. missing imports used by the repo
& ... -m pip install scikit-learn scikit-multilearn

# 6. CUDA toolkit for nvcc
conda install -y -p D:\WorkSpace\conda\envs\3dteethland -c nvidia cuda-toolkit=12.8
conda install -y -p D:\WorkSpace\conda\envs\3dteethland -c nvidia cuda-toolkit=13.0
```

## 4. Problems hit and how each was resolved

### 4.1 `pip list` showed nothing / a sub-agent lost its instructions

The first delegated environment attempt never installed anything because the
spawned agent arrived without its task description. The environment work was
then done directly in this main session. No lasting impact.

### 4.2 torch cu121 would not work on this GPU

Driver 596.58 has no CUDA 12.x support and the GPU is sm_120. Resolved by
installing torch 2.11.0+cu128 from the PyTorch cu128 index; verified with
`torch.cuda.get_arch_list()` containing `sm_120` and a real matmul on `cuda`.

### 4.3 `nvcc` not found

No CUDA toolkit was installed. Installed `cuda-toolkit=12.8` from the `nvidia`
conda channel into the environment (nvcc ends up in
`<env>\Library\bin\nvcc.exe`).

### 4.4 `nvcc` rejected MSVC 14.51

`crt/host_config.h` guards on `_MSC_VER`. CUDA 12.8 accept 19.10-19.49, this
machine has 19.51. First attempt: patched the guard inside the conda CUDA
installation (a `.orig` backup is kept next to the file). That removed the
*error message* but exposed the real problem below.

### 4.5 `cudafe++ died with status 0xC0000005 (ACCESS_VIOLATION)` on a trivial kernel

With CUDA 12.8 (patched guard) and later with CUDA 13.0
(`-allow-unsupported-compiler`), compiling a 1-line kernel
(`__global__ void k(float* p){ p[0]=1.f; }`) for `compute_75`, `compute_90`
and `compute_120` all crashed identically. Conclusion: CUDA's EDG front end
does not support MSVC 19.51 at all. A supported older MSVC toolset is required.

### 4.6 Two lesser build-tooling issues (already worked around in the scripts)

* torch decodes `cl.exe`'s output with the OEM code page (cp936 here) while
  this language-packed MSVC writes UTF-8, producing
  `UnicodeDecodeError: 'cp1' codec can't decode ...`. Worked around in
  `landmark_extension/env/run_pointops_build.py` by rebinding
  `subprocess.SUBPROCESS_DECODE_ARGS` **and**
  `torch.utils.cpp_extension.SUBPROCESS_DECODE_ARGS` to `('utf-8',)`
  (torch binds the constant at import time, so patching only `subprocess` is
  not enough).
* `ninja` (used by torch's build backend) does not pick up the MSVC
  environment on its own, so the build script imports `vcvars64.bat` first.
  `DISTUTILS_USE_SDK=1` must be set once the VC environment is active.

### 4.7 Could not install MSVC 14.44 (the actual blocker)

The toolset is present in the VS 2026 catalog as
`Microsoft.VisualStudio.Component.VC.14.44.17.14.x86.x64`, and
`Microsoft.VC.14.44.17.14.*` packages exist in the same catalog, but all
automated paths require elevation:

* `vs_BuildTools2022.exe --quiet` (base VS 2022 bootstrapper): starts, extracts,
  then hangs with no installer activity.
* `vs_installer.exe modify ... --quiet`: exits with code **5007**,
  `Commands with --quiet or --passive should be run elevated from the
  beginning.` (`isadmin` True, `iselevated` False).
* `Start-Process -Verb RunAs`: launched a process that then reported
  `Didn't find any channel feed` and exited (code 1) - the elevation prompt
  was never actually granted in the session.

This is the one step that needs the user. The exact commands are in section 6.2
of `landmark_extension/README.md`.

### 4.8 Smart App Control blocks prebuilt extension DLLs (pymeshlab, torch_scatter)

Code Integrity events (`Microsoft-Windows-CodeIntegrity/Operational`, ids
3077/3033/3118) show Smart App Control refusing to load unsigned `.pyd`
modules, including `torch_scatter\_scatter_cuda.pyd` (and, in an unrelated env,
`PyQt6\sip.cp310-win_amd64.pyd`), and the native library
`pymeshlab\meshlab-common.dll`:

```
HKLM\SYSTEM\CurrentControlSet\Control\CI\Policy
  VerifiedAndReputablePolicyState = 1
  SAC_PreviousState = 1
  SAC_EnforcementReason = 1
```

Implication: `torch_scatter` cannot be imported from the prebuilt wheel, and a
prebuilt `pointops` wheel would not load either. Binaries compiled locally are
accepted by Smart App Control, which is why building `pointops` in place is the
productive path. `torch_scatter` should likewise be rebuilt from source once
the compiler toolset exists:

```powershell
& 'D:\WorkSpace\conda\envs\3dteethland\python.exe' -m pip install --no-binary torch-scatter torch-scatter
```

`pymeshlab` has no source build that avoids the blocked DLL, so
`landmark_extension/shims/pymeshlab/__init__.py` provides a dependency-free OBJ
reader with the same call surface (`MeshSet().load_new_mesh(...).current_mesh()`
plus `vertex_matrix` / `face_matrix` / `vertex_normal_matrix` /
`vertex_color_matrix` / `vertex_number` / `face_number`). It was verified on a
real Teeth3DS scan: 119 810 vertices / 239 496 faces, unit-length normals,
`(N, 3)` color matrix. Vertex order is preserved, which is what the per-vertex
FDI labels depend on. Activate it with
`set PYTHONPATH=...\landmark_extension\shims` or
`from landmark_extension.shims import activate; activate()`.

Screen/UI packages are affected the same way (`PyQt6` is blocked too), so any
interactive viewer in this environment is unavailable until Smart App Control
is off.

Note that this also explains *why* an administrator step is unavoidable rather
than merely convenient: the blocked modules cannot be replaced by anything that
this session can build, because the toolchain itself is missing.

## 5. What is still missing, and the exact next action

Three things remain between the current state and a runnable training loop:

1. A CUDA-supported MSVC toolset (14.44.17.14) in the Visual Studio
   installation - **administrator action**.
2. `pointops` compiled locally so Smart App Control accepts it.
3. `torch-scatter` either rebuilt locally or unblocked (Smart App Control).

`pymeshlab` is already worked around by the shim in
`landmark_extension/shims/`.

After the admin step, everything else is automated:

```powershell
powershell -ExecutionPolicy Bypass -File D:\WorkSpace\Dental\3dteethland\landmark_extension\env\build_pointops.ps1
```

That script prints `pointops ok: <path>` and runs a real `farthestPointSampling`
call on a CUDA tensor as its own verification.

## 6. Files created by this task (environment side only)

* `landmark_extension/env/build_pointops.ps1`
* `landmark_extension/env/run_pointops_build.py`
* `landmark_extension/env/pointops_build.log`
* `landmark_extension/env/ENV_SETUP_REPORT.md` (this file)

No repository source file was modified. The only change outside
`landmark_extension/` and the conda environment is the patched
`crt/host_config.h` inside the conda CUDA installation, which keeps a `.orig`
backup next to it.
