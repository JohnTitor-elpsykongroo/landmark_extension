# Build the `pointops` CUDA extension of the 3dteethland repository on Windows.
#
# Why this script exists (see ENV_SETUP_REPORT.md for the full story):
#   * The repository requires the `pointops` CUDA extension (kNN / ball query /
#     FPS / stratified attention kernels). It must be compiled on the machine
#     that runs training.
#   * This machine has an RTX 5050 (compute capability sm_120). Only CUDA
#     12.8+ can target sm_120, and the NVIDIA driver here (596.58) has dropped
#     CUDA 12.x compatibility, so a torch built against CUDA 12.8+ is required.
#   * MSVC 14.51 (Visual Studio 18 BuildTools) is newer than CUDA 12.8's
#     official support window, so the MSVC version guard in the CUDA toolkit's
#     `crt/host_config.h` was patched (a `.orig` backup is kept next to it).
#   * `ninja` (used by torch's build backend) does not pick up the MSVC
#     environment by itself, so we import `vcvars64.bat` first.
#
# Usage:  powershell -ExecutionPolicy Bypass -File landmark_extension/env/build_pointops.ps1

$ErrorActionPreference = 'Stop'

$RepoRoot   = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)  # landmark_extension/env -> repo root
$EnvPrefix  = 'D:\WorkSpace\conda\envs\3dteethland'
$Python     = Join-Path $EnvPrefix 'python.exe'
$VCVars     = 'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat'

if (-not (Test-Path $Python))   { throw "python not found: $Python" }
if (-not (Test-Path $VCVars))   { throw "vcvars64.bat not found: $VCVars" }

# Import the MSVC environment (cl.exe, link.exe, INCLUDE, LIB, ...) into this shell
$vcEnv = cmd /c "`"$VCVars`" >nul 2>&1 && set"
foreach ($line in $vcEnv) {
    if ($line -match '^([^=]+)=(.*)$') {
        Set-Item -Path ("env:" + $matches[1]) -Value $matches[2] -ErrorAction SilentlyContinue
    }
}

# Point nvcc at the CUDA toolkit that lives inside the conda environment
$CudaRoot = Join-Path $EnvPrefix 'Library'
$env:CUDA_HOME = $CudaRoot
$env:CUDA_PATH = $CudaRoot
$env:TORCH_CUDA_ARCH_LIST = '12.0'   # sm_120
# Force English diagnostics: torch decodes cl.exe output with the ANSI code page
# (cp936 on this machine), which cannot decode Chinese MSVC messages.
$env:PATH = "$CudaRoot\bin;$EnvPrefix\Scripts;$EnvPrefix;$env:PATH"

Write-Host "python     : $Python"
Write-Host "CUDA_HOME  : $env:CUDA_HOME"
Write-Host "arch list  : $env:TORCH_CUDA_ARCH_LIST"

Push-Location $RepoRoot
try {
    # torch detects the MSVC version with `cl` and decodes its output with the
    # OEM code page; the Chinese-language MSVC here emits UTF-8, which crashes
    # that decode step. Patch the decode arguments for this build only.
    $buildScript = Join-Path $PSScriptRoot 'run_pointops_build.py'
    & $Python $buildScript
    if ($LASTEXITCODE -ne 0) { throw "build_ext failed with exit code $LASTEXITCODE" }

    & $Python -c "import pointops, torch as t; print('pointops ok:', pointops.__file__); x=t.randn(256,3,device='cuda'); idx,cnt=pointops.farthestPointSampling(x, t.tensor([256],device='cuda'), 0.25, 64); print('fps ok:', idx.shape, idx.dtype, idx.device)"
    if ($LASTEXITCODE -ne 0) { throw "pointops import/run failed" }
}
finally {
    Pop-Location
}
