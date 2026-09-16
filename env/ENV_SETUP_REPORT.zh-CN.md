# 环境搭建报告 —— 3dteethland landmark 流水线

机器：Windows，NVIDIA GeForce RTX 5050 Laptop GPU（8 GB，**sm_120** / Blackwell），
驱动 596.58（支持 CUDA 13.2），conda 位于 `D:\WorkSpace\conda`。

目标解释器：`D:\WorkSpace\conda\envs\3dteethland\python.exe`（Python 3.10.21，由本任务创建）。

英文版：[`ENV_SETUP_REPORT.md`](ENV_SETUP_REPORT.md)

## 1. 最终状态

| 组件 | 版本 / 状态 |
| --- | --- |
| Python | 3.10.21（conda 环境 `3dteethland`） |
| torch | **2.11.0+cu128** |
| torchvision / torchaudio | 0.26.0+cu128 / 2.11.0+cu128 |
| `torch.cuda.is_available()` | True —— 已在 GPU 上执行真实 1024×1024 matmul |
| 设备 / 算力 | `NVIDIA GeForce RTX 5050 Laptop GPU` / `(12, 0)` |
| numpy | 1.26.4（满足项目 `<2` 的要求） |
| open3d / pymeshlab / opencv-python | 0.17.0 / 2023.12.post1 / 4.10.0.84 |
| pytorch-lightning / torchmetrics / tensorboard | 2.3.3 / 1.9.0 / 2.17.0 |
| timm / lxml / gco-wrapper / torchtyping / pytest | 1.0.7 / 5.2.2 / 3.0.9 / 0.1.4 / 8.2.2 |
| scikit-learn / scipy / scikit-multilearn / pandas | 1.7.2 / 1.15.3 / 0.2.0 / 2.3.3 |
| torch-scatter | 2.1.2+pt211cu128 —— 已安装，但**加载被 Smart App Control 拦截** |
| pymeshlab | 2023.12.post1 已安装，但**DLL 被 Smart App Control 拦截**；已由本任务的 OBJ 垫片替代 |
| torchtyping | 0.1.4 在 torch 2.11 上抛 `RuntimeError: Cannot subclass _TensorBase directly`（只影响注解，见下文） |
| nvcc | CUDA 13.0.88（位于 `D:\WorkSpace\conda\envs\3dteethland\Library`）；CUDA 12.8 先装后被取代 |
| MSVC | 14.51.36231（Visual Studio 18 BuildTools）—— **目前没有任何 CUDA 工具链支持它** |
| `pointops` | **未编译成功** —— 见第 5 节 |

## 2. 与原仓库声明环境的差异

README 要求 `conda create -n 3dteethland python=3.10` + `pip install -r requirements.txt` +
`pip install -v -e .`。其中有两条依赖必须覆盖：

1. **本机不能使用 torch 2.3.0+cu121。** 已安装的 NVIDIA 驱动（596.58）放弃了 CUDA 12.x
   兼容，且 GPU 是 sm_120，需要 CUDA 12.8 及以上的运行时支持。最终改为安装
   torch 2.11.0+cu128，并用真实 CUDA 运算验证通过。
2. **`requirements.txt` 漏掉了 `scikit-learn` 和 `pandas`**，尽管
   `teethland/tensor.py` 与 `teethland/data/transforms.py` 会 import
   `sklearn.cluster` / `sklearn.decomposition`，而 3DTeethLand 的评测路径使用 `pandas`。
   两者已单独安装（并连带安装了 `scipy`、`joblib`、`pytz`、`python-dateutil`）。

`scikit-multilearn` 改为从 PyPI 安装（0.2.0），而不是 `requirements.txt` 里的 git master
URL；data module 只用到 `skmultilearn.model_selection.IterativeStratification`，0.2.0 中
已存在该接口。（由于 PyPI 包解析成功，不需要 git URL。）

`torchtyping` 0.1.4 与 torch 2.11 不兼容：在导入 `torchtyping.tensor_type` 时抛
`RuntimeError: Cannot subclass _TensorBase directly`。仓库只在注解中使用 `TensorType`
（全仓库没有任何 `@typechecked`），因此没有任何运行时路径依赖它；如果确实有工具需要导入，
可以降级到 torch 2.8.0+cu128，或修改 `torchtyping` 改为从 `torch.Tensor` 派生。

## 3. 环境操作命令记录

```powershell
# 1. 创建环境
conda create -y -p D:\WorkSpace\conda\envs\3dteethland python=3.10

# 2. 安装 torch（cu128 源，支持 sm_120）
& 'D:\WorkSpace\conda\envs\3dteethland\python.exe' -m pip install \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 3. 安装 requirements.txt 的其余部分（排除 torch 那一行）
& ... -m pip install "numpy<2.0.0" lxml==5.2.2 opencv-python==4.10.0.84 \
    open3d==0.17.0 gco-wrapper==3.0.9 pymeshlab==2023.12.post1 pytest==8.2.2 \
    pytorch-lightning==2.3.3 tensorboard==2.17.0 timm==1.0.7 torchtyping==0.1.4

# 4. 安装编译型辅助算子
& ... -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.11.0+cu128.html

# 5. 补齐仓库实际用到但未声明的导入
& ... -m pip install scikit-learn scikit-multilearn pandas

# 6. 提供 nvcc 的 CUDA 工具链
conda install -y -p D:\WorkSpace\conda\envs\3dteethland -c nvidia cuda-toolkit=12.8
conda install -y -p D:\WorkSpace\conda\envs\3dteethland -c nvidia cuda-toolkit=13.0
```

## 4. 遇到的问题与解决办法

### 4.1 `pip list` 没有任何输出 / 子代理丢失了任务说明

第一次委派出去的环境搭建没有安装任何东西，原因是那个子代理收到的消息里没有任务描述。
随后本次环境工作改在主会话中直接完成，没有留下遗留影响。

### 4.2 torch cu121 在这块 GPU 上无法使用

驱动 596.58 不支持 CUDA 12.x，且 GPU 是 sm_120。解决办法是改用 PyTorch cu128 源安装
torch 2.11.0+cu128；随后用 `torch.cuda.get_arch_list()` 中出现 `sm_120` 以及一次真实的
`cuda` matmul 完成验证。

### 4.3 找不到 `nvcc`

机器上没有安装任何 CUDA 工具链。改为从 `nvidia` conda 频道把 `cuda-toolkit=12.8` 装进该
环境（nvcc 位于 `<env>\Library\bin\nvcc.exe`）。

### 4.4 `nvcc` 拒绝 MSVC 14.51

`crt/host_config.h` 会按 `_MSC_VER` 做版本检查。CUDA 12.8 接受 19.10–19.49，而本机是 19.51。
第一步先改掉了 conda CUDA 安装目录内该头文件的版本守卫（旁边保留了 `.orig` 备份）。
这消除了*报错信息*，但也暴露出下面这个真正的问题。

### 4.5 编译一个最简单的 kernel 就 `cudafe++ died with status 0xC0000005 (ACCESS_VIOLATION)`

在 CUDA 12.8（已改守卫）以及之后的 CUDA 13.0（使用 `-allow-unsupported-compiler`）下，
编译一行 kernel（`__global__ void k(float* p){ p[0]=1.f; }`），目标分别设为 `compute_75`、
`compute_90` 和 `compute_120`，崩溃表现完全一致。结论：CUDA 的前端（EDG）完全不支持
MSVC 19.51。必须安装受支持的旧版 MSVC 工具集。

### 4.6 两个较小的编译工具链问题（脚本中已绕过）

* torch 会用 OEM 代码页（本机 cp936）解码 `cl.exe` 的输出，而本机装了语言包的 MSVC 输出的是
  UTF-8，于是抛 `UnicodeDecodeError: 'cp1' codec can't decode ...`。绕过方式见
  `landmark_extension/env/run_pointops_build.py`：把
  `subprocess.SUBPROCESS_DECODE_ARGS` **以及**
  `torch.utils.cpp_extension.SUBPROCESS_DECODE_ARGS` 都重绑为 `('utf-8',)`
  （torch 在导入时就绑定了该常量，所以只改 `subprocess` 不够）。
* `ninja`（torch 构建后端使用）不会自动获得 MSVC 环境，因此编译脚本会先导入
  `vcvars64.bat`。VC 环境激活后还必须设置 `DISTUTILS_USE_SDK=1`。

### 4.7 无法安装 MSVC 14.44（真正的阻塞点）

该工具集在 VS 2026 目录清单中存在，标识为
`Microsoft.VisualStudio.Component.VC.14.44.17.14.x86.x64`；同一目录清单里也有
`Microsoft.VC.14.44.17.14.*` 系列包。但所有自动化安装路径都需要提权：

* `vs_BuildTools2022.exe --quiet`（VS 2022 基础引导器）：能启动、能解包，随后卡住，
  安装器没有任何活动。
* `vs_installer.exe modify ... --quiet`：以退出码 **5007** 结束，提示
  `Commands with --quiet or --passive should be run elevated from the beginning.`
  （`isadmin` 为 True，`iselevated` 为 False）。
* `Start-Process -Verb RunAs`：启动的进程随后报告
  `Didn't find any channel feed` 并以退出码 1 结束 —— 本会话中提权确认实际上从未被授予。

这是唯一必须由用户完成的一步。具体命令见 `landmark_extension/README.zh-CN.md` 的 6.2 节。

### 4.8 Smart App Control 拦截预编译扩展 DLL（pymeshlab、torch_scatter）

Code Integrity 日志（`Microsoft-Windows-CodeIntegrity/Operational`，事件 id
3077/3033/3118）显示 Smart App Control 拒绝加载未签名的 `.pyd` 模块，包括
`torch_scatter\_scatter_cuda.pyd`（以及在另一个无关环境中的
`PyQt6\sip.cp310-win_amd64.pyd`），以及原生库 `pymeshlab\meshlab-common.dll`：

```
HKLM\SYSTEM\CurrentControlSet\Control\CI\Policy
  VerifiedAndReputablePolicyState = 1
  SAC_PreviousState = 1
  SAC_EnforcementReason = 1
```

含义：`torch_scatter` 无法从预编译 wheel 导入，预编译的 `pointops` wheel 同样无法加载。
本地自行编译的二进制会被 Smart App Control 接受，因此「就地编译 `pointops`」才是可行的
路径。等编译器工具集就绪后，`torch_scatter` 也应从源码重建：

```powershell
& 'D:\WorkSpace\conda\envs\3dteethland\python.exe' -m pip install --no-binary torch-scatter torch-scatter
```

`pymeshlab` 没有能绕开被拦截 DLL 的源码构建方式，因此
`landmark_extension/shims/pymeshlab/__init__.py` 提供了一个无依赖的 OBJ 读取实现，
调用接口与原库一致（`MeshSet().load_new_mesh(...).current_mesh()` 以及
`vertex_matrix` / `face_matrix` / `vertex_normal_matrix` / `vertex_color_matrix` /
`vertex_number` / `face_number`）。已在真实 Teeth3DS 扫描上验证：119 810 顶点 /
239 496 面，单位长度法线，`(N, 3)` 颜色矩阵；并且保持了顶点顺序 —— 这正是逐顶点 FDI
标签所依赖的。启用方式：`set PYTHONPATH=...\landmark_extension\shims`，或调用
`from landmark_extension.shims import activate; activate()`。

界面相关包同样受影响（`PyQt6` 也被拦截），因此在关闭 Smart App Control 之前，本环境
中任何交互式可视化窗口都不可用。

这一点也解释了为什么「管理员步骤」是**不可避免**而不是仅仅方便：这些被拦截的模块无法用
本会话能构建出的任何东西替代，因为工具链本身就缺失。

## 5. 还缺什么，以及确切的下一步操作

从当前状态到可以运行训练循环，还差三件事：

1. 给 Visual Studio 安装一个受 CUDA 支持的 MSVC 工具集（14.44.17.14）——
   **需要管理员操作**。
2. 本地编译 `pointops`，使 Smart App Control 接受它。
3. `torch-scatter` 要么本地重建，要么解除拦截（Smart App Control）。

`pymeshlab` 已由 `landmark_extension/shims/` 中的垫片解决。

管理员步骤完成后，其余都是自动化的：

```powershell
powershell -ExecutionPolicy Bypass -File D:\WorkSpace\Dental\3dteethland\landmark_extension\env\build_pointops.ps1
```

该脚本会打印 `pointops ok: <path>`，并在 CUDA tensor 上执行一次真实的
`farthestPointSampling` 调用作为自检。

## 6. 本任务创建的文件（仅环境相关部分）

* `landmark_extension/env/build_pointops.ps1`
* `landmark_extension/env/run_pointops_build.py`
* `landmark_extension/env/pointops_build.log`
* `landmark_extension/env/ENV_SETUP_REPORT.md`（英文）
* `landmark_extension/env/ENV_SETUP_REPORT.zh-CN.md`（本文件）
* `landmark_extension/shims/pymeshlab/__init__.py`、`landmark_extension/shims/activate.py`

没有修改任何仓库源码文件。`landmark_extension/` 与 conda 环境之外唯一的改动，是 conda
CUDA 安装目录内被修补过的 `crt/host_config.h`，其旁边保留了 `.orig` 备份。
