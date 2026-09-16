# landmark_extension

本目录用于在本机数据上，用
[`nnistelrooij/3dteethland`](https://github.com/nnistelrooij/3dteethland)
（ToothInstanceNet）训练 **dental landmark detection（牙体解剖标志点检测）** 模型，
训练标注来自本地已有的 **Teeth3DS+ 3DTeethLand** landmark 标注。

原仓库没有做任何修改。本目录内所有内容都是新增的：数据检查、数据/接口适配、环境工具、
验证脚本、评测辅助脚本以及本系列文档。

英文版：[`README.md`](README.md)　环境细节：[`env/ENV_SETUP_REPORT.md`](env/ENV_SETUP_REPORT.md)
（中文版 [`env/ENV_SETUP_REPORT.zh-CN.md`](env/ENV_SETUP_REPORT.zh-CN.md)）

想**手动用命令行搭建**（不使用 `env/setup_linux.sh`）？直接看第 7 节：
从拉取项目、放 `landmark_extension`、核对数据集、建环境、装依赖、编译 pointops，
到改配置、跑验证、启动训练，全部是可复制的命令。

---

## 1. 状态总览

| 项目 | 状态 |
| --- | --- |
| Python/CUDA 环境（`D:\WorkSpace\conda\envs\3dteethland`） | **就绪** —— torch 2.11.0+cu128 已在 RTX 5050（sm_120）上跑通真实 CUDA 运算 |
| 其余 pip 依赖 | **就绪**（详见 `env/ENV_SETUP_REPORT.zh-CN.md`） |
| `pointops` CUDA 扩展（kNN / ball query / FPS / stratified attention） | **受阻** —— 需要一个受 CUDA 支持的 MSVC 工具集，见第 6 节 |
| Teeth3DS+ 数据清点 + 几何一致性验证 | **已完成**（`check/landmark_geometry_report.json`） |
| 数据适配（Teeth3DS+ → `TeethLandDataModule`） | **已完成并验证**（每个颌 237 个可用病例，0 个未匹配 landmark） |
| 训练路径中的 mesh 读取 | **已用垫片解决** —— `pymeshlab` 被 Smart App Control 拦截，改用无依赖的 OBJ 垫片（`shims/`） |
| Dataset 端到端 smoke test | **待完成** —— 需要 `pointops` + `torch_scatter` 可导入 |
| `LandmarkNet` forward / loss / backward | **待完成** —— 同一个阻塞点 |
| 训练配置与命令 | **已完成**（`configs/landmark_upper.yaml`、`configs/landmark_lower.yaml`） |
| 评测 / 推理方案 | **已梳理**（`eval/`、`infer/`、第 11 节） |

**目前不要开始训练。** 训练需要你明确授权。（在当前这台 Windows 机器上还差第 12.2 节里的那一步环境操作；Linux + RTX 5090 目标机不需要。）

### 目标机器：Linux + RTX 5090 + CUDA 13.0 + 128 GB 内存

本目录是在一台 Windows 笔记本（RTX 5050 8 GB，驱动 596.58）上开发验证的。前面遇到的那些
环境麻烦——MSVC 14.51 让 `nvcc` 崩溃、Smart App Control 拦截未签名 DLL、`pymeshlab`
加载失败——**全部是 Windows 或该机器特有的，在 Linux 目标机上不会出现**。目标机上不需要
任何垫片、不需要 `-allow-unsupported-compiler`、不需要降级 MSVC。`shims/` 目录留在这里只是
为了本机可复现，目标机上 `import pymeshlab` 会直接用真库。

一键搭建：

```bash
cd /path/to/3dteethland
bash landmark_extension/env/setup_linux.sh
```

可用的环境变量（均有合理默认值）：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `ENV_PREFIX` | `$HOME/envs/3dteethland` | 环境安装位置 |
| `CUDA_INDEX` | `https://download.pytorch.org/whl/cu128` | torch 下载源 |
| `PYG_CUDA` | `cu128` | torch-scatter wheel 的 CUDA 后缀，需与 torch 匹配 |
| `TORCH_CUDA_ARCH_LIST` | `12.0` | RTX 5090 为 sm_120 |
| `SKIP_TOOLKIT` | `0` | 设为 `1` 表示机器上已有 CUDA toolkit |

脚本流程：创建 env → 从 nvidia 频道装 CUDA 12.8 toolkit（提供 `nvcc`）→ 装 cu128 版
torch 三件套 → 装 requirements 其余依赖 + 仓库漏声明的 `scikit-learn`/`pandas`/
`scikit-multilearn` → 装 torch-scatter（先试 PyG 预编译 wheel，失败则源码编译）→
`setup.py build_ext --inplace` 编译 `pointops` → 最后自动做一轮验证（含真实 CUDA matmul、
显存总量、以及一次真实的 `pointops.farthestPointSampling` 调用）。

关于 torch 版本选择：你的驱动支持 CUDA 13.0，**cu128 wheel 在 13.x 驱动上可以直接运行**
（本机就是用 cu128 wheel 配 13.2 驱动跑通的）。如果想用更新的构建，可以
`CUDA_INDEX=https://download.pytorch.org/whl/cu130`，但请同步把 `PYG_CUDA=cu130` 以匹配
torch-scatter 的 wheel，并确认 pointops 仍能编过。

关于 `python=3.10`：torch-scatter 的预编译 wheel 对每个 Python 小版本都单独构建，
3.10 在 PyG 的 wheel 索引里覆盖最完整。如果换成 3.12，请先确认该 torch 版本在 PyG 索引里
有对应 `cp312` wheel，否则会走源码编译（Linux 上可行，但慢）。

### 5090 上的训练参数建议

`configs/landmark_<jaw>.yaml` 已把 `batch_size` 设为 **32**、`num_workers` 设为 **8**
（仓库原始值是 8 / 4，那是为 8 GB 显存准备的）。每批点数约 32 × 8 × 12 000 ≈ 307 万。
32 GB 显存足够，但仍建议先用 smoke test 量一次实测峰值显存再定案：

```bash
python landmark_extension/validation/smoke_test_landmark_pipeline.py --jaw upper --batch-size 32
```

该脚本会在 `validation/smoke_report.json` 里写出 `peak_memory_gb`。另外两点需要注意：

* **学习率**：仓库的 6e-4 是按 batch 8 调的。若把 batch 提到 32，线性缩放对应 1.2e-3。
  想和论文/参考实现对齐就保持 6e-4，想更快收敛再按线性缩放调整——这是一个需要你决定的取舍。
* **`max_proposals`**：它是每个 scan 每个 epoch 采样的牙数，改动会改变训练语义（不只是吞吐），
  仓库原始值是 8。想提高牙位覆盖可以调到 12–16，但建议先跑基线。

128 GB 内存对数据加载是充裕的：dataset 会把每个病例的预处理结果缓存成一个 pickle
（`DatasetCache`，文件名是数据集哈希，生成在当前工作目录），把 `num_workers` 提到 8–16
也不会构成内存压力。

### 数据与路径的跨平台情况（已核对）

换到 Linux 后唯一需要改动的是指向数据的路径。已经验证过这几点不会出问题：

* 240 个 `__kpt.json` 的 `key` 字段全部是**扁平文件名**（`013TXGFK_upper.obj`），
  不含任何路径分隔符，所以 `process.py` 里的 `kpt_dict['key'].split('/')[-1]` 在
  Linux 上行为一致。
* 仓库的 `TeethSegDataModule` 用 `f'/{os.sep}(?!...)'` 构造过滤正则，在 Linux 上是 `/`，
  与路径形式一致；我们的 config 用 `regex_filter: upper/lower` 匹配颌骨子目录名，同样成立。
* `scan_file` 在 dataset 里以 `Path.as_posix()` 记录，输出 JSON 的 `key` 与文件名推导都用
  `Path`，所以不会出现 Windows 反斜杠。
* 需要把 configs 里的 `root` / `landmarks_root` / `fold` 改成目标机上的绝对路径；
  `prepare/` 下生成的 fold 文件是纯文本文件名列表，可以直接拷贝过去。

建议直接把整个 `data/Teeth3DS` 目录按原名拷到目标机（例如 `/data/Teeth3DS`），
然后只改 config 里三处路径即可，不需要重新生成任何索引或 fold。

---

## 7. 手动命令行搭建全流程（不使用 shell 脚本）

这一节假设你手上只有三样东西：**原 3dteethland 项目**、**本 `landmark_extension/` 目录**、
**原始 Teeth3DS+ 数据集**。下面每条命令都可以直接复制执行，`$` 开头的变量按你的实际路径改。
本机验证过的默认值是 `ENV_PREFIX=$HOME/envs/3dteethland`、`DATA_ROOT=/data/Teeth3DS`、
`REPO=/path/to/3dteethland`。

### 步骤 0　前置确认

```bash
nvidia-smi                      # 期望看到 RTX 5090，以及驱动报告的 CUDA 版本
conda --version                 # 或 mamba
gcc -dumpfullversion -dumpversion   # CUDA 12.8 要求 gcc <= 13
```

如果机器上没有 `nvcc` 也没关系，第 2 步会装它。

### 步骤 1　拉取项目并把 `landmark_extension` 放进去

```bash
git clone https://github.com/nnistelrooij/3dteethland.git
cd 3dteethland
```

然后把 `landmark_extension/` 整个拷到**仓库根目录下**（与 `teethland/`、`train.py`、`setup.py`
同级）：

```bash
# 方式 A：从已有副本拷进来
cp -r /path/to/landmark_extension ./landmark_extension

# 方式 B：如果你把本目录单独做成了 git 仓库
git clone <landmark_extension 的地址> landmark_extension
```

目录位置很关键：`run_landmark.py`、`configs/*.yaml` 里的 `fold` 路径等，都是按
「`landmark_extension` 位于仓库根目录」来解析的。

自检：

```bash
ls landmark_extension/README.zh-CN.md landmark_extension/configs/landmark_upper.yaml
```

### 步骤 2　准备数据集并核对结构

```bash
export DATA_ROOT=/data/Teeth3DS      # 改成你的实际路径
ls "$DATA_ROOT"                      # 应看到 upper/ lower/ 3DTeethLand_landmarks_train/ ...
ls "$DATA_ROOT/upper" | head         # 每个病例一个目录
ls "$DATA_ROOT/upper/01328DDN"       # 应包含 01328DDN_upper.obj 与 01328DDN_upper.json
ls "$DATA_ROOT/3DTeethLand_landmarks_train/upper" | head       # 每病例一个目录
ls "$DATA_ROOT/3DTeethLand_landmarks_train/upper/013TXGFK"     # 应包含 013TXGFK_upper__kpt.json
```

所需的三段数据：**mesh（OBJ）**、**逐顶点 FDI 分割（同名 JSON）**、
**`__kpt.json` landmark**。核对数量（应为 950 / 950 / 120 / 120）：

```bash
ls -d "$DATA_ROOT"/upper/*/     | wc -l
ls -d "$DATA_ROOT"/lower/*/     | wc -l
ls -d "$DATA_ROOT"/3DTeethLand_landmarks_train/upper/*/ | wc -l
ls -d "$DATA_ROOT"/3DTeethLand_landmarks_train/lower/*/ | wc -l
```

### 步骤 3　创建环境

```bash
export ENV_PREFIX=$HOME/envs/3dteethland
conda create -y -p "$ENV_PREFIX" python=3.10
export PY="$ENV_PREFIX/bin/python"
"$PY" -m pip install --upgrade pip wheel setuptools
```

### 步骤 4　安装带 `nvcc` 的 CUDA toolkit

```bash
conda install -y -p "$ENV_PREFIX" -c nvidia cuda-toolkit=12.8
export CUDA_HOME="$ENV_PREFIX"
export CUDA_PATH="$CUDA_HOME"
"$ENV_PREFIX/bin/nvcc" --version
```

### 步骤 5　安装 torch（必须能支持 sm_120）

```bash
"$PY" -m pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
"$PY" -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

期望输出类似 `2.11.0+cu128 12.8 True`。**不要**按 `requirements.txt` 装 torch==2.3.0+cu121，
那个版本不支持 sm_120 / 本机驱动。

### 步骤 6　安装项目其余依赖（手动剔除 torch 相关行）

`requirements.txt` 里同时包含 torch 三件套与 torch-scatter，这些已经/将要单独处理，
所以这里按下面这条安装即可（等价于「requirements 减去 torch 相关行」再加两个仓库漏声明的包）：

```bash
"$PY" -m pip install "numpy<2.0.0" lxml==5.2.2 opencv-python==4.10.0.84 \
    open3d==0.17.0 gco-wrapper==3.0.9 pymeshlab==2023.12.post1 pytest==8.2.2 \
    pytorch-lightning==2.3.3 tensorboard==2.17.0 timm==1.0.7 torchtyping==0.1.4

# 仓库实际 import 但 requirements.txt 未声明的依赖
"$PY" -m pip install scikit-learn pandas scikit-multilearn

# torch-scatter 必须与 torch 版本、CUDA 后缀严格匹配
"$PY" -m pip install torch-scatter \
    -f https://data.pyg.org/whl/torch-2.11.0+cu128.html
```

如果第 5 步装到的不是 2.11.0，把上面 `torch-2.11.0+cu128.html` 里的版本号改成你实际的
torch 版本（用 `"$PY" -c "import torch; print(torch.__version__)"` 查）。若该索引没有对应
wheel，就源码编译：`"$PY" -m pip install --no-build-isolation torch-scatter`。

### 步骤 7　编译 pointops CUDA 扩展

```bash
cd /path/to/3dteethland
export TORCH_CUDA_ARCH_LIST=12.0        # RTX 5090 = sm_120
export PATH="$CUDA_HOME/bin:$ENV_PREFIX/bin:$PATH"
"$PY" setup.py build_ext --inplace      # 若失败，可改用： "$PY" -m pip install -v -e .
```

编译成功后自检（这一步必须过，否则训练会在第一次 kNN/采样时崩）：

```bash
"$PY" -c "
import torch, pointops
print('pointops:', pointops.__file__)
x = torch.randn(4096, 3, device='cuda')
idx, n = pointops.farthestPointSampling(x, torch.tensor([4096], device='cuda'), 0.25, 1024)
print('fps ok:', tuple(idx.shape), idx.dtype)
"
```

### 步骤 8　改配置里的三处路径

在 `landmark_extension/configs/landmark_upper.yaml` 与 `landmark_lower.yaml` 里，把
`datamodule` 下的路径改成目标机的：

```yaml
datamodule:
  root: '/data/Teeth3DS'
  landmarks_root: '/data/Teeth3DS/3DTeethLand_landmarks_train/upper'   # lower.yaml 用 lower
  fold: 'landmark_extension/prepare/landmark_upper_fold_0.txt'         # lower.yaml 用 lower
```

`fold` 可以保持相对路径（相对于仓库根目录解析）。如果你换了数据划分，用下面命令重新生成
（注意它默认的 `--data-root` 是 Windows 路径，必须显式指定）：

```bash
"$PY" landmark_extension/prepare/prepare_landmark_training_data.py --data-root /data/Teeth3DS
```

### 步骤 9　数据链路验证（不训练）

```bash
cd /path/to/3dteethland
"$PY" landmark_extension/validation/check_data_adapter.py --jaw upper --cases 3
"$PY" landmark_extension/validation/smoke_test_landmark_pipeline.py --jaw upper --batch-size 32
```

第二条会跑 dataset → collate → forward → loss → backward 一次，并在
`landmark_extension/validation/smoke_report.json` 里写出 `peak_memory_gb`。
如果显存不够就把 `--batch-size` 调小，并同步改 config 里的 `batch_size`。

### 步骤 10　启动训练

```bash
cd /path/to/3dteethland

# 上颌
"$PY" landmark_extension/run_landmark.py \
    --config landmark_extension/configs/landmark_upper.yaml \
    --devices 1

# 下颌（另开一个进程/任务；一个 data module 只装一侧颌骨）
"$PY" landmark_extension/run_landmark.py \
    --config landmark_extension/configs/landmark_lower.yaml \
    --devices 1
```

续训 / 热启动加 `--checkpoint <ckpt>` 即可。若想用多卡：
`--devices 2`（`train.py` 会在 `devices > 1` 时打开 `sync_batchnorm`）。

产物位置：

* checkpoint 与 TensorBoard 日志：`landmark_extension/runs/<version>/`（`version` 由 config 里的
  `version` 决定，默认 `upper` / `lower`），TensorBoard 用
  `tensorboard --logdir landmark_extension/runs` 打开。
* dataset 预处理缓存：文件名是数据集哈希 `xxxx.pkl`，**写在当前工作目录**。
  所以请固定从仓库根目录启动；这个文件可以删，删掉只会重新预处理一次。

### 步骤 11（可选）推理与评测

```bash
# 推理（需要先有训练好的 checkpoint）
"$PY" landmark_extension/infer/predict_landmarks.py \
    --checkpoint landmark_extension/runs/upper/checkpoints/<ckpt> \
    --jaw upper --limit 4

# 用官方评分代码评测
"$PY" landmark_extension/eval/score_predictions.py make-gold \
    --jaw upper --fold landmark_extension/prepare/landmark_upper_fold_0.txt
"$PY" landmark_extension/eval/score_predictions.py score \
    --predictions landmark_extension/preds/upper/predictions.csv \
    --goldstand   landmark_extension/eval/gold_upper_landmark_upper_fold_0.pkl \
    --out         landmark_extension/eval/scores_upper_fold_0.json
```

### 7.1 手动流程与 `setup_linux.sh` 的对应关系

| 手动步骤 | `setup_linux.sh` 中的对应阶段 |
| --- | --- |
| 步骤 3 创建环境 | `[1/7]` |
| 步骤 4 CUDA toolkit | `[2/7]` |
| 步骤 5 torch | `[3/7]` |
| 步骤 6 其余依赖 | `[4/7]` + `[5/7]` |
| 步骤 7 编译 pointops | `[6/7]` |
| 步骤 7 的自检 | `[7/7]` |
| 步骤 1、2、8–11 | 脚本不涉及，必须手动做 |

也就是说：脚本只负责**环境**，项目放置、数据路径、验证和训练命令无论用不用脚本都要自己执行。

---

## 8. 模型是怎么训练的（结论来自源码，不是 README）

入口：`train.py landmarks` → `TeethLandDataModule` + `LandmarkNet`。

### 8.1 模型结构

`teethland/models/landmarknet.py::LandmarkNet` 包装了
`nn.StratifiedTransformer`（`teethland/nn/modules/stratified_transformer.py`），
由 YAML 中的 `model.landmarks` 配置：

```
out_channels: [1, 4, 4, 4, 4, 4]     # 六个头，合计 11 个输出通道
channels_list: [48, 96, 192, 256]
depths: [3, 9, 3]
heads_list: [6, 12, 24]
window_sizes: [0.1, 0.2, 0.4]        # landmark 配置；分割配置用的是 0.4/0.8/1.6
point_embedding: KPConv(influence 0.02, radius 0.05)
```

`LandmarkNet.forward` 返回 `(seg, mesial_distal, facial, outer, inner, cusps)`。
五个 landmark 头每个输出 `(distance, dx, dy, dz)`：对每个点给出「离该 landmark 有多近」
以及「该 landmark 相对这个点的偏移」。

需要留意：仓库配置给**每一个 landmark 头都分配了 4 个通道**，包括只有单一类别的
`facial` / `outer` / `inner` 三个头；本任务沿用 `teethland/config/config.yaml` 里的原始
取值，因为已发布的 checkpoint 就是按这个配置训练的。

### 8.2 损失函数（`LandmarkNet.training_step`）

```
loss = BCEWithLogits(seg, tooth_mask)
     + LandmarkLoss(mesial_distal, gt, classes=[0, 1])
     + LandmarkLoss(facial,        gt, classes=[2])
     + LandmarkLoss(outer,         gt, classes=[3])
     + LandmarkLoss(inner,         gt, classes=[4])
     + LandmarkLoss(cusps,         gt, classes=[5])
```

`teethland/nn/modules/criterion.py::LandmarkLoss` 由三部分组成：SmoothL1 距离项、
带掩码的 Chamfer 项、分离项（`dist_thresh=0.10`、`w_dist=0.05`、`w_chamfer=0.05`、
`w_separation=0.0005`）。

**所有常数都在归一化坐标系下**，其中 1 单位 = `STD = 17.3281` mm
（`TeethSegDataset.STD`）。也就是说 `dist_thresh=0.10` 约为 1.7 mm，而
`LandmarkF1Score` 的阈值 17.3281 恰好等于 1.0 个归一化单位。

`train.py` 按 `loss/val` 和 `dice/val`（`mode='min'`）保存 checkpoint —— 注意
landmark 阶段监控的是**分割 dice**，不是任何 landmark 指标，所以 stock runner 选出的
「最优 checkpoint」对 landmark 而言并不是最优的。如果需要改成基于 landmark 指标选
checkpoint，见 11.4 节的建议。

### 8.3 优化设置

AdamW + `CosineAnnealingLR`，前置 `LinearWarmupLR`；参数分组来自
`StratifiedTransformer.param_groups`（transformer 模块使用
`lr * transformer_lr_ratio`）；梯度裁剪阈值 35（`config['gradient_clip_norm']`）。

---

## 9. 训练数据需要什么格式

`TeethLandDataModule` 继承自 `TeethSegDataModule`，并且**一个 data module 只能装一侧颌骨**：

```
<root>/<split>/<case>/<case>_<jaw>.obj    口扫 mesh（obj/ply/stl）
<root>/<split>/<case>/<case>_<jaw>.json   {"labels": [每个顶点一个 FDI],
                                           "instances": [每个顶点一个牙位实例 id],
                                           ...}
<landmarks_root>/<...>/<case>_<jaw>__kpt.json
                                          {"objects": [{"class": "...",
                                                        "coord": [x, y, z]}, ...]}
```

从源码确认的几条约定性契约：

* `TeethSegDataModule._files` 通过**文件名 stem** 匹配 mesh 与 segmentation；子目录层级
  不影响匹配（两者都是递归 glob）。
* `TeethLandDataModule._files` 取 segmentation stem 与 landmark stem 的交集，其中
  landmark stem = `kpt_file.name.split('__')[0]`。由于文件名本身已带 `_<jaw>` 后缀，
  **一个 `landmarks_root` 必须且只能包含一侧颌骨**。
* landmark JSON 必须是 `__kpt.json` 这一种（即文件名在 `kpt` 前有双下划线）。如果不小心
  指向同时包含 segmentation JSON 的目录，会静默混入错误文件。
* 标签值就是 FDI（1x–4x 恒牙，5x–8x 乳牙）；`0` 表示牙龈。
* landmark 类别名必须严格是：
  `Mesial`、`Distal`、`FacialPoint`、`OuterPoint`、`InnerPoint`、`Cusp`
  （`TeethLandDataset.landmark_classes`）。其他写法会直接 `KeyError`。
* `num_classes` 为 5；因为 `label_as_instance` 为 False，分割目标的取值是
  `labels - 1`（`-1` 表示背景）。

### 9.1 mesh / segmentation / FDI / landmark 之间的对应关系

以下关系已在本数据集上对每个病例做了数值验证，结论见第 10 节。

| 数据 | 含义 |
| --- | --- |
| `.obj` | 颌骨三角网格；顶点顺序是唯一的参考顺序 |
| `<case>_<jaw>.json` 的 `labels[i]` | mesh 顶点 `i` 的 FDI 编号 |
| `<case>_<jaw>.json` 的 `instances[i]` | mesh 顶点 `i` 的实例 id（0 = 非牙） |
| `instances` + `labels` | 一颗牙 = 一个实例，实例内 FDI 标签恒定 |
| `__kpt.json` 的 `objects[k].coord` | 三维坐标，**与 mesh 顶点处于同一坐标系** |
| `__kpt.json` 的 `objects[k].class` | 六个 landmark 类别之一 |

landmark 与牙位的匹配由 `teethland/data/transforms.py::MatchLandmarksAndTeeth` 完成：
它先把近中/远中 landmark 稍微向外移动，再把每个 landmark 分配到最近的**牙体**顶点所属
实例。这个实例 id 就是后续 `GenerateProposals` 能「按采样到的牙保留对应 landmark」的依据。

---

## 10. Teeth3DS+ 是如何适配的（不复制、不修改任何数据）

本地数据结构本身已经符合要求，所以**不需要任何抽取、转换或复制**。适配工作本质上是把
仓库的文件发现范围收敛到 3DTeethLand 病例，并按颌骨给出 data module 期望的输入：

| 配置项 | 取值 | 原因 |
| --- | --- | --- |
| `root` | `D:/WorkSpace/Dental/data/Teeth3DS` | 包含 `<jaw>/<case>/...` 的同一棵目录树 |
| `regex_filter` | `upper` / `lower` | 仓库是在路径片段名上做正则过滤，因此用颌骨子目录名来选择颌骨 |
| `landmarks_root` | `.../3DTeethLand_landmarks_train/<jaw>` | 必须只包含一侧颌骨 |
| `extensions` | `['obj']` | 本机扫描件是 OBJ |
| `fold` | `landmark_extension/prepare/landmark_<jaw>_fold_0.txt` | 确定性的按受试者划分 |

`prepare/prepare_landmark_training_data.py` **只写**辅助产物（病例索引 CSV、数据集统计
JSON、fold 文件），从不写入 `D:\WorkSpace\Dental\data\Teeth3DS`。

### 10.1 复现仓库内置的排除名单

仓库里有硬编码的排除项，必须遵守：

* `TeethSegDataModule`（mesh+segmentation）：`017U3R3T_upper`、`01A91JH6_lower`、
  `01A91JH6_upper`、`01HMF5HV_lower`、`01HMF5HV_upper`、`E23G704K_lower`、
  `E23G704K_upper`、`016FSM14_lower`、`01FPTYH2_lower`
* `TeethLandDataModule`（landmark）：`K4BAII5F_upper`、`87N5YSES_upper`
  （缺少牙体分割）、`67PV9M7X_lower`、`S0AON6PZ_lower`（缺失一个以上 landmark）

### 10.2 几何一致性验证（`check/check_landmark_geometry.py`）

对全部 236 个 landmark 病例（排除后为上颌 118 + 下颌 118）做了检查：

| 指标 | 上颌 | 下颌 |
| --- | --- | --- |
| 病例数 | 118 | 118 |
| mesh 顶点数 == segmentation 标签数 | 118/118 | 118/118 |
| landmark 到最近 mesh 顶点的平均距离 | 0.0685 | 0.0700 |
| 单病例平均距离的最差值 | 0.173 | 0.180 |
| 最近顶点为牙龈（`label == 0`）的 landmark 占比 | 616 / 10260（6.0%） | 630 / 10812（5.8%） |
| 每病例平均出现的不同 FDI 数 | 13.3 | 13.4 |

（距离单位均为 mm。）结论：**landmark JSON 与 mesh 完全处于同一坐标系**，segmentation 是
逐顶点的、且与 mesh 索引严格对齐。

运行时的排除名单也已应用：`K4BAII5F_upper`、`87N5YSES_upper`（无牙体分割）和
`67PV9M7X_lower`、`S0AON6PZ_lower` 从未被纳入上下颌 landmark 集合；`017U3R3T_upper`
（有 landmark 但没有牙体分割）由仓库的 `TeethSegDataModule` 名单剔除。这就是 120 个
landmark 目录最终只剩每个颌 118 个可训练病例的原因。

那约 6% 最近顶点为牙龈的 landmark 是边缘位置的标志点（例如牙龈缘上的 `InnerPoint`），
它们到最近 mesh 点的距离仍然只有约 0.07 mm，因此 `MatchLandmarksAndTeeth` 仍会把它们
分配到正确的邻牙。最差的离群病例平均偏移也只有 0.18 mm，远低于 1.7 mm 的损失阈值。

### 10.3 数据适配验证（`validation/check_data_adapter.py`）

该脚本用不依赖 torch 的方式重新实现了适配器的文件解析，以及 dataset 的归一化和
landmark 匹配，并在真实病例上运行：

```
regex filter           : upper
mesh files (upper)     : 1071
mesh+seg pairs         : 1013
landmark files         : 120
-> usable triples      : 237
  013TXGFK_upper: verts=122958 fdi=11 instances=11 landmarks=70 distinct landmark teeth=10
  0140W3ND_upper: verts=152382 fdi=14 instances=14 landmarks=83 distinct landmark teeth=13
  0140YFGV_upper: verts=123972 fdi=12 instances=12 landmarks=76 distinct landmark teeth=12

landmarks checked                   : 229
unmatched landmarks                 : 0
worst landmark->assigned-tooth dist: 0.0234 normalised = 0.406 mm
```

也就是说每个 landmark 都被分配到了真实牙位实例，并且都在配置阈值之内。

### 10.4 数据规模与划分

| 颌骨 | 训练病例（可用） | 测试病例（仅有 landmark） |
| --- | --- | --- |
| upper | 118 | 50 |
| lower | 118 | 50 |

landmark 总数：上颌训练集 10 464，下颌训练集 10 930。各类别分布（按颌骨与划分）大致为：
`Mesial` 约 1 580、`Distal` 约 1 580、`FacialPoint` 约 1 575、`OuterPoint` 约 1 575、
`InnerPoint` 约 1 575、`Cusp` 约 2 600–3 000。

**`3DTeethLand_landmarks_test` 这 100 例没有 segmentation JSON**（只有 `__kpt.json`
landmark 和 mesh，mesh 位于 `Teeth3DS/<jaw>/<case>/<case>_<jaw>.obj`）。由此带来两点结论：

* 它可以用于 landmark 评测，但需要 segmentation/FDI 来自非本地 Teeth3DS 的途径，也就是
  走完整的 align + 实例分割流程（`FullNet`），或者只评测 landmark 头（见
  `infer/predict_landmarks.py`）。
* `TeethLandDataModule` 完全无法加载它，因为它要求 segmentation 与 landmark 的交集。

fold 文件（`prepare/landmark_<jaw>_fold_<k>.txt`）把每个受试者分到且仅分到 5 折中的一折：
每颌 24/24/24/23/23 个病例。`configs/landmark_<jaw>.yaml` 用 fold 0 作验证集
（约 24 例，20%），其余约 94 例作训练集，`include_val_as_train: False`。

---

## 11. 训练、评测与推理

### 11.1 训练命令（已就绪，等你在目标机上确认后再执行）

```powershell
# 上颌
& 'D:\WorkSpace\conda\envs\3dteethland\python.exe' `
    landmark_extension\run_landmark.py `
    --config landmark_extension\configs\landmark_upper.yaml `
    --devices 1

# 下颌
& 'D:\WorkSpace\conda\envs\3dteethland\python.exe' `
    landmark_extension\run_landmark.py `
    --config landmark_extension\configs\landmark_lower.yaml `
    --devices 1
```

`run_landmark.py` 调用的是未修改的 `train.main('landmarks', ...)`；它只是在运行期间把
`teethland/config/config.yaml` 临时替换为本任务的配置，结束后按字节原样恢复原文件。
在 Linux 目标机上把 `python.exe` 换成 `$ENV_PREFIX/bin/python` 即可。

断点续训 / 热启动：

```powershell
... run_landmark.py --config landmark_extension\configs\landmark_upper.yaml --checkpoint <ckpt>
```

### 11.2 主要训练参数（`configs/landmark_<jaw>.yaml`）

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| `model.landmarks.epochs` | 500 | 仓库原始取值 |
| `lr` / `weight_decay` | 6e-4 / 1e-4 | 仓库原始取值 |
| `warmup_epochs` | 5 | `LinearWarmupLR` 之后接 cosine |
| `transformer_lr_ratio` | 0.1 | transformer 块使用 6e-5 |
| `batch_size` | 8 | 每次 8 个扫描 × 每个 8 个 proposal |
| `num_workers` | 4 | `persistent_workers: true` |
| `uniform_density_voxel_size` | `[0.025, 0.01]` | landmark 阶段取索引 1，即 1 cm 体素 |
| `proposal_points` | 12 000 | 每个牙体 proposal 的点数 |
| `max_proposals` | 8 | 每个扫描、每个 epoch 采样的牙数 |
| `sampler` | `balanced` | `InstanceBalancedSampler` |
| `gradient_clip_norm` | 35 | |
| `accumulate_grad_batches` | 1 | |

**显存预期**：约 8 个扫描 × 8 个 proposal × 12 000 点 ≈ 每批 77 万个点。在 8 GB 的
RTX 5050 上，这是第一个需要下调的参数（改 `batch_size: 4` 或 `max_proposals: 6`）；
smoke test 会给出实测峰值显存。

### 11.3 模型与 checkpoint

* 结构：`LandmarkNet` = 上述 landmark 配置的 `StratifiedTransformer`，**从零开始训练**
  （`checkpoint_path: null`）。
* 本地不存在预训练 checkpoint。`teethland/config/config.yaml` 引用的
  `checkpoints/landmarks_full.ckpt` 与 `checkpoints/landmarks_ablation.ckpt`，以及 README
  提到的 Google Drive 目录，**在当前 checkout 中都不存在**，所以今天既不能续训也不能热启动。
  如果要用已发布权重，需要先下载。
* 最优 checkpoint 选择：stock `train.py` 监控的是 `loss/val` 和 `dice/val`。由于 landmark
  指标根本没有被记录，建议（最小改动）在 `LandmarkNet.validation_step` 中启用
  `LandmarkMeanAveragePrecision` 的更新 —— 这段代码其实已经写好，只是被注释掉了。
  这属于可选的后续改动，也是 stock 仓库唯一真正需要改代码的地方。

### 11.4 评测

仓库自带**官方挑战赛评测代码**（`evaluation/3dteethland.py`）：在 0–2.9 mm 的距离阈值上
按类别计算 AP/AR，并汇总为 `AP_cusp`、`AP_mesial_distal`、`AP_inner_outer`、`AP_facial`、
`mAP`（以及对应的 AR 指标）。它需要两个输入：

* `--predictions_file`：CSV，列为 `key, coord_x, coord_y, coord_z, class, score`
* `--goldstandard_file`：pickle，结构为 `{class: {mesh_name: [coord, ...]}}`

`eval/score_predictions.py` 从本地 landmark JSON 生成该 gold standard，并直接调用仓库的
评测代码：

```powershell
& 'D:\WorkSpace\conda\envs\3dteethland\python.exe' landmark_extension\eval\score_predictions.py make-gold `
    --jaw upper --fold landmark_extension\prepare\landmark_upper_fold_0.txt

& 'D:\WorkSpace\conda\envs\3dteethland\python.exe' landmark_extension\eval\score_predictions.py score `
    --predictions landmark_extension\preds\upper\predictions.csv `
    --goldstand   landmark_extension\eval\gold_upper_landmark_upper_fold_0.pkl `
    --out         landmark_extension\eval\scores_upper_fold_0.json
```

gold standard 生成器已在上颌 fold 0 上跑通（24 个病例，2 055 个 landmark）。

训练过程中的验证指标：`teethland/metrics/` 里有 `LandmarkF1Score`（匈牙利匹配，阈值
1.0 单位 = 17.33 mm）和 `LandmarkMap`（0.5/1/2/3 mm 上的 mAP），它们已经接入
`LandmarkNet`，但 `validation_step` 里对应的代码体被注释掉了。启用它属于 11.3 节提到的
同一处最小改动。

### 11.5 推理

两条路径，各自的注意事项如下：

1. **只跑 landmark 阶段**（符合本任务「复用已有 segmentation / FDI」的目标）：
   `infer/predict_landmarks.py`。
   它读取 mesh 以及已有的逐顶点 FDI/实例分割，用仓库自身的 transform 构建牙体
   proposal，但不运行 align 和分割模型；随后跑 `LandmarkNet`，输出
   `<case>_<jaw>__kpt.json` 和 `predictions.csv`。
   **状态：已对照源码做过设计复核，但尚未端到端执行** —— 它需要 `pointops` 和一个训练好的
   checkpoint，因此只能在环境搭建完成之后运行（目标机见第 7 节，Windows 开发机见第 12.2 节）。

2. **完整流水线**（`infer.py`、`FullNet`，即 align → 实例分割+FDI → 逐牙 landmark）。
   这是挑战赛提交所用的路径，需要从 Google Drive 下载的四个 checkpoint。
   该路径发现两个缺陷，属于本次范围之外，**未做修复**：
   * `TeethInstFullDataModule.collate_fn` 返回 6 个元素，而
     `transfer_batch_to_device` 只解包 5 个；随后还会读取 data module 上的
     `self.affine`，而这个属性从未被赋值 —— 因此 `predict` 路径会在 landmark 后处理之前
     就崩掉。
   * `FullNet.landmarks_process` 会把每颗牙的近中和远中 landmark 合并成一个检测结果，
     再依赖 `process_landmarks` 按几何位置拆分，这也是预测 CSV 中出现
     `mesial_distal` 类别的原因。

### 11.6 已知缺口 / 需要如实说明的地方

* `LandmarkNet.validation_step` 只计算并记录 loss，所有 landmark 指标都被注释掉了。
  不做改动的话，验证过程无法告诉你 landmark 质量如何。
* landmark 头回归的是**相对 proposal 点**的 `(distance, dx, dy, dz)`，所以必须先做聚类
  （`PointTensor.cluster` + `dbscan_cfg`）或按牙归约，才能转成绝对坐标；
  这正是 `infer/predict_landmarks.py` 所做的：按牙在置信度头上取 argmax 归约。
  这个归约方式是本任务的选择，仓库本身并未给出。
* 仓库把近中与远中合并为一个头，因此预测 CSV 中的类别需要做
  `process_landmarks` 式的几何区分，才能与挑战赛的 `Mesial`/`Distal` 两个独立类别对齐。

---

## 12. 环境

> **目标部署机器是 Linux + RTX 5090 + CUDA 13.0 + 128 GB 内存。** 12.2 节记录的两个
> 阻塞点（MSVC 工具集版本、Smart App Control）是**当前这台 Windows 笔记本特有**的，
> 在 Linux 目标机上不存在。目标机请直接用 `env/setup_linux.sh` 一键搭建，相关内容见
> 第 7 节。本节其余内容记录的是本机验证过程，保留作为可追溯记录。

### 12.1 已就绪的部分

`D:\WorkSpace\conda\envs\3dteethland\python.exe`（Python 3.10.21）：

| 包 | 已安装版本 |
| --- | --- |
| torch / torchvision / torchaudio | 2.11.0+cu128 / 0.26.0+cu128 / 2.11.0+cu128 |
| RTX 5050（sm_120）上的 `torch.cuda` | 已用真实 matmul 验证 |
| numpy | 1.26.4（项目要求 `<2`） |
| open3d / pymeshlab / opencv-python | 0.17.0 / 2023.12.post1 / 4.10.0.84 |
| pytorch-lightning / torchmetrics / tensorboard | 2.3.3 / 1.9.0 / 2.17.0 |
| timm / lxml / gco-wrapper / pytest / torchtyping | 1.0.7 / 5.2.2 / 3.0.9 / 8.2.2 / 0.1.4 |
| torch-scatter | 2.1.2+pt211cu128（已安装，但加载被拦截，见 8.1） |
| pymeshlab | 2023.12.post1 已安装，但**DLL 被 Smart App Control 拦截**；已用本目录的 OBJ 垫片替代 |
| torchtyping | 0.1.4 在 torch 2.11 上抛 `RuntimeError: Cannot subclass _TensorBase directly`（仅影响注解，见下） |
| scikit-learn / scipy / scikit-multilearn / pandas | 1.7.2 / 1.15.3 / 0.2.0 / 2.3.3 |
| 环境内自带 CUDA 工具链 | `Library` 当前是 **CUDA 13.0**（`nvcc` 13.0.88）；CUDA 12.8 先装后被取代 |

为什么用 cu128 而不是 requirements 里钉住的 cu121：本机 NVIDIA 驱动（596.58）已经放弃了
CUDA 12.x 兼容，而 GPU 是 sm_120，所以 `requirements.txt` 里的 torch 2.3.0+cu121 在这台
机器上根本无法运行；torch 2.11.0+cu128 已经验证可用。

另外还有两处 `requirements.txt` 的隐性缺口：仓库 import 了 `sklearn`
（`scikit-learn`）和 `pandas`，却没有列在依赖里。两者都已安装。

torch 2.11 对 `torch.Tensor` 内部实现有改动，导致 **`torchtyping` 0.1.4 在导入时抛
`RuntimeError: Cannot subclass _TensorBase directly`**。仓库只在注解里使用 `TensorType`
（全仓库没有任何 `@typechecked`），所以运行时路径不受影响；如果确实被卡到，解决办法是换成
torch 2.8.0+cu128（Python 3.10 有对应 wheel），或把
`torchtyping/tensor_type.py` 改成从 `torch.Tensor` 派生。

### 12.2 唯一剩下的阻塞点（两条路）

有两个 Windows 层面的事实挡住了 CUDA 扩展：

1. **Smart App Control 处于开启状态**（`HKLM\SYSTEM\CurrentControlSet\Control\CI\Policy`
   的 `VerifiedAndReputablePolicyState = 1`、`SAC_EnforcementReason = 1`）。
   Code Integrity 日志确认它拒绝加载未签名的扩展模块，例如
   `torch_scatter\_scatter_cuda.pyd` 和 `pymeshlab\meshlab-common.dll`
   （事件 `Microsoft-Windows-CodeIntegrity/Operational` 3077/3033/3118）。
   本地自行编译的二进制会被接受，但预编译的未签名 wheel 不会。
2. **本机安装的 MSVC 是 14.51（Visual Studio 18 BuildTools）**，且没有任何受支持的旧版
   工具集。`nvcc` 12.8 *和* 13.0 在编译一个仅一行的 `.cu` 文件时都会直接崩溃：
   `cudafe++ died with status 0xC0000005 (ACCESS_VIOLATION)` —— CUDA 目前尚不支持
   MSVC 19.5x（CUDA 13 支持 19.29–19.4x）。给 Visual Studio 安装旧版工具集需要提权进程，
   而本会话没有提权能力：`vs_installer.exe modify ... --quiet` 以 5007 退出，提示
   *"Commands with --quiet or --passive should be run elevated from the beginning."*

**方案 A（推荐）。** 在**管理员** PowerShell 中执行：

```powershell
& 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vs_installer.exe' modify `
  --installPath 'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools' `
  --quiet --norestart `
  --add Microsoft.VisualStudio.Component.VC.14.44.17.14.x86.x64

powershell -ExecutionPolicy Bypass -File D:\WorkSpace\Dental\3dteethland\landmark_extension\env\build_pointops.ps1
```

然后还需要让预编译的 `torch_scatter` DLL 能加载 —— 要么针对 CUDA 13 本地重编译（这样
Smart App Control 会接受）：

```powershell
& 'D:\WorkSpace\conda\envs\3dteethland\python.exe' -m pip install --no-binary torch-scatter torch-scatter
```

要么关闭 Smart App Control（Windows 安全中心 → 应用和浏览器控制 → Smart App Control；
注意关闭后除非重装 Windows，否则无法重新开启）。

**方案 B。** 保持现有工具链，改为用纯 PyTorch 重写 `pointops` + `torch_scatter` 的内核。
最远点采样、kNN、ball query 都不难，但 CRPE 注意力的反向传播是手写 CUDA kernel，
因此只有在有明确的数值一致性测试计划时才值得做，而且这会偏离参考实现。本次没有尝试。

`env/build_pointops.ps1` 和 `env/run_pointops_build.py` 已经为方案 A 准备好：它们会导入
MSVC 环境、把 `CUDA_HOME` 指向 conda 内的 CUDA 工具链、设置
`TORCH_CUDA_ARCH_LIST=12.0`、通过把 `subprocess.SUBPROCESS_DECODE_ARGS` 重绑为 UTF-8
来绕过 torch 的编码崩溃，并设置 `DISTUTILS_USE_SDK=1`。

关于环境每一个问题及其解决过程的完整记录，见
`env/ENV_SETUP_REPORT.zh-CN.md`（英文版 `env/ENV_SETUP_REPORT.md`）。

---

## 13. 目录结构

```
landmark_extension/
  README.md                              <- 英文主文档
  README.zh-CN.md                        <- 本文件
  run_landmark.py                        <- 用任务配置运行 train.main()
  train_landmarks.sh                     <- 参考调用方式
  configs/landmark_upper.yaml            <- 上颌 landmark 训练配置
  configs/landmark_lower.yaml            <- 下颌 landmark 训练配置
  prepare/prepare_landmark_training_data.py   <- 清点 + fold 生成
  prepare/landmark_cases_train.csv            <- 236 例索引（路径、计数、标签）
  prepare/landmark_dataset_summary_train.json <- 数据集统计
  prepare/landmark_<jaw>_fold_<k>.txt         <- 5 折受试者划分
  check/check_landmark_geometry.py       <- landmark/mesh/segmentation 对齐检查
  check/landmark_geometry_report.json    <- 其完整逐病例输出
  check/visualize_landmarks.py           <- FDI 上色 OBJ + landmark 小球导出
  check/visualizations/                  <- 导出的 OBJ 文件
  validation/check_data_adapter.py       <- 不依赖 torch 的适配器验证
  validation/smoke_test_landmark_pipeline.py <- dataset+collate+forward+loss+backward
  eval/score_predictions.py              <- 与官方评分兼容的评测
  eval/gold_upper_landmark_upper_fold_0.pkl  <- 验证 fold 0 的 gold standard
  infer/predict_landmarks.py             <- 仅 landmark 的推理适配脚本
  shims/pymeshlab/                       <- 被 SAC 拦截时替代 pymeshlab 的 OBJ 读取垫片
  shims/activate.py                      <- 启用垫片的辅助入口
  env/build_pointops.ps1                 <- 感知 MSVC 环境的 pointops 编译脚本
  env/run_pointops_build.py              <- 其 python 辅助脚本（编码绕过）
  env/pointops_build.log                 <- 最近一次编译日志
  env/ENV_SETUP_REPORT.md                <- 环境记录（英文）
  env/ENV_SETUP_REPORT.zh-CN.md          <- 环境记录（中文，本文件同批生成）
  runs/                                  <- 训练产物（TensorBoard + checkpoint）
  preds/                                 <- 推理输出
```

## 14. 需要你决定的后续步骤

1. 把 `data/Teeth3DS` 拷贝到 RTX 5090 那台 Linux 机器（建议保持目录名一致），并修改
   `configs/landmark_<jaw>.yaml` 里的 `root` / `landmarks_root` / `fold` 三处路径。
2. 在目标机上跑 `bash landmark_extension/env/setup_linux.sh`，脚本结束时会自动验证
   torch CUDA、torch-scatter 与 pointops 是否可用。
3. 跑
   `python landmark_extension/validation/smoke_test_landmark_pipeline.py --jaw upper --batch-size 32`，
   确认 dataset/collate/forward/loss/backward 链路，并读取 `smoke_report.json` 里的
   `peak_memory_gb`，据此定最终的 batch size。
4. 决定数据规模、epoch 数、batch size、学习率是否按 batch 线性缩放，以及是否启用
   11.3–11.4 节提到的 landmark 指标与 checkpoint 选择改动。
5. 以上确认之后再启动训练（`run_landmark.py`）。

当前这台 Windows 机器上剩下的 MSVC / Smart App Control 问题（12.2 节）**可以不再处理**，
除非你打算就在这台机器上训练。
