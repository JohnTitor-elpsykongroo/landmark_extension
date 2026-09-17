# RTX 5090 / Linux：Teeth3DS+ 位点训练与测试识别

本文针对本仓库当前代码，使用 Python 3.10、`.venv`、RTX 5090 和 CUDA 13.0。**命令尚未在目标机器验证**；先完成一例数据与一轮训练的检查，再扩大训练。所有新配置、构建物和输出都放在 `landmark_extension`；原 3dteethland 文件及 Teeth3DS+ 原始数据只读。

## 1. 数据和运行边界

以下用 `$PROJECT` 表示本仓库路径，`$DATA` 表示 Teeth3DS+ 路径。训练位点需要三件同名材料：`ID_upper.obj`、`ID_upper.json`（逐顶点 `labels`、`instances`）和 `ID_upper__kpt.json`（`objects` 中的 `class`、`coord`）。下颌同理。**只有 OBJ 和分割 JSON，不能监督训练位点。** `landmarks_root` 中可以只有一部分病例，程序取三者交集。测试识别只需要 OBJ，但完整推理还需要一个牙齿实例分割 checkpoint；位点 checkpoint 单独不能从整颌网格直接输出位点。

本文假定数据中有 `upper/ID/ID_upper.{obj,json}`、`lower/ID/ID_lower.{obj,json}` 和 `3DTeethLand_landmarks_train/{upper,lower}/ID/ID_(upper|lower)__kpt.json`。先用 `find "$DATA" -name '*__kpt.json' | head` 核对实际目录。若 Teeth3DS+ 的目录或标注名不同，先建立只包含适用文件的只读软链接视图，不要移动原始数据。官方测试集不能并入训练或验证。

## 2. 安装环境

原 `requirements.txt` 固定 PyTorch 2.3 / CUDA 12.1，不能直接照装。以下选择有 CUDA 13.0 Linux wheel 且有对应 `torch-scatter` wheel 的 PyTorch 2.12.0；它是**候选组合**，本仓库旧版 Lightning、自定义 CUDA 算子仍须在目标机编译并做冒烟验证。[PyTorch 版本目录](https://download.pytorch.org/whl/cu130/torch/)、[PyG 安装矩阵](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)。

```bash
export PROJECT=/absolute/path/to/3dteethland
export DATA=/absolute/path/to/Teeth3DS+
export EXT="$PROJECT/landmark_extension"
cd "$EXT"
nvidia-smi
nvcc --version                         # 应显示 CUDA 13.0；只有驱动不够编译 pointops
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel ninja
python -m pip install torch==2.12.0 torchvision==0.27.0 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install --no-index torch-scatter \
  -f 'https://data.pyg.org/whl/torch-2.12.0+cu130.html'
sed -E '/^--extra-index-url /d; /^-f /d; /^torch==/d; /^torchaudio/d; /^torchvision/d; /^torch-scatter/d' \
  "$PROJECT/requirements.txt" > "$EXT/requirements-linux.txt"
python -m pip install -r "$EXT/requirements-linux.txt"
python -m pip install PyYAML

# 在扩展目录中的副本上编译，避免安装过程在原项目目录生成构建文件。
mkdir -p "$EXT/build/pointops"
cp -a "$PROJECT/src" "$PROJECT/setup.py" "$EXT/build/pointops/"
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=8
python -m pip install --no-build-isolation "$EXT/build/pointops"
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
python - <<'PY'
import torch, torch_scatter, pointops, pytorch_lightning, open3d, pymeshlab
print('torch', torch.__version__, 'CUDA', torch.version.cuda)
print('GPU', torch.cuda.get_device_name(0), 'capability', torch.cuda.get_device_capability(0))
assert torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (12, 0)
PY
```

Linux 还需可用的 C++ 编译器及 `git`；缺少时安装发行版相应的编译工具包。若 wheel 下载、导入或编译失败，停在环境验证阶段，记录完整错误和 `python -m pip freeze`，不要改用仓库旧版 cu121 依赖。PyTorch wheel 自带 CUDA 运行时；`nvcc` / `CUDA_HOME` 用于本仓库 `pointops` 编译。

## 3. 准备数据视图与配置

在扩展目录运行。下面把每个 OBJ/JSON 文件链接到平坦视图，不复制或修改原始数据。训练读取器会按排序配对文件，因此要先核对缺失配对。不同病例的文件名必须唯一。

```bash
mkdir -p "$EXT/data_view"
find "$DATA/upper" "$DATA/lower" -type f \
  \( -name '*_upper.obj' -o -name '*_upper.json' -o -name '*_lower.obj' -o -name '*_lower.json' \) \
  -print0 | while IFS= read -r -d '' file; do ln -s "$file" "$EXT/data_view/$(basename "$file")"; done
python - <<'PY'
import os
from pathlib import Path
root = Path(os.environ['EXT']) / 'data_view'
meshes = {p.stem for p in root.glob('*.obj')}
labels = {p.stem for p in root.glob('*.json')}
print('OBJ', len(meshes), 'JSON', len(labels), 'unpaired', len(meshes ^ labels))
assert meshes and meshes == labels, 'missing OBJ/JSON pairs'
PY
```

复制原配置的模型结构到扩展目录，改写路径及初始显存参数。`fold: 0` 表示按病人 ID 做 5 折划分中的第 0 折；生成的 `zainab_fold_*.txt` 会落在当前扩展目录。第一次用少量 epoch 冒烟，再复制配置为正式运行版，给每次运行不同的 `version`。不要使用原配置中作者机器的路径。

```bash
python - <<'PY'
import os
from pathlib import Path
import yaml
project, data, ext = (Path(os.environ[k]) for k in ('PROJECT', 'DATA', 'EXT'))
cfg = yaml.safe_load((project / 'teethland/config/config.yaml').read_text())
cfg['version'] = 'landmarks_smoke_001'
cfg['work_dir'] = str(ext / 'runs')
cfg['datamodule'].update(
    root=str(ext / 'data_view'),
    landmarks_root=str(data / '3DTeethLand_landmarks_train'),
    fold=0, include_val_as_train=False,
    batch_size=1, num_workers=4, persistent_workers=True,
    proposal_points=8000, max_proposals=4,
)
cfg['model']['landmarks']['epochs'] = 2
cfg['accumulate_grad_batches'] = 4
(ext / 'config-landmarks-smoke.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
```

用下面的只读检查确认三方交集非零；每个 `__kpt.json` 的类别应属于 `Mesial, Distal, FacialPoint, OuterPoint, InnerPoint, Cusp`。还应抽查 OBJ 顶点数等于分割 JSON 的 `labels` / `instances` 长度。

```bash
python - <<'PY'
import os
from pathlib import Path
r = Path(os.environ['EXT']) / 'data_view'
l = Path(os.environ['DATA']) / '3DTeethLand_landmarks_train'
seg = {p.stem for p in r.rglob('*.obj')}
kpt = {p.name.split('__')[0] for p in l.rglob('*__kpt.json')}
print('mesh/seg cases:', len(seg), 'landmark cases:', len(kpt), 'usable:', len(seg & kpt))
assert seg & kpt
PY
```

## 4. 训练位点模型

从 `landmark_extension` 执行扩展入口；它调用原项目的 `TeethLandDataModule` 和 `LandmarkNet`，从外部 YAML 读取配置，把 checkpoint 存入新运行目录。

```bash
cd "$EXT"
source .venv/bin/activate
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
python run_landmark_train.py --config config-landmarks-smoke.yaml --devices 1 \
  2>&1 | tee landmarks_smoke_001.log
```

检查日志中训练/验证病例数、两轮 `loss/val`、`runs/landmarks_smoke_001/checkpoints/` 的 `last.ckpt` 与最佳 `landmarks-*.ckpt`。再复制配置，设置新运行名与计划轮数，例如：

```bash
python - <<'PY'
import os
from pathlib import Path
import yaml
ext = Path(os.environ['EXT'])
cfg = yaml.safe_load((ext / 'config-landmarks-smoke.yaml').read_text())
cfg['version'] = 'landmarks_full_001'
cfg['model']['landmarks']['epochs'] = 500
(ext / 'config-landmarks-full.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
python run_landmark_train.py --config config-landmarks-full.yaml --devices 1 \
  2>&1 | tee landmarks_full_001.log
```

显存允许后再增加 `batch_size`、`max_proposals` 或 `proposal_points`。显存不足先降低后三项。恢复时指定 `--resume /path/to/last.ckpt`，并使用**新的** `version` 目录。保存最终 YAML、`pip freeze`、训练日志和所用病例清单，保证结果可追溯。

## 5. 用训练权重识别测试 OBJ

原 `infer.py landmarks` 使用完整模型：实例分割 checkpoint + 上述位点 checkpoint。若没有实例分割 checkpoint，先从原项目 README 所述来源取得兼容权重，或另行训练实例分割；不能仅填位点权重。若不提供配准 checkpoint，下面关闭 `do_align`，要求测试 OBJ 与训练数据使用相同的姿态/尺度约定。测试文件名必须含 `_upper` 或 `_lower`，例如 `CASE_upper.obj`。

```bash
export TEST=/absolute/path/to/test_meshes   # 测试目录中只放待识别的 OBJ
export INST_CKPT=/absolute/path/to/instance.ckpt
export LAND_CKPT=/absolute/path/to/the/best/landmarks-XXX.ckpt
python - <<'PY'
import os
from pathlib import Path
import yaml
ext = Path(os.environ['EXT'])
test = Path(os.environ['TEST'])
cfg = yaml.safe_load((ext / 'config-landmarks-full.yaml').read_text())
cfg['version'] = 'landmarks_test_001'
cfg['datamodule'].update(root=str(test), fold=str(ext / 'test_files.txt'),
                         batch_size=1, num_workers=0, persistent_workers=False)
cfg['model']['do_align'] = False
cfg['model']['instseg']['checkpoint_path'] = os.environ['INST_CKPT']
cfg['model']['landmarks']['checkpoint_path'] = os.environ['LAND_CKPT']
cfg['out_dir'] = str(ext / 'predictions' / cfg['version'])
files = sorted(test.rglob('*.obj'))
assert files and all(p.stem.endswith(('_upper', '_lower')) for p in files)
(ext / 'test_files.txt').write_text(''.join(p.name + '\n' for p in files))
(ext / 'config-test.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
print('test meshes:', len(files))
PY
cd "$EXT"
python "$PROJECT/infer.py" landmarks --devices 1 --config config-test.yaml \
  2>&1 | tee landmarks_test_001.log
find "$EXT/predictions/landmarks_test_001" -name '*__kpt.json' | wc -l
```

每例预期产生 `CASE_upper__kpt.json`（`objects` 中有 `class`、`coord`、`score`）。核对输出文件数是否等于输入数，并抽看坐标与网格是否对齐；空 `objects`、少文件或异常日志都算失败。**原 `FullNet.predict_step` 会吞掉部分逐例异常**，所以不能只凭进程退出码判断成功。不要在原始测试目录中写预测结果；每次推理使用新的 `version` 和输出目录。

训练 checkpoint 的兼容性取决于模型配置和实例分割类别设置。若载入失败，先核对两次运行的完整 YAML、checkpoint 文件和 PyTorch 版本；不要强行忽略缺失权重。
