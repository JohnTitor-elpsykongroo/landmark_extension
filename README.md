# landmark_extension

Task-local work for training a **dental landmark detection model** with
[`nnistelrooij/3dteethland`](https://github.com/nnistelrooij/3dteethland)
(ToothInstanceNet) on the **Teeth3DS+ 3DTeethLand** landmark annotations that
already exist locally.

Nothing in the original repository was modified. Everything in this folder is
additive: data checking, data/interface adaptation, environment tooling,
validation, evaluation helpers and this documentation.

Chinese version: [`README.zh-CN.md`](README.zh-CN.md)
(environment report: [`env/ENV_SETUP_REPORT.zh-CN.md`](env/ENV_SETUP_REPORT.zh-CN.md)).

---

## 1. Status summary

| Item | Status |
| --- | --- |
| Python/CUDA environment (`D:\WorkSpace\conda\envs\3dteethland`) | **READY** - torch 2.11.0+cu128 runs real CUDA ops on the RTX 5050 (sm_120) |
| Remaining pip dependencies | **READY** (see `env/ENV_SETUP_REPORT.md`) |
| `pointops` CUDA extension (kNN / ball query / FPS / stratified attention) | **BLOCKED** - needs a supported MSVC toolset (see section 6) |
| Teeth3DS+ data inventory + geometry validation | **DONE** (`check/landmark_geometry_report.json`) |
| Data adapter (Teeth3DS+ -> `TeethLandDataModule`) | **DONE and verified** (237 usable cases per jaw split, 0 unmatched landmarks) |
| Mesh loading in the training path | **WORKAROUND READY** - `pymeshlab` is blocked by Smart App Control; a dependency-free OBJ shim replaces it (`shims/`) |
| Dataset smoke test end-to-end | **PENDING** - requires `pointops` + `torch_scatter` to import |
| `LandmarkNet` forward / loss / backward | **PENDING** - same blocker |
| Training configs and command | **DONE** (`configs/landmark_upper.yaml`, `configs/landmark_lower.yaml`) |
| Evaluation / inference approach | **DOCUMENTED** (`eval/`, `infer/`, section 5) |

**Do not start training yet.** Training requires the user's explicit go-ahead.
(On the Windows development machine it also requires the environment workaround
in section 8.1; the Linux/RTX 5090 target does not.)

### Target machine: Linux + RTX 5090 + CUDA 13.0 + 128 GB RAM

This folder was developed on a Windows laptop (RTX 5050 8 GB, driver 596.58).
Every environment problem described below - the MSVC 14.51 host-compiler
rejection, Smart App Control blocking unsigned DLLs, `pymeshlab` failing to
load - is **specific to Windows and to that machine**. None of it exists on the
Linux target:

```bash
bash landmark_extension/env/setup_linux.sh
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `ENV_PREFIX` | `$HOME/envs/3dteethland` | where the environment is created |
| `CUDA_INDEX` | `https://download.pytorch.org/whl/cu128` | torch wheel index |
| `PYG_CUDA` | `cu128` | torch-scatter wheel suffix, must match torch |
| `TORCH_CUDA_ARCH_LIST` | `12.0` | RTX 5090 is sm_120 |
| `SKIP_TOOLKIT` | `0` | set to `1` when a CUDA toolkit is already present |

The script creates the env, installs a CUDA 12.8 toolkit (for `nvcc`), installs
cu128 torch, installs the project requirements plus the undeclared
`scikit-learn` / `pandas` / `scikit-multilearn` imports, installs
`torch-scatter` (prebuilt wheel, falling back to a source build), builds
`pointops` in place, and finishes with a self-check that runs a real CUDA matmul
and a real `pointops.farthestPointSampling` call.

`configs/landmark_<jaw>.yaml` is already tuned for the 5090
(`batch_size: 32`, `num_workers: 8`; the repository baseline is 8 / 4), and the
config comments explain when to revisit `lr` and `max_proposals`. See section 7
of the Chinese README (`README.zh-CN.md`) for the fuller rationale, including the
cross-platform data/path findings (all 240 landmark JSON `key` fields are flat
filenames, so nothing depends on Windows path separators).

**Prefer to do it by hand?** See section 7 "Manual command-line setup" - the
same steps as `setup_linux.sh`, written out as copy-pasteable commands, plus
the steps the script does *not* cover (where to put `landmark_extension/`, how
to verify the dataset, how to repoint the configs, how to validate, how to
start training, how to run inference/evaluation).

### This directory is its own git repository

Remote: <https://github.com/JohnTitor-elpsykongroo/landmark_extension>
(branch `main`).

Its git directory deliberately lives **outside** the work tree, so the
surrounding 3dteethland checkout sees `landmark_extension/` as a plain
directory rather than a nested repository / gitlink:

```
<3dteethland>/
  landmark_extension/        <- work tree (this directory, regular files, no .git)
  landmark_extension.git/    <- git directory (HEAD, objects, refs, ...)
  .git/
  .gitignore                 <- contains landmark_extension/ and landmark_extension.git/
```

So do **not** move `landmark_extension.git` back to
`landmark_extension/.git` - that would make the outer repository treat it as a
nested repo again.

To commit or push from this repository:

```bash
# preferred: the bundled wrapper
bash landmark_extension.git/git.sh status -sb
bash landmark_extension.git/git.sh add -A
bash landmark_extension.git/git.sh commit -m "message"
bash landmark_extension.git/git.sh push

# or by hand
git --git-dir=landmark_extension.git --work-tree=landmark_extension status -sb
git --git-dir=landmark_extension.git --work-tree=landmark_extension commit -am "message"
```

On another machine (for example the RTX 5090 box) you can simply
`git clone https://github.com/JohnTitor-elpsykongroo/landmark_extension` into a
scratch directory when you only need to edit and push back; the work-tree layout
above is only needed when it has to sit inside the 3dteethland checkout.

---

## 2. How the model trains (from the source, not the README)

Entry point: `train.py landmarks` -> `TeethLandDataModule` + `LandmarkNet`.

### 2.1 Model

`teethland/models/landmarknet.py::LandmarkNet` wraps
`nn.StratifiedTransformer` (`teethland/nn/modules/stratified_transformer.py`)
configured by `model.landmarks` in the YAML:

```
out_channels: [1, 4, 4, 4, 4, 4]     # -> six heads, 11 output channels in total
channels_list: [48, 96, 192, 256]
depths: [3, 9, 3]
heads_list: [6, 12, 24]
window_sizes: [0.1, 0.2, 0.4]        # landmark config; the seg config uses 0.4/0.8/1.6
point_embedding: KPConv(influence 0.02, radius 0.05)
```

`LandmarkNet.forward` returns `(seg, mesial_distal, facial, outer, inner, cusps)`.
Each of the five landmark heads emits `(distance, dx, dy, dz)`: per point, how
close it is to the landmark and how far the landmark is from that point.

Note that the repository config assigns **4 channels to every landmark head**,
including the single-class `facial` / `outer` / `inner` heads; the stock config
in `teethland/config/config.yaml` is what this task keeps, since it is what the
published checkpoints were trained with.

### 2.2 Losses (`LandmarkNet.training_step`)

```
loss = BCEWithLogits(seg, tooth_mask)
     + LandmarkLoss(mesial_distal, gt, classes=[0, 1])
     + LandmarkLoss(facial,        gt, classes=[2])
     + LandmarkLoss(outer,         gt, classes=[3])
     + LandmarkLoss(inner,         gt, classes=[4])
     + LandmarkLoss(cusps,         gt, classes=[5])
```

`teethland/nn/modules/criterion.py::LandmarkLoss` combines a SmoothL1 distance
term with a masked Chamfer term and a separation term
(`dist_thresh=0.10`, `w_dist=0.05`, `w_chamfer=0.05`, `w_separation=0.0005`).
**All constants are in the normalised frame**, where 1 unit = `STD = 17.3281` mm
(`TeethSegDataset.STD`), so `dist_thresh=0.10` is about 1.7 mm and the
`LandmarkF1Score` threshold of 17.3281 is 1.0 normalised unit.

`train.py` checkpoints on `loss/val` and on `dice/val` (`mode='min'`) - note
that the landmark stage monitors the **segmentation dice**, not a landmark
metric, so "best checkpoint" for landmarks is not what the stock runner picks.
See section 5.4 for the recommended change if you want landmark-based
checkpoint selection.

### 2.3 Optimisation

AdamW + `CosineAnnealingLR` after a `LinearWarmupLR`, parameter groups from
`StratifiedTransformer.param_groups` (transformers get `lr * transformer_lr_ratio`),
gradient clipping at 35 (`config['gradient_clip_norm']`).

---

## 3. What the training data must look like

`TeethLandDataModule` extends `TeethSegDataModule` and needs a **single jaw per
data module**:

```
<root>/<split>/<case>/<case>_<jaw>.obj     intra-oral scan mesh  (obj/ply/stl)
<root>/<split>/<case>/<case>_<jaw>.json    {"labels": [FDI per vertex],
                                            "instances": [tooth id per vertex],
                                            ...}
<landmarks_root>/<...>/<case>_<jaw>__kpt.json
                                           {"objects": [{"class": "...",
                                                         "coord": [x, y, z]}, ...]}
```

Key contract points confirmed in the source:

* `TeethSegDataModule._files` matches mesh and segmentation **by file stem**;
  the subdirectory layout is irrelevant (both are globbed recursively).
* `TeethLandDataModule._files` intersects the segmentation stems with the
  landmark stems, taking `kpt_file.name.split('__')[0]` as the stem. Because the
  filename already ends in `_<jaw>`, a **single `landmarks_root` must contain
  exactly one jaw**.
* The landmark JSON must be the `__kpt.json` variant, i.e. the file whose name
  contains the double underscore before `kpt`. Passing the landmark tree that
  also contains segmentation JSONs would silently mix them in.
* Label values are FDI (1x-4x permanent, 5x-8x primary); `0` is gingiva.
* The landmark class names must be exactly:
  `Mesial`, `Distal`, `FacialPoint`, `OuterPoint`, `InnerPoint`, `Cusp`
  (`TeethLandDataset.landmark_classes`). Anything else raises `KeyError`.
* `num_classes` is 5 and the segmentation targets are `labels - 1`
  (`-1` = background) because `label_as_instance` is False.

### 3.1 Correspondence between mesh / segmentation / FDI / landmark

This was verified numerically for every case in this dataset; see section 10.

| Artifact | Meaning |
| --- | --- |
| `.obj` | triangle mesh of the jaw; vertex order is the reference order |
| `<case>_<jaw>.json` `labels[i]` | FDI number of mesh vertex `i` |
| `<case>_<jaw>.json` `instances[i]` | instance id of mesh vertex `i` (0 = not a tooth) |
| `instances` + `labels` | one tooth = one instance with a constant FDI label |
| `__kpt.json` `objects[k].coord` | 3D point **in the same coordinate frame as the mesh vertices** |
| `__kpt.json` `objects[k].class` | one of the six landmark classes |

Landmarks are matched to teeth by `teethland/data/transforms.py::MatchLandmarksAndTeeth`,
which moves mesial/distal landmarks slightly outward and then assigns each
landmark to the nearest **tooth** vertex's instance. The instance id is what
later makes `GenerateProposals` keep "the landmarks of the sampled teeth".

---

## 4. How Teeth3DS+ is adapted (no data is copied or modified)

The local data already has the right layout, so **no extraction, conversion or
duplication is needed**. What the adapter does is narrow the repository's file
discovery down to the 3DTeethLand cases and give the data module the jaw-scoped
inputs it expects:

| Config key | Value | Reason |
| --- | --- | --- |
| `root` | `D:/WorkSpace/Dental/data/Teeth3DS` | one tree containing `<jaw>/<case>/...` |
| `regex_filter` | `upper` / `lower` | the repo filters on the path segment name, so the jaw subdirectory selects the jaw |
| `landmarks_root` | `.../3DTeethLand_landmarks_train/<jaw>` | must contain exactly one jaw |
| `extensions` | `['obj']` | local scans are OBJ |
| `fold` | `landmark_extension/prepare/landmark_<jaw>_fold_0.txt` | deterministic subject-level split |

`prepare/prepare_landmark_training_data.py` only *writes* auxiliary artifacts
(case index CSV, dataset summary JSON, fold files). It never writes to
`D:\WorkSpace\Dental\data\Teeth3DS`.

### 4.1 Reproduced repository exclusions

The repository hardcodes exclusions that must be respected:

* `TeethSegDataModule` (mesh+segmentation): `017U3R3T_upper`, `01A91JH6_lower`,
  `01A91JH6_upper`, `01HMF5HV_lower`, `01HMF5HV_upper`, `E23G704K_lower`,
  `E23G704K_upper`, `016FSM14_lower`, `01FPTYH2_lower`
* `TeethLandDataModule` (landmarks): `K4BAII5F_upper`, `87N5YSES_upper`
  (missing tooth segmentation), `67PV9M7X_lower`, `S0AON6PZ_lower`
  (missing more than one landmark)

### 4.2 Geometry validation (`check/check_landmark_geometry.py`)

All 236 landmark cases (118 upper + 118 lower, after exclusions) were checked:

| Metric | Upper | Lower |
| --- | --- | --- |
| cases | 118 | 118 |
| mesh-vertex count == segmentation-label count | 118/118 | 118/118 |
| mean distance landmark -> nearest mesh vertex | 0.0685 | 0.0700 |
| worst per-case mean distance | 0.173 | 0.180 |
| landmarks whose nearest vertex is gingiva (`label == 0`) | 616 / 10260 (6.0%) | 630 / 10812 (5.8%) |
| average distinct FDI labels per case | 13.3 | 13.4 |

(All distances in mm.) The conclusion is that **the landmark JSONs share the
mesh coordinate frame exactly** and that the segmentation is per vertex and
index-aligned with the mesh.

The runtime exclusion lists were applied: `K4BAII5F_upper` and `87N5YSES_upper`
(no tooth segmentation) and `67PV9M7X_lower` and `S0AON6PZ_lower` are dropped
from the upper/lower landmark sets, and `017U3R3T_upper` - which has landmarks
but no tooth segmentation either - is dropped by the repository's
`TeethSegDataModule` list. That is why 120 landmark directories yield 118
trainable cases per jaw.

The ~6% of landmarks whose nearest vertex is gingiva are marginal landmarks
(e.g. `InnerPoint` on the gingival margin); their distance to the nearest mesh
point is still ~0.07 mm, so `MatchLandmarksAndTeeth` still assigns them to the
correct adjacent tooth. The worst outlier case has a mean offset of only
0.18 mm, i.e. well below the 1.7 mm loss threshold.

### 4.3 Data adapter validation (`validation/check_data_adapter.py`)

A torch-free reimplementation of the adapter's file resolution plus the
dataset's normalisation and landmark matching, run on real cases:

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

So every landmark is assigned to a real tooth instance and lands well inside
the configured thresholds.

### 4.4 Dataset scale and split

| Jaw | Train cases (usable) | Test cases (landmark only) |
| --- | --- | --- |
| upper | 118 | 50 |
| lower | 118 | 50 |

Landmarks per split: 10 464 (upper train), 10 930 (lower train);
class distribution per jaw and split: ~1 580 `Mesial`, ~1 580 `Distal`,
~1 575 `FacialPoint`, ~1 575 `OuterPoint`, ~1 575 `InnerPoint`,
~2 600-3 000 `Cusp`.

**The `3DTeethLand_landmarks_test` split has no segmentation JSONs** (only the
`__kpt.json` landmarks and the meshes, which do exist in
`Teeth3DS/<jaw>/<case>/<case>_<jaw>.obj`). Consequence:

* it can be used for landmark evaluation, but only with a segmentation/FDI
  source that is not the local Teeth3DS folder, i.e. with the full
  alignment+instance pipeline (`FullNet`), or by evaluating the landmark head
  alone (see `infer/predict_landmarks.py`).
* `TeethLandDataModule` cannot load it at all, because it requires the
  segmentation+landmark intersection.

Fold files (`prepare/landmark_<jaw>_fold_<k>.txt`) put each subject in exactly
one of 5 folds: 24/24/24/23/23 cases per jaw. `configs/landmark_<jaw>.yaml`
uses fold 0 as validation (~24 cases, 20%) and the other ~94 as training, with
`include_val_as_train: False`.

---

## 5. Training, evaluation and inference

### 5.1 Training command (ready, but do not run yet)

```powershell
# upper jaw
& 'D:\WorkSpace\conda\envs\3dteethland\python.exe' `
    landmark_extension\run_landmark.py `
    --config landmark_extension\configs\landmark_upper.yaml `
    --devices 1

# lower jaw
& 'D:\WorkSpace\conda\envs\3dteethland\python.exe' `
    landmark_extension\run_landmark.py `
    --config landmark_extension\configs\landmark_lower.yaml `
    --devices 1
```

`run_landmark.py` calls the unmodified `train.main('landmarks', ...)`; it only
swaps `teethland/config/config.yaml` for the task config for the duration of the
run and restores the original file afterwards (byte-for-byte).

Resume / warm start:

```powershell
... run_landmark.py --config landmark_extension\configs\landmark_upper.yaml --checkpoint <ckpt>
```

### 5.2 Main training parameters (`configs/landmark_<jaw>.yaml`)

| Parameter | Value | Note |
| --- | --- | --- |
| `model.landmarks.epochs` | 500 | stock value |
| `lr` / `weight_decay` | 6e-4 / 1e-4 | stock values |
| `warmup_epochs` | 5 | `LinearWarmupLR` then cosine |
| `transformer_lr_ratio` | 0.1 | transformer blocks get 6e-5 |
| `batch_size` | 8 | 8 proposals per scan x 8 scans |
| `num_workers` | 4 | `persistent_workers: true` |
| `uniform_density_voxel_size` | `[0.025, 0.01]` | the landmark stage uses index 1 = 1 cm voxels |
| `proposal_points` | 12 000 | points per tooth proposal |
| `max_proposals` | 8 | teeth sampled per scan and epoch |
| `sampler` | `balanced` | `InstanceBalancedSampler` |
| `gradient_clip_norm` | 35 | |
| `accumulate_grad_batches` | 1 | |

**Expected memory**: ~8 scans x 8 proposals x 12 000 points = ~770 k points per
batch. This is the parameter to lower first on the 8 GB RTX 5050
(`batch_size: 4` or `max_proposals: 6`); the smoke test will report the actual
peak memory.

### 5.3 Model / checkpoint

* Architecture: `LandmarkNet` = `StratifiedTransformer` with the landmark head
  configuration above, trained **from scratch** (`checkpoint_path: null`).
* No pretrained checkpoint is present locally. `teethland/config/config.yaml`
  references `checkpoints/landmarks_full.ckpt` and `checkpoints/landmarks_ablation.ckpt`,
  and the README points to a Google Drive folder; **neither exists in this
  checkout**, so nothing can be resumed from or warm-started with today. If you
  want the published weights, they have to be downloaded first.
* Best-checkpoint selection: the stock `train.py` monitors `loss/val` and
  `dice/val`. Because the landmark metric is not logged, the recommended
  (minimal) change is to add a `LandmarkMeanAveragePrecision` update in
  `LandmarkNet.validation_step` - that code already exists but is commented out.
  Treat that as an optional follow-up; it is the only place where the stock
  repository would need a real change.

### 5.4 Evaluation

The repository ships the **official challenge scorer**
(`evaluation/3dteethland.py`): AP/AR per class over distance thresholds
0-2.9 mm, summarised as `AP_cusp`, `AP_mesial_distal`, `AP_inner_outer`,
`AP_facial`, `mAP` (and the AR equivalents). It is driven by two inputs:

* `--predictions_file`: CSV `key, coord_x, coord_y, coord_z, class, score`
* `--goldstandard_file`: pickle `{class: {mesh_name: [coord, ...]}}`

`eval/score_predictions.py` builds that gold standard from the local landmark
JSONs and calls the repository scorer directly:

```powershell
& 'D:\WorkSpace\conda\envs\3dteethland\python.exe' landmark_extension\eval\score_predictions.py make-gold `
    --jaw upper --fold landmark_extension\prepare\landmark_upper_fold_0.txt

& 'D:\WorkSpace\conda\envs\3dteethland\python.exe' landmark_extension\eval\score_predictions.py score `
    --predictions landmark_extension\preds\upper\predictions.csv `
    --goldstand   landmark_extension\eval\gold_upper_landmark_upper_fold_0.pkl `
    --out         landmark_extension\eval\scores_upper_fold_0.json
```

The gold-standard builder has already been run for fold 0 of the upper jaw
(24 cases, 2 055 landmarks) and works.

In-training validation metrics: `LandmarkF1Score` (Hungarian matching at a
1.0-unit = 17.33 mm threshold) and `LandmarkMap` (mAP at 0.5/1/2/3 mm) exist in
`teethland/metrics/`, wired into `LandmarkNet` but bypassed in
`validation_step` (the body is commented out). Enabling it is the same minimal
change as in 5.3.

### 5.5 Inference

Two options, and their current caveats:

1. **Landmark stage only** (matches this task's "reuse segmentation/FDI"
   requirement): `infer/predict_landmarks.py`.
   It reads the mesh plus the existing per-vertex FDI/instance segmentation,
   builds the tooth proposals with the repository's own transforms without
   running the alignment or segmentation models, runs `LandmarkNet`, and writes
   `<case>_<jaw>__kpt.json` plus `predictions.csv`.
   **Status: design-verified against the source, not yet executed** - it needs
   `pointops` and a trained checkpoint, so it can only be run once the
   environment is complete (section 7 or 8.1, depending on the machine).

2. **Full pipeline** (`infer.py`, `FullNet`, i.e. alignment -> instance
   segmentation+FDI -> per-tooth landmarks). This is what the challenge
   submission uses, and it needs the four checkpoints from Google Drive.
   Two defects were found in that path and are *not* fixed (out of scope):
   * `TeethInstFullDataModule.collate_fn` returns 6 elements but
     `transfer_batch_to_device` unpacks 5, and `self.affine` is then read from
     the data module where it never gets assigned - so the `predict` path
     crashes before landmark post-processing.
   * `FullNet.landmarks_process` merges each tooth's mesial and distal landmark
     into one detection and relies on `process_landmarks` to split them by
     geometry, which is the source of the `mesial_distal` class in the
     prediction CSV.

### 5.6 Known gaps / honest caveats

* `LandmarkNet.validation_step` computes and logs losses only; all landmark
  metrics are commented out. Without a change, validation will not tell you how
  good the landmarks are.
* The landmark heads regress `(distance, dx, dy, dz)` **relative to proposal
  points**, so predictions must be clustered (`PointTensor.cluster` with
  `dbscan_cfg`) or reduced per tooth before converting to absolute
  coordinates; this is what `infer/predict_landmarks.py` does per tooth with an
  argmax over the confidence head. That reduction is a design choice, not
  something the repository fixes for you.
* The repository's landmark classes treat mesial and distal as one head, so the
  class labels in the prediction CSV need `process_landmarks`-style geometric
  disambiguation to match the challenge's separate `Mesial`/`Distal` classes.

---

## 6. Environment

> **The target deployment machine is Linux + RTX 5090 + CUDA 13.0 + 128 GB RAM.**
> The two blockers in section 8.1 (MSVC toolset version, Smart App Control) are
> specific to the current Windows laptop and do not exist on Linux. Use
> `env/setup_linux.sh` there - see the target-machine section at the top.
> The rest of this section is the record of how the Windows machine was
> brought up, kept for traceability.

### 6.1 What is ready

`D:\WorkSpace\conda\envs\3dteethland\python.exe` (Python 3.10.21):

| Package | Installed |
| --- | --- |
| torch / torchvision / torchaudio | 2.11.0+cu128 / 0.26.0+cu128 / 2.11.0+cu128 |
| `torch.cuda` on RTX 5050 (sm_120) | verified with a real matmul |
| numpy | 1.26.4 (project pins `<2`) |
| open3d / pymeshlab / opencv-python | 0.17.0 / 2023.12.post1 / 4.10.0.84 |
| pytorch-lightning / torchmetrics / tensorboard | 2.3.3 / 1.9.0 / 2.17.0 |
| timm / lxml / gco-wrapper / pytest / torchtyping | 1.0.7 / 5.2.2 / 3.0.9 / 8.2.2 / 0.1.4 |
| torch-scatter | 2.1.2+pt211cu128 (installed, but blocked at load - see 8.1) |
| scikit-learn, scipy, scikit-multilearn | 1.7.2, 1.15.3, 0.2.0 (scikit-learn is imported by the repo but missing from `requirements.txt`) |
| CUDA toolkits in the env | `Library` currently holds **CUDA 13.0** (`nvcc` 13.0.88); CUDA 12.8 was installed first and is superseded |

Why cu128 and not the pinned cu121: this machine's NVIDIA driver (596.58)
has dropped CUDA 12.x compatibility and the GPU is sm_120, so torch 2.3.0+cu121
from `requirements.txt` cannot run here. torch 2.11.0+cu128 verifiably does.

Two more silent `requirements.txt` gaps: the repository imports `sklearn`
(`scikit-learn`) and `pandas` without listing them. Both are installed.

torch 2.11 also changes `torch.Tensor` internals enough that **`torchtyping`
0.1.4 raises `RuntimeError: Cannot subclass _TensorBase directly`** at import
time. The repository only uses `TensorType` in annotations, so this is only a
problem for tooling that actually imports torchtyping; if it bites, the fix is
either torch 2.8.0+cu128 (available for Python 3.10) or patching
`torchtyping/tensor_type.py` to derive from `torch.Tensor` instead of
`torch._TensorBase`.

## 7. Manual command-line setup (no shell script, Linux target)

This section assumes you have exactly three things: the original **3dteethland**
repository, this **`landmark_extension/`** folder, and the raw **Teeth3DS+**
dataset. Every command below is copy-pasteable. Adjust the three variables to
your machine; the values used while developing this were
`ENV_PREFIX=$HOME/envs/3dteethland`, `DATA_ROOT=/data/Teeth3DS`,
`REPO=/path/to/3dteethland`.

### Step 0 - preconditions

```bash
nvidia-smi                          # expect the RTX 5090 + the CUDA version the driver reports
conda --version                     # or mamba
gcc -dumpfullversion -dumpversion   # CUDA 12.8 requires gcc <= 13
```

No `nvcc` yet is fine - step 4 installs it.

### Step 1 - get the repository and drop `landmark_extension` into it

```bash
git clone https://github.com/nnistelrooij/3dteethland.git
cd 3dteethland

# place landmark_extension next to teethland/, train.py and setup.py
cp -r /path/to/landmark_extension ./landmark_extension
# or, if you keep it as its own repo:
# git clone <landmark_extension-url> landmark_extension

ls landmark_extension/README.md landmark_extension/configs/landmark_upper.yaml
```

The location matters: `run_landmark.py` and the `fold` paths in
`configs/*.yaml` are resolved assuming `landmark_extension/` sits at the
repository root.

### Step 2 - stage the dataset and check the layout

```bash
export DATA_ROOT=/data/Teeth3DS      # change to your path
ls "$DATA_ROOT"                      # upper/ lower/ 3DTeethLand_landmarks_train/ ...
ls "$DATA_ROOT/upper" | head         # one directory per case
ls "$DATA_ROOT/upper/01328DDN"       # 01328DDN_upper.obj + 01328DDN_upper.json
ls "$DATA_ROOT/3DTeethLand_landmarks_train/upper" | head
ls "$DATA_ROOT/3DTeethLand_landmarks_train/upper/013TXGFK"   # 013TXGFK_upper__kpt.json

# expected counts: 950 / 950 / 120 / 120
ls -d "$DATA_ROOT"/upper/*/     | wc -l
ls -d "$DATA_ROOT"/lower/*/     | wc -l
ls -d "$DATA_ROOT"/3DTeethLand_landmarks_train/upper/*/ | wc -l
ls -d "$DATA_ROOT"/3DTeethLand_landmarks_train/lower/*/ | wc -l
```

### Step 3 - create the environment

```bash
export ENV_PREFIX=$HOME/envs/3dteethland
conda create -y -p "$ENV_PREFIX" python=3.10
export PY="$ENV_PREFIX/bin/python"
"$PY" -m pip install --upgrade pip wheel setuptools
```

### Step 4 - install a CUDA toolkit that provides `nvcc`

```bash
conda install -y -p "$ENV_PREFIX" -c nvidia cuda-toolkit=12.8
export CUDA_HOME="$ENV_PREFIX"
export CUDA_PATH="$CUDA_HOME"
"$ENV_PREFIX/bin/nvcc" --version
```

### Step 5 - install torch with sm_120 support

```bash
"$PY" -m pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
"$PY" -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Expect something like `2.11.0+cu128 12.8 True`. Do **not** install
`torch==2.3.0` from `requirements.txt`; it has no sm_120 support.

### Step 6 - install the remaining requirements by hand

`requirements.txt` mixes the torch packages (handled above) with plain
dependencies, so install the equivalent set directly:

```bash
"$PY" -m pip install "numpy<2.0.0" lxml==5.2.2 opencv-python==4.10.0.84 \
    open3d==0.17.0 gco-wrapper==3.0.9 pymeshlab==2023.12.post1 pytest==8.2.2 \
    pytorch-lightning==2.3.3 tensorboard==2.17.0 timm==1.0.7 torchtyping==0.1.4

# imported by the repo but missing from requirements.txt
"$PY" -m pip install scikit-learn pandas scikit-multilearn

# must match your torch version and CUDA suffix exactly
"$PY" -m pip install torch-scatter \
    -f https://data.pyg.org/whl/torch-2.11.0+cu128.html
```

If step 5 gave you a different torch version, change the `2.11.0` in that URL
(check with `"$PY" -c "import torch; print(torch.__version__)"`). If no wheel
exists there, build from source:
`"$PY" -m pip install --no-build-isolation torch-scatter`.

### Step 7 - build the pointops CUDA extension

```bash
cd /path/to/3dteethland
export TORCH_CUDA_ARCH_LIST=12.0        # RTX 5090 = sm_120
export PATH="$CUDA_HOME/bin:$ENV_PREFIX/bin:$PATH"
"$PY" setup.py build_ext --inplace      # fallback: "$PY" -m pip install -v -e .

# must pass, otherwise training dies on the first kNN / sampling call
"$PY" -c "
import torch, pointops
print('pointops:', pointops.__file__)
x = torch.randn(4096, 3, device='cuda')
idx, n = pointops.farthestPointSampling(x, torch.tensor([4096], device='cuda'), 0.25, 1024)
print('fps ok:', tuple(idx.shape), idx.dtype)
"
```

### Step 8 - repoint the three paths in the configs

In `landmark_extension/configs/landmark_upper.yaml` and `landmark_lower.yaml`:

```yaml
datamodule:
  root: '/data/Teeth3DS'
  landmarks_root: '/data/Teeth3DS/3DTeethLand_landmarks_train/upper'  # lower.yaml uses lower
  fold: 'landmark_extension/prepare/landmark_upper_fold_0.txt'        # lower.yaml uses lower
```

`fold` may stay relative (it resolves against the repository root). To
regenerate the folds for a different data root, pass `--data-root` explicitly -
the script default is the Windows path:

```bash
"$PY" landmark_extension/prepare/prepare_landmark_training_data.py --data-root /data/Teeth3DS
```

### Step 9 - validate the data path (no training)

```bash
cd /path/to/3dteethland
"$PY" landmark_extension/validation/check_data_adapter.py --jaw upper --cases 3
"$PY" landmark_extension/validation/smoke_test_landmark_pipeline.py --jaw upper --batch-size 32
```

The second one runs dataset -> collate -> forward -> loss -> backward once and
writes `peak_memory_gb` into `landmark_extension/validation/smoke_report.json`.
If it does not fit, lower `--batch-size` and update `batch_size` in the config.

### Step 10 - start training

```bash
cd /path/to/3dteethland

"$PY" landmark_extension/run_landmark.py \
    --config landmark_extension/configs/landmark_upper.yaml --devices 1

"$PY" landmark_extension/run_landmark.py \
    --config landmark_extension/configs/landmark_lower.yaml --devices 1
```

Add `--checkpoint <ckpt>` to resume. Multi-GPU: `--devices 2` (`train.py`
enables `sync_batchnorm` when `devices > 1`).

Outputs:

* checkpoints and TensorBoard logs: `landmark_extension/runs/<version>/`
  (`version` comes from the config: `upper` / `lower`); view with
  `tensorboard --logdir landmark_extension/runs`.
* the dataset preprocessing cache is a `<hash>.pkl` written to the **current
  working directory**, so always start from the repository root. Deleting it is
  safe - it just re-preprocesses.

### Step 11 (optional) - inference and scoring

```bash
"$PY" landmark_extension/infer/predict_landmarks.py \
    --checkpoint landmark_extension/runs/upper/checkpoints/<ckpt> --jaw upper --limit 4

"$PY" landmark_extension/eval/score_predictions.py make-gold \
    --jaw upper --fold landmark_extension/prepare/landmark_upper_fold_0.txt
"$PY" landmark_extension/eval/score_predictions.py score \
    --predictions landmark_extension/preds/upper/predictions.csv \
    --goldstand   landmark_extension/eval/gold_upper_landmark_upper_fold_0.pkl \
    --out         landmark_extension/eval/scores_upper_fold_0.json
```

### 7.1 How the manual steps map onto `setup_linux.sh`

| Manual step | Stage in `setup_linux.sh` |
| --- | --- |
| step 3 env | `[1/7]` |
| step 4 CUDA toolkit | `[2/7]` |
| step 5 torch | `[3/7]` |
| step 6 other deps | `[4/7]` + `[5/7]` |
| step 7 pointops build | `[6/7]` |
| step 7 self-check | `[7/7]` |
| steps 1, 2, 8-11 | not covered by the script - always manual |

In other words the script only covers the *environment*; placing the project,
pointing at the data, validating and the training command are manual either way.

## 8. Environment record (Windows development machine)

### 8.1 The one remaining blocker (two options)

Two Windows-level facts block the CUDA extensions:

1. **Smart App Control is ON** (`HKLM\SYSTEM\CurrentControlSet\Control\CI\Policy`
   `VerifiedAndReputablePolicyState = 1`, `SAC_EnforcementReason = 1`).
   Code Integrity logs confirm it refusing to load unsigned extension modules,
   e.g. `torch_scatter\_scatter_cuda.pyd`
   (events `Microsoft-Windows-CodeIntegrity/Operational` 3077/3033/3118).
   Locally compiled binaries are accepted, but prebuilt unsigned wheels are not.
2. **The installed MSVC is 14.51 (Visual Studio 18 BuildTools)** and no
   supported older toolset is present. `nvcc` 12.8 *and* 13.0 both abort with
   `cudafe++ died with status 0xC0000005 (ACCESS_VIOLATION)` on a trivial
   `.cu` file - CUDA does not support MSVC 19.5x yet
   (CUDA 13 supports 19.29-19.4x). Adding the older toolset to the Visual
   Studio installation needs an elevated process, which this session does not
   have: `vs_installer.exe modify ... --quiet` exits with code 5007,
   *"Commands with --quiet or --passive should be run elevated from the
   beginning."*

**Option A (recommended).** In an **administrator** PowerShell:

```powershell
& 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vs_installer.exe' modify `
  --installPath 'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools' `
  --quiet --norestart `
  --add Microsoft.VisualStudio.Component.VC.14.44.17.14.x86.x64

powershell -ExecutionPolicy Bypass -File D:\WorkSpace\Dental\3dteethland\landmark_extension\env\build_pointops.ps1
```

Then also allow the prebuilt `torch_scatter` DLL to load - either rebuild it
locally against CUDA 13 (so Smart App Control accepts it):

```powershell
& 'D:\WorkSpace\conda\envs\3dteethland\python.exe' -m pip install --no-binary torch-scatter torch-scatter
```

or turn Smart App Control off (Windows Security -> App & browser control ->
Smart App Control; note this cannot be re-enabled without reinstalling Windows).

**Option B.** Keep the current toolchain and instead implement a pure-PyTorch
replacement for the `pointops` + `torch_scatter` kernels. Farthest point
sampling, kNN and ball query are straightforward, but the CRPE attention
backward passes are hand-written CUDA kernels, so this is only worth doing with
a deliberate numerical-parity test plan, and it would deviate from the
reference model. Not attempted here.

`env/build_pointops.ps1` and `env/run_pointops_build.py` are already prepared
for Option A: they import the MSVC environment, point `CUDA_HOME` at the conda
CUDA toolkit, set `TORCH_CUDA_ARCH_LIST=12.0`, work around a torch/locale
crash by rebinding `subprocess.SUBPROCESS_DECODE_ARGS` to UTF-8, and set
`DISTUTILS_USE_SDK=1`.

See `env/ENV_SETUP_REPORT.md` for the full chronological record of every
environment problem and its resolution.

---

## 9. Folder layout

```
landmark_extension/
  README.md                              <- this document
  README.zh-CN.md                        <- Chinese version of this document
  run_landmark.py                        <- runs train.main() with a task config
  train_landmarks.sh                     <- reference invocation
  configs/landmark_upper.yaml            <- upper-jaw landmark training config
  configs/landmark_lower.yaml            <- lower-jaw landmark training config
  prepare/prepare_landmark_training_data.py   <- inventory + fold generation
  prepare/landmark_cases_train.csv            <- 236-case index (paths, counts, labels)
  prepare/landmark_dataset_summary_train.json <- dataset statistics
  prepare/landmark_<jaw>_fold_<k>.txt         <- 5-fold subject splits
  check/check_landmark_geometry.py       <- landmark/mesh/segmentation alignment check
  check/landmark_geometry_report.json    <- its full per-case output
  check/visualize_landmarks.py           <- FDI-coloured OBJ + landmark spheres
  check/visualizations/                  <- exported OBJ files
  validation/check_data_adapter.py       <- torch-free adapter verification
  validation/smoke_test_landmark_pipeline.py <- dataset+collate+forward+loss+backward
  eval/score_predictions.py              <- official-score-compatible evaluation
  eval/gold_upper_landmark_upper_fold_0.pkl  <- gold standard for validation fold 0
  infer/predict_landmarks.py             <- landmark-only inference adapter
  env/build_pointops.ps1                 <- MSVC-aware pointops build
  env/run_pointops_build.py              <- its python helper (locale workaround)
  env/pointops_build.log                 <- last build log
  env/ENV_SETUP_REPORT.md                <- environment record
  env/ENV_SETUP_REPORT.zh-CN.md          <- environment record (Chinese)
  runs/                                  <- training runs (TensorBoard + checkpoints)
  preds/                                 <- inference output
```

## 10. Next steps for the user

1. Copy `data/Teeth3DS` to the RTX 5090 Linux machine (keeping the directory
   layout) and point `root` / `landmarks_root` / `fold` in
   `configs/landmark_<jaw>.yaml` at it.
2. Run `bash landmark_extension/env/setup_linux.sh`; it self-verifies torch
   CUDA, torch-scatter and pointops at the end.
3. Run
   `validation/smoke_test_landmark_pipeline.py --jaw upper --batch-size 32`
   for the dataset/collate/forward/loss/backward evidence, and read
   `peak_memory_gb` from `validation/smoke_report.json` before fixing the final
   batch size.
4. Decide on data scale, epochs, batch size, whether to scale the learning rate
   with the batch size, and whether to enable the landmark metrics/checkpointing
   change from section 5.3-5.4.
5. Only then start training.

The remaining MSVC / Smart App Control work in section 8.1 can be skipped
entirely unless you intend to train on this Windows machine.
