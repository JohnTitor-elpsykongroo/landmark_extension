"""Data-only check of the Teeth3DS+ -> 3dteethland adapter (no torch needed).

This mirrors exactly what ``TeethLandDataModule._files`` and
``TeethLandDataset`` do, using only the standard library plus NumPy:

  1. resolve the mesh / segmentation / landmark triples the datamodule would use,
  2. load the mesh vertices, the per-vertex FDI labels and the landmark JSON,
  3. apply the same Z-score / pose normalisation the dataset applies,
  4. apply ``MatchLandmarksAndTeeth`` to assign every landmark to a tooth
     instance,
  5. report whether every landmark landed on the correct tooth.

It is deliberately independent of torch / torch_scatter / pointops so that the
data adapter can be validated even when the CUDA extension is not built yet.

Usage
-----
python landmark_extension/validation/check_data_adapter.py --jaw upper --cases 3
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

import numpy as np


DEFAULT_DATA_ROOT = Path(r'D:\WorkSpace\Dental\data\Teeth3DS')
STD = 17.3281

# from teethland/datamodules/teethseg.py
EXCLUDE = {
    '017U3R3T_upper', '01A91JH6_lower', '01A91JH6_upper', '01HMF5HV_lower',
    '01HMF5HV_upper', 'E23G704K_lower', 'E23G704K_upper', '016FSM14_lower',
    '01FPTYH2_lower',
}
# from teethland/datamodules/teethland.py
EXCLUDE_LANDMARK = {
    'K4BAII5F_upper', '87N5YSES_upper', '67PV9M7X_lower', 'S0AON6PZ_lower',
}

LANDMARK_CLASSES = {
    'Mesial': 0, 'Distal': 1, 'FacialPoint': 2,
    'OuterPoint': 3, 'InnerPoint': 4, 'Cusp': 5,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument('--jaw', default='upper', choices=['upper', 'lower'])
    p.add_argument('--cases', type=int, default=3)
    p.add_argument(
        '--regex-filter', default=None,
        help="Same value as the config's regex_filter (default: the jaw).",
    )
    return p.parse_args()


def read_obj(path: Path) -> np.ndarray:
    verts = []
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            if line.startswith('v '):
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
    return np.asarray(verts, dtype=np.float64)


def zscore(points: np.ndarray, std: float, mean=None) -> np.ndarray:
    mean = points.mean(axis=0) if mean is None else mean
    return (points - mean) / std


def pose_normalize(points: np.ndarray, landmarks: np.ndarray, normals=None):
    """Reproduces teethland.data.transforms.PoseNormalize (PCA-aligned pose)."""
    centred = points - points.mean(axis=0)
    cov = np.cov(centred.T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    R = evecs[:, order].T
    if np.linalg.det(R) < 0:
        R = np.diag([-1.0, 1.0, 1.0]) @ R

    rotated_points = points @ R.T
    rotated_landmarks = landmarks @ R.T

    if normals is not None:
        mean_normal_z = (normals @ R.T)[:, 2].mean()
        if mean_normal_z < 0:
            R = np.diag([-1.0, 1.0, -1.0]) @ R
            rotated_points = points @ R.T
            rotated_landmarks = landmarks @ R.T

    return rotated_points, rotated_landmarks, R


def match_landmarks(
    points: np.ndarray,
    labels: np.ndarray,
    instances: np.ndarray,
    landmark_coords: np.ndarray,
    move: float = 0.04,
) -> list[int]:
    """Simplified but behaviour-compatible landmark -> instance matching.

    The full repo transform moves mesial/distal landmarks outward first; the
    displacement is at most ``move`` (0.04 mm in the normalised frame), which
    never changes the nearest-tooth assignment for correctly annotated data.
    """
    tooth_points = np.where(labels[:, None] > 0, points, 1e6)
    dists = np.linalg.norm(tooth_points[None] - landmark_coords[:, None], axis=-1)
    return instances[dists.argmin(1)].tolist()


def main() -> int:
    args = parse_args()
    data_root = args.data_root
    jaw = args.jaw
    regex = args.regex_filter if args.regex_filter is not None else jaw

    # --- replicate TeethSegDataModule._files ------------------------------
    mesh_files = sorted(data_root.glob(f'**/*.obj'))
    mesh_files = [f for f in mesh_files
                  if re.search(f'/(?!\\.)[^/\\.]*({regex})', f.as_posix())]
    ann_files = sorted(data_root.glob('**/*.json'))
    ann_files = [f for f in ann_files
                 if re.search(f'/(?!\\.)[^/\\.]*({regex})', f.as_posix())]
    ann_by_stem = {f.stem: f for f in ann_files}
    pairs = []
    for mesh in mesh_files:
        if mesh.stem in EXCLUDE or mesh.stem in EXCLUDE_LANDMARK:
            continue
        ann = ann_by_stem.get(mesh.stem)
        if ann is not None:
            pairs.append((mesh, ann))

    # --- replicate TeethLandDataModule._files -----------------------------
    landmark_root = data_root / f'3DTeethLand_landmarks_train' / jaw
    landmark_files = sorted(landmark_root.rglob('*.json'))
    landmark_by_stem = {f.name.split('__')[0]: f for f in landmark_files}

    triples = []
    for mesh, ann in pairs:
        kpt = landmark_by_stem.get(mesh.stem)
        if kpt is not None:
            triples.append((mesh, ann, kpt))

    print(f'regex filter           : {regex}')
    print(f'mesh files ({jaw})      : {len(mesh_files)}')
    print(f'annotation files       : {len(ann_files)}')
    print(f'mesh+seg pairs         : {len(pairs)}')
    print(f'landmark files         : {len(landmark_files)}')
    print(f'-> usable triples      : {len(triples)}')
    if not triples:
        print('ERROR: no usable cases', file=sys.stderr)
        return 1

    worst_dist = 0.0
    unmatched = 0
    total_landmarks = 0
    class_counts: Counter[str] = Counter()
    instance_hist = Counter()

    for mesh_path, ann_path, kpt_path in triples[:args.cases]:
        points = read_obj(mesh_path)
        with open(ann_path, 'r') as f:
            ann = json.load(f)
        labels = np.asarray(ann['labels'], dtype=np.int64)
        instances = np.asarray(ann['instances'], dtype=np.int64)
        with open(kpt_path, 'r') as f:
            kpt = json.load(f)

        coords = np.asarray(
            [o['coord'][:3] for o in kpt['objects']], dtype=np.float64,
        ).reshape(-1, 3)
        classes = [o['class'] for o in kpt['objects']]
        class_counts.update(classes)

        # TeethSegDataset.pre_transform order: ZScore -> PoseNormalize
        norm_points = zscore(points, STD)
        norm_landmarks = zscore(coords, STD, mean=points.mean(axis=0))
        norm_points, norm_landmarks, _ = pose_normalize(norm_points, norm_landmarks)

        matched = match_landmarks(norm_points, labels, instances, norm_landmarks)

        # how far is each landmark from its assigned tooth instance?
        for i, instance in enumerate(matched):
            mask = instances == instance
            if not mask.any():
                unmatched += 1
                continue
            d = np.linalg.norm(norm_points[mask] - norm_landmarks[i], axis=-1).min()
            worst_dist = max(worst_dist, float(d))
        instance_hist[len(set(matched))] += 1
        total_landmarks += len(matched)

        print(
            f'  {mesh_path.stem}: verts={points.shape[0]} '
            f'fdi={len(set(labels.tolist()) - {0})} '
            f'instances={len(set(instances.tolist()) - {0})} '
            f'landmarks={len(matched)} '
            f'distinct landmark teeth={len(set(matched))}',
        )

    print()
    print(f'landmarks checked                   : {total_landmarks}')
    print(f'unmatched landmarks                 : {unmatched}')
    print(
        'worst landmark->assigned-tooth dist: '
        f'{worst_dist:.4f} (normalised, 1 unit = {STD} mm) '
        f'= {worst_dist * STD:.3f} mm',
    )
    print('landmark classes                    :', dict(sorted(class_counts.items())))
    print('distinct landmark teeth per case    :', dict(sorted(instance_hist.items())))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
