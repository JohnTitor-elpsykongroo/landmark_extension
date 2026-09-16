"""Landmark-only inference adapter for the 3dteethland LandmarkNet.

Why this exists
---------------
The repository's ``infer.py`` bundles the full ToothInstanceNet pipeline
(alignment -> instance segmentation + FDI -> per-tooth landmarks) through
``FullNet``, which needs four checkpoints and has a data-module bug in its
``predict`` path. For a landmark-focused workflow on Teeth3DS+ - where tooth
segmentation and FDI already exist per scan - the landmark stage can be run
directly on proposals that are built from the *known* segmentation and FDI
labels. That removes both the alignment and the segmentation/FDI stages, which
is exactly the "avoid duplicating segmentation / tooth identification" goal.

Input per case (already in the local Teeth3DS+ layout):
    <root>/<jaw>/<case>/<case>_<jaw>.obj        mesh
    <root>/<jaw>/<case>/<case>_<jaw>.json       per-vertex FDI labels + instances

Output per case:
    <out-dir>/<case>_<jaw>__kpt.json            landmarks (challenge format)
    <out-dir>/predictions.csv                   flat table for the official scorer

STATUS: design-reviewed against the source but NOT executed end to end. It
cannot run until `pointops` and `torch_scatter` load (see README section 6) and
a trained checkpoint exists. Read `README.md` section 5.6 before trusting its
output:
  * mesial and distal come out of a single head and are written as one
    `MesialAndDistal` class; the geometric split into `Mesial`/`Distal` still
    has to be applied (the repository does it in
    `TeethInstFullDataModule.process_landmarks`).
  * the per-tooth landmark is reduced with an argmax over the confidence head
    after masking by the known segmentation instance, which is a task-local
    choice rather than a repository behaviour.
  * the proposal -> tooth-instance association is recovered from the proposal
    centroids, because `GenerateProposals` samples `max_proposals` random
    instances per scan.

Usage
-----
python landmark_extension/infer/predict_landmarks.py --checkpoint <ckpt> --jaw upper --limit 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import teethland.data.transforms as T  # noqa: E402
from teethland import PointTensor  # noqa: E402
from teethland.data.datasets import TeethLandDataset  # noqa: E402
from teethland.models import LandmarkNet  # noqa: E402


LANDMARK_CLASS_NAMES = [
    'MesialAndDistal', 'FacialPoint', 'OuterPoint', 'InnerPoint', 'Cusp',
]
OUTPUT_CLASS_NAMES = ['Mesial', 'Distal', 'FacialPoint', 'OuterPoint',
                      'InnerPoint', 'Cusp']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--jaw', default='upper', choices=['upper', 'lower'])
    parser.add_argument('--config', type=Path, default=None)
    parser.add_argument('--out-dir', type=Path, default=None)
    parser.add_argument('--limit', type=int, default=None,
                        help='Only run the first N cases (quick check).')
    parser.add_argument('--device', default='cuda')
    parser.add_argument(
        '--conf-thresh', type=float, default=0.15,
        help='Distance threshold in normalised units (0.15 = 2.6 mm), '
             'matching FullNet.landmarks_process.',
    )
    parser.add_argument(
        '--no-seg-gate', action='store_true',
        help='Do not require the segmentation head to mark a point as tooth.',
    )
    return parser.parse_args()


def build_datamodule(cfg: dict, jaw: str, cases: list[str]):
    from teethland.datamodules import TeethLandDataModule

    datamodule_cfg = dict(cfg['datamodule'])
    datamodule_cfg['num_workers'] = 0
    datamodule_cfg['persistent_workers'] = False
    datamodule_cfg['sampler'] = 'default'
    datamodule_cfg['batch_size'] = 1

    dm = TeethLandDataModule(seed=cfg['seed'], **datamodule_cfg)
    files = dm._files('fit')
    files = [f for f in files if f[0].stem in set(cases)]
    return dm, files


def main() -> int:
    args = parse_args()
    config_path = args.config or (
        REPO_ROOT / 'landmark_extension' / 'configs' / f'landmark_{args.jaw}.yaml'
    )
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    data_root = Path(cfg['datamodule']['root'])
    jaw = args.jaw
    out_dir = args.out_dir or Path(cfg['out_dir'])
    out_dir = out_dir if out_dir.is_absolute() else REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = sorted(
        p.name for p in (data_root / jaw).iterdir()
        if p.is_dir() and (p / f'{p.name}_{jaw}.json').exists()
    )
    if args.limit:
        cases = cases[:args.limit]
    print(f'{len(cases)} cases to run')

    # Build the dataset with the full (non-random) evaluation transforms so the
    # proposals are generated from the known segmentation/FDI labels.
    dm, files = build_datamodule(cfg, jaw, cases)
    if not files:
        raise SystemExit('no files matched the requested cases')

    rng = np.random.default_rng(cfg['seed'])
    default_transforms = T.Compose(
        T.UniformDensityDownsample(dm.uniform_density_voxel_size, inplace=True),
        T.GenerateProposals(dm.proposal_points, dm.max_proposals, rng=rng),
        dm.default_transforms,
    )
    dataset = TeethLandDataset(
        stage='fit',
        seg_root=data_root,
        landmarks_root=Path(cfg['datamodule']['landmarks_root']),
        files=files,
        clean=cfg['datamodule']['clean'],
        transform=default_transforms,
    )

    model_cfg = dict(cfg['model']['landmarks'])
    model_cfg.pop('checkpoint_path', None)
    model = LandmarkNet.load_from_checkpoint(
        str(args.checkpoint),
        in_channels=dm.num_channels,
        num_classes=dm.num_classes,
        dbscan_cfg=cfg['model']['dbscan_cfg'],
        **model_cfg,
    )
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    print(f'model on {device}')

    # InstanceCentroids/GenerateProposals need the confidence for the
    # boundary-aware downsample only when that option is used; it is off here.
    rows = []
    for index in range(len(dataset)):
        t0 = time.perf_counter()
        sample = dataset[index]
        scan_file = str(sample['scan_file'])
        print(f'[{index + 1}/{len(dataset)}] {scan_file}')

        points = torch.from_numpy(sample['points']).float()
        features = torch.from_numpy(sample['features']).float()
        centroids = torch.from_numpy(sample['centroids']).float()
        labels = torch.from_numpy(sample['labels']).long().flatten()
        instances = torch.from_numpy(sample['instances']).long()
        point_idxs = torch.from_numpy(sample['point_idxs']).long()
        affine = torch.from_numpy(sample['affine']).float()
        point_counts = torch.as_tensor(sample['point_count']).long()

        x = PointTensor(
            coordinates=points.to(device),
            features=features.to(device),
            batch_counts=point_counts.to(device),
        )

        with torch.no_grad():
            seg, mesial_distal, facial, outer, inner, cusps = model(x)

        heads = [mesial_distal, facial, outer, inner, cusps]
        seg_scores = torch.sigmoid(seg.F[:, 0])
        num_proposals = int(sample['instance_count'])
        proposal_instances = np.unique(sample['instances'])[1:][:num_proposals]
        # GenerateProposals selects max_proposals random instances; recover them
        # from the proposal centroids instead.
        proposal_centroids = centroids.detach().cpu().numpy()

        landmarks = []
        for proposal_idx in range(num_proposals):
            mask = point_idxs[proposal_idx].numpy()
            instance = None
            distances = np.linalg.norm(
                points.numpy() - proposal_centroids[proposal_idx], axis=-1,
            )
            nearest = int(np.argmin(distances))
            if distances[nearest] < 1e-6:
                instance = int(instances[nearest])

            for head_idx, head in enumerate(heads):
                scores = head.F[:, 0].detach().cpu().numpy()
                offsets = head.F[:, 1:4].detach().cpu().numpy()
                if head_idx >= 5:
                    continue
                candidate_idxs = np.arange(mask.shape[0])
                if not args.no_seg_gate:
                    gate = seg_scores.detach().cpu().numpy()[mask] >= 0.5
                else:
                    gate = np.ones(mask.shape[0], dtype=bool)
                if gate.sum() == 0:
                    continue
                # only consider points that belong to this proposal's instance
                if instance is not None:
                    local_instances = instances[mask].numpy()
                    gate &= local_instances == instance
                    if gate.sum() == 0:
                        continue
                d = scores[candidate_idxs]
                d_masked = np.where(gate, d, np.inf)
                best = int(np.argmin(d_masked))
                if d_masked[best] > args.conf_thresh:
                    continue
                coord = points.numpy()[mask][best] + offsets[candidate_idxs[best]]
                landmarks.append((head_idx, float(d_masked[best]), coord, instance))

        # undo the preprocessing affine (PoseNormalize) to return original coords
        results = []
        for head_idx, score, coord, instance in landmarks:
            homogeneous = np.concatenate((coord, [1.0]))
            original = (affine.numpy() @ homogeneous)[:3]
            results.append({
                'key': Path(scan_file).stem,
                'coord_x': float(original[0]),
                'coord_y': float(original[1]),
                'coord_z': float(original[2]),
                'class': LANDMARK_CLASS_NAMES[head_idx],
                'score': score,
                'instance_id': instance,
            })

        out_name = Path(scan_file).with_suffix('').name + '__kpt.json'
        template = {
            'version': '1.1',
            'description': 'landmarks',
            'key': scan_file,
            'objects': [
                {
                    'key': f'uuid_{i}',
                    'score': r['score'],
                    'class': r['class'],
                    'coord': [r['coord_x'], r['coord_y'], r['coord_z']],
                    'instance_id': r['instance_id'],
                }
                for i, r in enumerate(results)
            ],
        }
        with open(out_dir / out_name, 'w') as f:
            json.dump(template, f, indent=2)
        rows.extend(results)
        print(f'  -> {len(results)} landmarks in {time.perf_counter() - t0:.1f}s')

    if rows:
        import pandas as pd
        pd.DataFrame(rows).to_csv(out_dir / 'predictions.csv', index=False)
        print(f'wrote {out_dir / "predictions.csv"} ({len(rows)} rows)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
