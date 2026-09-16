"""End-to-end smoke test for the 3DTeethLand landmark pipeline.

This script intentionally does NOT train. It only proves that the chain

    Teeth3DS scan + segmentation + landmark JSON
        -> TeethLandDataModule / TeethLandDataset (adapter)
        -> transforms (normalise, downsample, proposals, landmark matching)
        -> collate
        -> LandmarkNet.forward
        -> LandmarkLoss / BCE loss
        -> one backward pass

is wired up correctly for at most a couple of cases.

Usage
-----
python landmark_extension/validation/smoke_test_landmark_pipeline.py --jaw upper
python landmark_extension/validation/smoke_test_landmark_pipeline.py --jaw lower --batch-size 2
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jaw', default='upper', choices=['upper', 'lower'])
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--proposal-points', type=int, default=None)
    parser.add_argument('--max-proposals', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--backward', action='store_true', default=True)
    parser.add_argument(
        '--report', type=Path,
        default=REPO_ROOT / 'landmark_extension' / 'validation' / 'smoke_report.json',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from teethland.datamodules import TeethLandDataModule
    from teethland.models import LandmarkNet

    config_path = REPO_ROOT / 'landmark_extension' / 'configs' / f'landmark_{args.jaw}.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    datamodule_cfg = dict(config['datamodule'])
    datamodule_cfg['batch_size'] = args.batch_size
    datamodule_cfg['num_workers'] = args.num_workers
    datamodule_cfg['persistent_workers'] = args.num_workers > 0
    datamodule_cfg['sampler'] = 'default'
    if args.proposal_points:
        datamodule_cfg['proposal_points'] = args.proposal_points
    if args.max_proposals:
        datamodule_cfg['max_proposals'] = args.max_proposals

    report: dict = {
        'config': str(config_path),
        'jaw': args.jaw,
        'batch_size': args.batch_size,
        'proposal_points': datamodule_cfg['proposal_points'],
        'max_proposals': datamodule_cfg['max_proposals'],
    }

    t0 = time.perf_counter()
    dm = TeethLandDataModule(seed=config['seed'], **datamodule_cfg)
    dm.setup('fit')
    report['num_train_files'] = len(dm.train_dataset)
    report['num_val_files'] = len(dm.val_dataset)
    report['num_channels'] = dm.num_channels
    report['num_classes'] = dm.num_classes
    report['setup_seconds'] = time.perf_counter() - t0
    print(
        f"[data] train={report['num_train_files']} val={report['num_val_files']} "
        f"channels={report['num_channels']} classes={report['num_classes']} "
        f"({report['setup_seconds']:.1f}s)",
    )

    # ---- dataset level ----------------------------------------------------
    # Use a full-size proposal budget so the first sample is representative.
    dataset = dm.train_dataset
    t0 = time.perf_counter()
    sample = dataset[0]
    report['sample_seconds'] = time.perf_counter() - t0

    report['sample'] = {
        'scan_file': str(sample['scan_file']),
        'is_lower': bool(sample['is_lower']),
        'num_points': int(sample['points'].shape[0]),
        'num_features': int(sample['features'].shape[1]),
        'num_proposals': int(sample['instance_count']),
        'proposal_points': int(sample['point_count'][0]),
        'num_landmarks': int(sample['landmarks'].shape[0]) if 'landmarks' in sample else 0,
        'landmark_class_counts': (
            np.bincount(sample['landmarks'][:, 3].astype(np.int64),
                        minlength=6).tolist()
            if 'landmarks' in sample else None
        ),
        'num_labeled_points': int((sample['labels'] >= 0).sum()),
        'num_tooth_points': int((sample['labels'] > 0).sum()),
        'num_instances_in_scan': int(sample['instance_count']),
    }
    print(f"[dataset] sample 0: {json.dumps(report['sample'], indent=None)}")

    # ---- dataloader / collate --------------------------------------------
    loader = dm.train_dataloader()
    t0 = time.perf_counter()
    batch = next(iter(loader))
    report['collate_seconds'] = time.perf_counter() - t0

    scan_file, is_lower, x, (landmarks, labels) = batch
    report['batch'] = {
        'scan_file': str(scan_file),
        'points': list(x.C.shape),
        'features': list(x.F.shape),
        'batch_counts': x.batch_counts.tolist(),
        'labels': list(labels.F.shape),
        'landmarks_coords': list(landmarks.C.shape),
        'landmarks_features': list(landmarks.F.shape),
        'landmark_batch_counts': landmarks.batch_counts.tolist(),
    }
    print(f"[collate] {json.dumps(report['batch'])}")

    # ---- model forward / loss / backward ---------------------------------
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model_cfg = dict(config['model']['landmarks'])
    model_cfg.pop('checkpoint_path', None)
    model = LandmarkNet(
        in_channels=dm.num_channels,
        num_classes=dm.num_classes,
        dbscan_cfg=config['model']['dbscan_cfg'],
        **model_cfg,
    )
    model = model.to(device)
    model.train()

    x = x.to(device)
    labels = labels.to(device)
    landmarks = landmarks.to(device)

    t0 = time.perf_counter()
    outputs = model(x)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    report['forward_seconds'] = time.perf_counter() - t0

    seg, mesial_distal, facial, outer, inner, cusps = outputs
    report['outputs'] = {
        'seg': list(seg.F.shape),
        'mesial_distal': list(mesial_distal.F.shape),
        'facial': list(facial.F.shape),
        'outer': list(outer.F.shape),
        'inner': list(inner.F.shape),
        'cusps': list(cusps.F.shape),
    }
    print(f"[forward] {json.dumps(report['outputs'])} ({report['forward_seconds']:.1f}s)")

    seg_loss = model.seg_criterion(seg, labels)
    md_loss = model.landmark_criterion(mesial_distal, landmarks, [0, 1])
    facial_loss = model.landmark_criterion(facial, landmarks, [2])
    outer_loss = model.landmark_criterion(outer, landmarks, [3])
    inner_loss = model.landmark_criterion(inner, landmarks, [4])
    cusps_loss = model.landmark_criterion(cusps, landmarks, [5])

    total_loss = seg_loss + md_loss + facial_loss + outer_loss + inner_loss + cusps_loss
    report['losses'] = {
        'seg': float(seg_loss.detach().cpu()),
        'mesial_distal': float(md_loss.detach().cpu()),
        'facial': float(facial_loss.detach().cpu()),
        'outer': float(outer_loss.detach().cpu()),
        'inner': float(inner_loss.detach().cpu()),
        'cusps': float(cusps_loss.detach().cpu()),
        'total': float(total_loss.detach().cpu()),
        'finite': bool(torch.isfinite(total_loss).item()),
    }
    print(f"[loss] {json.dumps(report['losses'])}")

    if args.backward:
        t0 = time.perf_counter()
        total_loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1e9,
        ).detach().cpu())
        report['backward_seconds'] = time.perf_counter() - t0
        report['grad_norm'] = grad_norm
        report['gradients_finite'] = bool(np.isfinite(grad_norm))
        print(
            f"[backward] grad_norm={grad_norm:.4g} "
            f"({report['backward_seconds']:.1f}s)",
        )

    if device.type == 'cuda':
        report['peak_memory_gb'] = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"[memory] peak={report['peak_memory_gb']:.2f} GB")

    report['status'] = 'OK'
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[report] {args.report}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
