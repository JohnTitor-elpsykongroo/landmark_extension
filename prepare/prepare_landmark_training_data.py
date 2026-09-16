"""Prepare Teeth3DS+ landmark data for the 3dteethland repository.

This script does NOT copy or modify the raw Teeth3DS / Teeth3DS+ data. It only
generates auxiliary artifacts inside ``landmark_extension/``:

1. A case index (CSV + JSON) of every case that has mesh + segmentation + landmarks.
2. Deterministic subject-level folds (`landmark_<jaw>_fold_<k>.txt`) in exactly
   the format ``TeethSegDataModule._split`` expects for a ``fold`` config value.
3. A machine-readable summary for the report.

Usage
-----
python landmark_extension/prepare/prepare_landmark_training_data.py
python landmark_extension/prepare/prepare_landmark_training_data.py --split test
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
from pathlib import Path
import sys


DEFAULT_DATA_ROOT = Path(r'D:\WorkSpace\Dental\data\Teeth3DS')

# Reproduced from teethland/datamodules/teethland.py (TeethLandDataModule._files)
# and teethland/datamodules/teethseg.py (TeethSegDataModule._files).
EXCLUDE_LANDMARK_CASES = [
    # missing tooth segmentation
    'K4BAII5F_upper',
    '87N5YSES_upper',
    '67PV9M7X_lower',
    # missing more than one landmark
    'S0AON6PZ_lower',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--data-root', type=Path, default=DEFAULT_DATA_ROOT,
        help='Root of the Teeth3DS data set (read only).',
    )
    parser.add_argument(
        '--out-dir', type=Path, default=None,
        help='Output directory (default: landmark_extension/prepare).',
    )
    parser.add_argument('--split', default='train', choices=['train', 'test'])
    parser.add_argument(
        '--folds', type=int, default=5,
        help='Number of folds written to disk (the repo hardcodes 5).',
    )
    return parser.parse_args()


def find_mesh(scan_dir: Path, stem: str) -> Path | None:
    for ext in ('obj', 'ply', 'stl'):
        candidate = scan_dir / f'{stem}.{ext}'
        if candidate.exists():
            return candidate
    return None


def count_obj_vertices(path: Path) -> int:
    """Count OBJ vertices (fast, no mesh library needed)."""
    n = 0
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            if line.startswith('v '):
                n += 1
    return n


def load_json(path: Path) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def collect_cases(data_root: Path, split: str) -> list[dict]:
    """Collect every case that has mesh + segmentation + landmarks."""
    cases: list[dict] = []

    for jaw in ('upper', 'lower'):
        landmark_root = data_root / f'3DTeethLand_landmarks_{split}' / jaw
        scan_root = data_root / jaw
        if not landmark_root.exists():
            print(f'[warning] missing landmark root: {landmark_root}')
            continue

        for kpt_file in sorted(landmark_root.rglob('*.json')):
            stem = kpt_file.name.split('__')[0]
            if stem in EXCLUDE_LANDMARK_CASES:
                continue

            case_id = stem.split('_')[0]
            scan_dir = scan_root / case_id
            mesh_file = find_mesh(scan_dir, stem)
            seg_file = scan_dir / f'{stem}.json'
            if mesh_file is None or not seg_file.exists():
                print(
                    f'[warning] skipping {stem}: '
                    f'mesh={mesh_file is not None} seg={seg_file.exists()}',
                )
                continue

            landmark = load_json(kpt_file)
            segmentation = load_json(seg_file)
            labels = segmentation.get('labels', [])
            instances = segmentation.get('instances', [])

            class_counts = Counter(
                obj.get('class') for obj in landmark.get('objects', [])
            )
            fdis = sorted({int(v) for v in labels if int(v) > 0})
            n_instances = len({int(v) for v in instances if int(v) > 0})

            cases.append({
                'case_id': case_id,
                'stem': stem,
                'jaw': jaw,
                'mesh': str(mesh_file),
                'mesh_vertices': count_obj_vertices(mesh_file),
                'segmentation': str(seg_file),
                'segmentation_vertices': len(labels),
                'landmarks': str(kpt_file),
                'landmark_count': len(landmark.get('objects', [])),
                'landmark_classes': dict(sorted(class_counts.items())),
                'fdi_labels': fdis,
                'num_instances': n_instances,
                # The repo resolves these relative to its own data module root.
                'rel_mesh': (
                    Path(jaw) / mesh_file.parent.name / mesh_file.name
                ).as_posix(),
                'rel_landmarks': (
                    Path(jaw) / kpt_file.parent.name / kpt_file.name
                ).as_posix(),
            })

    return cases


def write_folds(
    cases: list[dict],
    out_dir: Path,
    folds: int,
) -> dict:
    """Write deterministic folds; every subject appears in exactly one fold.

    Subjects are ordered by a hash-free deterministic key so that the split is
    reproducible across machines: sort by case id, then deal round-robin.
    """
    written: dict[str, list[str]] = {}
    by_jaw: dict[str, list[dict]] = {}
    for case in cases:
        by_jaw.setdefault(case['jaw'], []).append(case)

    for jaw, jaw_cases in by_jaw.items():
        subjects = sorted({c['case_id'] for c in jaw_cases})
        fold_of = {
            subject: idx % folds
            for idx, subject in enumerate(subjects)
        }
        for fold in range(folds):
            selected = [
                c for c in jaw_cases
                if fold_of[c['case_id']] == fold
            ]
            name = f'landmark_{jaw}_fold_{fold}.txt'
            path = out_dir / name
            with open(path, 'w', newline='') as f:
                for case in sorted(selected, key=lambda c: c['stem']):
                    # TeethSegDataModule._split strips '.obj' and keeps the
                    # first two '_'-separated tokens -> '<case>_<jaw>'.
                    f.write(f"{case['stem']}.obj\n")
            written[name] = sorted(c['stem'] for c in selected)
            print(f'  wrote {name}: {len(selected)} cases')

    return written


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir or Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'data root : {args.data_root}')
    print(f'split     : {args.split}')
    print(f'out dir   : {out_dir}')

    cases = collect_cases(args.data_root, args.split)
    if not cases:
        print('ERROR: no usable cases found', file=sys.stderr)
        return 1

    csv_path = out_dir / f'landmark_cases_{args.split}.csv'
    fields = [
        'case_id', 'stem', 'jaw', 'mesh', 'mesh_vertices',
        'segmentation', 'segmentation_vertices', 'landmarks',
        'landmark_count', 'num_instances', 'rel_mesh', 'rel_landmarks',
        'landmark_classes', 'fdi_labels',
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            row = dict(case)
            row['landmark_classes'] = json.dumps(row['landmark_classes'])
            row['fdi_labels'] = json.dumps(row['fdi_labels'])
            writer.writerow(row)
    print(f'wrote {csv_path}: {len(cases)} cases')

    folds = write_folds(cases, out_dir, args.folds)

    per_jaw = Counter(c['jaw'] for c in cases)
    summary = {
        'split': args.split,
        'data_root': str(args.data_root),
        'num_cases': len(cases),
        'num_cases_per_jaw': dict(per_jaw),
        'total_landmarks': sum(c['landmark_count'] for c in cases),
        'landmark_classes': dict(sorted(Counter(
            cls for c in cases for cls in c['landmark_classes']
        ).items())),
        'excluded_cases': EXCLUDE_LANDMARK_CASES,
        'folds': {name: stems for name, stems in folds.items()},
    }
    summary_path = out_dir / f'landmark_dataset_summary_{args.split}.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f'wrote {summary_path}')

    print()
    print('cases per jaw :', dict(per_jaw))
    print('total landmark:', summary['total_landmarks'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
