"""Verify that Teeth3DS+ landmark JSON, mesh and segmentation actually line up.

For every landmark in every case this script reports

  * distance to the closest mesh vertex,
  * the FDI label of that closest vertex,
  * whether the FDI label of the closest vertex is 0 (gingiva) - this is the
    main failure mode if the landmark file belongs to a different coordinate
    frame or a different mesh revision than the scan file,
  * whether the number of mesh vertices equals the number of segmentation
    labels (they must, the segmentation is per vertex),
  * how many distinct FDI-labelled instances each case has.

The results are aggregated per jaw and printed as a table; a JSON report is
written next to this script.

Requires: numpy only (the OBJ reader below is deliberately dependency free so
that the geometry check runs before the training environment is installed).

Usage
-----
python landmark_extension/check/check_landmark_geometry.py
python landmark_extension/check/check_landmark_geometry.py --case 013TXGFK --jaw upper
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np


DEFAULT_DATA_ROOT = Path(r'D:\WorkSpace\Dental\data\Teeth3DS')

EXCLUDE_LANDMARK_CASES = [
    'K4BAII5F_upper', '87N5YSES_upper', '67PV9M7X_lower', 'S0AON6PZ_lower',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--split', default='train', choices=['train', 'test'])
    parser.add_argument(
        '--out', type=Path,
        default=Path(__file__).resolve().parent / 'landmark_geometry_report.json',
    )
    parser.add_argument('--case', default=None, help='Only check one case id.')
    parser.add_argument(
        '--jaw', default=None, choices=['upper', 'lower'],
        help='Only check one jaw.',
    )
    parser.add_argument(
        '--max-cases', type=int, default=None,
        help='Optional cap for a quick run.',
    )
    return parser.parse_args()


def read_obj_vertices(path: Path) -> np.ndarray:
    """Read OBJ vertices in file order (no re-ordering, no de-duplication)."""
    vertices = []
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return np.asarray(vertices, dtype=np.float64)


def load_landmarks(path: Path) -> tuple[np.ndarray, list[str]]:
    with open(path, 'r') as f:
        data = json.load(f)
    coords, classes = [], []
    for obj in data.get('objects', []):
        coord = obj.get('coord')
        if not coord or len(coord) < 3:
            continue
        coords.append([float(c) for c in coord[:3]])
        classes.append(obj.get('class', 'Unknown'))
    return np.asarray(coords, dtype=np.float64).reshape(-1, 3), classes


def nearest_vertex_indices(
    vertices: np.ndarray,
    query: np.ndarray,
    block: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Brute-force nearest vertex search in blocks (no scipy dependency)."""
    dists = np.empty(query.shape[0], dtype=np.float64)
    idxs = np.empty(query.shape[0], dtype=np.int64)
    v2 = np.einsum('ij,ij->i', vertices, vertices)
    for start in range(0, query.shape[0], block):
        stop = min(start + block, query.shape[0])
        q = query[start:stop]
        d2 = v2[None, :] - 2.0 * (q @ vertices.T) + np.einsum('ij,ij->i', q, q)[:, None]
        best = np.argmin(d2, axis=1)
        idxs[start:stop] = best
        dists[start:stop] = np.sqrt(np.maximum(d2[np.arange(best.shape[0]), best], 0.0))
    return dists, idxs


def check_case(
    data_root: Path,
    split: str,
    jaw: str,
    case_id: str,
) -> dict | None:
    stem = f'{case_id}_{jaw}'
    scan_dir = data_root / jaw / case_id
    mesh_path = scan_dir / f'{stem}.obj'
    seg_path = scan_dir / f'{stem}.json'
    kpt_path = data_root / f'3DTeethLand_landmarks_{split}' / jaw / case_id / f'{stem}__kpt.json'

    if stem in EXCLUDE_LANDMARK_CASES:
        return None
    if not (mesh_path.exists() and seg_path.exists() and kpt_path.exists()):
        return None

    vertices = read_obj_vertices(mesh_path)
    with open(seg_path, 'r') as f:
        segmentation = json.load(f)
    labels = np.asarray(segmentation['labels'], dtype=np.int64)
    instances = np.asarray(segmentation['instances'], dtype=np.int64)

    coords, classes = load_landmarks(kpt_path)
    dists, idxs = nearest_vertex_indices(vertices, coords)
    nearest_labels = labels[idxs]

    counts = Counter(classes)
    return {
        'case_id': case_id,
        'jaw': jaw,
        'num_vertices': int(vertices.shape[0]),
        'num_labels': int(labels.shape[0]),
        'vertex_label_match': bool(vertices.shape[0] == labels.shape[0]),
        'num_instances': int(len({int(v) for v in instances if int(v) > 0})),
        'num_fdi': int(len({int(v) for v in labels if int(v) > 0})),
        'num_landmarks': int(coords.shape[0]),
        'landmark_classes': dict(sorted(counts.items())),
        'nearest_vertex_dist_mean': float(dists.mean()),
        'nearest_vertex_dist_p95': float(np.percentile(dists, 95)),
        'nearest_vertex_dist_max': float(dists.max()),
        'landmarks_on_gingiva': int((nearest_labels == 0).sum()),
        'instance_label_zero_points': int((labels == 0).sum()),
    }


def main() -> int:
    args = parse_args()
    data_root = args.data_root
    jaws = [args.jaw] if args.jaw else ['upper', 'lower']

    cases: list[tuple[str, str]] = []
    for jaw in jaws:
        landmark_root = data_root / f'3DTeethLand_landmarks_{args.split}' / jaw
        if not landmark_root.exists():
            continue
        ids = sorted(p.name for p in landmark_root.iterdir() if p.is_dir())
        if args.case:
            ids = [i for i in ids if i == args.case]
        cases.extend((jaw, case_id) for case_id in ids)
    if args.max_cases:
        cases = cases[:args.max_cases]

    results = []
    for i, (jaw, case_id) in enumerate(cases, start=1):
        out = check_case(data_root, args.split, jaw, case_id)
        if out is None:
            print(f'[{i}/{len(cases)}] {case_id}_{jaw}: skipped (missing file)')
            continue
        results.append(out)
        print(
            f"[{i}/{len(cases)}] {case_id}_{jaw}: "
            f"verts={out['num_vertices']} labels={out['num_labels']} "
            f"fdi={out['num_fdi']} inst={out['num_instances']} "
            f"kpt={out['num_landmarks']} "
            f"d(mean/p95/max)={out['nearest_vertex_dist_mean']:.3f}/"
            f"{out['nearest_vertex_dist_p95']:.3f}/{out['nearest_vertex_dist_max']:.3f} "
            f"on_gingiva={out['landmarks_on_gingiva']}",
        )

    if not results:
        print('no cases checked', file=sys.stderr)
        return 1

    print()
    print('=== aggregate ===')
    for jaw in sorted({r['jaw'] for r in results}):
        jaw_results = [r for r in results if r['jaw'] == jaw]
        mismatch = sum(not r['vertex_label_match'] for r in jaw_results)
        dists_mean = np.array([r['nearest_vertex_dist_mean'] for r in jaw_results])
        dists_p95 = np.array([r['nearest_vertex_dist_p95'] for r in jaw_results])
        gingiva = sum(r['landmarks_on_gingiva'] for r in jaw_results)
        landmarks = sum(r['num_landmarks'] for r in jaw_results)
        fdis = sum(r['num_fdi'] for r in jaw_results) / len(jaw_results)
        print(
            f'{jaw}: cases={len(jaw_results)} '
            f'vertex/label mismatch={mismatch} '
            f'mean nearest-vertex dist={dists_mean.mean():.4f} '
            f'(worst case {dists_mean.max():.4f}) '
            f'worst-case p95={dists_p95.max():.4f} '
            f'landmarks on gingiva={gingiva}/{landmarks} '
            f'avg distinct FDI per case={fdis:.1f}',
        )

    class_counts: Counter[str] = Counter()
    for r in results:
        class_counts.update(r['landmark_classes'])
    print('landmark classes:', dict(sorted(class_counts.items())))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'split': args.split, 'cases': results}, f, indent=2)
    print('report:', args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
