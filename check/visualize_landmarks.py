"""Export a Teeth3DS+ case as an OBJ with FDI-coloured teeth and landmark markers.

The output is a single ASCII OBJ with per-vertex colours, so it opens directly
in MeshLab, Blender, 3D Viewer or the Windows 3D app - no Python viewer and no
extra dependency beyond NumPy.

Usage
-----
python landmark_extension/check/visualize_landmarks.py --case 013TXGFK --jaw upper
python landmark_extension/check/visualize_landmarks.py --case 013TXGFK --jaw upper --open
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np


DEFAULT_DATA_ROOT = Path(r'D:\WorkSpace\Dental\data\Teeth3DS')
OUT_ROOT = Path(__file__).resolve().parent / 'visualizations'

JAW_COLORS = {
    0: (150, 105, 105),   # gingiva
    1: (240, 70, 70),
    2: (240, 110, 60),
    3: (240, 150, 55),
    4: (225, 190, 55),
    5: (175, 210, 55),
    6: (110, 210, 70),
    7: (60, 195, 105),
    8: (50, 180, 145),
}
SECONDARY_COLORS = {
    1: (45, 180, 205),
    2: (45, 145, 235),
    3: (70, 105, 235),
    4: (105, 80, 235),
    5: (155, 70, 220),
    6: (195, 70, 195),
    7: (220, 70, 150),
    8: (235, 70, 115),
}

LANDMARK_COLORS = {
    'Mesial': (255, 30, 30),
    'Distal': (30, 70, 255),
    'Cusp': (255, 225, 20),
    'InnerPoint': (20, 230, 50),
    'OuterPoint': (255, 40, 200),
    'FacialPoint': (20, 220, 255),
    'Unknown': (255, 255, 255),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument('--jaw', default='upper', choices=['upper', 'lower'])
    p.add_argument('--case', required=True)
    p.add_argument('--split', default='train', choices=['train', 'test'])
    p.add_argument('--out', type=Path, default=None)
    p.add_argument(
        '--max-faces', type=int, default=0,
        help='Optional face cap (0 = all faces).',
    )
    p.add_argument('--open', action='store_true',
                   help='Open the result with the default application.')
    return p.parse_args()


def read_obj(path: Path):
    vertices, faces = [], []
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            if line.startswith('v '):
                p = line.split()
                vertices.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith('f '):
                idx = []
                for token in line.split()[1:]:
                    v = int(token.split('/')[0])
                    idx.append(v - 1 if v > 0 else len(vertices) + v)
                if len(idx) == 3:
                    faces.append(idx)
                elif len(idx) > 3:
                    for i in range(1, len(idx) - 1):
                        faces.append([idx[0], idx[i], idx[i + 1]])
    return np.asarray(vertices, np.float64), np.asarray(faces, np.int64)


def fdi_color(fdi: int):
    quadrant, tooth = divmod(fdi, 10)
    palette = SECONDARY_COLORS if quadrant % 2 == 0 else JAW_COLORS
    return palette.get(tooth, (190, 190, 190))


def write_obj(path, vertices, faces, colors):
    with open(path, 'w') as f:
        f.write('# Teeth3DS+ landmark visualisation\n')
        for v, c in zip(vertices, colors):
            f.write(
                f'v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f} '
                f'{c[0] / 255:.5f} {c[1] / 255:.5f} {c[2] / 255:.5f}\n',
            )
        for face in faces:
            f.write(f'f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n')


def icosphere(center, radius, subdivisions=1):
    phi = (1 + 5 ** 0.5) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=np.float64)
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int64)
    for _ in range(subdivisions):
        verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
        edge_to_mid = {}
        new_faces = []
        for face in faces:
            mids = []
            for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                key = (min(a, b), max(a, b))
                if key not in edge_to_mid:
                    edge_to_mid[key] = len(verts)
                    verts = np.vstack((verts, (verts[a] + verts[b]) / 2))
                mids.append(edge_to_mid[key])
            a, b, c = face
            ab, bc, ca = mids
            new_faces.extend([
                [a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca],
            ])
        faces = np.asarray(new_faces, dtype=np.int64)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    return verts * radius + center, faces


def main() -> int:
    args = parse_args()
    stem = f'{args.case}_{args.jaw}'
    scan_dir = args.data_root / args.jaw / args.case
    mesh_path = scan_dir / f'{stem}.obj'
    seg_path = scan_dir / f'{stem}.json'
    kpt_path = (
        args.data_root / f'3DTeethLand_landmarks_{args.split}' / args.jaw
        / args.case / f'{stem}__kpt.json'
    )
    for path in (mesh_path, seg_path, kpt_path):
        if not path.exists():
            print(f'missing: {path}', file=sys.stderr)
            return 1

    vertices, faces = read_obj(mesh_path)
    with open(seg_path) as f:
        segmentation = json.load(f)
    labels = np.asarray(segmentation['labels'], dtype=np.int64)
    with open(kpt_path) as f:
        landmark_json = json.load(f)
    landmarks = landmark_json['objects']

    print(f'{stem}: {vertices.shape[0]} vertices, {faces.shape[0]} faces, '
          f'{len(landmarks)} landmarks')
    print('landmark classes:', dict(Counter(o['class'] for o in landmarks)))

    # per-vertex colour from the per-vertex FDI labels
    colors = np.array(
        [fdi_color(int(v)) if v > 0 else (150, 105, 105) for v in labels],
        dtype=np.uint8,
    )

    if args.max_faces and faces.shape[0] > args.max_faces:
        faces = faces[:args.max_faces]

    out_path = args.out or OUT_ROOT / f'{stem}_landmarks.obj'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_obj(out_path, vertices, faces, colors)

    # append landmark spheres as a second file so the mesh stays lightweight
    diagonal = float(np.linalg.norm(vertices.max(0) - vertices.min(0)))
    radius = diagonal * 0.004
    sphere_verts, sphere_faces, sphere_colors = [], [], []
    for landmark in landmarks:
        center = np.asarray(landmark['coord'][:3], dtype=np.float64)
        verts, local_faces = icosphere(center, radius)
        base = len(sphere_verts)
        sphere_verts.extend(verts)
        sphere_faces.extend(local_faces + base)
        color = LANDMARK_COLORS.get(landmark['class'], LANDMARK_COLORS['Unknown'])
        sphere_colors.extend([color] * len(verts))

    kpt_out = out_path.with_name(out_path.stem + '_kpts.obj')
    write_obj(
        kpt_out,
        np.asarray(sphere_verts, dtype=np.float64),
        np.asarray(sphere_faces, dtype=np.int64),
        np.asarray(sphere_colors, dtype=np.uint8),
    )
    print('wrote:', out_path)
    print('wrote:', kpt_out)
    print('   (load both files in MeshLab to see the FDI colours and the landmarks)')

    if args.open:
        import os
        os.startfile(out_path)  # noqa: S606 - Windows-only convenience
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
