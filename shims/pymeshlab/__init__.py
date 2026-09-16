"""Drop-in replacement for the small part of pymeshlab that 3dteethland uses.

Why this exists
---------------
Windows **Smart App Control** on this machine refuses to load
``pymeshlab/meshlab-common.dll`` (CodeIntegrity events 3077/3033/3118), so
``import pymeshlab`` fails even though the wheel is installed:

    ImportError: DLL load failed while importing pmeshlab

The 3dteethland code only ever uses pymeshlab for one thing: loading a mesh file
and reading its vertices / faces / normals / colors in file order. This module
implements exactly that surface with a dependency-free OBJ reader that preserves
vertex order (``process=False`` semantics).

Activate it by putting this directory first on ``sys.path``/``PYTHONPATH``:

    set PYTHONPATH=D:\\WorkSpace\\Dental\\3dteethland\\landmark_extension\\shims

or by calling ``landmark_extension/shims/activate.py`` from your own script.

Limits: OBJ only (which is what Teeth3DS ships), no mesh editing operations, no
saving. If pymeshlab can be loaded (for example after turning Smart App Control
off), do not use this shim - the real library is better.
"""

from __future__ import annotations

import numpy as np


__version__ = 'shim-1.0'
__all__ = ['MeshSet', 'Matrix', 'MeshSetVersion', 'PymeshlabShimWarning']


class PymeshlabShimWarning(RuntimeWarning):
    """Warning class used by this shim."""


def _compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals (the usual definition)."""
    normals = np.zeros_like(vertices)
    if faces.size:
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        face_normals = np.cross(v1 - v0, v2 - v0)
        for corner in range(3):
            np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    return normals / lengths


def _read_obj(path: str):
    """Parse an OBJ file, keeping the original vertex order."""
    vertices, faces, normals, colors = [], [], [], []
    has_vertex_colors = False
    has_vertex_normals = False

    with open(path, 'r', errors='ignore') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
                if len(parts) >= 7:
                    has_vertex_colors = True
                    colors.append((float(parts[4]), float(parts[5]), float(parts[6])))
            elif line.startswith('vn '):
                parts = line.split()
                normals.append((float(parts[1]), float(parts[2]), float(parts[3])))
                has_vertex_normals = True
            elif line.startswith('f '):
                idx_v = []
                for token in line.split()[1:]:
                    v = int(token.split('/')[0])
                    idx_v.append(v - 1 if v > 0 else len(vertices) + v)
                if len(idx_v) == 3:
                    faces.append(idx_v)
                elif len(idx_v) > 3:
                    for i in range(1, len(idx_v) - 1):
                        faces.append([idx_v[0], idx_v[i], idx_v[i + 1]])

    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)

    if has_vertex_normals and len(normals) == len(vertices):
        normals = np.asarray(normals, dtype=np.float64)
    else:
        normals = _compute_vertex_normals(vertices, faces)

    if has_vertex_colors and len(colors) == len(vertices):
        colors = np.asarray(colors, dtype=np.float64)
    else:
        # Teeth3DS OBJ files carry no vertex colors. pymeshlab returns a
        # per-vertex RGB matrix (default gray) for OBJ files, and the dataset
        # slices `vertex_color_matrix()[:, :3]`, so return an explicit
        # (N, 3) matrix instead of an empty one to keep feature shapes right.
        colors = np.zeros((len(vertices), 3), dtype=np.float64)

    return vertices, faces, normals, colors


class Matrix(np.ndarray):
    """NumPy array that also supports pymeshlab's ``matrix_as_np()``."""

    def matrix_as_np(self):
        return np.asarray(self)


class Mesh:
    """Minimal stand-in for ``pymeshlab.Mesh``."""

    def __init__(self, vertices, faces, normals, colors):
        self._vertices = Matrix(vertices.shape, dtype=vertices.dtype, buffer=vertices)
        self._faces = Matrix(faces.shape, dtype=faces.dtype, buffer=faces)
        self._normals = Matrix(normals.shape, dtype=normals.dtype, buffer=normals)
        self._colors = Matrix(colors.shape, dtype=colors.dtype, buffer=colors)

    def compact(self):
        return None

    def vertex_matrix(self) -> np.ndarray:
        return self._vertices

    def face_matrix(self) -> np.ndarray:
        return self._faces

    def vertex_normal_matrix(self) -> np.ndarray:
        return self._normals

    def vertex_color_matrix(self) -> np.ndarray:
        return self._colors

    def vertex_number(self) -> int:
        return int(self._vertices.shape[0])

    def face_number(self) -> int:
        return int(self._faces.shape[0])


class MeshSet:
    """Minimal stand-in for ``pymeshlab.MeshSet`` (load + read only)."""

    def __init__(self, *_, **__):
        self._meshes = []
        self._current = -1

    def load_new_mesh(self, path: str, *_, **__):
        suffix = str(path).lower().rsplit('.', 1)[-1]
        if suffix != 'obj':
            raise NotImplementedError(
                f'pymeshlab shim only supports OBJ, got ".{suffix}" for {path}',
            )
        self._meshes.append(Mesh(*_read_obj(str(path))))
        self._current = len(self._meshes) - 1

    def current_mesh(self) -> Mesh:
        if self._current < 0:
            raise RuntimeError('no mesh loaded')
        return self._meshes[self._current]

    def mesh_number(self) -> int:
        return len(self._meshes)

    def __len__(self) -> int:
        return len(self._meshes)


class MeshSetVersion:
    """Placeholder for ``pymeshlab.MeshSetVersion`` (unused by 3dteethland)."""

    def __str__(self) -> str:
        return __version__
