"""Predict landmarks on held-out meshes using their ground-truth tooth instances."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import yaml

from teethland import PointTensor
from teethland.data import transforms as T
from teethland.data.datasets import TeethSegDataset
from teethland.models import LandmarkNet


OFFSET_NAMES = ("MesialDistal", "FacialPoint", "OuterPoint", "InnerPoint", "Cusp")


def discover(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    if args.mesh:
        mesh = args.mesh.resolve()
        annotation = (args.annotation or mesh.with_suffix(".json")).resolve()
        pairs = [(mesh, annotation)]
    else:
        meshes = sorted({p.resolve() for root in args.input_root for p in root.rglob("*.obj")})
        pairs = [(p, p.with_suffix(".json")) for p in meshes]
    if not pairs:
        raise ValueError("No OBJ files found.")
    stems = [mesh.stem for mesh, _ in pairs]
    if len(stems) != len(set(stems)):
        raise ValueError("Duplicate mesh stems would overwrite output files.")
    for mesh, annotation in pairs:
        if not mesh.is_file() or not annotation.is_file():
            raise FileNotFoundError(f"Missing mesh or matching annotation: {mesh}, {annotation}")
        if not mesh.stem.endswith(("_upper", "_lower")):
            raise ValueError(f"Mesh name must end in _upper or _lower: {mesh}")
    return pairs


def landmark_stems(root: Path) -> set[str]:
    if not root.is_dir():
        raise FileNotFoundError(f"Landmark exclusion directory does not exist: {root}")
    return {p.name.split("__")[0] for p in root.rglob("*__kpt.json")}


def nearest_surface(scene: o3d.t.geometry.RaycastingScene, xyz: np.ndarray) -> list[float]:
    query = o3d.core.Tensor(xyz.astype(np.float32)[None], dtype=o3d.core.Dtype.Float32)
    points = scene.compute_closest_points(query)["points"].numpy()
    return points[0].astype(float).tolist()


def mesial_class(fdi: int, xyz: np.ndarray, centroid: np.ndarray) -> str:
    quadrant, tooth = divmod(fdi, 10)
    if tooth in (1, 2):
        mesial = xyz[0] > centroid[0] if quadrant in (2, 4, 6, 8) else xyz[0] < centroid[0]
    elif tooth == 3:
        delta = xyz[:2] - centroid[:2]
        if quadrant in (2, 4, 6, 8):
            mesial = float(delta @ np.array([1, -1])) > float(delta @ np.array([-1, 1]))
        else:
            mesial = float(delta @ np.array([-1, -1])) > float(delta @ np.array([1, 1]))
    else:
        mesial = xyz[1] < centroid[1]
    return "Mesial" if mesial else "Distal"


def predict_case(
    mesh: Path,
    annotation: Path,
    model: LandmarkNet,
    cfg: dict,
    device: torch.device,
) -> list[dict]:
    dm_cfg = cfg["datamodule"]
    dataset = TeethSegDataset(
        stage="predict", root=Path("/"), files=[(mesh, annotation)],
        norm=dm_cfg["norm"], clean=dm_cfg["clean"],
    )
    data = dataset[0]
    raw = json.loads(annotation.read_text(encoding="utf-8"))
    vertex_count = len(data["points"])
    if len(raw["labels"]) != vertex_count or len(raw["instances"]) != vertex_count:
        raise ValueError(f"OBJ/JSON vertex count differs: {mesh}")

    voxel_size = dm_cfg["uniform_density_voxel_size"][1]
    data = T.UniformDensityDownsample(voxel_size, inplace=True)(**data)
    tooth_ids = np.unique(data["instances"])
    if 0 not in tooth_ids:
        raise ValueError(f"Background instance 0 is required by GenerateProposals: {mesh}")
    tooth_ids = tooth_ids[tooth_ids != 0]
    if tooth_ids.size == 0:
        raise ValueError(f"No tooth instances after downsampling: {mesh}")
    fdis = [int(data["instance_labels"][i]) for i in tooth_ids]
    if any(fdi <= 0 for fdi in fdis) or len(fdis) != len(set(fdis)):
        raise ValueError(f"Missing or repeated FDI label in {mesh}")
    for tooth_id, fdi in zip(tooth_ids, fdis):
        observed = np.unique(data["labels"][data["instances"] == tooth_id])
        if observed.tolist() != [fdi]:
            raise ValueError(f"Instance {tooth_id} has inconsistent FDI labels: {mesh}")
    centroids = data["instance_centroids"][tooth_ids].copy()
    affine = data["affine"].copy()
    data = T.GenerateProposals(
        proposal_points=dm_cfg["proposal_points"],
        max_proposals=len(tooth_ids),
        rng=np.random.default_rng(cfg["seed"]),
    )(**data)
    points = data["points"]
    normals = data["normals"]
    colors = data["colors"]
    if len(points) != len(tooth_ids):
        raise RuntimeError(f"Not all teeth received proposals: {mesh}")

    surface = o3d.io.read_triangle_mesh(str(mesh))
    if not surface.has_triangles():
        raise ValueError(f"Mesh has no triangles: {mesh}")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(surface))
    inverse_affine = np.linalg.inv(affine)
    output = []
    for index, (tooth_id, fdi) in enumerate(zip(tooth_ids, fdis)):
        xyz = points[index].astype(np.float32)
        normal = normals[index].astype(np.float32)
        normal /= np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-8
        features = [xyz, normal]
        if dm_cfg["with_color"]:
            features.append(colors[index].astype(np.float32))
        features.append(xyz - centroids[index].astype(np.float32))
        x = PointTensor(
            coordinates=torch.from_numpy(xyz).to(device),
            features=torch.from_numpy(np.concatenate(features, axis=-1)).to(device),
        )
        _, md, facial, outer, inner, cusp = model(x)
        for kind, offsets in zip(OFFSET_NAMES, (md, facial, outer, inner, cusp)):
            mask = offsets.F[:, 0] < 0.15
            if not torch.any(mask):
                continue
            candidate_xyz = x.C + offsets.F[:, 1:]
            weights = (0.15 - offsets.F[:, 0].clamp(0, 0.15)) / 0.15
            candidates = PointTensor(candidate_xyz[mask], weights[mask])
            cluster_cfg = cfg["model"]["dbscan_cfg"]
            cluster_ids = candidates.cluster(**cluster_cfg, return_index=True)
            for cluster_id in torch.unique(cluster_ids):
                members = cluster_ids == cluster_id
                if int(members.sum()) < cluster_cfg["min_points"]:
                    continue  # The original weighted clustering also assigns IDs to noise.
                member_xyz = candidates.C[members]
                member_weights = candidates.F[members]
                if cluster_cfg["weighted_average"]:
                    normalized = (member_xyz * member_weights[:, None]).sum(0) / member_weights.sum()
                else:
                    normalized = member_xyz.mean(0)
                normalized = normalized.cpu().numpy()
                score = float(member_weights.max())
                label = mesial_class(fdi, normalized, centroids[index]) if kind == "MesialDistal" else kind
                original = (np.append(normalized, 1.0) @ inverse_affine.T)[:3]
                output.append({
                    "key": f"uuid_{len(output)}",
                    "score": score,
                    "class": label,
                    "coord": nearest_surface(scene, original),
                    "fdi": fdi,
                    "loader_instance_id": int(tooth_id),
                })
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--mesh", type=Path)
    source.add_argument("--input-root", type=Path, nargs="+")
    parser.add_argument("--annotation", type=Path, help="Defaults to a JSON beside --mesh")
    parser.add_argument("--exclude-landmarks-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.annotation and not args.mesh:
        parser.error("--annotation requires --mesh")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the original LandmarkNet.")
    pairs = discover(args)
    excluded = landmark_stems(args.exclude_landmarks_root)
    if args.mesh and pairs[0][0].stem in excluded:
        raise ValueError("This case has landmark annotations and is excluded from held-out testing.")
    excluded_count = sum(mesh.stem in excluded for mesh, _ in pairs)
    pairs = [(mesh, ann) for mesh, ann in pairs if mesh.stem not in excluded]
    if not pairs:
        raise ValueError("No unannotated landmark cases remain after exclusion.")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dm_cfg = cfg["datamodule"]
    model = LandmarkNet.load_from_checkpoint(
        str(args.checkpoint), map_location="cpu",
        in_channels=9 + 3 * dm_cfg["with_color"], num_classes=5,
        dbscan_cfg=cfg["model"]["dbscan_cfg"],
        **cfg["model"]["landmarks"],
    ).to("cuda").eval()
    summary = {
        "status": "RUNNING", "mode": "GT_FDI_INSTANCE_CONDITIONED",
        "excluded_landmark_annotated_cases": excluded_count, "cases": [],
    }
    failures = 0
    for i, (mesh, ann) in enumerate(pairs, 1):
        print(f"[{i}/{len(pairs)}] {mesh.name}", flush=True)
        entry = {"mesh": str(mesh), "annotation": str(ann)}
        try:
            with torch.inference_mode():
                objects = predict_case(mesh, ann, model, cfg, torch.device("cuda"))
            if not objects:
                raise RuntimeError("No landmarks predicted")
            destination = out_dir / f"{mesh.stem}__kpt.json"
            destination.write_text(json.dumps({
                "version": "1.1", "description": "ground-truth FDI/instance conditioned landmarks",
                "key": mesh.name, "conditioning": "ground_truth_fdi_and_instances",
                "objects": objects,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            entry.update(status="OK", landmarks=len(objects), output=str(destination))
        except Exception as exc:
            failures += 1
            entry.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
            print(f"FAILED {mesh.name}: {entry['error']}", file=sys.stderr, flush=True)
        summary["cases"].append(entry)
        summary["status"] = "FAILED" if failures else "RUNNING"
        (out_dir / "status.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["status"] = "FAILED" if failures else "COMPLETED_NO_LANDMARK_GROUND_TRUTH"
    (out_dir / "status.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{summary['status']}: {len(pairs) - failures}/{len(pairs)} cases", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
