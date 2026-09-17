"""Keep upstream landmark data behavior while fixing path and norm handling."""

import re
from pathlib import Path
from typing import Optional

import numpy as np

from teethland.data.datasets import TeethLandDataset
from teethland.data import transforms as T
from teethland.datamodules.teethland import TeethLandDataModule


EXCLUDED_STEMS = {
    "017U3R3T_upper", "01A91JH6_lower", "01A91JH6_upper",
    "01HMF5HV_lower", "01HMF5HV_upper", "E23G704K_lower",
    "E23G704K_upper", "016FSM14_lower", "01FPTYH2_lower",
    "K4BAII5F_upper", "87N5YSES_upper", "67PV9M7X_lower", "S0AON6PZ_lower",
}


class PortableTeethLandDataModule(TeethLandDataModule):
    def __init__(self, regex_filter: str, **kwargs):
        super().__init__(regex_filter=regex_filter, **kwargs)
        self.path_regex_filter = regex_filter

    def _files(self, stage: str, exclude=None):
        meshes = {}
        for extension in self.extensions:
            for path in sorted(self.root.rglob(f"*.{extension}")):
                if path.stem in EXCLUDED_STEMS or (exclude and path.stem in exclude):
                    continue
                relative = path.relative_to(self.root)
                if self.path_regex_filter and not re.search(self.path_regex_filter, relative.as_posix()):
                    continue
                if path.stem in meshes and meshes[path.stem] != relative:
                    raise ValueError(f"Duplicate mesh stem: {path.stem}")
                meshes[path.stem] = relative

        landmarks = {}
        for path in sorted(self.landmarks_root.rglob("*__kpt.json")):
            stem = path.name.split("__")[0]
            if stem in landmarks:
                raise ValueError(f"Duplicate landmark stem: {stem}")
            landmarks[stem] = path.relative_to(self.landmarks_root)

        files = []
        for stem in sorted(meshes.keys() & landmarks.keys()):
            mesh = meshes[stem]
            annotation = mesh.with_suffix(".json")
            if not (self.root / annotation).is_file():
                raise FileNotFoundError(f"Missing segmentation annotation: {annotation}")
            files.append((mesh, annotation, landmarks[stem]))
        return files

    def setup(self, stage: Optional[str] = None):
        if stage not in (None, "fit"):
            raise ValueError(f"Unsupported stage: {stage}")
        rng = np.random.default_rng(self.seed)
        default_transforms = T.Compose(
            T.UniformDensityDownsample(self.uniform_density_voxel_size, inplace=True),
            T.GenerateProposals(self.proposal_points, self.max_proposals, rng=rng),
            self.default_transforms,
        )
        files = self._files("fit")
        print("Total number of landmark-annotated files:", len(files), flush=True)
        train_files, val_files = self._split(files)
        if not train_files or not val_files:
            raise ValueError(f"Empty training or validation split: {len(train_files)}, {len(val_files)}")
        train_transforms = T.Compose(
            T.RandomXAxisFlip(rng=rng),
            T.RandomScale(rng=rng),
            T.RandomZAxisRotate(rng=rng),
            default_transforms,
        )
        common = dict(seg_root=self.root, landmarks_root=self.landmarks_root,
                      norm=self.norm, clean=self.clean)
        self.train_dataset = TeethLandDataset(
            stage="fit", files=train_files, transform=train_transforms, **common,
        )
        self.val_dataset = TeethLandDataset(
            stage="fit", files=val_files, transform=default_transforms, **common,
        )
