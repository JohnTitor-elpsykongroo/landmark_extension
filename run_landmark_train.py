"""Train the original LandmarkNet with a config outside the upstream tree."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if Path.cwd().resolve() != PROJECT_ROOT:
    raise RuntimeError(f"Run from the project root: cd {PROJECT_ROOT}")
sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT_ROOT))

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from teethland.datamodules import TeethLandDataModule
from teethland.data.datasets.cache import DatasetCache
from teethland.models import LandmarkNet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--devices", type=int, default=1)
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)

    fold = cfg["datamodule"]["fold"]
    if not isinstance(fold, str) or not Path(fold).is_file():
        raise ValueError("Set datamodule.fold to an existing split file in landmark_extension.")
    output = Path(cfg["work_dir"]).expanduser().resolve() / cfg["version"]
    output.mkdir(parents=True, exist_ok=False)
    cache_dir = output / "cache"
    cache_dir.mkdir()
    original_cache_init = DatasetCache.__init__

    def scoped_cache_init(self, dataset, cache_path, disable=False):
        original_cache_init(self, dataset, cache_dir / cache_path.name, disable)

    DatasetCache.__init__ = scoped_cache_init

    pl.seed_everything(cfg["seed"], workers=True)
    dm = TeethLandDataModule(seed=cfg["seed"], **cfg["datamodule"])
    model = LandmarkNet(
        in_channels=dm.num_channels,
        num_classes=dm.num_classes,
        dbscan_cfg=cfg["model"]["dbscan_cfg"],
        **cfg["model"]["landmarks"],
    )

    logger = TensorBoardLogger(save_dir=str(output), name="tb")
    checkpoints = ModelCheckpoint(
        dirpath=str(output / "checkpoints"),
        filename="landmarks-{epoch:03d}",
        auto_insert_metric_name=False,
        monitor="loss/val",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        default_root_dir=str(output),
        max_epochs=cfg["model"]["landmarks"]["epochs"],
        logger=logger,
        callbacks=[checkpoints, LearningRateMonitor(logging_interval="epoch")],
        accumulate_grad_batches=cfg["accumulate_grad_batches"],
        gradient_clip_val=cfg["gradient_clip_norm"],
    )
    trainer.fit(model, datamodule=dm, ckpt_path=str(args.resume) if args.resume else None)
    print(f"best_checkpoint={checkpoints.best_model_path}")


if __name__ == "__main__":
    main()
