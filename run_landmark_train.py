"""Train the original LandmarkNet with a config outside the upstream tree."""

import argparse
from pathlib import Path

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from teethland.datamodules import TeethLandDataModule
from teethland.models import LandmarkNet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--devices", type=int, default=1)
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)

    pl.seed_everything(cfg["seed"], workers=True)
    dm = TeethLandDataModule(seed=cfg["seed"], **cfg["datamodule"])
    model = LandmarkNet(
        in_channels=dm.num_channels,
        num_classes=dm.num_classes,
        dbscan_cfg=cfg["model"]["dbscan_cfg"],
        **cfg["model"]["landmarks"],
    )

    output = Path(cfg["work_dir"]).expanduser().resolve() / cfg["version"]
    output.mkdir(parents=True, exist_ok=False)
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
