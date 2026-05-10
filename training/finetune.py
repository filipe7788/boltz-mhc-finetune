"""Fine-tune Boltz-1 on MHC Class I peptide complexes."""

import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


def get_args():
    parser = argparse.ArgumentParser(description="Fine-tune Boltz-1 on MHC-peptide complexes")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to Boltz-1 base checkpoint (boltz1_conf.ckpt)")
    parser.add_argument("--data_dir", type=str, default="data/boltz_processed",
                        help="Path to preprocessed data dir (output of to_boltz_npz.py)")
    parser.add_argument("--ccd_path", type=str, default=None,
                        help="Path to ccd.pkl (default: ~/.boltz/ccd.pkl)")
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Max tokens per crop (reduce for memory)")
    parser.add_argument("--max_atoms", type=int, default=4096)
    parser.add_argument("--samples_per_epoch", type=int, default=1000)
    parser.add_argument("--partial", action="store_true",
                        help="Freeze Evoformer (msa_module + pairformer_module), train structure module only")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_ccd_path(ccd_path: str | None) -> str:
    if ccd_path is not None:
        return ccd_path
    default = Path.home() / ".boltz" / "ccd.pkl"
    if not default.exists():
        raise FileNotFoundError(
            f"CCD file not found at {default}. "
            "Run 'boltz predict' once to download it, or pass --ccd_path."
        )
    return str(default)


def load_model(checkpoint: str, partial: bool) -> "Boltz1":
    from boltz.model.models.boltz1 import Boltz1

    # Boltz-1 is a LightningModule — load_from_checkpoint rehydrates it
    # fully (hparams + weights); torch.load only gives the raw state dict.
    model = Boltz1.load_from_checkpoint(checkpoint, map_location="cpu", weights_only=False)

    if partial:
        # Freeze Evoformer trunk: MSA stack + Pairformer.
        # Only the structure module + diffusion decoder are updated.
        for name, param in model.named_parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if any(k in name for k in ("structure_module", "diffusion_module")):
                param.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Partial fine-tune: {trainable:,} / {total:,} parameters trainable")
    else:
        total = sum(p.numel() for p in model.parameters())
        print(f"Full fine-tune: {total:,} parameters")

    return model


def build_data_module(args, ccd_path: str):
    from boltz.data.crop.boltz import BoltzCropper
    from boltz.data.feature.featurizer import BoltzFeaturizer
    from boltz.data.feature.symmetry import get_symmetries
    from boltz.data.filter.dynamic.filter import DynamicFilter
    from boltz.data.module.training import (
        BoltzTrainingDataModule,
        DataConfig,
        DatasetConfig,
    )
    from boltz.data.sample.random import RandomSampler
    from boltz.data.tokenize.boltz import BoltzTokenizer

    data_dir = Path(args.data_dir)
    val_split_file = data_dir / "val_split.txt"

    dataset_cfg = DatasetConfig(
        target_dir=str(data_dir),
        msa_dir=str(data_dir / "msa"),
        prob=1.0,
        sampler=RandomSampler(),
        cropper=BoltzCropper(min_neighborhood=0, max_neighborhood=40),
        split=str(val_split_file) if val_split_file.exists() else None,
        manifest_path=str(data_dir / "manifest.json"),
    )

    cfg = DataConfig(
        datasets=[dataset_cfg],
        filters=[],  # no dynamic filters; structures are already validated
        featurizer=BoltzFeaturizer(),
        tokenizer=BoltzTokenizer(),
        max_atoms=args.max_atoms,
        max_tokens=args.max_tokens,
        max_seqs=1,        # pMHC: no MSA signal; one sequence per chain
        samples_per_epoch=args.samples_per_epoch,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_seed=args.seed,
        pin_memory=torch.cuda.is_available(),
        symmetries=ccd_path,   # get_symmetries reads CCD pkl for ligand symmetries
        atoms_per_window_queries=32,
        min_dist=2.0,
        max_dist=22.0,
        num_bins=64,
        val_batch_size=1,
    )

    return BoltzTrainingDataModule(cfg)


def pick_accelerator() -> tuple[str, int]:
    """Return (accelerator, devices) for the current hardware."""
    if torch.cuda.is_available():
        return "gpu", 1
    if torch.backends.mps.is_available():
        # MPS: Apple Silicon — Lightning uses accelerator="mps"
        return "mps", 1
    return "cpu", 1


def main():
    args = get_args()
    pl.seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    ccd_path = resolve_ccd_path(args.ccd_path)
    model = load_model(args.checkpoint, args.partial)
    data_module = build_data_module(args, ccd_path)

    accelerator, devices = pick_accelerator()
    print(f"Accelerator: {accelerator}")

    callbacks = [
        ModelCheckpoint(
            dirpath=args.out_dir,
            filename="boltz_mhc_{epoch:02d}_{val_loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Boltz1.configure_optimizers already sets up Adam + AlphaFoldLRScheduler
    # when structure_prediction_training=True; we let Lightning call it.
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        gradient_clip_val=1.0,         # matches Boltz training norm clip
        log_every_n_steps=10,
        default_root_dir=args.out_dir,
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
