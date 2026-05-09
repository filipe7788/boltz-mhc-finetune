"""Fine-tune Boltz-1 on MHC Class I peptide complexes."""

import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from dataset import MHCPeptideDataset


def get_args():
    parser = argparse.ArgumentParser(description="Fine-tune Boltz-1 on MHC-peptide complexes")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Boltz-1 base checkpoint")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--partial", action="store_true", help="Freeze backbone, train only structure module")
    return parser.parse_args()


def load_model(checkpoint: str, partial: bool):
    # Boltz-1 is a Lightning module — must use load_from_checkpoint, not torch.load,
    # otherwise you get the raw dict instead of the model object.
    from boltz.model.models.boltz1 import Boltz1
    model = Boltz1.load_from_checkpoint(checkpoint, map_location="cpu")

    if partial:
        for name, param in model.named_parameters():
            param.requires_grad = False
        # Unfreeze only the structure/diffusion output layers
        for name, param in model.named_parameters():
            if "structure_module" in name or "diffusion" in name:
                param.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Partial fine-tune: {trainable:,} trainable parameters")
    else:
        trainable = sum(p.numel() for p in model.parameters())
        print(f"Full fine-tune: {trainable:,} parameters")

    return model


def train(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = load_model(args.checkpoint, args.partial)
    model = model.to(device)

    train_ds = MHCPeptideDataset("train", args.data_dir)
    val_ds = MHCPeptideDataset("val", args.data_dir)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            # Boltz-1 training step — adapt to actual model API
            loss = model.training_step(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                loss = model.validation_step(batch)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{args.epochs} — train: {train_loss:.4f} | val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(args.out_dir, "best_mhc.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved best checkpoint → {ckpt_path}")


if __name__ == "__main__":
    args = get_args()
    train(args)
