"""
Run inference on the test set and compute pRMSD / interface-RMSD.

Usage
-----
# 1. Run predictions on test structures (skips already-done ones)
# 2. Collect Boltz output paths
# 3. Compute pRMSD and iRMSD vs. reference CIF files

python evaluation/run_evaluation.py \
    --checkpoint checkpoints/boltz_mhc_best.ckpt \
    --data_dir data \
    --out_dir results/eval \
    [--splits data/splits.json] \
    [--split test]          # or val
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from metrics import evaluate_dataset


def predict_structure(
    yaml_path: str,
    checkpoint: str | None,
    out_dir: str,
) -> bool:
    """Run boltz predict for a single YAML input. Returns True on success."""
    cmd = ["boltz", "predict", yaml_path, "--out_dir", out_dir]
    if checkpoint:
        cmd += ["--checkpoint", checkpoint]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  boltz predict failed:\n{e.stderr.decode()[-500:]}")
        return False


def collect_prediction_path(out_dir: str, pdb_id: str) -> str | None:
    """
    Boltz writes predictions to:
        {out_dir}/predictions/{pdb_id}/{pdb_id}_model_0.cif   (mmcif, rank-0 = best)

    Returns the path if it exists, else None.
    """
    path = Path(out_dir) / "predictions" / pdb_id / f"{pdb_id}_model_0.cif"
    return str(path) if path.exists() else None


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Boltz-1 on pMHC test set")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Fine-tuned checkpoint (.ckpt). Omit to use base Boltz-1.")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Root data directory (contains raw/, boltz_inputs/, splits.json)")
    parser.add_argument("--out_dir", type=str, default="results/eval",
                        help="Where to write predictions and the metrics CSV")
    parser.add_argument("--splits", type=str, default=None,
                        help="Path to splits.json (default: <data_dir>/splits.json)")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"],
                        help="Which split to evaluate (default: test)")
    parser.add_argument("--alpha_chain", type=str, default="A",
                        help="Chain ID for MHC alpha (default: A)")
    parser.add_argument("--peptide_chain", type=str, default="C",
                        help="Chain ID for peptide (default: C)")
    parser.add_argument("--skip_inference", action="store_true",
                        help="Skip boltz predict and only (re-)compute metrics from existing predictions")
    return parser.parse_args()


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    splits_path = args.splits or os.path.join(args.data_dir, "splits.json")
    with open(splits_path) as f:
        splits = json.load(f)

    pdb_ids = splits[args.split]
    print(f"Evaluating {len(pdb_ids)} structures from '{args.split}' split.")

    yaml_dir = Path(args.data_dir) / "boltz_inputs"
    raw_dir  = Path(args.data_dir) / "raw"

    # ── Step 1: run inference ────────────────────────────────────────────────
    if not args.skip_inference:
        failed_pred = []
        for pdb_id in tqdm(pdb_ids, desc="Running predictions"):
            yaml_path = yaml_dir / f"{pdb_id}.yaml"
            if not yaml_path.exists():
                print(f"  WARNING: no YAML for {pdb_id}, skipping")
                failed_pred.append(pdb_id)
                continue
            # boltz predict is idempotent — skips if output already exists
            ok = predict_structure(str(yaml_path), args.checkpoint, args.out_dir)
            if not ok:
                failed_pred.append(pdb_id)

        if failed_pred:
            print(f"\n{len(failed_pred)} predictions failed: {failed_pred[:5]}{'...' if len(failed_pred) > 5 else ''}")

    # ── Step 2: collect paths ────────────────────────────────────────────────
    predictions = {}
    references  = {}
    missing     = []

    for pdb_id in pdb_ids:
        pred_path = collect_prediction_path(args.out_dir, pdb_id)
        ref_path  = str(raw_dir / f"{pdb_id}.cif")

        if pred_path is None:
            missing.append(pdb_id)
            continue
        if not os.path.exists(ref_path):
            print(f"  WARNING: no reference CIF for {pdb_id}")
            continue

        predictions[pdb_id] = pred_path
        references[pdb_id]  = ref_path

    if missing:
        print(f"\n{len(missing)} structures have no prediction yet "
              f"(run without --skip_inference to generate them).")

    print(f"\nComputing metrics for {len(predictions)} structures...")

    # ── Step 3: compute metrics ──────────────────────────────────────────────
    results = evaluate_dataset(
        predictions=predictions,
        references=references,
        alpha_chain=args.alpha_chain,
        peptide_chain=args.peptide_chain,
    )

    df = pd.DataFrame(results)

    # ── Step 4: print summary ────────────────────────────────────────────────
    if df.empty:
        print("No results computed.")
        return

    print(f"\n{'─'*45}")
    print(f"  pRMSD  — mean: {df['peptide_rmsd'].mean():.3f} Å  "
          f"median: {df['peptide_rmsd'].median():.3f} Å  "
          f"< 2Å: {(df['peptide_rmsd'] < 2.0).mean()*100:.1f}%")
    print(f"  iRMSD  — mean: {df['interface_rmsd'].mean():.3f} Å  "
          f"median: {df['interface_rmsd'].median():.3f} Å  "
          f"< 2Å: {(df['interface_rmsd'] < 2.0).mean()*100:.1f}%")
    print(f"{'─'*45}")

    out_csv = os.path.join(args.out_dir, f"metrics_{args.split}.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nPer-structure results saved to {out_csv}")


if __name__ == "__main__":
    main()
