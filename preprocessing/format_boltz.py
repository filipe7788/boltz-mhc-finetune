"""Generate Boltz-1 YAML input files for each MHC-peptide complex."""

import os
import yaml
import pandas as pd
from tqdm import tqdm


def make_boltz_yaml(pdb_id: str, alpha_seq: str, b2m_seq: str, peptide_seq: str) -> dict:
    return {
        "version": 1,
        "sequences": [
            {"protein": {"id": "A", "sequence": alpha_seq}},
            {"protein": {"id": "B", "sequence": b2m_seq}},
            {"protein": {"id": "C", "sequence": peptide_seq}},
        ],
    }


def generate_yaml_inputs(sequences_csv: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(sequences_csv)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        pdb_id = row["pdb_id"]
        config = make_boltz_yaml(
            pdb_id=pdb_id,
            alpha_seq=row["alpha_seq"],
            b2m_seq=row["b2m_seq"],
            peptide_seq=row["peptide_seq"],
        )
        out_path = os.path.join(out_dir, f"{pdb_id}.yaml")
        with open(out_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    print(f"Generated {len(df)} YAML files in {out_dir}/")


def split_dataset(sequences_csv: str, out_dir: str, test_size: float = 0.15, val_size: float = 0.15):
    """Split by unique alpha-chain sequence to avoid HLA allele leakage."""
    from sklearn.model_selection import GroupShuffleSplit
    import json

    df = pd.read_csv(sequences_csv)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size + val_size, random_state=42)
    train_idx, temp_idx = next(gss.split(df, groups=df["alpha_seq"]))

    temp_df = df.iloc[temp_idx]
    val_fraction = val_size / (test_size + val_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=42)
    test_idx_r, val_idx_r = next(gss2.split(temp_df, groups=temp_df["alpha_seq"]))

    splits = {
        "train": df.iloc[train_idx]["pdb_id"].tolist(),
        "val": temp_df.iloc[val_idx_r]["pdb_id"].tolist(),
        "test": temp_df.iloc[test_idx_r]["pdb_id"].tolist(),
    }

    out_path = os.path.join(out_dir, "splits.json")
    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Split — train: {len(splits['train'])} | val: {len(splits['val'])} | test: {len(splits['test'])}")
    return splits


def main():
    generate_yaml_inputs(
        sequences_csv="data/sequences/sequences.csv",
        out_dir="data/boltz_inputs",
    )
    split_dataset(
        sequences_csv="data/sequences/sequences.csv",
        out_dir="data",
    )


if __name__ == "__main__":
    main()
