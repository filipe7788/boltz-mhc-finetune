"""PyTorch dataset for MHC-peptide complexes."""

import json
import os
from pathlib import Path
from torch.utils.data import Dataset


class MHCPeptideDataset(Dataset):
    def __init__(self, split: str, data_dir: str = "data"):
        splits_path = os.path.join(data_dir, "splits.json")
        with open(splits_path) as f:
            splits = json.load(f)

        self.pdb_ids = splits[split]
        self.yaml_dir = Path(data_dir) / "boltz_inputs"
        self.raw_dir = Path(data_dir) / "raw"

    def __len__(self) -> int:
        return len(self.pdb_ids)

    def __getitem__(self, idx: int) -> dict:
        pdb_id = self.pdb_ids[idx]
        return {
            "pdb_id": pdb_id,
            "yaml_path": str(self.yaml_dir / f"{pdb_id}.yaml"),
            "cif_path": str(self.raw_dir / f"{pdb_id}.cif"),
        }
