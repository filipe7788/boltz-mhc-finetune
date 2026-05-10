"""Convert filtered mmCIF structures to Boltz's internal npz format + manifest."""

import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def convert_cif_to_npz(cif_path: str, out_dir: str, ccd_path: str) -> bool:
    """Parse a mmCIF file using Boltz's internal parser and save as npz."""
    from boltz.data.parse.mmcif import parse_mmcif
    from boltz.data.mol import load_molecules

    pdb_id = Path(cif_path).stem
    out_path = Path(out_dir) / "structures" / f"{pdb_id}.npz"
    if out_path.exists():
        return True

    try:
        with open(ccd_path, "rb") as f:
            import pickle
            ccd = pickle.load(f)

        molecules = load_molecules(ccd)
        result = parse_mmcif(
            path=cif_path,
            molecules=molecules,
            symmetries=None,
        )
        if result is None:
            return False

        structure = result.structure
        np.savez(
            out_path,
            atoms=structure.atoms,
            bonds=structure.bonds,
            residues=structure.residues,
            chains=structure.chains,
            connections=structure.connections,
            interfaces=structure.interfaces,
            mask=structure.mask,
        )
        return True
    except Exception:
        return False


def build_manifest(filtered_csv: str, structures_dir: str) -> list[dict]:
    """Build manifest records for structures that were successfully converted."""
    from boltz.data.types import Record, ChainInfo, Manifest

    df = pd.read_csv(filtered_csv)
    records = []

    for _, row in df.iterrows():
        pdb_id = row["pdb_id"]
        npz_path = Path(structures_dir) / f"{pdb_id}.npz"
        if not npz_path.exists():
            continue

        try:
            data = np.load(npz_path, allow_pickle=True)
            chains = data["chains"]
            chain_infos = []
            for chain in chains:
                chain_infos.append(
                    ChainInfo(
                        chain_id=str(chain["chain_id"]),
                        chain_type=int(chain["chain_type"]),
                        entity_id=str(chain["entity_id"]),
                        msa_id=-1,  # no MSA for any chain in pMHC training
                        num_residues=int(chain["num_residues"]),
                        valid=True,
                    )
                )
            records.append(
                Record(id=pdb_id, chains=chain_infos, interfaces=[])
            )
        except Exception:
            continue

    return records


def get_ccd_path() -> str:
    """Find the CCD file downloaded by Boltz."""
    cache = Path.home() / ".boltz"
    ccd_path = cache / "ccd.pkl"
    if not ccd_path.exists():
        print("CCD file not found. Run 'boltz predict' once to download it.")
        sys.exit(1)
    return str(ccd_path)


def main():
    filtered_csv = "data/filtered_structures.csv"
    raw_dir = "data/raw"
    out_dir = "data/boltz_processed"
    msa_dir = "data/boltz_processed/msa"

    os.makedirs(f"{out_dir}/structures", exist_ok=True)
    os.makedirs(msa_dir, exist_ok=True)

    ccd_path = get_ccd_path()
    df = pd.read_csv(filtered_csv)

    print(f"Converting {len(df)} CIF files to Boltz npz format...")
    failed = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        pdb_id = row["pdb_id"]
        cif_path = os.path.join(raw_dir, f"{pdb_id}.cif")
        if not os.path.exists(cif_path):
            failed += 1
            continue
        ok = convert_cif_to_npz(cif_path, out_dir, ccd_path)
        if not ok:
            failed += 1

    print(f"Converted: {len(df) - failed} | Failed: {failed}")

    print("Building manifest...")
    records = build_manifest(filtered_csv, f"{out_dir}/structures")

    # Load split to annotate val records
    with open("data/splits.json") as f:
        splits = json.load(f)
    val_ids = set(splits["val"] + splits["test"])

    manifest_data = {"records": []}
    for r in records:
        manifest_data["records"].append({
            "id": r.id,
            "chains": [
                {
                    "chain_id": c.chain_id,
                    "chain_type": c.chain_type,
                    "entity_id": c.entity_id,
                    "msa_id": c.msa_id,
                    "num_residues": c.num_residues,
                    "valid": c.valid,
                }
                for c in r.chains
            ],
            "interfaces": [],
        })

    with open(f"{out_dir}/manifest.json", "w") as f:
        json.dump(manifest_data, f, indent=2)

    # Write val split file (list of IDs that are validation)
    val_in_manifest = [r.id for r in records if r.id in val_ids]
    with open(f"{out_dir}/val_split.txt", "w") as f:
        f.write("\n".join(val_in_manifest))

    print(f"Manifest: {len(records)} records | Val split: {len(val_in_manifest)}")
    print(f"Output: {out_dir}/")


if __name__ == "__main__":
    main()
