"""Convert filtered mmCIF structures to Boltz's internal npz format + manifest."""

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def _encode_atom_name(name: str) -> tuple:
    """Encode a 4-char atom name string to the 4xi1 int format Boltz uses."""
    name = name.strip()
    ints = [ord(c) - 32 for c in name]
    ints += [0] * (4 - len(ints))
    return tuple(ints[:4])


def _atoms_v2_to_v1(atoms_v2: np.ndarray) -> np.ndarray:
    """Convert AtomV2 structured array to v1 Atom dtype expected by the featurizer."""
    from boltz.data.types import Atom

    out = np.zeros(len(atoms_v2), dtype=np.dtype(Atom))
    for i, a in enumerate(atoms_v2):
        out[i]["name"] = _encode_atom_name(str(a["name"]))
        out[i]["coords"] = a["coords"]
        out[i]["conformer"] = a["coords"]   # best available; no separate conformer in v2
        out[i]["is_present"] = a["is_present"]
        # element / charge / chirality not in AtomV2; leave as 0 (unknown)
    return out


def _bonds_v2_to_v1(bonds_v2: np.ndarray) -> np.ndarray:
    """Convert BondV2 structured array to v1 Bond dtype (atom_1, atom_2, type)."""
    from boltz.data.types import Bond

    out = np.zeros(len(bonds_v2), dtype=np.dtype(Bond))
    out["atom_1"] = bonds_v2["atom_1"]
    out["atom_2"] = bonds_v2["atom_2"]
    out["type"] = bonds_v2["type"]
    return out


def convert_cif_to_npz(cif_path: str, out_dir: str, ccd: dict) -> bool:
    """Parse a mmCIF file using Boltz's parser and save as v1-compatible npz."""
    from boltz.data.parse.mmcif import parse_mmcif
    from boltz.data.types import Connection

    pdb_id = Path(cif_path).stem
    out_path = Path(out_dir) / "structures" / f"{pdb_id}.npz"
    if out_path.exists():
        return True

    try:
        # parse_mmcif returns ParsedStructure(data: StructureV2, info, sequences)
        result = parse_mmcif(path=cif_path, mols=ccd)
        if result is None:
            return False

        s = result.data  # StructureV2

        # Convert v2 atom/bond arrays to v1 format required by BoltzTrainingDataModule
        atoms_v1 = _atoms_v2_to_v1(s.atoms)
        bonds_v1 = _bonds_v2_to_v1(s.bonds)

        # StructureV2 has no connections field; synthesise an empty v1 connections array
        empty_connections = np.array([], dtype=np.dtype(Connection))

        np.savez_compressed(
            out_path,
            atoms=atoms_v1,
            bonds=bonds_v1,
            residues=s.residues,
            chains=s.chains,
            connections=empty_connections,
            interfaces=s.interfaces,
            mask=s.mask,
        )
        return True
    except Exception:
        return False


def build_manifest(filtered_csv: str, structures_dir: str) -> list:
    """Build manifest records for structures that were successfully converted."""
    from boltz.data.types import ChainInfo, Record, StructureInfo

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
                        chain_id=int(chain["asym_id"]),
                        chain_name=str(chain["name"]).strip(),
                        mol_type=int(chain["mol_type"]),
                        cluster_id=-1,
                        msa_id=-1,
                        num_residues=int(chain["res_num"]),
                        entity_id=int(chain["entity_id"]),
                        valid=True,
                    )
                )
            records.append(
                Record(
                    id=pdb_id,
                    structure=StructureInfo(),
                    chains=chain_infos,
                    interfaces=[],
                )
            )
        except Exception:
            continue

    return records


def get_ccd(ccd_path: str | None = None) -> dict:
    """Load the CCD dict from disk."""
    if ccd_path is None:
        ccd_path = Path.home() / ".boltz" / "ccd.pkl"
    else:
        ccd_path = Path(ccd_path)
    if not ccd_path.exists():
        print("CCD file not found. Run 'boltz predict' once to download it.")
        sys.exit(1)
    with open(ccd_path, "rb") as f:
        return pickle.load(f)


def main():
    filtered_csv = "data/filtered_structures.csv"
    raw_dir = "data/raw"
    out_dir = "data/boltz_processed"
    msa_dir = "data/boltz_processed/msa"

    os.makedirs(f"{out_dir}/structures", exist_ok=True)
    os.makedirs(msa_dir, exist_ok=True)

    ccd = get_ccd()
    df = pd.read_csv(filtered_csv)

    print(f"Converting {len(df)} CIF files to Boltz npz format...")
    failed = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        pdb_id = row["pdb_id"]
        cif_path = os.path.join(raw_dir, f"{pdb_id}.cif")
        if not os.path.exists(cif_path):
            failed += 1
            continue
        ok = convert_cif_to_npz(cif_path, out_dir, ccd)
        if not ok:
            failed += 1

    print(f"Converted: {len(df) - failed} | Failed: {failed}")

    print("Building manifest...")
    records = build_manifest(filtered_csv, f"{out_dir}/structures")

    with open("data/splits.json") as f:
        splits = json.load(f)
    val_ids = set(splits["val"])

    manifest_data = {"records": [r.to_dict() for r in records]}

    with open(f"{out_dir}/manifest.json", "w") as f:
        json.dump(manifest_data, f, indent=2)

    val_in_manifest = [r.id for r in records if r.id in val_ids]
    with open(f"{out_dir}/val_split.txt", "w") as f:
        f.write("\n".join(val_in_manifest))

    print(f"Manifest: {len(records)} records | Val split: {len(val_in_manifest)}")
    print(f"Output: {out_dir}/")


if __name__ == "__main__":
    main()
