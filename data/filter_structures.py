"""Filter downloaded PDB structures for valid MHC Class I complexes."""

import os
import json
import pandas as pd
from tqdm import tqdm
from Bio.PDB.MMCIFParser import MMCIFParser

PEPTIDE_LENGTH_RANGE = (8, 12)
ALPHA_CHAIN_LENGTH_RANGE = (170, 210)
B2M_LENGTH_RANGE = (90, 110)


def count_residues(chain) -> int:
    return sum(1 for r in chain.get_residues() if r.id[0] == " ")


def classify_chains(structure) -> dict | None:
    """Attempt to identify alpha, B2M, and peptide chains."""
    chains = list(structure[0].get_chains())
    result = {"alpha": None, "b2m": None, "peptide": None}

    for chain in chains:
        n = count_residues(chain)
        if ALPHA_CHAIN_LENGTH_RANGE[0] <= n <= ALPHA_CHAIN_LENGTH_RANGE[1]:
            result["alpha"] = chain.id
        elif B2M_LENGTH_RANGE[0] <= n <= B2M_LENGTH_RANGE[1]:
            result["b2m"] = chain.id
        elif PEPTIDE_LENGTH_RANGE[0] <= n <= PEPTIDE_LENGTH_RANGE[1]:
            result["peptide"] = chain.id

    if all(v is not None for v in result.values()):
        return result
    return None


def has_missing_residues(chain, structure, model_id=0) -> bool:
    residues = [r for r in structure[model_id][chain].get_residues() if r.id[0] == " "]
    seq_nums = [r.id[1] for r in residues]
    if not seq_nums:
        return True
    expected = set(range(min(seq_nums), max(seq_nums) + 1))
    return len(expected - set(seq_nums)) > 0


def filter_structures(raw_dir: str, out_path: str):
    parser = MMCIFParser(QUIET=True)
    records = []
    failed = []

    cif_files = [f for f in os.listdir(raw_dir) if f.endswith(".cif")]
    print(f"Filtering {len(cif_files)} structures...")

    for fname in tqdm(cif_files):
        pdb_id = fname.replace(".cif", "")
        path = os.path.join(raw_dir, fname)
        try:
            structure = parser.get_structure(pdb_id, path)
            chains = classify_chains(structure)
            if chains is None:
                continue
            if has_missing_residues(chains["peptide"], structure):
                continue
            records.append({"pdb_id": pdb_id, **chains})
        except Exception as e:
            failed.append((pdb_id, str(e)))

    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False)
    print(f"\nPassed: {len(df)} | Failed/skipped: {len(failed)}")
    print(f"Saved to {out_path}")
    return df


def main():
    df = filter_structures("data/raw", "data/filtered_structures.csv")
    print(df.head())


if __name__ == "__main__":
    main()
