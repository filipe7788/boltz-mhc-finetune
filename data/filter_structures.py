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
    candidates = {"alpha": [], "b2m": [], "peptide": []}

    for chain in chains:
        n = count_residues(chain)
        if ALPHA_CHAIN_LENGTH_RANGE[0] <= n <= ALPHA_CHAIN_LENGTH_RANGE[1]:
            candidates["alpha"].append(chain.id)
        if B2M_LENGTH_RANGE[0] <= n <= B2M_LENGTH_RANGE[1]:
            candidates["b2m"].append(chain.id)
        if PEPTIDE_LENGTH_RANGE[0] <= n <= PEPTIDE_LENGTH_RANGE[1]:
            candidates["peptide"].append(chain.id)

    # Reject if any role is ambiguous or missing
    if not all(len(v) == 1 for v in candidates.values()):
        return None

    return {role: ids[0] for role, ids in candidates.items()}


def has_missing_residues(chain, structure, model_id=0) -> bool:
    residues = [r for r in structure[model_id][chain].get_residues() if r.id[0] == " "]
    # Use full residue id (seq_num, icode) to handle insertion codes correctly;
    # plain integer comparison would treat 100A and 100B as the same residue.
    ids = [r.id[1:] for r in residues]  # (seq_num, icode) tuples
    if not ids:
        return True
    seq_nums = [i[0] for i in ids]
    expected_count = max(seq_nums) - min(seq_nums) + 1
    unique_ids = len(set(ids))
    return unique_ids < expected_count


def filter_structures(raw_dir: str, out_path: str):
    parser = MMCIFParser(QUIET=True)
    records = []
    failed = []

    cif_files = [f for f in os.listdir(raw_dir) if f.endswith(".cif")]
    print(f"Filtering {len(cif_files)} structures...")

    skipped = 0
    for fname in tqdm(cif_files):
        pdb_id = fname.replace(".cif", "")
        path = os.path.join(raw_dir, fname)
        try:
            structure = parser.get_structure(pdb_id, path)
            chains = classify_chains(structure)
            if chains is None:
                skipped += 1
                continue
            if has_missing_residues(chains["peptide"], structure) or \
               has_missing_residues(chains["alpha"], structure):
                skipped += 1
                continue
            records.append({"pdb_id": pdb_id, **chains})
        except Exception as e:
            failed.append((pdb_id, str(e)))

    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False)
    print(f"\nPassed: {len(df)} | Skipped (chain/quality): {skipped} | Parse errors: {len(failed)}")
    print(f"Saved to {out_path}")
    return df


def main():
    df = filter_structures("data/raw", "data/filtered_structures.csv")
    print(df.head())


if __name__ == "__main__":
    main()
