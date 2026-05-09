"""Extract sequences from MHC Class I complexes."""

import os
import pandas as pd
from tqdm import tqdm
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.Polypeptide import PPBuilder

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def chain_to_sequence(chain) -> str:
    residues = [r for r in chain.get_residues() if r.id[0] == " "]
    return "".join(THREE_TO_ONE.get(r.resname, "X") for r in residues)


def extract_sequences(filtered_csv: str, raw_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(filtered_csv)
    parser = MMCIFParser(QUIET=True)
    records = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        pdb_id = row["pdb_id"]
        cif_path = os.path.join(raw_dir, f"{pdb_id}.cif")
        if not os.path.exists(cif_path):
            continue

        structure = parser.get_structure(pdb_id, cif_path)
        model = structure[0]

        alpha_seq = chain_to_sequence(model[row["alpha"]])
        b2m_seq = chain_to_sequence(model[row["b2m"]])
        peptide_seq = chain_to_sequence(model[row["peptide"]])

        fasta_path = os.path.join(out_dir, f"{pdb_id}.fasta")
        with open(fasta_path, "w") as f:
            f.write(f">{pdb_id}_alpha\n{alpha_seq}\n")
            f.write(f">{pdb_id}_b2m\n{b2m_seq}\n")
            f.write(f">{pdb_id}_peptide\n{peptide_seq}\n")

        records.append({
            "pdb_id": pdb_id,
            "alpha_seq": alpha_seq,
            "b2m_seq": b2m_seq,
            "peptide_seq": peptide_seq,
            "peptide_length": len(peptide_seq),
        })

    out_df = pd.DataFrame(records)
    out_df.to_csv(os.path.join(out_dir, "sequences.csv"), index=False)
    print(f"Extracted sequences for {len(out_df)} complexes.")
    return out_df


def main():
    extract_sequences(
        filtered_csv="data/filtered_structures.csv",
        raw_dir="data/raw",
        out_dir="data/sequences",
    )


if __name__ == "__main__":
    main()
