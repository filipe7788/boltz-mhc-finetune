"""Structural evaluation metrics for MHC-peptide complexes."""

import numpy as np
from Bio.PDB import PDBParser, Superimposer


def get_ca_atoms(structure, chain_id: str) -> list:
    return [
        atom
        for residue in structure[0][chain_id].get_residues()
        if residue.id[0] == " "
        for atom in residue.get_atoms()
        if atom.name == "CA"
    ]


def rmsd(atoms_pred: list, atoms_ref: list) -> float:
    coords_pred = np.array([a.get_vector().get_array() for a in atoms_pred])
    coords_ref = np.array([a.get_vector().get_array() for a in atoms_ref])
    diff = coords_pred - coords_ref
    return float(np.sqrt((diff ** 2).sum(axis=1).mean()))


def peptide_rmsd(pred_pdb: str, ref_pdb: str, alpha_chain: str = "A", peptide_chain: str = "C") -> float:
    """
    Align structures by MHC alpha chain, then compute RMSD on peptide only.
    Lower is better; < 1.0 Å is excellent, < 2.0 Å is acceptable.
    """
    parser = PDBParser(QUIET=True)
    pred = parser.get_structure("pred", pred_pdb)
    ref = parser.get_structure("ref", ref_pdb)

    pred_alpha = get_ca_atoms(pred, alpha_chain)
    ref_alpha = get_ca_atoms(ref, alpha_chain)

    min_len = min(len(pred_alpha), len(ref_alpha))
    sup = Superimposer()
    sup.set_atoms(ref_alpha[:min_len], pred_alpha[:min_len])
    sup.apply(list(pred[0].get_atoms()))

    pred_pep = get_ca_atoms(pred, peptide_chain)
    ref_pep = get_ca_atoms(ref, peptide_chain)
    min_pep = min(len(pred_pep), len(ref_pep))

    return rmsd(pred_pep[:min_pep], ref_pep[:min_pep])


def interface_rmsd(pred_pdb: str, ref_pdb: str, peptide_chain: str = "C", cutoff: float = 8.0) -> float:
    """RMSD restricted to residues at the MHC-peptide interface (within cutoff Å)."""
    parser = PDBParser(QUIET=True)
    ref = parser.get_structure("ref", ref_pdb)
    pred = parser.get_structure("pred", pred_pdb)

    def interface_residues(structure, chain_a: str, chain_b: str, cutoff: float) -> set:
        residues = set()
        for res_a in structure[0][chain_a].get_residues():
            for res_b in structure[0][chain_b].get_residues():
                for atom_a in res_a.get_atoms():
                    for atom_b in res_b.get_atoms():
                        if atom_a - atom_b < cutoff:
                            residues.add(res_a.id[1])
                            residues.add(res_b.id[1])
        return residues

    interface = interface_residues(ref, "A", peptide_chain, cutoff)

    def get_interface_ca(structure, chain_id):
        return [
            atom
            for residue in structure[0][chain_id].get_residues()
            if residue.id[1] in interface and residue.id[0] == " "
            for atom in residue.get_atoms()
            if atom.name == "CA"
        ]

    pred_ca = get_interface_ca(pred, peptide_chain)
    ref_ca = get_interface_ca(ref, peptide_chain)
    min_len = min(len(pred_ca), len(ref_ca))
    if min_len == 0:
        return float("nan")

    return rmsd(pred_ca[:min_len], ref_ca[:min_len])


def evaluate_dataset(predictions: dict[str, str], references: dict[str, str]) -> list[dict]:
    """
    predictions: {pdb_id: path_to_predicted_pdb}
    references:  {pdb_id: path_to_reference_pdb}
    """
    results = []
    for pdb_id in predictions:
        if pdb_id not in references:
            continue
        p_rmsd = peptide_rmsd(predictions[pdb_id], references[pdb_id])
        i_rmsd = interface_rmsd(predictions[pdb_id], references[pdb_id])
        results.append({"pdb_id": pdb_id, "peptide_rmsd": p_rmsd, "interface_rmsd": i_rmsd})

    return results
