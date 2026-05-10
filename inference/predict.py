"""Run Boltz-1 inference on new MHC-peptide pairs."""

import argparse
import os
import subprocess
import yaml


def make_input_yaml(hla_sequence: str, peptide_sequence: str, out_path: str):
    config = {
        "version": 1,
        "sequences": [
            {"protein": {"id": "A", "sequence": hla_sequence}},
            {
                "protein": {
                    "id": "B",
                    # beta-2 microglobulin (canonical human sequence)
                    # Mature B2M sequence (signal peptide cleaved, 96 aa)
                    "sequence": (
                        "IQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKNGER"
                        "VEHSDLSFSKDWSFYLLYYTEFTPTEKDEYACRVNHVTLSQPKIVKWDRDM"
                    ),
                }
            },
            {"protein": {"id": "C", "sequence": peptide_sequence}},
        ],
    }
    with open(out_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def predict(
    hla_sequence: str,
    peptide_sequence: str,
    checkpoint: str | None,
    out_dir: str,
    use_msa_server: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)
    yaml_path = os.path.join(out_dir, "input.yaml")
    make_input_yaml(hla_sequence, peptide_sequence, yaml_path)

    cmd = ["boltz", "predict", yaml_path, "--out_dir", out_dir]
    if checkpoint:
        cmd += ["--checkpoint", checkpoint]
    if use_msa_server:
        cmd.append("--use_msa_server")

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"\nResults saved to {out_dir}/")


def get_args():
    parser = argparse.ArgumentParser(description="Predict MHC-peptide structure with Boltz-1")
    parser.add_argument("--hla_sequence", type=str, required=True, help="HLA alpha chain sequence (FASTA string)")
    parser.add_argument("--peptide", type=str, required=True, help="Peptide sequence (8-12 amino acids)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Fine-tuned checkpoint (optional)")
    parser.add_argument("--out_dir", type=str, default="results/prediction")
    parser.add_argument("--no_msa_server", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    predict(
        hla_sequence=args.hla_sequence,
        peptide_sequence=args.peptide,
        checkpoint=args.checkpoint,
        out_dir=args.out_dir,
        use_msa_server=not args.no_msa_server,
    )
