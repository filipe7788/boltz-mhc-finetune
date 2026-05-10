"""Download MHC Class I structures from the PDB."""

import os
import json
import requests
from tqdm import tqdm

PDB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
PDB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.cif"


def query_mhc_structures() -> list[str]:
    """Query RCSB for MHC Class I entries with a bound peptide."""
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "struct_keywords.pdbx_keywords",
                        "operator": "contains_words",
                        "value": "MHC",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                        "operator": "greater_or_equal",
                        "value": 2,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": 3.0,
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": 10000}},
    }

    response = requests.post(PDB_SEARCH_URL, json=query, timeout=60)
    response.raise_for_status()
    data = response.json()
    total = data.get("total_count", 0)
    results = data.get("result_set", [])
    if total > len(results):
        print(f"WARNING: query matched {total} entries but only {len(results)} returned. "
              "Increase rows in paginate or implement pagination to avoid missing entries.")
    return [r["identifier"] for r in results]


def download_cif(pdb_id: str, out_dir: str) -> str:
    out_path = os.path.join(out_dir, f"{pdb_id}.cif")
    if os.path.exists(out_path):
        return out_path

    url = PDB_DOWNLOAD_URL.format(pdb_id=pdb_id)
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    with open(out_path, "wb") as f:
        f.write(response.content)
    return out_path


def main():
    out_dir = "data/raw"
    os.makedirs(out_dir, exist_ok=True)

    print("Querying PDB for MHC Class I structures...")
    pdb_ids = query_mhc_structures()
    print(f"Found {len(pdb_ids)} entries.")

    with open("data/pdb_ids.json", "w") as f:
        json.dump(pdb_ids, f)

    print("Downloading mmCIF files...")
    failed = []
    for pdb_id in tqdm(pdb_ids):
        try:
            download_cif(pdb_id, out_dir)
        except Exception as e:
            print(f"  Failed {pdb_id}: {e}")
            failed.append(pdb_id)

    print(f"\nDone. {len(pdb_ids) - len(failed)} downloaded, {len(failed)} failed.")


if __name__ == "__main__":
    main()
