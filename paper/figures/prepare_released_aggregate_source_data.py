"""Prepare aggregate, release-safe source data for manuscript figures 2 and 4.

This script reads the already released D1-D3 aggregate metric payloads. It does
not read row-level predictions, fit models, tune hyperparameters, or alter any
scientific artifact. The resulting CSV files are the portable plotting inputs
included with the manuscript figure package.
"""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent / "source_data"
RUNTIME = Path(
    os.environ.get(
        "SCAFFOLDSEAL_D123_METRICS_DIR",
        str(ROOT / "scaffoldseal/artifacts/d123_v1/d123_sealed_oof_metrics"),
    )
).resolve()

METRIC_IDENTITIES = {
    "D1": "42b731c71b158d78c0a16d7fe5e699c1ed061ef16e90063cf824ee8f89121cf7",
    "D2": "4de38252f4112e81a7d362043f72a645996b205138bdf0276a5dcf866b3f46e5",
    "D3": "d12be4360feb4a1ce324e495e462d1edf7758900bc50c0a4f426a584ed0a0b51",
}

EXPECTED_HASHES = {
    "D1": "cc42326c402b64d8ec4a15a53c66bfc322b9d0913ab3d1ef4cab7f1ffba2901a",
    "D2": "2542c9ba6cfeb6f4e348fbbb5060e0f935a7bb6da5727f195cdf6b496bfe0aa8",
    "D3": "8067a1fbce9737c921819fbc37eb33d9daf6dc578e8526b014c801cbfabf89de",
}


def load_payload(variant: str) -> dict:
    path = RUNTIME / METRIC_IDENTITIES[variant] / "attempt_001" / "metrics.json"
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if payload["variant"] != variant:
        raise ValueError(f"Variant mismatch for {variant}: {payload['variant']}")
    if hashlib.sha256(raw).hexdigest() != EXPECTED_HASHES[variant]:
        raise ValueError(f"Released metric file hash mismatch for {variant}")
    if not payload["labels_loaded_after_prediction_sealing"]:
        raise ValueError(f"Prediction-sealing flag is false for {variant}")
    return payload


def main() -> None:
    SOURCE.mkdir(parents=True, exist_ok=True)
    payloads = {variant: load_payload(variant) for variant in METRIC_IDENTITIES}

    seed_rows = []
    for variant, payload in payloads.items():
        for seed_index, mae in payload["source_macro_mae_by_seed"].items():
            seed_rows.append(
                {"variant": variant, "seed_index": int(seed_index), "source_macro_mae": float(mae)}
            )
    pd.DataFrame(seed_rows).sort_values(["variant", "seed_index"]).to_csv(
        SOURCE / "figure4a_seed_metrics.csv", index=False
    )

    source_tables = []
    for variant, payload in payloads.items():
        frame = pd.DataFrame(payload["per_source"])
        means = frame.groupby("source", as_index=False)["mae"].mean()
        source_tables.append(means.rename(columns={"mae": f"{variant.lower()}_mae"}))
    source = source_tables[0]
    for table in source_tables[1:]:
        source = source.merge(table, on="source", how="inner", validate="one_to_one")
    source["d3_minus_d1"] = source["d3_mae"] - source["d1_mae"]
    source.sort_values("d3_minus_d1").to_csv(SOURCE / "figure4b_source_deltas.csv", index=False)

    block_tables = []
    for variant, payload in payloads.items():
        frame = pd.DataFrame(payload["per_block"])
        keys = ["outer_fold", "sealed_block_id", "n"]
        means = frame.groupby(keys, as_index=False)["mae"].mean()
        block_tables.append(means.rename(columns={"mae": f"{variant.lower()}_mae"}))
    block = block_tables[0]
    for table in block_tables[1:]:
        block = block.merge(table, on=["outer_fold", "sealed_block_id", "n"], how="inner", validate="one_to_one")
    block["d3_minus_d1"] = block["d3_mae"] - block["d1_mae"]
    block.sort_values("outer_fold").to_csv(SOURCE / "figure4c_block_deltas.csv", index=False)

    geometry = pd.read_csv(ROOT / "scaffoldseal" / "artifacts" / "source_component_blocks.csv")
    geometry = geometry[
        ["sealed_block_id", "n_curated_rows", "n_unique_molecules", "n_analogue_components", "n_sources"]
    ].sort_values("n_curated_rows", ascending=False)
    geometry.to_csv(SOURCE / "figure2_block_geometry.csv", index=False)


if __name__ == "__main__":
    main()
