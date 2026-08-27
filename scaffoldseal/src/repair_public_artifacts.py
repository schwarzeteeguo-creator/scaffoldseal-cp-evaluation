"""Sanitize the tracked public curated-record artifact without touching the vault.

This repair is deliberately narrow and idempotent. It reads and rewrites only
artifacts/curated_records_public.csv and writes a non-label audit record. It
does not read the split manifest, development labels, or the external vault.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from build_manifests import CURATED_PUBLIC_COLUMNS, OUTCOME_DERIVED_COLUMNS


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PATH = ROOT / "artifacts" / "curated_records_public.csv"
RECORD_PATH = ROOT / "artifacts" / "public_artifact_repair.json"
REQUIRED_REMOVALS = ("replicate_min", "replicate_max", "replicate_spread")


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    before = PUBLIC_PATH.read_bytes()
    frame = pd.read_csv(PUBLIC_PATH, low_memory=False)
    present = [column for column in REQUIRED_REMOVALS if column in frame.columns]
    if not present and RECORD_PATH.exists():
        existing = json.loads(RECORD_PATH.read_text(encoding="utf-8"))
        if existing.get("output_sha256") != sha256(before):
            raise RuntimeError("Current artifact does not match the recorded repaired hash")
        print(json.dumps(existing, indent=2, sort_keys=True))
        return 0
    unexpected_outcomes = sorted(
        OUTCOME_DERIVED_COLUMNS.intersection(
            {str(column).strip().lower() for column in frame.columns}
        )
        - set(REQUIRED_REMOVALS)
    )
    if unexpected_outcomes:
        raise RuntimeError(f"Unexpected outcome-derived columns: {unexpected_outcomes}")
    repaired = frame.drop(columns=present)
    actual = tuple(repaired.columns)
    if actual != CURATED_PUBLIC_COLUMNS:
        raise RuntimeError(
            f"Repaired schema does not match allowlist: actual={actual!r}, "
            f"expected={CURATED_PUBLIC_COLUMNS!r}"
        )
    payload = repaired.to_csv(index=False, lineterminator="\n").encode("utf-8")
    temporary = PUBLIC_PATH.with_suffix(".csv.repairing")
    temporary.write_bytes(payload)
    temporary.replace(PUBLIC_PATH)
    record = {
        "repair": "remove_outcome_derived_replicate_statistics",
        "source_artifact": "artifacts/curated_records_public.csv",
        "input_sha256": sha256(before),
        "output_sha256": sha256(payload),
        "removed_columns": present,
        "final_schema": list(actual),
        "rows_before": int(len(frame)),
        "rows_after": int(len(repaired)),
        "non_removed_values_changed": False,
        "external_vault_read": False,
        "split_regenerated": False,
    }
    RECORD_PATH.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
