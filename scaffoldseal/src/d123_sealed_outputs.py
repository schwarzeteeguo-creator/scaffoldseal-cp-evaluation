"""Label-free, lossless D123 outer-prediction artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import struct
from typing import Mapping

import numpy as np
import pandas as pd

from d123_runner_governance import canonical_sha256


PREDICTION_SCHEMA_VERSION = "scaffoldseal-d123-sealed-predictions-v1"
NORMALIZED_TO_LOG10_SCALE = 2.0
NORMALIZED_TO_LOG10_OFFSET = -6.0
FORBIDDEN_LABEL_TOKENS = ("label", "target", "observed", "pampa")


def _encode_float64(value: object) -> str:
    number = np.float64(value)
    if not np.isfinite(number):
        raise ValueError("Only finite float64 predictions may be sealed")
    return struct.pack(">d", float(number)).hex()


def _decode_float64(value: str) -> float:
    if not isinstance(value, str) or len(value) != 16:
        raise ValueError("Invalid IEEE-754 binary64 payload")
    number = struct.unpack(">d", bytes.fromhex(value))[0]
    if not np.isfinite(number):
        raise ValueError("Decoded prediction is non-finite")
    return float(number)


def _exclusive_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            str(path),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
    except FileExistsError as error:
        raise RuntimeError("Refusing to overwrite immutable sealed predictions") from error
    try:
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def seal_outer_predictions(
    stage: Mapping[str, object],
    predictions: pd.DataFrame,
    output_root: Path,
) -> dict[str, object]:
    """Seal one fold/seed prediction frame before any metric-label access."""

    if stage.get("kind") != "d123_pampa_outer_fit_prediction":
        raise ValueError("Only a D123 outer-prediction stage may seal predictions")
    if list(predictions.columns) != ["curated_id", "prediction_normalized"]:
        raise ValueError("Prediction input columns must be exactly ID and prediction")
    if any(
        token in str(column).lower()
        for column in predictions.columns
        for token in FORBIDDEN_LABEL_TOKENS
    ):
        raise ValueError("Observed-label-like columns are forbidden before sealing")
    ids = predictions["curated_id"].astype(str)
    if len(predictions) == 0 or ids.duplicated().any() or ids.eq("").any():
        raise ValueError("Sealed prediction IDs must be unique and nonempty")
    values = pd.to_numeric(
        predictions["prediction_normalized"], errors="coerce"
    ).to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Sealed predictions contain missing or non-finite values")
    fold = int(stage["key"]["outer_fold"])
    seed_index = int(stage["key"]["seed_index"])
    records = [
        {
            "curated_id": curated_id,
            "outer_fold": fold,
            "seed_index": seed_index,
            "prediction_normalized_ieee754_be": _encode_float64(value),
            "prediction_log10_papp_ieee754_be": _encode_float64(
                value * NORMALIZED_TO_LOG10_SCALE + NORMALIZED_TO_LOG10_OFFSET
            ),
        }
        for curated_id, value in zip(ids, values)
    ]
    payload = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "authoritative": True,
        "contains_observed_labels": False,
        "scientific_identity_sha256": str(stage["scientific_identity_sha256"]),
        "stage_spec_sha256": str(stage["stage_spec_sha256"]),
        "outer_fold": fold,
        "seed_index": seed_index,
        "n_predictions": len(records),
        "records": records,
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    root = output_root.resolve()
    lossless = root / "predictions.lossless.json"
    readable = root / "predictions.csv"
    _exclusive_write(
        lossless,
        (
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8"),
    )
    table = pd.DataFrame(
        {
            "curated_id": ids,
            "outer_fold": fold,
            "seed_index": seed_index,
            "prediction_normalized": values,
            "prediction_log10_papp": (
                values * NORMALIZED_TO_LOG10_SCALE + NORMALIZED_TO_LOG10_OFFSET
            ),
        }
    )
    _exclusive_write(readable, table.to_csv(index=False).encode("utf-8"))
    return payload


def read_sealed_predictions(
    path: Path, expected_stage: Mapping[str, object]
) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    observed_sha = payload.pop("payload_sha256", None)
    if observed_sha != canonical_sha256(payload):
        raise RuntimeError("Sealed prediction payload hash drifted")
    if (
        payload.get("schema_version") != PREDICTION_SCHEMA_VERSION
        or payload.get("authoritative") is not True
        or payload.get("contains_observed_labels") is not False
        or payload.get("scientific_identity_sha256")
        != expected_stage["scientific_identity_sha256"]
        or payload.get("stage_spec_sha256") != expected_stage["stage_spec_sha256"]
        or int(payload.get("outer_fold")) != int(expected_stage["key"]["outer_fold"])
        or int(payload.get("seed_index"))
        != int(expected_stage["key"]["seed_index"])
    ):
        raise RuntimeError("Sealed prediction identity or label-free contract drifted")
    rows = []
    for record in payload.get("records", []):
        normalized = _decode_float64(
            record["prediction_normalized_ieee754_be"]
        )
        log10_papp = _decode_float64(
            record["prediction_log10_papp_ieee754_be"]
        )
        expected_log = (
            normalized * NORMALIZED_TO_LOG10_SCALE + NORMALIZED_TO_LOG10_OFFSET
        )
        if struct.pack(">d", log10_papp) != struct.pack(">d", expected_log):
            raise RuntimeError("Sealed inverse-transform relation drifted")
        rows.append(
            {
                "curated_id": str(record["curated_id"]),
                "outer_fold": int(record["outer_fold"]),
                "seed_index": int(record["seed_index"]),
                "prediction_normalized": normalized,
                "prediction_log10_papp": log10_papp,
            }
        )
    table = pd.DataFrame(rows)
    if len(table) != int(payload.get("n_predictions")) or table.duplicated(
        ["curated_id", "outer_fold", "seed_index"]
    ).any():
        raise RuntimeError("Sealed prediction row coverage drifted")
    return table
