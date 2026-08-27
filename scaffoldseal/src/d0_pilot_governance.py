"""Zero-training governance helpers for the frozen ScaffoldSeal-CP D0 pilot.

This module does not import the model implementation and has no training or
full-LOBO dispatcher.  It provides immutable attempt inventories, a one-shot
cross-process dispatch ledger, exact-row resource projection, read-only
checkpoint accounting, and a lossless prediction interchange format.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
from typing import Iterable, Mapping, Sequence

import pandas as pd


LOSSLESS_PREDICTION_SCHEMA = "scaffoldseal-predictions-ieee754-v1"
ATTEMPT_INVENTORY_SCHEMA = "scaffoldseal-d0-attempt-inventory-v1"
DISPATCH_LEDGER_SCHEMA = "scaffoldseal-single-dispatch-ledger-v1"
CORRECTED_PROJECTION_SCHEMA = "scaffoldseal-d0-resource-projection-v2"
_COPYABLE_TRACE_NAMES = frozenset({"pretraining_trace.json", "training_trace.json"})


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def stream_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _utc_iso_from_ns(value: int) -> str:
    return datetime.fromtimestamp(value / 1_000_000_000, tz=timezone.utc).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z")


def _file_kind(relative_path: str) -> str:
    name = Path(relative_path).name
    if name in _COPYABLE_TRACE_NAMES:
        return "scientific_trace_json"
    if name == "checkpoint1.pt":
        return "model_checkpoint_binary"
    if name == "tasks.json" or name.endswith("metadata.json"):
        return "dataset_metadata_json"
    if relative_path.endswith(".json"):
        return "provenance_json"
    if relative_path.endswith(".csv"):
        return "materialized_tabular_input"
    if relative_path.endswith((".npy", ".gzip")):
        return "materialized_dataset_cache"
    return "other"


def inventory_attempt_roots(
    attempt_roots: Mapping[str, Path],
    archive_dir: Path,
    *,
    copy_trace_limit_bytes: int = 1_048_576,
) -> dict[str, object]:
    """Hash every retained file and copy only small scientific trace JSON.

    Each file is stat'ed before and after hashing.  Enumeration is repeated at
    the end so a concurrent mutation or path addition cannot be silently
    accepted into the inventory.
    """

    archive_dir = archive_dir.resolve()
    trace_root = archive_dir / "traces"
    attempts: list[dict[str, object]] = []
    all_file_records: list[dict[str, object]] = []
    copied: list[dict[str, object]] = []

    for label, supplied_root in sorted(attempt_roots.items()):
        root = supplied_root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Retained attempt root is missing: {root}")
        initial_paths = sorted(
            (path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())
        )
        root_records: list[dict[str, object]] = []
        for relative_path in initial_paths:
            path = root / Path(relative_path)
            before = path.stat()
            digest = stream_sha256(path)
            after = path.stat()
            before_key = (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            after_key = (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            if before_key != after_key:
                raise RuntimeError(f"Attempt file changed while hashing: {path}")
            birth_ns = int(getattr(before, "st_birthtime_ns", before.st_ctime_ns))
            record = {
                "attempt": label,
                "relative_path": relative_path,
                "size_bytes": int(before.st_size),
                "created_utc": _utc_iso_from_ns(birth_ns),
                "created_time_ns": birth_ns,
                "modified_utc": _utc_iso_from_ns(int(before.st_mtime_ns)),
                "modified_time_ns": int(before.st_mtime_ns),
                "sha256": digest,
                "kind": _file_kind(relative_path),
            }
            root_records.append(record)
            all_file_records.append(record)

            if (
                Path(relative_path).name in _COPYABLE_TRACE_NAMES
                and before.st_size <= copy_trace_limit_bytes
            ):
                destination = trace_root / label / Path(relative_path)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
                if stream_sha256(destination) != digest:
                    raise RuntimeError(f"Copied trace hash differs: {destination}")
                copied.append(
                    {
                        "attempt": label,
                        "source_relative_path": relative_path,
                        "archive_relative_path": destination.relative_to(archive_dir).as_posix(),
                        "size_bytes": int(before.st_size),
                        "sha256": digest,
                    }
                )

        final_paths = sorted(
            (path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())
        )
        if initial_paths != final_paths:
            raise RuntimeError(f"Attempt path set changed while inventorying: {root}")
        attempts.append(
            {
                "attempt": label,
                "retained_root": str(root),
                "file_count": len(root_records),
                "total_size_bytes": sum(int(item["size_bytes"]) for item in root_records),
                "earliest_modified_utc": min(
                    (str(item["modified_utc"]) for item in root_records), default=None
                ),
                "latest_modified_utc": max(
                    (str(item["modified_utc"]) for item in root_records), default=None
                ),
                "scientific_trace_count": sum(
                    item["kind"] == "scientific_trace_json" for item in root_records
                ),
                "checkpoint_count": sum(
                    item["kind"] == "model_checkpoint_binary" for item in root_records
                ),
            }
        )

    payload: dict[str, object] = {
        "schema_version": ATTEMPT_INVENTORY_SCHEMA,
        "scope": "retained local bounded-pilot attempts; read-only inventory",
        "attempts": attempts,
        "files": all_file_records,
        "copied_small_trace_json": copied,
        "copy_policy": {
            "allowed_names": sorted(_COPYABLE_TRACE_NAMES),
            "maximum_bytes": copy_trace_limit_bytes,
            "large_dataset_or_checkpoint_copied": False,
        },
        "totals": {
            "file_count": len(all_file_records),
            "size_bytes": sum(int(item["size_bytes"]) for item in all_file_records),
            "copied_trace_count": len(copied),
        },
    }
    payload["files_canonical_sha256"] = canonical_json_sha256(all_file_records)
    return payload


def frozen_pilot_identity(protocol: Mapping[str, object]) -> str:
    identity = {
        "stage": "scaffoldseal-cp-d0-bounded-runtime-pilot",
        "protocol_sha256": canonical_json_sha256(protocol),
        "baseline_commit": protocol["baseline_commit"],
        "model": protocol["model"],
        "pilot_case": protocol["pilot_case"],
        "pretraining": protocol["pretraining"],
        "authorized_scientific_fit_counts": protocol["authorized_scientific_fit_counts"],
    }
    return canonical_json_sha256(identity)


class DispatchAlreadyRecorded(RuntimeError):
    """Raised when a scientific identity already has an immutable attempt."""


@dataclass(frozen=True)
class DispatchClaim:
    identity: str
    ledger_path: Path


class SingleDispatchLedger:
    """One-shot process-safe scientific-dispatch ledger.

    The first process atomically creates ``<identity>.json`` with O_EXCL.  The
    record is never deleted, including after failure, so changing a work/output
    namespace cannot relaunch the same scientific identity.
    """

    def __init__(self, root: Path):
        self.root = root.resolve()

    def claim(self, identity: str, attempt: Mapping[str, object]) -> DispatchClaim:
        if len(identity) != 64 or any(char not in "0123456789abcdef" for char in identity):
            raise ValueError("Scientific identity must be a lowercase SHA-256 digest")
        self.root.mkdir(parents=True, exist_ok=True)
        ledger_path = self.root / f"{identity}.json"
        record = {
            "schema_version": DISPATCH_LEDGER_SCHEMA,
            "scientific_identity_sha256": identity,
            "status": "CLAIMED_BEFORE_SCIENTIFIC_EXECUTION",
            "attempt": dict(attempt),
            "one_shot": True,
        }
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        try:
            descriptor = os.open(str(ledger_path), flags, 0o600)
        except FileExistsError as error:
            raise DispatchAlreadyRecorded(
                f"Scientific identity already has a dispatch record: {identity}"
            ) from error
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(json.dumps(record, indent=2, sort_keys=True).encode("utf-8") + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            # A partially written record still blocks relaunch and is evidence
            # that this identity was claimed.  It must never be deleted here.
            raise
        return DispatchClaim(identity=identity, ledger_path=ledger_path)

    @staticmethod
    def update_status(
        claim: DispatchClaim, status: str, details: Mapping[str, object] | None = None
    ) -> None:
        existing = json.loads(claim.ledger_path.read_text(encoding="utf-8"))
        if existing["scientific_identity_sha256"] != claim.identity:
            raise RuntimeError("Dispatch ledger identity changed")
        existing["status"] = status
        if details is not None:
            existing["details"] = dict(details)
        temporary = claim.ledger_path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(existing, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, claim.ledger_path)


def inspect_checkpoint_read_only(
    checkpoint_path: Path,
    *,
    restored_checkpoint_path: Path,
    training_rows: int,
    batch_size: int,
) -> dict[str, object]:
    """Read checkpoint counters on CPU without constructing a model/optimizer.

    The checkpoint and restored-pretraining checkpoint are hashed before and
    after ``torch.load``.  The returned delta is a lower bound because an
    interrupted fit can perform updates after its most recently saved best
    checkpoint.
    """

    import torch

    checkpoint_path = checkpoint_path.resolve()
    restored_checkpoint_path = restored_checkpoint_path.resolve()
    before = {
        "checkpoint": stream_sha256(checkpoint_path),
        "restored_checkpoint": stream_sha256(restored_checkpoint_path),
    }
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    restored = torch.load(restored_checkpoint_path, map_location="cpu")
    after = {
        "checkpoint": stream_sha256(checkpoint_path),
        "restored_checkpoint": stream_sha256(restored_checkpoint_path),
    }
    if before != after:
        raise RuntimeError("Checkpoint bytes changed during read-only inspection")
    checkpoint_step = int(checkpoint["global_step"])
    restored_step = int(restored["global_step"])
    optimizer_steps = sorted(
        {
            int(state["step"].item() if hasattr(state["step"], "item") else state["step"])
            for state in checkpoint["optimizer_state_dict"]["state"].values()
            if "step" in state
        }
    )
    delta = checkpoint_step - restored_step
    updates_per_epoch = math.ceil(training_rows / batch_size)
    if delta < 0 or delta % updates_per_epoch:
        raise RuntimeError("Checkpoint step delta is inconsistent with exact batch accounting")
    return {
        "schema_version": "scaffoldseal-zero-update-checkpoint-inspection-v1",
        "inspection_mode": "torch.load(map_location='cpu'); no model, optimizer, or scheduler constructed",
        "checkpoint_path": str(checkpoint_path),
        "restored_checkpoint_path": str(restored_checkpoint_path),
        "checkpoint_sha256_before_after": before["checkpoint"],
        "restored_checkpoint_sha256_before_after": before["restored_checkpoint"],
        "checkpoint_global_step": checkpoint_step,
        "restored_checkpoint_global_step": restored_step,
        "optimizer_state_step_values": optimizer_steps,
        "training_rows": int(training_rows),
        "batch_size": int(batch_size),
        "updates_per_epoch": updates_per_epoch,
        "minimum_stage_optimizer_updates": delta,
        "minimum_stage_epoch_fit_calls": delta // updates_per_epoch,
        "lower_bound_reason": (
            "The partial attempt has no final trace; updates after the retained best checkpoint, "
            "if any, are not recoverable."
        ),
        "weight_or_optimizer_update_performed_by_inspection": False,
    }


def exact_outer_training_rows(outer_assignments: pd.DataFrame) -> pd.Series:
    required = {"curated_id", "outer_fold"}
    if not required.issubset(outer_assignments.columns):
        raise ValueError(f"Outer assignments require columns: {sorted(required)}")
    frame = outer_assignments.loc[:, ["curated_id", "outer_fold"]].copy()
    if frame["curated_id"].astype(str).duplicated().any():
        raise ValueError("Outer assignments contain duplicate curated_id values")
    folds = pd.to_numeric(frame["outer_fold"], errors="raise").astype(int)
    observed = tuple(sorted(folds.unique().tolist()))
    if observed != tuple(range(1, 19)):
        raise ValueError("Exact D0 projection requires outer folds 1..18")
    test_rows = folds.value_counts().sort_index()
    return (len(frame) - test_rows).astype(int).rename("n_outer_training_rows")


def corrected_joint_lobo_projection(
    inner_traces: Iterable[Mapping[str, object]],
    outer_trace: Mapping[str, object],
    outer_assignments: pd.DataFrame,
    pretrain_seconds: float,
) -> dict[str, object]:
    inner = list(inner_traces)
    if len(inner) != 4:
        raise ValueError("The frozen bounded pilot requires exactly four inner traces")
    exact_rows = exact_outer_training_rows(outer_assignments)
    pilot_inner_seconds = sum(float(item["runtime_seconds"]) for item in inner)
    pilot_outer_rows = int(outer_trace["n_train"])
    pilot_outer_seconds = float(outer_trace["runtime_seconds"])
    pilot_outer_fold = int(outer_trace["outer_fold"])
    if pilot_outer_fold not in exact_rows.index:
        raise ValueError("Pilot outer fold is absent from exact outer assignments")
    if int(exact_rows.loc[pilot_outer_fold]) != pilot_outer_rows:
        raise ValueError("Pilot outer-training row count differs from exact assignments")
    runtime_values = [
        *(float(item["runtime_seconds"]) for item in inner),
        pilot_outer_seconds,
        float(pretrain_seconds),
    ]
    if pilot_outer_rows <= 0 or any(not math.isfinite(value) or value < 0 for value in runtime_values):
        raise ValueError("Projection runtimes/row counts must be non-negative")
    inner_seconds = pilot_inner_seconds * len(exact_rows)
    outer_seconds = pilot_outer_seconds * 5.0 * float(exact_rows.sum()) / pilot_outer_rows
    pretraining_seconds = float(pretrain_seconds) * 5.0
    total_hours = (inner_seconds + outer_seconds + pretraining_seconds) / 3600.0
    return {
        "schema_version": CORRECTED_PROJECTION_SCHEMA,
        "status": "CANDIDATE_RESOURCE_EVIDENCE_ONLY",
        "original_v3_summary_modified": False,
        "formula": (
            "(sum(pilot four-inner runtimes)*18 + pilot outer runtime*5*"
            "sum(exact outer-training rows)/pilot outer-training rows + "
            "pilot pretraining-A runtime*5) / 3600"
        ),
        "inputs": {
            "outer_folds": len(exact_rows),
            "outer_seeds": 5,
            "pretraining_seeds": 5,
            "pilot_inner_runtime_seconds": pilot_inner_seconds,
            "pilot_outer_runtime_seconds": pilot_outer_seconds,
            "pilot_outer_fold": pilot_outer_fold,
            "pilot_outer_training_rows": pilot_outer_rows,
            "exact_outer_training_rows_by_fold": {
                str(int(fold)): int(rows) for fold, rows in exact_rows.items()
            },
            "exact_outer_training_rows_sum": int(exact_rows.sum()),
            "pilot_pretraining_a_runtime_seconds": float(pretrain_seconds),
        },
        "components_seconds": {
            "inner": inner_seconds,
            "outer": outer_seconds,
            "pretraining": pretraining_seconds,
        },
        "projected_joint_lobo_gpu_hours": total_hours,
        "frozen_limit_gpu_hours": 72.0,
        "within_frozen_limit": total_hours <= 72.0,
        "caveat": (
            "v2 and v3 overlapped, so the v3 runtime is not an isolated-process measurement; "
            "this projection is sizing evidence, not authorization to run full D0."
        ),
    }


def project_joint_lobo_hours_exact(
    inner_traces: Iterable[Mapping[str, object]],
    outer_trace: Mapping[str, object],
    outer_assignments: pd.DataFrame,
    pretrain_seconds: float,
) -> float:
    return float(
        corrected_joint_lobo_projection(
            inner_traces, outer_trace, outer_assignments, pretrain_seconds
        )["projected_joint_lobo_gpu_hours"]
    )


def _encode_float64(value: object) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Lossless prediction artifacts require finite float64 values")
    return struct.pack(">d", number).hex()


def _decode_float64(value: str) -> float:
    if len(value) != 16:
        raise ValueError("Invalid IEEE-754 binary64 hex payload")
    return struct.unpack(">d", bytes.fromhex(value))[0]


def write_lossless_prediction_artifacts(
    table: pd.DataFrame, lossless_path: Path, readable_csv_path: Path
) -> dict[str, object]:
    """Write exact float64 bits to JSON plus a human-readable CSV companion."""

    required = (
        "curated_id",
        "outer_fold",
        "seed",
        "observed_log10_papp",
        "prediction_normalized",
        "prediction_log10_papp",
    )
    if tuple(table.columns) != required:
        raise ValueError(f"Prediction columns must exactly equal {required}")
    if table["curated_id"].astype(str).duplicated().any():
        raise ValueError("Prediction curated_id values must be unique")
    records = []
    for row in table.itertuples(index=False, name=None):
        records.append(
            {
                "curated_id": str(row[0]),
                "outer_fold": int(row[1]),
                "seed": int(row[2]),
                "observed_log10_papp_ieee754_be": _encode_float64(row[3]),
                "prediction_normalized_ieee754_be": _encode_float64(row[4]),
                "prediction_log10_papp_ieee754_be": _encode_float64(row[5]),
            }
        )
    payload = {
        "schema_version": LOSSLESS_PREDICTION_SCHEMA,
        "encoding": "finite IEEE-754 binary64, big-endian hexadecimal",
        "columns": list(required),
        "record_count": len(records),
        "records": records,
        "records_sha256": canonical_json_sha256(records),
    }
    write_json(lossless_path, payload)
    readable_csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(readable_csv_path, index=False, float_format="%.17g", lineterminator="\n")
    return {
        "lossless_path": str(lossless_path.resolve()),
        "lossless_file_sha256": stream_sha256(lossless_path),
        "records_sha256": payload["records_sha256"],
        "readable_csv_path": str(readable_csv_path.resolve()),
        "readable_csv_sha256": stream_sha256(readable_csv_path),
        "record_count": len(records),
    }


def read_lossless_prediction_artifact(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != LOSSLESS_PREDICTION_SCHEMA:
        raise ValueError("Unsupported lossless prediction schema")
    records = payload.get("records")
    if not isinstance(records, list) or canonical_json_sha256(records) != payload.get(
        "records_sha256"
    ):
        raise ValueError("Lossless prediction record hash differs")
    decoded = []
    for record in records:
        decoded.append(
            {
                "curated_id": str(record["curated_id"]),
                "outer_fold": int(record["outer_fold"]),
                "seed": int(record["seed"]),
                "observed_log10_papp": _decode_float64(
                    record["observed_log10_papp_ieee754_be"]
                ),
                "prediction_normalized": _decode_float64(
                    record["prediction_normalized_ieee754_be"]
                ),
                "prediction_log10_papp": _decode_float64(
                    record["prediction_log10_papp_ieee754_be"]
                ),
            }
        )
    frame = pd.DataFrame(decoded, columns=payload["columns"])
    if len(frame) != int(payload["record_count"]):
        raise ValueError("Lossless prediction record count differs")
    return frame


def summarize_cumulative_execution(
    attempt_roots: Mapping[str, Path], partial_checkpoint: Mapping[str, object]
) -> dict[str, object]:
    """Aggregate all complete traces and the read-only partial lower bound."""

    attempt_summaries: list[dict[str, object]] = []
    totals = {
        "delaney_pretraining_complete_runs": 0,
        "pampa_inner_complete_runs": 0,
        "pampa_inner_partial_runs": 0,
        "pampa_outer_complete_runs": 0,
        "pretraining_epoch_fit_calls": 0,
        "pampa_inner_epoch_fit_calls": 0,
        "pampa_outer_epoch_fit_calls": 0,
        "optimizer_updates": 0,
    }
    for label, supplied_root in sorted(attempt_roots.items()):
        root = supplied_root.resolve()
        stages: list[dict[str, object]] = []
        trace_paths = sorted(root.rglob("*training_trace.json"))
        for trace_path in trace_paths:
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            relative = trace_path.relative_to(root).as_posix()
            if trace_path.name == "pretraining_trace.json":
                stage = "delaney_pretraining"
                totals["delaney_pretraining_complete_runs"] += 1
                totals["pretraining_epoch_fit_calls"] += int(trace["fit_calls"])
            else:
                stage = str(trace["role"])
                if stage == "inner":
                    totals["pampa_inner_complete_runs"] += 1
                    totals["pampa_inner_epoch_fit_calls"] += int(trace["fit_calls"])
                elif stage == "outer_refit":
                    totals["pampa_outer_complete_runs"] += 1
                    totals["pampa_outer_epoch_fit_calls"] += int(trace["fit_calls"])
                else:
                    raise ValueError(f"Unknown training trace role: {stage}")
            totals["optimizer_updates"] += int(trace["optimizer_updates"])
            stages.append(
                {
                    "stage": stage,
                    "relative_trace_path": relative,
                    "status": "COMPLETE_TRACE",
                    "fit_calls": int(trace["fit_calls"]),
                    "optimizer_updates": int(trace["optimizer_updates"]),
                    "checkpoint_sha256": trace["checkpoint_sha256"],
                }
            )

        if label == "v2":
            totals["pampa_inner_partial_runs"] += 1
            totals["pampa_inner_epoch_fit_calls"] += int(
                partial_checkpoint["minimum_stage_epoch_fit_calls"]
            )
            totals["optimizer_updates"] += int(
                partial_checkpoint["minimum_stage_optimizer_updates"]
            )
            stages.append(
                {
                    "stage": "inner",
                    "inner_basket": 4,
                    "status": "PARTIAL_CHECKPOINT_LOWER_BOUND",
                    "relative_checkpoint_path": (
                        "pampa/outer_01/inner_04/d0_locked/seed_0/checkpoint1.pt"
                    ),
                    "minimum_fit_calls": int(
                        partial_checkpoint["minimum_stage_epoch_fit_calls"]
                    ),
                    "minimum_optimizer_updates": int(
                        partial_checkpoint["minimum_stage_optimizer_updates"]
                    ),
                    "checkpoint_global_step": int(partial_checkpoint["checkpoint_global_step"]),
                    "restored_checkpoint_global_step": int(
                        partial_checkpoint["restored_checkpoint_global_step"]
                    ),
                }
            )
        checkpoint_count = len(list(root.rglob("checkpoint1.pt")))
        attempt_summaries.append(
            {
                "attempt": label,
                "status": (
                    "MATERIALIZATION_ONLY_NO_FIT_TRACE_OR_CHECKPOINT"
                    if not stages and checkpoint_count == 0
                    else "COMPLETE_CANDIDATE"
                    if any(stage["stage"] == "outer_refit" for stage in stages)
                    else "PARTIAL_SCIENTIFIC_ATTEMPT"
                ),
                "checkpoint_count": checkpoint_count,
                "stages": stages,
            }
        )
    totals["scientific_logical_fits_complete"] = (
        totals["delaney_pretraining_complete_runs"]
        + totals["pampa_inner_complete_runs"]
        + totals["pampa_outer_complete_runs"]
    )
    totals["scientific_logical_fits_including_partial"] = (
        totals["scientific_logical_fits_complete"] + totals["pampa_inner_partial_runs"]
    )
    totals["epoch_fit_calls_lower_bound"] = (
        totals["pretraining_epoch_fit_calls"]
        + totals["pampa_inner_epoch_fit_calls"]
        + totals["pampa_outer_epoch_fit_calls"]
    )
    return {
        "schema_version": "scaffoldseal-d0-cumulative-execution-v1",
        "accounting_scope": "all retained v1/v2/v3 local attempt namespaces",
        "attempts": attempt_summaries,
        "cumulative": totals,
        "lower_bound": True,
        "lower_bound_reason": partial_checkpoint["lower_bound_reason"],
        "full_lobo_started": False,
        "accepted_model_result": False,
        "v3_use": "resource sizing evidence only under D-036",
    }
