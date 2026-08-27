"""No-training probe for restoring one accepted D3 inner checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def accepted_attempt(root: Path) -> Path:
    accepted = []
    for attempt in sorted(root.glob("attempt_*")):
        manifest_path = attempt / "artifact_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != "scaffoldseal-d123-artifact-manifest-v1":
            raise RuntimeError(f"Unexpected D123 artifact manifest schema: {manifest_path}")
        for record in manifest["artifacts"]:
            path = attempt / record["relative_path"]
            if path.stat().st_size != int(record["size_bytes"]) or sha256(path) != record["sha256"]:
                raise RuntimeError(f"Accepted artifact mismatch: {path}")
        accepted.append(attempt)
    if len(accepted) != 1:
        raise RuntimeError(f"Expected one accepted attempt, found {len(accepted)}")
    return accepted[0]


def forbidden(*_args, **_kwargs):
    raise RuntimeError("ZERO_TRAINING_GUARD: a forbidden learning/write path was called")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.runtime_root.resolve()
    plan = json.loads((root / "d123_plan_candidate.json").read_text(encoding="utf-8"))
    stages = [
        stage for stage in plan["stages"]
        if stage["kind"] == "d123_pampa_inner_fit"
        and stage["key"]["variant"] == "D3"
        and int(stage["key"]["outer_fold"]) == 1
        and int(stage["key"]["inner_basket"]) == 1
    ]
    if len(stages) != 1:
        raise RuntimeError("Probe stage identity is not unique")
    stage = stages[0]
    attempt = accepted_attempt(
        root / "artifacts/d123_v1/d123_pampa_inner_fit" / stage["scientific_identity_sha256"]
    )
    checkpoint = attempt / "checkpoint1.pt"
    transform_path = attempt / "descriptor_transform.json"

    import sys
    sys.path.insert(0, str(root / "src"))
    from d123_dmpnn_integration import D123FitPayload, build_ordered_role_dataset, create_pinned_dmpnn_model
    from d123_features import FitScopedDescriptorTransform
    from d123_fold_data import load_fold_frames
    import deepchem as dc

    frames = load_fold_frames(plan, stage, root)
    validation = frames["validation"].sort_values("curated_id").head(16).copy()
    transform_serialized = json.loads(transform_path.read_text(encoding="utf-8"))
    transform = FitScopedDescriptorTransform.from_dict(transform_serialized)
    if transform.transform_sha256 != json.loads((attempt / "training_trace.json").read_text(encoding="utf-8"))["descriptor_transform_sha256"]:
        raise RuntimeError("Saved transform and trace are not bound")
    raw_descriptors = pd.read_csv(root / "artifacts/d123_inputs_v1/raw_descriptors.csv")
    payload = D123FitPayload(
        variant="D3",
        training_ids=tuple(),
        weights=np.empty((0, 1), dtype=np.float32),
        global_features=np.empty((0, 27), dtype=np.float32),
        descriptor_transform=transform,
        payload_sha256="checkpoint-only-probe",
    )
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="d3-checkpoint-only-probe-", dir=str(root / "runs")) as model_dir:
        model = create_pinned_dmpnn_model("D3", model_dir=Path(model_dir), batch_size=32)
        model.restore(str(checkpoint))
        model.fit = forbidden
        model.fit_generator = forbidden
        model.save_checkpoint = forbidden
        optimizer = getattr(model, "_pytorch_optimizer", None)
        if optimizer is None:
            raise RuntimeError("Restored model has no optimizer state to guard")
        optimizer.step = forbidden
        module = model.model
        module.eval()
        before = state_sha256(module)
        dataset = build_ordered_role_dataset(
            frame=validation,
            fit_payload=payload,
            raw_descriptors=raw_descriptors,
            featurizer=dc.feat.DMPNNFeaturizer(),
            target_column="normalized_pampa",
        )
        with torch.no_grad():
            prediction = np.asarray(model.predict(dataset), dtype=np.float64).reshape(-1)
        after = state_sha256(module)
        if before != after:
            raise RuntimeError("Model parameters changed during checkpoint-only inference")
    if len(prediction) != len(validation) or not np.isfinite(prediction).all():
        raise RuntimeError("Probe prediction coverage is invalid")
    observed = validation["normalized_pampa"].to_numpy(np.float64)
    result = {
        "schema_version": "scaffoldseal-d3-checkpoint-only-probe-v1",
        "status": "PASS",
        "frozen_plan_sha256": plan["plan_sha256"],
        "stage_identity": stage["scientific_identity_sha256"],
        "outer_fold": 1,
        "inner_basket": 1,
        "accepted_attempt": attempt.name,
        "checkpoint_sha256": sha256(checkpoint),
        "descriptor_transform_sha256": transform.transform_sha256,
        "n_probe_rows": len(validation),
        "probe_ids_sha256": hashlib.sha256("".join(f"{value}\n" for value in validation["curated_id"]).encode()).hexdigest(),
        "prediction_min_normalized": float(prediction.min()),
        "prediction_max_normalized": float(prediction.max()),
        "probe_mae_log10_papp": float(np.abs(prediction - observed).mean() * 2.0),
        "model_state_sha256_before": before,
        "model_state_sha256_after": after,
        "zero_training_guards": ["fit", "fit_generator", "save_checkpoint", "optimizer.step"],
        "runtime_seconds": time.perf_counter() - started,
        "training_performed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
