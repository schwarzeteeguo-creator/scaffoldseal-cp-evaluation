"""Reconstruct frozen D3 inner-validation residuals without any fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import torch

from d3_checkpoint_only_probe import accepted_attempt, forbidden, sha256, state_sha256


def frame_sha256(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.runtime_root.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)

    plan = json.loads((root / "d123_plan_candidate.json").read_text(encoding="utf-8"))
    if plan["plan_sha256"] != "6b73097ee83c000a48ec4dfc646ddbcad3a561d3276c33e3661d3bf8b9d8f5c5":
        raise RuntimeError("Frozen plan binding changed")
    stages = sorted(
        (
            stage for stage in plan["stages"]
            if stage["kind"] == "d123_pampa_inner_fit" and stage["key"]["variant"] == "D3"
        ),
        key=lambda stage: (int(stage["key"]["outer_fold"]), int(stage["key"]["inner_basket"])),
    )
    if len(stages) != 72:
        raise RuntimeError(f"Expected 72 D3 inner stages, found {len(stages)}")

    sys.path.insert(0, str(root / "src"))
    from d123_dmpnn_integration import D123FitPayload, build_ordered_role_dataset, create_pinned_dmpnn_model
    from d123_features import FitScopedDescriptorTransform
    from d123_fold_data import canonical_id_hash, load_fold_frames
    import deepchem as dc

    raw_descriptors = pd.read_csv(root / "artifacts/d123_inputs_v1/raw_descriptors.csv")
    featurizer = dc.feat.DMPNNFeaturizer()
    rows: list[pd.DataFrame] = []
    receipts: list[dict[str, object]] = []
    started = time.perf_counter()

    for number, stage in enumerate(stages, start=1):
        fold = int(stage["key"]["outer_fold"])
        basket = int(stage["key"]["inner_basket"])
        identity = stage["scientific_identity_sha256"]
        attempt = accepted_attempt(root / "artifacts/d123_v1/d123_pampa_inner_fit" / identity)
        checkpoint = attempt / "checkpoint1.pt"
        transform_path = attempt / "descriptor_transform.json"
        trace = json.loads((attempt / "training_trace.json").read_text(encoding="utf-8"))
        transform = FitScopedDescriptorTransform.from_dict(json.loads(transform_path.read_text(encoding="utf-8")))
        if transform.transform_sha256 != trace["descriptor_transform_sha256"]:
            raise RuntimeError(f"Transform binding mismatch for {identity}")
        frames = load_fold_frames(plan, stage, root)
        validation = frames["validation"].sort_values("curated_id").reset_index(drop=True)
        if canonical_id_hash(validation["curated_id"]) != stage["key"]["validation_ids_sha256"]:
            raise RuntimeError(f"Validation identity mismatch for {identity}")
        payload = D123FitPayload(
            variant="D3", training_ids=tuple(), weights=np.empty((0, 1), dtype=np.float32),
            global_features=np.empty((0, 27), dtype=np.float32), descriptor_transform=transform,
            payload_sha256="checkpoint-only-reconstruction",
        )
        with tempfile.TemporaryDirectory(prefix="d3-checkpoint-only-reconstruct-", dir=str(root / "runs")) as model_dir:
            model = create_pinned_dmpnn_model("D3", model_dir=Path(model_dir), batch_size=32)
            model.restore(str(checkpoint))
            model.fit = forbidden
            model.fit_generator = forbidden
            model.save_checkpoint = forbidden
            optimizer = getattr(model, "_pytorch_optimizer", None)
            if optimizer is None:
                raise RuntimeError(f"Restored optimizer unavailable for {identity}")
            optimizer.step = forbidden
            module = model.model
            module.eval()
            before = state_sha256(module)
            dataset = build_ordered_role_dataset(
                frame=validation, fit_payload=payload, raw_descriptors=raw_descriptors,
                featurizer=featurizer, target_column="normalized_pampa",
            )
            with torch.no_grad():
                prediction = np.asarray(model.predict(dataset), dtype=np.float64).reshape(-1)
            after = state_sha256(module)
        if before != after or len(prediction) != len(validation) or not np.isfinite(prediction).all():
            raise RuntimeError(f"Inference invariant failed for {identity}")
        observed = validation["normalized_pampa"].to_numpy(np.float64)
        part = pd.DataFrame({
            "curated_id": validation["curated_id"].astype(str), "outer_fold": fold,
            "inner_basket": basket, "prediction_normalized": prediction,
            "observed_normalized": observed, "absolute_residual_log10_papp": np.abs(prediction - observed) * 2.0,
        })
        rows.append(part)
        receipts.append({
            "ordinal": number, "stage_identity": identity, "outer_fold": fold, "inner_basket": basket,
            "accepted_attempt": attempt.name, "checkpoint_sha256": sha256(checkpoint),
            "descriptor_transform_sha256": transform.transform_sha256, "n_validation": len(validation),
            "model_state_sha256_before": before, "model_state_sha256_after": after,
        })
        print(
            f"checkpoint_only_progress={number}/72 outer_fold={fold} inner_basket={basket} n={len(validation)}",
            flush=True,
        )

    residuals = pd.concat(rows, ignore_index=True).sort_values(["outer_fold", "inner_basket", "curated_id"]).reset_index(drop=True)
    if residuals.duplicated(["outer_fold", "curated_id"]).any():
        raise RuntimeError("An outer-training record received duplicate validation predictions")
    fold_counts = []
    for fold in range(1, 19):
        stage = next(item for item in stages if int(item["key"]["outer_fold"]) == fold)
        expected = load_fold_frames(plan, stage, root)["outer_train"]
        observed_ids = residuals.loc[residuals["outer_fold"].eq(fold), "curated_id"]
        if set(observed_ids) != set(expected["curated_id"]):
            raise RuntimeError(f"Fold {fold} does not exactly cover its outer-training records")
        fold_counts.append({"outer_fold": fold, "n_residuals": len(observed_ids)})

    output.mkdir(parents=True)
    residual_path = output / "inner_validation_residuals.csv"
    residuals.to_csv(residual_path, index=False, lineterminator="\n")
    summary = {
        "schema_version": "scaffoldseal-d3-checkpoint-only-reconstruction-v1", "status": "PASS",
        "frozen_plan_sha256": plan["plan_sha256"], "n_checkpoints": len(receipts),
        "n_residual_rows": len(residuals), "fold_counts": fold_counts,
        "residuals_sha256": sha256(residual_path), "canonical_frame_sha256": frame_sha256(residuals),
        "runtime_seconds": time.perf_counter() - started, "training_performed": False,
        "zero_training_guards": ["fit", "fit_generator", "save_checkpoint", "optimizer.step"],
        "receipts": receipts,
    }
    (output / "reconstruction_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
