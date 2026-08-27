"""Locked D1/D2/D3 adapter over the accepted DeepChem DMPNN training loop."""

from __future__ import annotations

import math
from pathlib import Path
import time

import numpy as np
import pandas as pd

from d123_dmpnn_integration import (
    Variant,
    build_ordered_fit_dataset,
    build_ordered_role_dataset,
    canonical_sha256,
    create_pinned_dmpnn_model,
)
from r1c0_dmpnn_pilot import (
    file_sha256,
    global_step,
    gpu_snapshot,
    reset_gpu_peak,
    reset_rng,
    write_json,
)
from split_safe import FramePredictionOutput, FreshTrainingState


class D123LockedAdapter:
    def __init__(
        self,
        *,
        baseline_root: Path,
        source_hashes: dict[str, str],
        variant: Variant,
        group_metadata: pd.DataFrame,
        raw_descriptors: pd.DataFrame,
    ) -> None:
        if variant not in ("D1", "D2", "D3"):
            raise ValueError("Unknown frozen D123 variant")
        self.baseline_root = str(baseline_root.resolve())
        self.source_hashes = tuple(sorted(source_hashes.items()))
        self.variant = variant
        self.group_metadata = group_metadata.copy(deep=True)
        self.raw_descriptors = raw_descriptors.copy(deep=True)
        self.transform_sha256 = canonical_sha256(
            {
                "variant": variant,
                "fit_transform": "training-ID-bound D123 payload",
                "role_transform": "reuse exact fit payload",
            }
        )
        self.fitted_transform_sha256 = self.transform_sha256
        self.model_config_sha256 = canonical_sha256(
            {
                "variant": variant,
                "model": "DeepChem DMPNNModel",
                "global_features_size": 27 if variant in ("D2", "D3") else 0,
                "sources": dict(self.source_hashes),
            }
        )

    def create_fresh_state(self, *, run_context):
        reset_rng(run_context.seed)
        model = create_pinned_dmpnn_model(
            self.variant,
            model_dir=Path(run_context.checkpoint_dir),
            batch_size=32,
        )
        if run_context.pretrained_checkpoint is None:
            raise RuntimeError("D123 requires an explicit seed-specific checkpoint")
        model.restore(run_context.pretrained_checkpoint)
        optimizer = getattr(model, "_pytorch_optimizer", None)
        if optimizer is None:
            raise RuntimeError("Pinned checkpoint did not restore a fresh torch optimizer")
        return FreshTrainingState(model=model, optimizer=optimizer, scheduler=None)

    def _fit_dataset(self, frame, target_column):
        import deepchem as dc

        return build_ordered_fit_dataset(
            variant=self.variant,
            fit_frame=frame,
            group_metadata=self.group_metadata,
            raw_descriptors=self.raw_descriptors,
            target_column=target_column,
            featurizer=dc.feat.DMPNNFeaturizer(),
        )

    def _role_dataset(self, frame, fit_payload, target_column):
        import deepchem as dc

        return build_ordered_role_dataset(
            frame=frame,
            fit_payload=fit_payload,
            raw_descriptors=self.raw_descriptors,
            featurizer=dc.feat.DMPNNFeaturizer(),
            target_column=target_column,
        )

    def fit_inner(
        self,
        *,
        state,
        train_frame,
        validation_frame,
        target_column,
        feature_columns,
        sample_weight,
        maximum_epochs,
        options,
        run_context,
        recorder,
    ):
        if tuple(feature_columns) != ("SMILES",):
            raise RuntimeError("D123 accepts only frozen SMILES plus internal descriptors")
        if sample_weight is not None or maximum_epochs != 2000:
            raise RuntimeError("D123 inner configuration changed")
        reset_gpu_peak()
        started = time.perf_counter()
        train, payload = self._fit_dataset(train_frame, target_column)
        valid = self._role_dataset(validation_frame, payload, target_column)
        state.model._scaffoldseal_validation_dataset = valid
        state.model._scaffoldseal_validation_ids = tuple(map(str, valid.ids))
        state.model._scaffoldseal_fit_payload_sha256 = payload.payload_sha256
        if payload.descriptor_transform is not None:
            write_json(
                Path(run_context.checkpoint_dir) / "descriptor_transform.json",
                payload.descriptor_transform.to_dict(),
            )
        epoch_seconds: list[float] = []
        best = float("inf")
        best_epoch = 0
        non_improving = 0
        start_step = global_step(state.model)
        for epoch in range(1, maximum_epochs + 1):
            tick = time.perf_counter()
            state.model.fit(train, nb_epoch=1, checkpoint_interval=0, restore=False)
            event = recorder.evaluate_frame_epoch(epoch, self, state)
            epoch_seconds.append(time.perf_counter() - tick)
            if event.loss < best:
                best, best_epoch, non_improving = event.loss, epoch, 0
                state.model.save_checkpoint(
                    max_checkpoints_to_keep=1,
                    model_dir=run_context.checkpoint_dir,
                )
            else:
                non_improving += 1
            if non_improving > 200:
                break
        end_step = global_step(state.model)
        checkpoint = Path(run_context.checkpoint_dir) / "checkpoint1.pt"
        if not checkpoint.is_file():
            raise RuntimeError("D123 inner fit did not produce a best checkpoint")
        state.model.restore(str(checkpoint))
        del state.model._scaffoldseal_validation_dataset
        del state.model._scaffoldseal_validation_ids
        write_json(
            Path(run_context.checkpoint_dir) / "training_trace.json",
            {
                "role": "inner",
                "variant": self.variant,
                "outer_fold": run_context.outer_fold,
                "inner_basket": run_context.inner_basket,
                "seed": run_context.seed,
                "n_train": len(train),
                "n_validation": len(valid),
                "epochs_run": len(epoch_seconds),
                "best_epoch": best_epoch,
                "best_validation_mse": best,
                "fit_calls": len(epoch_seconds),
                "optimizer_updates": end_step - start_step,
                "epoch_seconds": epoch_seconds,
                "runtime_seconds": time.perf_counter() - started,
                "checkpoint_sha256": file_sha256(checkpoint),
                "fit_payload_sha256": payload.payload_sha256,
                "descriptor_transform_sha256": (
                    None
                    if payload.descriptor_transform is None
                    else payload.descriptor_transform.transform_sha256
                ),
                "gpu": gpu_snapshot(),
            },
        )

    def predict_validation(
        self, *, state, validation_frame, feature_columns, run_context
    ):
        expected = tuple(validation_frame["curated_id"].astype(str))
        if getattr(state.model, "_scaffoldseal_validation_ids", None) != expected:
            raise RuntimeError("D123 validation IDs differ from split-local dataset")
        return np.asarray(
            state.model.predict(state.model._scaffoldseal_validation_dataset),
            dtype=float,
        ).reshape(-1)

    def fit_outer(
        self,
        *,
        state,
        train_frame,
        target_column,
        feature_columns,
        sample_weight,
        fixed_epoch,
        options,
        run_context,
    ):
        if tuple(feature_columns) != ("SMILES",):
            raise RuntimeError("D123 accepts only frozen SMILES plus internal descriptors")
        if sample_weight is not None or fixed_epoch < 1:
            raise RuntimeError("D123 outer configuration changed")
        reset_gpu_peak()
        started = time.perf_counter()
        train, payload = self._fit_dataset(train_frame, target_column)
        state.model._scaffoldseal_outer_fit_payload = payload
        if payload.descriptor_transform is not None:
            write_json(
                Path(run_context.checkpoint_dir) / "descriptor_transform.json",
                payload.descriptor_transform.to_dict(),
            )
        epoch_seconds: list[float] = []
        start_step = global_step(state.model)
        for _ in range(fixed_epoch):
            tick = time.perf_counter()
            state.model.fit(train, nb_epoch=1, checkpoint_interval=0, restore=False)
            epoch_seconds.append(time.perf_counter() - tick)
        end_step = global_step(state.model)
        state.model.save_checkpoint(
            max_checkpoints_to_keep=1, model_dir=run_context.checkpoint_dir
        )
        checkpoint = Path(run_context.checkpoint_dir) / "checkpoint1.pt"
        write_json(
            Path(run_context.checkpoint_dir) / "training_trace.json",
            {
                "role": "outer_refit",
                "variant": self.variant,
                "outer_fold": run_context.outer_fold,
                "seed": run_context.seed,
                "n_train": len(train),
                "fixed_epoch": fixed_epoch,
                "fit_calls": fixed_epoch,
                "optimizer_updates": end_step - start_step,
                "epoch_seconds": epoch_seconds,
                "runtime_seconds": time.perf_counter() - started,
                "checkpoint_sha256": file_sha256(checkpoint),
                "fit_payload_sha256": payload.payload_sha256,
                "gpu": gpu_snapshot(),
            },
        )

    def predict_outer(
        self, *, state, outer_test_frame, feature_columns, run_context
    ):
        started = time.perf_counter()
        payload = getattr(state.model, "_scaffoldseal_outer_fit_payload", None)
        if payload is None:
            raise RuntimeError("D123 outer prediction lacks its sealed fit payload")
        dataset = self._role_dataset(outer_test_frame, payload, None)
        ids = tuple(map(str, dataset.ids))
        values = np.asarray(state.model.predict(dataset), dtype=float).reshape(-1)
        return FramePredictionOutput(
            pd.Series(
                values,
                index=pd.Index(ids, name="curated_id"),
                name="prediction",
            ),
            {
                "outer_predict_calls": 1,
                "prediction_runtime_seconds": time.perf_counter() - started,
                "fit_payload_sha256": payload.payload_sha256,
            },
        )
