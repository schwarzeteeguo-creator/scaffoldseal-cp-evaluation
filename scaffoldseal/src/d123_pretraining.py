"""Seed-specific Delaney pretraining for the descriptor DMPNN variants.

The pretraining task must not learn, fit, or otherwise inspect PAMPA
descriptors.  Every Delaney graph receives the same immutable 27-dimensional
zero vector so the checkpoint architecture matches D2/D3 while the scientific
pretraining task remains the accepted Delaney task.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import time

import numpy as np

from d123_dmpnn_integration import (
    N_GLOBAL_FEATURES,
    attach_global_features,
    create_pinned_dmpnn_model,
)
from r1c0_dmpnn_pilot import (
    MODEL_CONFIG,
    canonical_array_hash,
    dataset_ids_hash,
    file_sha256,
    global_step,
    gpu_snapshot,
    reset_gpu_peak,
    reset_rng,
)


DELANEY_TARGET = "measured log solubility in mols per litre"


def zero_descriptor_matrix(n_rows: int) -> np.ndarray:
    if isinstance(n_rows, bool) or not isinstance(n_rows, int) or n_rows < 1:
        raise ValueError("Delaney zero-descriptor row count must be a positive integer")
    values = np.zeros((n_rows, N_GLOBAL_FEATURES), dtype=np.float32)
    values.setflags(write=False)
    return values


def attach_zero_descriptors(graphs: np.ndarray) -> np.ndarray:
    values = zero_descriptor_matrix(len(graphs))
    attached = attach_global_features(np.asarray(graphs, dtype=object), values)
    observed = np.stack(
        [np.asarray(graph.global_features, dtype=np.float32) for graph in attached]
    )
    if observed.shape != values.shape or np.any(observed):
        raise RuntimeError("Delaney descriptor pretraining features are not exact 27D zeros")
    return attached


class ZeroDescriptorFeaturizer:
    """Wrap the pinned DMPNN featurizer without exposing PAMPA descriptors."""

    def __init__(self, base_featurizer) -> None:
        self.base_featurizer = base_featurizer

    def featurize(self, datapoints, **kwargs):
        graphs = np.asarray(
            self.base_featurizer.featurize(datapoints, **kwargs), dtype=object
        )
        if len(graphs) != len(datapoints):
            raise RuntimeError("Pinned Delaney featurizer changed the input row count")
        return attach_zero_descriptors(graphs)

    def __call__(self, datapoints, **kwargs):
        """Match the callable featurizer protocol required by CSVLoader."""
        return self.featurize(datapoints, **kwargs)


def create_descriptor_pretraining_model(run_dir: Path):
    import deepchem as dc

    featurizer = ZeroDescriptorFeaturizer(dc.feat.DMPNNFeaturizer())
    model = create_pinned_dmpnn_model(
        "D2", model_dir=run_dir, batch_size=64
    )
    if float(model.optimizer.learning_rate) != 0.001 or hasattr(
        model.optimizer, "scheduler"
    ):
        raise RuntimeError("Pinned descriptor-pretraining optimizer identity changed")
    return featurizer, model


def train_descriptor_delaney_once(
    baseline_root: Path,
    run_dir: Path,
    repeat_name: str,
    *,
    seed_index: int,
    effective_rng_seed: int | None = None,
) -> dict[str, object]:
    """Run one authorized D2/D3-compatible Delaney pretraining identity.

    This function is deliberately not called by construction or pre-fit tests.
    The execution runner may call it only after the frozen acceptance gate is
    true and the user has separately re-authorized GPU training.
    """

    import deepchem as dc

    if run_dir.exists():
        raise RuntimeError(f"Pretraining namespace already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    if seed_index not in range(5):
        raise ValueError("Descriptor pretraining seed index is outside 0..4")
    effective_seed = seed_index if effective_rng_seed is None else effective_rng_seed
    reset_rng(effective_seed)
    reset_gpu_peak()
    started = time.perf_counter()
    featurizer, model = create_descriptor_pretraining_model(run_dir)
    raw_path = baseline_root / "CSV" / "PreTrainData" / "delaney-processed.csv"
    if not raw_path.is_file():
        raise RuntimeError("Local Delaney CSV is missing")
    loader = dc.data.CSVLoader(
        tasks=[DELANEY_TARGET],
        feature_field="smiles",
        featurizer=featurizer,
    )
    raw = loader.create_dataset(str(raw_path), data_dir=str(run_dir / "raw_dataset"))
    train, valid, test = dc.splits.RandomSplitter().train_valid_test_split(
        raw,
        seed=None,
        train_dir=str(run_dir / "train_untransformed"),
        valid_dir=str(run_dir / "valid_untransformed"),
        test_dir=str(run_dir / "test_untransformed"),
    )
    transformer = dc.trans.NormalizationTransformer(transform_y=True, dataset=train)
    train = transformer.transform(train, out_dir=str(run_dir / "train"))
    valid = transformer.transform(valid, out_dir=str(run_dir / "valid"))
    test = transformer.transform(test, out_dir=str(run_dir / "test"))
    for role, dataset in (("train", train), ("validation", valid), ("test", test)):
        observed = np.stack(
            [np.asarray(graph.global_features, dtype=np.float32) for graph in dataset.X]
        )
        if observed.shape != (len(dataset), N_GLOBAL_FEATURES) or np.any(observed):
            raise RuntimeError(f"{role} Delaney dataset lost exact 27D zero features")

    metric = dc.metrics.Metric(dc.metrics.score_function.rms_score, name="rms_score")
    history: list[float] = []
    best = float("inf")
    best_epoch = 0
    non_improving = 0
    start_step = global_step(model)
    for epoch in range(1, MODEL_CONFIG["maximum_epochs"] + 1):
        model.fit(train, nb_epoch=1, checkpoint_interval=0, restore=False)
        score = float(model.evaluate(valid, [metric], [transformer])["rms_score"])
        history.append(score)
        if score < best:
            best, best_epoch, non_improving = score, epoch, 0
            model.save_checkpoint(max_checkpoints_to_keep=1, model_dir=str(run_dir))
        else:
            non_improving += 1
        if non_improving > MODEL_CONFIG["patience"]:
            break
    end_step = global_step(model)
    checkpoint = run_dir / "checkpoint1.pt"
    if not checkpoint.is_file() or best_epoch < 1:
        raise RuntimeError("Descriptor Delaney pretraining produced no best checkpoint")
    model.restore(str(checkpoint))
    probe = np.asarray(
        model.predict(test, transformers=[transformer]), dtype=float
    ).reshape(-1)
    zero_sha256 = hashlib.sha256(
        np.zeros((1, N_GLOBAL_FEATURES), dtype=np.float32).tobytes()
    ).hexdigest()
    return {
        "repeat": repeat_name,
        "seed_index": seed_index,
        "effective_rng_seed": effective_seed,
        "raw_csv_sha256": file_sha256(raw_path),
        "target": DELANEY_TARGET,
        "global_features_size": N_GLOBAL_FEATURES,
        "global_features_policy": "all_zero_no_pampa_descriptor_access",
        "single_row_zero_sha256": zero_sha256,
        "split_ids": {
            "train": dataset_ids_hash(train),
            "validation": dataset_ids_hash(valid),
            "test": dataset_ids_hash(test),
        },
        "split_counts": {
            "train": len(train),
            "validation": len(valid),
            "test": len(test),
        },
        "history": history,
        "history_sha256": canonical_array_hash(history),
        "best_epoch": best_epoch,
        "stopping_epoch": len(history),
        "best_validation_rmse": best,
        "optimizer_updates": end_step - start_step,
        "checkpoint_sha256": file_sha256(checkpoint),
        "test_probe_sha256": canonical_array_hash(probe),
        "runtime_seconds": time.perf_counter() - started,
        "gpu": gpu_snapshot(),
    }
