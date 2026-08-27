"""Fail-closed D1/D2/D3 integration at the pinned DeepChem DMPNN boundary."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from d123_features import (
    DESCRIPTOR_COLUMNS,
    N_GLOBAL_FEATURES,
    FitScopedDescriptorTransform,
    group_balanced_weights,
)


Variant = Literal["D1", "D2", "D3"]


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class D123FitPayload:
    variant: Variant
    training_ids: tuple[str, ...]
    weights: np.ndarray
    global_features: np.ndarray
    descriptor_transform: FitScopedDescriptorTransform | None
    payload_sha256: str


def build_fit_payload(
    variant: Variant,
    fit_frame: pd.DataFrame,
    group_metadata: pd.DataFrame,
    raw_descriptors: pd.DataFrame,
) -> D123FitPayload:
    if variant not in ("D1", "D2", "D3"):
        raise ValueError("Unknown frozen D123 variant")
    if "curated_id" not in fit_frame or fit_frame["curated_id"].duplicated().any():
        raise ValueError("Fit frame must have unique curated_id values")
    ids = tuple(fit_frame["curated_id"].astype(str))
    if not ids:
        raise ValueError("Fit frame cannot be empty")

    groups = group_metadata.set_index("curated_id", verify_integrity=True)
    if not set(ids).issubset(groups.index):
        raise ValueError("Fit IDs are not fully covered by group metadata")
    if variant in ("D1", "D3"):
        weighted = groups.loc[list(ids)].reset_index()
        series = group_balanced_weights(weighted)
        weights = series.loc[list(ids)].to_numpy(np.float32).reshape(-1, 1)
    else:
        weights = np.ones((len(ids), 1), dtype=np.float32)

    transform = None
    if variant in ("D2", "D3"):
        transform = FitScopedDescriptorTransform.fit(raw_descriptors, ids)
        global_features = transform.transform(raw_descriptors, ids)
    else:
        global_features = np.empty((len(ids), 0), dtype=np.float32)

    payload = {
        "variant": variant,
        "training_ids": list(ids),
        "weights_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
        "global_features_sha256": hashlib.sha256(global_features.tobytes()).hexdigest(),
        "descriptor_transform_sha256": (
            None if transform is None else transform.transform_sha256
        ),
        "descriptor_columns": (
            [] if transform is None else list(DESCRIPTOR_COLUMNS)
        ),
    }
    return D123FitPayload(
        variant=variant,
        training_ids=ids,
        weights=weights,
        global_features=global_features,
        descriptor_transform=transform,
        payload_sha256=canonical_sha256(payload),
    )


def attach_global_features(graphs: np.ndarray, values: np.ndarray) -> np.ndarray:
    if len(graphs) != len(values):
        raise ValueError("Graph/global-feature row count mismatch")
    if values.ndim != 2 or values.shape[1] not in (0, N_GLOBAL_FEATURES):
        raise ValueError("Global-feature width is outside the frozen D123 contract")
    copied = np.asarray(
        [copy.deepcopy(graph) for graph in np.asarray(graphs, dtype=object)],
        dtype=object,
    )
    for graph, vector in zip(copied, values):
        graph.global_features = np.asarray(vector, dtype=np.float32).copy()
    return copied


def make_deepchem_dataset(
    *,
    graphs: np.ndarray,
    graph_ids: tuple[str, ...],
    labels: np.ndarray,
    label_ids: tuple[str, ...],
    ids: tuple[str, ...],
    payload: D123FitPayload,
):
    import deepchem as dc

    if ids != payload.training_ids:
        raise ValueError("Dataset IDs differ from the complete fit-scoped payload")
    if graph_ids != payload.training_ids:
        raise ValueError("Graph IDs differ from the complete fit-scoped payload")
    if label_ids != payload.training_ids:
        raise ValueError("Label IDs differ from the complete fit-scoped payload")
    if len(graph_ids) != len(graphs):
        raise ValueError("Graph ID count mismatch")
    y = np.asarray(labels, dtype=np.float32).reshape(-1, 1)
    if len(y) != len(ids):
        raise ValueError("Dataset label count mismatch")
    x = attach_global_features(graphs, payload.global_features)
    dataset = dc.data.NumpyDataset(X=x, y=y, w=payload.weights, ids=np.asarray(ids))
    observed_ids = tuple(map(str, dataset.ids))
    if observed_ids != ids:
        raise RuntimeError("DeepChem changed the exact fit ID order")
    if not np.array_equal(np.asarray(dataset.w), payload.weights):
        raise RuntimeError("D1/D3 weights did not reach DeepChem dataset.w exactly")
    observed_widths = tuple(len(np.asarray(graph.global_features)) for graph in dataset.X)
    expected_width = N_GLOBAL_FEATURES if payload.variant in ("D2", "D3") else 0
    if observed_widths != (expected_width,) * len(ids):
        raise RuntimeError("D2/D3 global features did not reach graph readout inputs")
    return dataset


def build_ordered_fit_dataset(
    *,
    variant: Variant,
    fit_frame: pd.DataFrame,
    group_metadata: pd.DataFrame,
    raw_descriptors: pd.DataFrame,
    target_column: str,
    featurizer,
):
    """Atomically derive graphs, labels, IDs, weights and descriptors."""

    required = {"curated_id", "SMILES", target_column}
    if not required.issubset(fit_frame.columns):
        raise ValueError(f"Fit frame lacks required columns: {sorted(required - set(fit_frame))}")
    if fit_frame["curated_id"].duplicated().any() or len(fit_frame) == 0:
        raise ValueError("Ordered fit materialization requires unique nonempty IDs")
    ids = tuple(fit_frame["curated_id"].astype(str))
    labels = pd.to_numeric(fit_frame[target_column], errors="coerce").to_numpy(
        np.float32
    )
    if not np.isfinite(labels).all():
        raise ValueError("Fit target contains missing or non-finite values")
    smiles = tuple(fit_frame["SMILES"].astype(str))
    graphs = np.asarray(featurizer.featurize(list(smiles)), dtype=object)
    if len(graphs) != len(ids):
        raise RuntimeError("Pinned featurizer changed the complete fit row count")
    payload = build_fit_payload(variant, fit_frame, group_metadata, raw_descriptors)
    dataset = make_deepchem_dataset(
        graphs=graphs,
        graph_ids=ids,
        labels=labels,
        label_ids=ids,
        ids=ids,
        payload=payload,
    )
    return dataset, payload


def build_ordered_role_dataset(
    *,
    frame: pd.DataFrame,
    fit_payload: D123FitPayload,
    raw_descriptors: pd.DataFrame,
    featurizer,
    target_column: str | None,
):
    """Materialize validation/test data with the associated fit transform only."""

    required = {"curated_id", "SMILES"}
    if target_column is not None:
        required.add(target_column)
    if not required.issubset(frame.columns):
        raise ValueError(f"Role frame lacks required columns: {sorted(required - set(frame))}")
    if frame["curated_id"].duplicated().any():
        raise ValueError("Role frame IDs must be unique")
    ids = tuple(frame["curated_id"].astype(str))
    graphs = np.asarray(
        featurizer.featurize(frame["SMILES"].astype(str).tolist()), dtype=object
    )
    if len(graphs) != len(ids):
        raise RuntimeError("Pinned featurizer changed the role row count")
    if fit_payload.variant in ("D2", "D3"):
        transform = fit_payload.descriptor_transform
        if transform is None:
            raise RuntimeError("Descriptor variant lacks its fit-scoped transform")
        global_features = transform.transform(raw_descriptors, ids)
    else:
        global_features = np.empty((len(ids), 0), dtype=np.float32)
    x = attach_global_features(graphs, global_features)
    import deepchem as dc

    if target_column is None:
        dataset = dc.data.NumpyDataset(X=x, ids=np.asarray(ids))
    else:
        labels = pd.to_numeric(frame[target_column], errors="coerce").to_numpy(
            np.float32
        )
        if not np.isfinite(labels).all():
            raise ValueError("Validation target contains missing or non-finite values")
        dataset = dc.data.NumpyDataset(
            X=x,
            y=labels.reshape(-1, 1),
            w=np.ones((len(ids), 1), dtype=np.float32),
            ids=np.asarray(ids),
        )
    if tuple(map(str, dataset.ids)) != ids:
        raise RuntimeError("DeepChem changed the exact role ID order")
    return dataset


def create_pinned_dmpnn_model(
    variant: Variant, *, model_dir: Path, batch_size: int
):
    import deepchem as dc

    if variant not in ("D1", "D2", "D3"):
        raise ValueError("Unknown frozen D123 variant")
    width = N_GLOBAL_FEATURES if variant in ("D2", "D3") else 0
    return dc.models.DMPNNModel(
        n_tasks=1,
        mode="regression",
        model_dir=str(model_dir),
        batch_size=batch_size,
        global_features_size=width,
    )


def descriptor_definition_payload() -> dict[str, object]:
    return {
        "schema_version": "scaffoldseal-d123-descriptor-definition-v1",
        "output_columns": list(DESCRIPTOR_COLUMNS),
        "global_features_size": N_GLOBAL_FEATURES,
        "attachment": "GraphData.global_features",
        "readout_contract": "DMPNN enc_hidden concatenated with global_features",
    }
