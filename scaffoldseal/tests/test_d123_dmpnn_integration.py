from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import d123_dmpnn_integration as MOD


def _frames():
    ids = ("a", "b", "c", "d")
    fit = pd.DataFrame({"curated_id": ids})
    groups = pd.DataFrame(
        {
            "curated_id": ids,
            "source": ["s1", "s1", "s2", "s2"],
            "analogue_component_id": ["x", "y", "z", "z"],
        }
    )
    raw = pd.DataFrame({"curated_id": ids})
    for index, column in enumerate(MOD.DESCRIPTOR_COLUMNS):
        raw[column] = (
            [1.0, 0.0, 0.0, 1.0]
            if column.startswith("topology__")
            else np.arange(4, dtype=float) + index
        )
    return fit, groups, raw


def test_d1_weights_reach_dataset_w_and_are_fixed(monkeypatch):
    fit, groups, raw = _frames()
    payload = MOD.build_fit_payload("D1", fit, groups, raw)

    class Graph:
        global_features = np.empty(0)

    class Dataset:
        def __init__(self, X, y, w, ids):
            self.X, self.y, self.w, self.ids = X, y, w, ids

    fake = type("DC", (), {"data": type("Data", (), {"NumpyDataset": Dataset})})
    monkeypatch.setitem(sys.modules, "deepchem", fake)
    dataset = MOD.make_deepchem_dataset(
        graphs=np.asarray([Graph() for _ in payload.training_ids], dtype=object),
        graph_ids=payload.training_ids,
        labels=np.arange(4),
        label_ids=payload.training_ids,
        ids=payload.training_ids,
        payload=payload,
    )
    assert np.array_equal(dataset.w, payload.weights)
    before = dataset.w.copy()
    for minibatch in ([0, 2], [1, 3]):
        assert np.array_equal(dataset.w, before)
        assert dataset.w[minibatch].shape == (2, 1)


def test_d2_global_features_are_attached_in_frozen_order(monkeypatch):
    fit, groups, raw = _frames()
    payload = MOD.build_fit_payload("D2", fit, groups, raw)

    class Graph:
        global_features = np.empty(0)

    class Dataset:
        def __init__(self, X, y, w, ids):
            self.X, self.y, self.w, self.ids = X, y, w, ids

    fake = type("DC", (), {"data": type("Data", (), {"NumpyDataset": Dataset})})
    monkeypatch.setitem(sys.modules, "deepchem", fake)
    original = np.asarray([Graph() for _ in payload.training_ids], dtype=object)
    dataset = MOD.make_deepchem_dataset(
        graphs=original,
        graph_ids=payload.training_ids,
        labels=np.arange(4),
        label_ids=payload.training_ids,
        ids=payload.training_ids,
        payload=payload,
    )
    observed = np.stack([graph.global_features for graph in dataset.X])
    assert observed.shape == (4, 27)
    assert np.array_equal(observed, payload.global_features)
    assert all(len(graph.global_features) == 0 for graph in original)


def test_heldout_mutation_cannot_change_fit_payload():
    fit, groups, raw = _frames()
    inner_fit = fit.iloc[:3].copy()
    first = MOD.build_fit_payload("D3", inner_fit, groups, raw)
    mutated = raw.copy()
    mutated.loc[mutated["curated_id"].eq("d"), MOD.DESCRIPTOR_COLUMNS] = 999999.0
    second = MOD.build_fit_payload("D3", inner_fit, groups, mutated)
    assert first.payload_sha256 == second.payload_sha256
    assert first.descriptor_transform.transform_sha256 == second.descriptor_transform.transform_sha256
    assert np.array_equal(first.global_features, second.global_features)


def test_model_width_is_variant_locked(monkeypatch, tmp_path):
    observed = []

    def model(**kwargs):
        observed.append(kwargs)
        return kwargs

    fake = type("DC", (), {"models": type("Models", (), {"DMPNNModel": model})})
    monkeypatch.setitem(sys.modules, "deepchem", fake)
    MOD.create_pinned_dmpnn_model("D1", model_dir=tmp_path / "d1", batch_size=16)
    MOD.create_pinned_dmpnn_model("D2", model_dir=tmp_path / "d2", batch_size=16)
    assert observed[0]["global_features_size"] == 0
    assert observed[1]["global_features_size"] == 27


def test_dataset_rejects_graph_or_label_reordering():
    fit, groups, raw = _frames()
    payload = MOD.build_fit_payload("D1", fit, groups, raw)
    graphs = np.asarray([object() for _ in payload.training_ids], dtype=object)
    with pytest.raises(ValueError, match="Graph IDs differ"):
        MOD.make_deepchem_dataset(
            graphs=graphs[::-1],
            graph_ids=tuple(reversed(payload.training_ids)),
            labels=np.arange(4),
            label_ids=payload.training_ids,
            ids=payload.training_ids,
            payload=payload,
        )


def test_atomic_fit_materialization_uses_one_ordered_frame(monkeypatch):
    fit, groups, raw = _frames()
    fit["SMILES"] = ["CC", "CCC", "CCCC", "CCCCC"]
    fit["target"] = [0.1, 0.2, 0.3, 0.4]

    class Graph:
        global_features = np.empty(0)

    class Featurizer:
        def featurize(self, smiles):
            assert smiles == fit["SMILES"].tolist()
            return [Graph() for _ in smiles]

    class Dataset:
        def __init__(self, X, y, w, ids):
            self.X, self.y, self.w, self.ids = X, y, w, ids

    fake = type("DC", (), {"data": type("Data", (), {"NumpyDataset": Dataset})})
    monkeypatch.setitem(sys.modules, "deepchem", fake)
    dataset, payload = MOD.build_ordered_fit_dataset(
        variant="D3",
        fit_frame=fit,
        group_metadata=groups,
        raw_descriptors=raw,
        target_column="target",
        featurizer=Featurizer(),
    )
    assert tuple(dataset.ids) == payload.training_ids
    assert np.array_equal(dataset.y.reshape(-1), fit["target"].to_numpy(np.float32))
    assert np.array_equal(dataset.w, payload.weights)


def test_validation_reuses_training_transform_without_refit(monkeypatch):
    fit, groups, raw = _frames()
    fit["SMILES"] = ["CC", "CCC", "CCCC", "CCCCC"]
    fit["target"] = [0.1, 0.2, 0.3, 0.4]

    class Graph:
        global_features = np.empty(0)

    class Featurizer:
        def featurize(self, smiles):
            return [Graph() for _ in smiles]

    class Dataset:
        def __init__(self, X, y=None, w=None, ids=None):
            self.X, self.y, self.w, self.ids = X, y, w, ids

    fake = type("DC", (), {"data": type("Data", (), {"NumpyDataset": Dataset})})
    monkeypatch.setitem(sys.modules, "deepchem", fake)
    _, payload = MOD.build_ordered_fit_dataset(
        variant="D2",
        fit_frame=fit.iloc[:3],
        group_metadata=groups,
        raw_descriptors=raw,
        target_column="target",
        featurizer=Featurizer(),
    )
    before = payload.descriptor_transform.transform_sha256
    validation = MOD.build_ordered_role_dataset(
        frame=fit.iloc[3:],
        fit_payload=payload,
        raw_descriptors=raw,
        featurizer=Featurizer(),
        target_column="target",
    )
    assert payload.descriptor_transform.transform_sha256 == before
    assert len(validation.X[0].global_features) == 27
    assert np.array_equal(validation.w, np.ones((1, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="Label IDs differ"):
        MOD.make_deepchem_dataset(
            graphs=graphs,
            graph_ids=payload.training_ids,
            labels=np.arange(4)[::-1],
            label_ids=tuple(reversed(payload.training_ids)),
            ids=payload.training_ids,
            payload=payload,
        )
