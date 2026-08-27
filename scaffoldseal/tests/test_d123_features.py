import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).parents[1] / "src/d123_features.py"
SPEC = importlib.util.spec_from_file_location("d123_features", MODULE_PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def test_group_balanced_weights_equalize_sources_and_components():
    frame = pd.DataFrame(
        {
            "curated_id": list("abcdefg"),
            "source": ["A", "A", "A", "A", "B", "B", "B"],
            "analogue_component_id": ["x", "x", "x", "y", "z", "z", "z"],
        }
    )
    weights = MOD.group_balanced_weights(frame)
    joined = frame.assign(weight=weights.to_numpy())
    source = joined.groupby("source")["weight"].sum()
    component = joined.groupby(["source", "analogue_component_id"])["weight"].sum()
    assert np.isclose(weights.mean(), 1.0)
    assert np.isclose(source["A"], source["B"])
    assert np.isclose(component[("A", "x")], component[("A", "y")])


def test_fit_transform_roundtrip_and_hash_tamper_rejection():
    raw = descriptor_fixture()
    transform = MOD.FitScopedDescriptorTransform.fit(raw, ("id0", "id1", "id2"))
    serialized = transform.to_dict()
    restored = MOD.FitScopedDescriptorTransform.from_dict(serialized)
    assert restored == transform
    tampered = dict(serialized)
    tampered["means"] = dict(serialized["means"])
    first = next(iter(tampered["means"]))
    tampered["means"][first] += 1.0
    try:
        MOD.FitScopedDescriptorTransform.from_dict(tampered)
    except ValueError as error:
        assert "hash mismatch" in str(error)
    else:
        raise AssertionError("Tampered descriptor transform was accepted")


def descriptor_fixture():
    rows = []
    for index, value in enumerate([1.0, 2.0, np.nan, 4.0]):
        row = {"curated_id": f"id{index}"}
        for column in MOD.TOPOLOGY_COLUMNS:
            row[column] = float(column == MOD.TOPOLOGY_COLUMNS[0])
        for column in MOD.CONTINUOUS_DESCRIPTORS:
            row[column] = value
            row[f"{column}__missing"] = float(np.isnan(value))
        row["formal_charge"] = 7.0
        rows.append(row)
    return pd.DataFrame(rows, columns=["curated_id", *MOD.DESCRIPTOR_COLUMNS])


def test_descriptor_transform_uses_training_only_statistics():
    raw = descriptor_fixture()
    fitted = MOD.FitScopedDescriptorTransform.fit(raw, ("id0", "id1", "id2"))
    before = fitted.transform(raw, ("id3",))
    changed = raw.copy()
    changed.loc[changed["curated_id"] == "id3", "ring_size"] = 4000.0
    after = fitted.transform(changed, ("id3",))
    assert fitted.transform_sha256 == MOD.FitScopedDescriptorTransform.fit(
        raw, ("id0", "id1", "id2")
    ).transform_sha256
    assert not np.array_equal(before, after)


def test_zero_variance_slot_is_zero_for_unseen_heldout_value():
    raw = descriptor_fixture()
    fitted = MOD.FitScopedDescriptorTransform.fit(raw, ("id0", "id1", "id2"))
    assert "formal_charge" in fitted.inactive_columns
    changed = raw.copy()
    changed.loc[changed["curated_id"] == "id3", "formal_charge"] = -99.0
    output = fitted.transform(changed, ("id3",))
    index = list(MOD.DESCRIPTOR_COLUMNS).index("formal_charge")
    assert output[0, index] == 0.0


def test_descriptor_width_is_frozen():
    assert len(MOD.DESCRIPTOR_COLUMNS) == 27
    assert MOD.N_GLOBAL_FEATURES == 27
