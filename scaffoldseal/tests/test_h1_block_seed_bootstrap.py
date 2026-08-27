import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).parents[1] / "src/h1_block_seed_bootstrap.py"
SPEC = importlib.util.spec_from_file_location("h1_bootstrap", MODULE_PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def toy_table() -> pd.DataFrame:
    rows = []
    values = {
        "lobo": {0: {"A": [2.0, 4.0], "B": [6.0]}, 1: {"A": [4.0, 6.0], "B": [8.0]}},
        "random": {0: {"A": [1.0, 2.0], "B": [3.0]}, 1: {"A": [2.0, 3.0], "B": [4.0]}},
    }
    for arm, seeds in values.items():
        for seed, blocks in seeds.items():
            for block, errors in blocks.items():
                for index, error in enumerate(errors):
                    rows.append(
                        {
                            "arm": arm,
                            "seed": seed,
                            "sealed_block_id": block,
                            "source": f"{block}-{index}",
                            "source_mae": error,
                        }
                    )
    return pd.DataFrame(rows)


def test_point_estimate_matches_arm_seed_then_seed_mean():
    point = MOD.point_estimates(toy_table())
    assert np.isclose(point["lobo_source_macro_mae"], 5.0)
    assert np.isclose(point["random_source_macro_mae"], 2.5)
    assert np.isclose(point["gap_lobo_minus_random"], 2.5)


def test_bootstrap_is_deterministic():
    table = toy_table()
    first = MOD.bootstrap(table, ["A", "B"], [0, 1], 123, n_replicates=50)
    second = MOD.bootstrap(table, ["A", "B"], [0, 1], 123, n_replicates=50)
    pd.testing.assert_frame_equal(first, second)


def test_paired_resampling_preserves_positive_gap():
    result = MOD.bootstrap(toy_table(), ["A", "B"], [0, 1], 7, n_replicates=100)
    assert (result["gap_lobo_minus_random"] > 0).all()


def test_canonical_hash_is_order_invariant_for_dicts():
    assert MOD.canonical_sha({"a": 1, "b": 2}) == MOD.canonical_sha({"b": 2, "a": 1})


def test_atomic_publish_removes_failed_attempt_and_never_exposes_canonical(tmp_path=None):
    import tempfile

    parent = Path(tempfile.mkdtemp()) if tmp_path is None else tmp_path
    target = parent / "final"

    def broken(temp):
        (temp / "bootstrap_replicates.csv").write_text("partial\n")
        raise RuntimeError("injected")

    try:
        MOD.atomic_publish(target, broken)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Injected failure did not propagate")
    assert not target.exists()
    assert not list(parent.glob(".final.nonfinal-attempt-*"))


def test_atomic_publish_refuses_existing_target(tmp_path=None):
    import tempfile

    parent = Path(tempfile.mkdtemp()) if tmp_path is None else tmp_path
    target = parent / "final"
    target.mkdir()
    try:
        MOD.atomic_publish(target, lambda _: None)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Existing canonical target was not rejected")


def test_atomic_publish_verifies_manifest_and_commits_complete_bundle(tmp_path=None):
    import tempfile

    parent = Path(tempfile.mkdtemp()) if tmp_path is None else tmp_path
    target = parent / "final"

    def build(temp):
        (temp / "bootstrap_replicates.csv").write_text("replicate,gap\n0,1.0\n")
        (temp / "bootstrap_summary.json").write_text("{}\n")
        files = [
            MOD.file_record(temp / name, temp)
            for name in ("bootstrap_replicates.csv", "bootstrap_summary.json")
        ]
        manifest = {
            "schema_version": "test",
            "files": files,
            "files_sha256": MOD.canonical_sha(files),
        }
        (temp / "artifact_manifest.json").write_text(json.dumps(manifest))

    MOD.atomic_publish(target, build)
    assert target.is_dir()
    manifest = json.loads((target / "artifact_manifest.json").read_text())
    for record in manifest["files"]:
        path = target / record["relative_path"]
        assert path.stat().st_size == record["size_bytes"]
        assert MOD.sha256(path) == record["sha256"]
