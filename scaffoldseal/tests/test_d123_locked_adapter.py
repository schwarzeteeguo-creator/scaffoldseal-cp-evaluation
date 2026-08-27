from pathlib import Path
import sys

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from d123_locked_adapter import D123LockedAdapter


def test_adapter_variant_width_and_config_hash_are_distinct():
    groups = pd.DataFrame(
        {
            "curated_id": ["a"],
            "source": ["s"],
            "analogue_component_id": ["c"],
        }
    )
    raw = pd.DataFrame({"curated_id": ["a"]})
    adapters = [
        D123LockedAdapter(
            baseline_root=Path("."),
            source_hashes={"x": "0" * 64},
            variant=variant,
            group_metadata=groups,
            raw_descriptors=raw,
        )
        for variant in ("D1", "D2", "D3")
    ]
    assert len({adapter.model_config_sha256 for adapter in adapters}) == 3
