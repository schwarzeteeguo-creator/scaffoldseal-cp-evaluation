from __future__ import annotations

import numpy as np
import pytest

from d123_features import N_GLOBAL_FEATURES
from d123_pretraining import (
    DELANEY_TARGET,
    ZeroDescriptorFeaturizer,
    attach_zero_descriptors,
    zero_descriptor_matrix,
)


class Graph:
    def __init__(self, marker):
        self.marker = marker
        self.global_features = np.empty(0, dtype=np.float32)


class Featurizer:
    def featurize(self, datapoints, **kwargs):
        return np.asarray([Graph(value) for value in datapoints], dtype=object)


def test_zero_descriptor_matrix_is_exact_immutable_27d():
    values = zero_descriptor_matrix(3)
    assert values.shape == (3, N_GLOBAL_FEATURES) == (3, 27)
    assert values.dtype == np.float32
    assert not values.flags.writeable
    assert np.count_nonzero(values) == 0


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_zero_descriptor_matrix_rejects_invalid_counts(bad):
    with pytest.raises(ValueError):
        zero_descriptor_matrix(bad)


def test_zero_attachment_is_deep_copied_and_never_uses_pampa_values():
    original = np.asarray([Graph("A"), Graph("B")], dtype=object)
    attached = attach_zero_descriptors(original)
    assert all(len(graph.global_features) == 0 for graph in original)
    assert all(graph is not source for graph, source in zip(attached, original))
    observed = np.stack([graph.global_features for graph in attached])
    assert observed.shape == (2, 27)
    assert np.count_nonzero(observed) == 0


def test_wrapped_featurizer_preserves_order_and_attaches_only_zeros():
    wrapped = ZeroDescriptorFeaturizer(Featurizer())
    result = wrapped.featurize(["C", "N", "O"])
    assert [graph.marker for graph in result] == ["C", "N", "O"]
    assert np.stack([graph.global_features for graph in result]).shape == (3, 27)
    assert not np.stack([graph.global_features for graph in result]).any()


def test_wrapped_featurizer_is_callable_for_csvloader():
    wrapped = ZeroDescriptorFeaturizer(Featurizer())
    result = wrapped(["C", "N", "O"])
    assert [graph.marker for graph in result] == ["C", "N", "O"]
    assert np.stack([graph.global_features for graph in result]).shape == (3, 27)
    assert not np.stack([graph.global_features for graph in result]).any()


def test_delaney_target_is_the_accepted_baseline_task():
    assert DELANEY_TARGET == "measured log solubility in mols per litre"
