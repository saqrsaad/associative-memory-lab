"""Unit tests for Hebbian weight accumulation."""

from __future__ import annotations

import numpy as np
import pytest

from associative_memory.algorithms.encoding import Encoding
from associative_memory.algorithms.hebbian import (
    accumulate_hebbian,
    hebbian_weight_matrix,
)


def test_hebbian_accumulation_matches_reference_case(reference_case):
    case = reference_case("hebbian_sum_two_pairs")
    pairs = list(zip(case["inputs"]["s"], case["inputs"]["t"]))

    result = accumulate_hebbian(pairs, encoding=case["encoding"], trace=True)

    assert result.pair_count == 2
    assert result.source_dimension == 3
    assert result.target_dimension == 2
    assert result.encoding is Encoding.BIPOLAR
    assert np.array_equal(
        result.weight_matrix,
        np.asarray(case["expected"]["W"], dtype=float),
    )
    assert result.contributions is not None
    assert len(result.contributions) == 2
    for contribution, expected in zip(
        result.contributions,
        case["expected"]["contributions"],
    ):
        assert np.array_equal(contribution, np.asarray(expected, dtype=float))


def test_hebbian_omits_contributions_when_trace_is_not_requested(reference_case):
    case = reference_case("hebbian_sum_two_pairs")
    pairs = list(zip(case["inputs"]["s"], case["inputs"]["t"]))

    result = accumulate_hebbian(pairs, encoding=case["encoding"])

    assert result.contributions is None
    assert np.array_equal(
        result.weight_matrix,
        np.asarray(case["expected"]["W"], dtype=float),
    )


def test_hebbian_weight_matrix_wrapper_returns_same_values(reference_case):
    case = reference_case("hebbian_sum_two_pairs")
    pairs = list(zip(case["inputs"]["s"], case["inputs"]["t"]))

    weights = hebbian_weight_matrix(pairs, encoding=case["encoding"])

    assert np.array_equal(weights, np.asarray(case["expected"]["W"], dtype=float))


def test_hebbian_accumulation_preserves_negative_contributions():
    weights = hebbian_weight_matrix(
        [([1, -1], [-1, 1])],
        encoding=Encoding.BIPOLAR,
    )

    assert np.array_equal(weights, np.array([[-1.0, 1.0], [1.0, -1.0]]))


def test_hebbian_rejects_empty_training_data():
    with pytest.raises(ValueError):
        accumulate_hebbian([])


@pytest.mark.parametrize(
    "pairs",
    [
        [([1, 2], [1]), ([1, 2, 3], [1])],
        [([1, 2], [1]), ([1, 2], [1, 2])],
    ],
)
def test_hebbian_rejects_inconsistent_dimensions(pairs):
    with pytest.raises(ValueError):
        accumulate_hebbian(pairs)


@pytest.mark.parametrize(
    "pairs",
    [
        [([1, 2],)],
        [([1, 2], [1], [3])],
    ],
)
def test_hebbian_rejects_malformed_pairs(pairs):
    with pytest.raises(ValueError):
        accumulate_hebbian(pairs)


def test_hebbian_rejects_invalid_encoding_values():
    with pytest.raises(ValueError):
        accumulate_hebbian([([0, 1], [0, 1])], encoding=Encoding.BIPOLAR)


def test_hebbian_rejects_unknown_encoding():
    with pytest.raises(ValueError):
        accumulate_hebbian([([1, 0], [1, 0])], encoding="ternary")


def test_hebbian_rejects_non_finite_values():
    with pytest.raises(ValueError):
        accumulate_hebbian([([1, np.nan], [1])])


def test_hebbian_does_not_mutate_caller_arrays():
    source = np.array([1.0, -1.0])
    target = np.array([-1.0, 1.0])
    source_before = source.copy()
    target_before = target.copy()
    pairs = [(source, target), (source, target)]

    accumulate_hebbian(pairs, encoding=Encoding.BIPOLAR, trace=True)

    assert np.array_equal(source, source_before)
    assert np.array_equal(target, target_before)
