"""Unit tests for the outer-product calculation."""

from __future__ import annotations

import numpy as np
import pytest

from associative_memory.algorithms.encoding import Encoding
from associative_memory.algorithms.outer_product import outer_product


def test_single_outer_product_matches_reference_case(reference_case):
    case = reference_case("single_outer_product")

    result = outer_product(
        case["inputs"]["s"],
        case["inputs"]["t"],
        encoding=case["encoding"],
    )

    assert result.shape == tuple(case["expected"]["dimensions"]["W"])
    assert np.array_equal(result, np.asarray(case["expected"]["W"], dtype=float))


def test_outer_product_preserves_negative_bipolar_contributions():
    result = outer_product([1, -1], [-1, 1], encoding=Encoding.BIPOLAR)
    assert np.array_equal(result, np.array([[-1.0, 1.0], [1.0, -1.0]]))


def test_outer_product_supports_different_source_and_target_lengths():
    result = outer_product([1, 2, 3], [4, 5])

    assert result.shape == (3, 2)
    assert np.array_equal(
        result,
        np.array([[4.0, 5.0], [8.0, 10.0], [12.0, 15.0]]),
    )


@pytest.mark.parametrize(
    ("source", "target", "encoding"),
    [
        ([0, 1, 2], [1, 0], Encoding.BINARY),
        ([1, 0, -1], [1, 0], Encoding.BINARY),
        ([0, 1], [0, 1], Encoding.BIPOLAR),
        ([-1, 1], [2, -1], Encoding.BIPOLAR),
    ],
)
def test_outer_product_rejects_invalid_encoding_values(source, target, encoding):
    with pytest.raises(ValueError):
        outer_product(source, target, encoding=encoding)


def test_outer_product_rejects_unknown_encoding():
    with pytest.raises(ValueError):
        outer_product([0, 1], [0, 1], encoding="ternary")


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (np.zeros((2, 2)), [1, 0]),
        ([1, 0], np.zeros((2, 2))),
        ([], [1, 0]),
        ([1, 0], []),
        ([1, np.nan], [1, 0]),
        ([1, 0], [1, np.inf]),
    ],
)
def test_outer_product_rejects_invalid_vectors(source, target):
    with pytest.raises(ValueError):
        outer_product(source, target)


def test_outer_product_rejects_non_numeric_input():
    with pytest.raises(TypeError):
        outer_product(["a", "b"], [1, 0])


def test_outer_product_does_not_mutate_caller_arrays():
    source = np.array([1.0, -1.0, 1.0])
    target = np.array([-1.0, 1.0])
    source_before = source.copy()
    target_before = target.copy()

    outer_product(source, target)

    assert np.array_equal(source, source_before)
    assert np.array_equal(target, target_before)
