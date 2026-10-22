"""Unit tests for the recall comparison metrics."""

from __future__ import annotations

import numpy as np
import pytest

from associative_memory.algorithms.encoding import Encoding
from associative_memory.algorithms.metrics import PatternComparison, compare_vectors


def test_compare_vectors_matches_reference_case(reference_case):
    case = reference_case("autoassociative_interference_failure")

    comparison = compare_vectors(
        case["expected"]["output"],
        case["expected"]["target"],
        Encoding.BIPOLAR,
    )

    assert isinstance(comparison, PatternComparison)
    assert comparison.element_count == 4
    assert comparison.mismatch_count == case["expected"]["mismatch_count"]
    assert comparison.match_ratio == pytest.approx(0.75)
    assert comparison.exact_match is case["expected"]["exact_recall"]


def test_perfect_match_reports_zero_mismatches_and_full_ratio():
    expected = np.array([1.0, 0.0, 1.0, 0.0])

    comparison = compare_vectors(expected, np.array([1.0, 0.0, 1.0, 0.0]), "binary")

    assert comparison.element_count == 4
    assert comparison.mismatch_count == 0
    assert comparison.match_ratio == 1.0
    assert comparison.exact_match is True


def test_partial_match_uses_the_documented_mismatch_formula():
    expected = np.array([1.0, 1.0, 0.0, 0.0])

    comparison = compare_vectors(expected, np.array([1.0, 0.0, 0.0, 0.0]), "binary")

    assert comparison.mismatch_count == 1
    assert comparison.match_ratio == 0.75
    assert comparison.exact_match is False


def test_complete_mismatch_reports_a_genuine_zero_match_ratio():
    comparison = compare_vectors([0, 0], [1, 1], Encoding.BINARY)

    assert comparison.match_ratio == 0.0
    assert comparison.mismatch_count == 2
    assert comparison.exact_match is False


def test_incompatible_lengths_are_reported_not_scored():
    with pytest.raises(ValueError, match="Comparison requires equal lengths"):
        compare_vectors([1, 0, 1], [1, 1], Encoding.BINARY)


def test_values_that_violate_the_encoding_are_rejected():
    with pytest.raises(ValueError, match="not valid for binary encoding"):
        compare_vectors([1, 2], [1, 0], Encoding.BINARY)
    with pytest.raises(ValueError, match="not valid for bipolar encoding"):
        compare_vectors([1, 0], [1, 0], Encoding.BIPOLAR)


def test_non_numeric_or_wrong_shape_inputs_are_rejected():
    with pytest.raises(TypeError):
        compare_vectors(["a", "b"], [1, 0], Encoding.BINARY)
    with pytest.raises(ValueError, match="one-dimensional"):
        compare_vectors([[1, 0]], [1, 0], Encoding.BINARY)


def test_caller_arrays_are_not_mutated():
    output = np.array([1.0, 0.0])

    compare_vectors(output, np.array([1.0, 0.0]), Encoding.BINARY)

    assert output.tolist() == [1.0, 0.0]
