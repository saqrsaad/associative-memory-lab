"""Unit tests for autoassociative memory training and inference."""

from __future__ import annotations

import numpy as np
import pytest

from associative_memory.algorithms.autoassociative import (
    train_autoassociative,
)
from associative_memory.algorithms.encoding import Encoding


def test_autoassociative_training_and_inference_match_reference_case(reference_case):
    case = reference_case("autoassociative_inference")
    pattern = case["inputs"]["x"]

    memory = train_autoassociative([pattern], encoding=case["encoding"])
    result = memory.infer(pattern)

    assert memory.weight_matrix.shape == (3, 3)
    assert np.array_equal(
        memory.weight_matrix,
        np.asarray(case["inputs"]["W"], dtype=float),
    )
    assert np.array_equal(
        np.diag(memory.weight_matrix),
        np.array([1.0, 1.0, 1.0]),
    )
    assert np.array_equal(
        result.net_input,
        np.asarray(case["expected"]["net_input"], dtype=float),
    )
    assert np.array_equal(
        result.activated_output,
        np.asarray(case["expected"]["output"], dtype=float),
    )
    assert result.source_dimension == 3
    assert result.target_dimension == 3
    assert result.threshold == 0.0


def test_autoassociative_interference_is_a_legitimate_numerical_outcome(reference_case):
    case = reference_case("autoassociative_interference_failure")

    memory = train_autoassociative(
        case["inputs"]["patterns"],
        encoding=case["encoding"],
    )
    result = memory.infer(case["inputs"]["query"])

    assert np.array_equal(
        memory.weight_matrix,
        np.asarray(case["expected"]["W"], dtype=float),
    )
    assert np.array_equal(
        np.diag(memory.weight_matrix),
        np.array([2.0, 2.0, 2.0, 2.0]),
    )
    assert np.array_equal(
        result.net_input,
        np.asarray(case["expected"]["net_input"], dtype=float),
    )
    assert np.array_equal(
        result.activated_output,
        np.asarray(case["expected"]["output"], dtype=float),
    )

    target = np.asarray(case["expected"]["target"], dtype=float)
    mismatch = int(np.count_nonzero(result.activated_output != target))
    assert mismatch == case["expected"]["mismatch_count"] == 1
    assert not np.array_equal(result.activated_output, target)
    assert case["expected"]["exact_recall"] is False


def test_autoassociative_rejects_empty_training_data():
    with pytest.raises(ValueError):
        train_autoassociative([], encoding=Encoding.BIPOLAR)


def test_autoassociative_requires_explicit_encoding():
    with pytest.raises(TypeError):
        train_autoassociative([[1, -1, 1]])


def test_autoassociative_rejects_inconsistent_pattern_dimensions():
    with pytest.raises(ValueError):
        train_autoassociative(
            [[1, -1, 1], [1, -1]],
            encoding=Encoding.BIPOLAR,
        )


def test_autoassociative_rejects_invalid_encoding_values():
    with pytest.raises(ValueError):
        train_autoassociative(
            [[0, 1, 0]],
            encoding=Encoding.BIPOLAR,
        )


def test_autoassociative_inference_rejects_wrong_query_length():
    memory = train_autoassociative(
        [[1, -1, 1]],
        encoding=Encoding.BIPOLAR,
    )

    with pytest.raises(ValueError):
        memory.infer([1, -1])


def test_autoassociative_inference_rejects_invalid_query_values():
    memory = train_autoassociative(
        [[1, -1, 1]],
        encoding=Encoding.BIPOLAR,
    )

    with pytest.raises(ValueError):
        memory.infer([1, 0, 1])


def test_autoassociative_training_does_not_mutate_caller_arrays():
    pattern = np.array([1.0, -1.0, 1.0])
    pattern_before = pattern.copy()

    train_autoassociative([pattern, pattern], encoding=Encoding.BIPOLAR, trace=True)

    assert np.array_equal(pattern, pattern_before)
