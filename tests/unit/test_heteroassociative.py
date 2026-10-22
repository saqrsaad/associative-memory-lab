"""Unit tests for heteroassociative memory training and inference."""

from __future__ import annotations

import numpy as np
import pytest

from associative_memory.algorithms.encoding import Encoding
from associative_memory.algorithms.heteroassociative import (
    train_heteroassociative,
)


def test_heteroassociative_training_and_inference_match_reference_case(reference_case):
    case = reference_case("heteroassociative_inference")

    memory = train_heteroassociative(
        [([1, -1, 1], [-1, 1])],
        encoding=case["encoding"],
    )
    result = memory.infer(case["inputs"]["x"])

    assert memory.weight_matrix.shape == (3, 2)
    assert np.array_equal(
        memory.weight_matrix,
        np.asarray(case["inputs"]["W"], dtype=float),
    )
    assert np.array_equal(
        result.input_pattern,
        np.asarray(case["inputs"]["x"], dtype=float),
    )
    assert np.array_equal(
        result.weight_matrix,
        np.asarray(case["inputs"]["W"], dtype=float),
    )
    assert np.array_equal(
        result.net_input,
        np.asarray(case["expected"]["net_input"], dtype=float),
    )
    assert np.array_equal(
        result.activated_output,
        np.asarray(case["expected"]["output"], dtype=float),
    )
    assert result.encoding is Encoding.BIPOLAR
    assert result.activation_name == "bipolar_activation"
    assert result.threshold == 0.0
    assert result.source_dimension == 3
    assert result.target_dimension == 2
    assert result.output is result.activated_output
    assert result.query is result.input_pattern

    mismatch = int(
        np.count_nonzero(
            result.activated_output
            != np.asarray(case["expected"]["target"], dtype=float)
        )
    )
    assert mismatch == case["expected"]["mismatch_count"] == 0


def test_heteroassociative_training_accumulates_two_pairs(reference_case):
    case = reference_case("hebbian_sum_two_pairs")
    pairs = list(zip(case["inputs"]["s"], case["inputs"]["t"]))

    memory = train_heteroassociative(pairs, encoding=case["encoding"])

    assert memory.pair_count == 2
    assert memory.source_dimension == 3
    assert memory.target_dimension == 2
    assert np.array_equal(
        memory.weight_matrix,
        np.asarray(case["expected"]["W"], dtype=float),
    )


def test_heteroassociative_output_dimension_matches_target_dimension():
    memory = train_heteroassociative(
        [([1, -1, 1], [-1, 1])],
        encoding=Encoding.BIPOLAR,
    )

    result = memory.infer([1, -1, 1])

    assert result.net_input.shape == (2,)
    assert result.activated_output.shape == (2,)
    assert result.target_dimension == 2


def test_heteroassociative_supports_binary_encoding():
    memory = train_heteroassociative(
        [([0, 1], [1, 0])],
        encoding=Encoding.BINARY,
    )

    result = memory.infer([0, 1])

    assert np.array_equal(result.net_input, np.array([1.0, 0.0]))
    assert np.array_equal(result.activated_output, np.array([1.0, 0.0]))


def test_heteroassociative_rejects_empty_training_data():
    with pytest.raises(ValueError):
        train_heteroassociative([], encoding=Encoding.BIPOLAR)


def test_heteroassociative_requires_explicit_encoding():
    with pytest.raises(TypeError):
        train_heteroassociative([([1, -1], [-1, 1])])


def test_heteroassociative_rejects_invalid_encoding_values():
    with pytest.raises(ValueError):
        train_heteroassociative(
            [([0, 1], [0, 1])],
            encoding=Encoding.BIPOLAR,
        )


def test_heteroassociative_inference_rejects_wrong_query_length():
    memory = train_heteroassociative(
        [([1, -1, 1], [-1, 1])],
        encoding=Encoding.BIPOLAR,
    )

    with pytest.raises(ValueError):
        memory.infer([1, -1])


def test_heteroassociative_inference_rejects_invalid_query_values():
    memory = train_heteroassociative(
        [([1, -1, 1], [-1, 1])],
        encoding=Encoding.BIPOLAR,
    )

    with pytest.raises(ValueError):
        memory.infer([1, 0, 1])


def test_heteroassociative_training_does_not_mutate_caller_arrays():
    source = np.array([1.0, -1.0, 1.0])
    target = np.array([-1.0, 1.0])
    source_before = source.copy()
    target_before = target.copy()

    train_heteroassociative(
        [(source, target), (source, target)],
        encoding=Encoding.BIPOLAR,
        trace=True,
    )

    assert np.array_equal(source, source_before)
    assert np.array_equal(target, target_before)
