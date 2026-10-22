"""Integration tests for the application training workflow (GUI-free)."""

from __future__ import annotations

import numpy as np
import pytest

from associative_memory.algorithms.encoding import Encoding
from associative_memory.application.dto.training_result import MemoryType
from associative_memory.application.services.training_service import (
    TrainingDataset,
    TrainingError,
    TrainingService,
)

SERVICE = TrainingService()


def _hetero_dataset() -> TrainingDataset:
    """Two hand-calculated bipolar source/target pairs."""

    return TrainingDataset.from_pairs(
        [
            ([1, -1], [1, 1]),
            ([-1, 1], [1, -1]),
        ],
        name="hetero-demo",
    )


def test_hetero_training_workflow_produces_the_hand_calculated_matrix():
    result = SERVICE.train(_hetero_dataset(), encoding=Encoding.BIPOLAR)

    # W = outer(s1, t1) + outer(s2, t2) = [[0, 2], [0, -2]].
    assert np.array_equal(result.weight_matrix, np.array([[0.0, 2.0], [0.0, -2.0]]))
    assert result.memory_type is MemoryType.HETEROASSOCIATIVE
    assert result.pair_count == 2
    assert result.input_dimension == 2
    assert result.output_dimension == 2


def test_training_metadata_is_complete_and_correct():
    result = SERVICE.train(_hetero_dataset(), encoding="bipolar")

    assert result.encoding is Encoding.BIPOLAR
    assert result.activation_name == "bipolar_activation"
    assert result.threshold == 0.0
    assert result.dataset_name == "hetero-demo"
    assert result.memory.source_dimension == 2
    assert result.memory.target_dimension == 2


def test_autoassociative_training_workflow_stores_patterns_with_themselves():
    dataset = TrainingDataset.from_patterns([[1, -1, 1], [-1, 1, 1]])

    result = SERVICE.train(dataset, encoding=Encoding.BIPOLAR)

    assert result.memory_type is MemoryType.AUTOASSOCIATIVE
    assert result.pair_count == 2
    assert result.input_dimension == 3
    assert result.output_dimension == 3
    # W = outer(p, p) summed over both patterns, full diagonal retained.
    expected = np.outer([1.0, -1, 1], [1.0, -1, 1]) + np.outer(
        [-1.0, 1, 1], [-1.0, 1, 1]
    )
    assert np.array_equal(result.weight_matrix, expected)


def test_trace_request_preserves_read_only_outer_product_contributions():
    result = SERVICE.train(_hetero_dataset(), encoding=Encoding.BIPOLAR, trace=True)

    assert result.contributions is not None
    assert len(result.contributions) == 2
    assert np.array_equal(result.contributions[0], np.array([[1.0, 1.0], [-1.0, -1.0]]))
    assert np.array_equal(result.contributions[1], np.array([[-1.0, 1.0], [1.0, -1.0]]))
    for contribution in result.contributions:
        assert contribution.flags.writeable is False


def test_without_trace_contributions_are_none():
    result = SERVICE.train(_hetero_dataset(), encoding=Encoding.BIPOLAR)

    assert result.contributions is None


def test_result_weight_matrix_is_a_read_only_array():
    result = SERVICE.train(_hetero_dataset(), encoding=Encoding.BIPOLAR)

    assert result.weight_matrix.flags.writeable is False
    with pytest.raises(ValueError):
        result.weight_matrix[0, 0] = 99.0


def test_caller_owned_dataset_arrays_are_not_mutated():
    source = np.array([1.0, -1.0])
    target = np.array([1.0, 1.0])
    dataset = TrainingDataset.from_pairs([(source, target)])

    SERVICE.train(dataset, encoding=Encoding.BIPOLAR)

    assert source.tolist() == [1.0, -1.0]
    assert target.tolist() == [1.0, 1.0]


def test_memory_type_must_match_the_dataset():
    with pytest.raises(
        TrainingError,
        match="does not match the dataset memory type 'heteroassociative'",
    ):
        SERVICE.train(
            _hetero_dataset(), encoding=Encoding.BIPOLAR, memory_type="autoassociative"
        )
    with pytest.raises(TrainingError, match="Unknown memory type 'strange'"):
        SERVICE.train(
            _hetero_dataset(), encoding=Encoding.BIPOLAR, memory_type="strange"
        )


def test_missing_training_examples_are_rejected():
    with pytest.raises(TrainingError, match="at least one pair"):
        SERVICE.train(TrainingDataset.from_pairs([]), encoding=Encoding.BINARY)
    with pytest.raises(TrainingError, match="at least one pattern"):
        SERVICE.train(
            TrainingDataset.from_patterns([]), encoding=Encoding.BINARY
        )


def test_incompatible_source_dimensions_are_rejected():
    dataset = TrainingDataset.from_pairs(
        [
            ([1, -1], [1, 1]),
            ([1, -1, 1], [1, 1]),
        ]
    )

    with pytest.raises(TrainingError, match="pairs\\[1\\].source has length 3"):
        SERVICE.train(dataset, encoding=Encoding.BIPOLAR)


def test_incompatible_target_dimensions_are_rejected():
    dataset = TrainingDataset.from_pairs(
        [
            ([1, -1], [1, 1]),
            ([1, -1], [1]),
        ]
    )

    with pytest.raises(TrainingError, match="pairs\\[1\\].target has length 1"):
        SERVICE.train(dataset, encoding=Encoding.BIPOLAR)


def test_invalid_encoding_name_is_rejected():
    with pytest.raises(TrainingError, match="Unsupported encoding 'ternary'"):
        SERVICE.train(_hetero_dataset(), encoding="ternary")


def test_pattern_values_that_violate_the_encoding_are_rejected():
    dataset = TrainingDataset.from_pairs([([2, -1], [1, 1])])

    with pytest.raises(
        TrainingError, match="not valid for bipolar encoding"
    ):
        SERVICE.train(dataset, encoding=Encoding.BIPOLAR)


def test_autoassociative_dataset_requires_self_associated_pairs():
    dataset = TrainingDataset(
        patterns=(([1.0, -1.0], [1.0, 1.0]),),
        memory_type=MemoryType.AUTOASSOCIATIVE,
    )

    with pytest.raises(TrainingError, match="not self-associated"):
        SERVICE.train(dataset, encoding=Encoding.BIPOLAR)
