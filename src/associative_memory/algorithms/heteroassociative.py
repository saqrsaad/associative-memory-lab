"""Heteroassociative memory training and inference.

Heteroassociative memory learns ``s -> t`` associations where the source and
target patterns may have different lengths. The learned matrix follows the
row-vector convention: source shape ``(n,)``, target shape ``(m,)``, weight
matrix shape ``(n, m)``, and inference uses ``y_in = x @ W``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .activation import ACTIVATION_THRESHOLD, activate, activation_name
from .encoding import Encoding, coerce_encoding, validate_vector
from .hebbian import accumulate_hebbian
from .results import InferenceResult


@dataclass(frozen=True)
class HeteroassociativeMemory:
    """A trained heteroassociative weight matrix and its configuration.

    Attributes:
        weight_matrix: Learned weights with shape ``(source_dimension,
            target_dimension)``.
        source_dimension: Source pattern length ``n``.
        target_dimension: Target pattern length ``m``.
        encoding: Encoding used for training and inference.
        pair_count: Number of training pairs used.
        contributions: Individual outer-product contributions in training
            order, or ``None`` when tracing was not requested.
    """

    weight_matrix: NDArray[np.float64]
    source_dimension: int
    target_dimension: int
    encoding: Encoding
    pair_count: int
    contributions: tuple[NDArray[np.float64], ...] | None = None

    def infer(self, query: ArrayLike) -> InferenceResult:
        """Run recall for ``query`` using this trained memory."""

        return infer_heteroassociative(self, query)


def train_heteroassociative(
    pairs: Iterable[tuple[ArrayLike, ArrayLike]],
    *,
    encoding: Encoding | str,
    trace: bool = False,
) -> HeteroassociativeMemory:
    """Train a heteroassociative memory from source-target pairs.

    The encoding is explicit configuration and is used to validate every
    source and target pattern.

    Raises:
        TypeError: If a pattern does not contain real numeric values.
        ValueError: If the dataset is empty, dimensions are inconsistent, or
            a vector violates the selected encoding.
    """

    normalized_encoding = coerce_encoding(encoding)
    result = accumulate_hebbian(
        pairs,
        encoding=normalized_encoding,
        trace=trace,
    )
    return HeteroassociativeMemory(
        weight_matrix=result.weight_matrix,
        source_dimension=result.source_dimension,
        target_dimension=result.target_dimension,
        encoding=normalized_encoding,
        pair_count=result.pair_count,
        contributions=result.contributions,
    )


def infer_heteroassociative(
    memory: HeteroassociativeMemory,
    query: ArrayLike,
) -> InferenceResult:
    """Recall a target from ``query`` using a trained heteroassociative memory.

    The query must match the trained source dimension. Net input is computed
    with ``query @ W`` and then passed through the configured activation.

    Raises:
        TypeError: If the query does not contain real numeric values.
        ValueError: If the query or the stored weight matrix has incompatible
            dimensions, is empty, or contains non-finite values.
    """

    query_vector = validate_vector(
        query,
        name="query",
        encoding=memory.encoding,
    )
    if query_vector.shape[0] != memory.source_dimension:
        raise ValueError(
            f"Query length {query_vector.shape[0]} does not match the trained "
            f"source dimension {memory.source_dimension}."
        )

    expected_weight_shape = (memory.source_dimension, memory.target_dimension)
    weight_matrix = np.asarray(memory.weight_matrix, dtype=np.float64)
    if weight_matrix.ndim != 2 or weight_matrix.shape != expected_weight_shape:
        raise ValueError(
            f"Stored weight matrix has shape {weight_matrix.shape}, "
            f"expected {expected_weight_shape}."
        )
    if not np.all(np.isfinite(weight_matrix)):
        raise ValueError("Stored weight matrix contains non-finite values.")

    net_input = query_vector @ weight_matrix
    if net_input.shape != (memory.target_dimension,):
        raise ValueError(
            f"Net input has shape {net_input.shape}, expected "
            f"({memory.target_dimension},)."
        )

    activated_output = activate(net_input, memory.encoding)
    return InferenceResult(
        input_pattern=query_vector,
        weight_matrix=weight_matrix,
        net_input=net_input,
        activated_output=activated_output,
        encoding=memory.encoding,
        activation_name=activation_name(memory.encoding),
        threshold=ACTIVATION_THRESHOLD,
        source_dimension=memory.source_dimension,
        target_dimension=memory.target_dimension,
    )
