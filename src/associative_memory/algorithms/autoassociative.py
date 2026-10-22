"""Autoassociative memory training and inference.

Autoassociative memory stores each pattern in association with itself. The
learned matrix follows the row-vector convention: pattern shape ``(n,)``,
weight matrix shape ``(n, n)``, and inference uses ``y_in = x @ W``. The full
diagonal is retained; it is never silently zeroed.
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
class AutoassociativeMemory:
    """A trained autoassociative weight matrix and its configuration.

    Attributes:
        weight_matrix: Learned weights with shape ``(pattern_dimension,
            pattern_dimension)``.
        pattern_dimension: Stored pattern length ``n``.
        encoding: Encoding used for training and inference.
        pattern_count: Number of stored patterns.
        contributions: Individual outer-product contributions in training
            order, or ``None`` when tracing was not requested.
    """

    weight_matrix: NDArray[np.float64]
    pattern_dimension: int
    encoding: Encoding
    pattern_count: int
    contributions: tuple[NDArray[np.float64], ...] | None = None

    def infer(self, query: ArrayLike) -> InferenceResult:
        """Run recall for ``query`` using this trained memory."""

        return infer_autoassociative(self, query)


def train_autoassociative(
    patterns: Iterable[ArrayLike],
    *,
    encoding: Encoding | str,
    trace: bool = False,
) -> AutoassociativeMemory:
    """Train an autoassociative memory from stored patterns.

    The weight matrix is ``W = sum(outer(pattern, pattern))``. The encoding is
    explicit configuration. The diagonal of ``W`` is retained.

    Raises:
        TypeError: If a pattern does not contain real numeric values.
        ValueError: If the dataset is empty, dimensions are inconsistent, or
            a pattern violates the selected encoding.
    """

    normalized_encoding = coerce_encoding(encoding)
    pattern_list = list(patterns)
    if not pattern_list:
        raise ValueError("Autoassociative training requires at least one pattern.")

    pairs = [(pattern, pattern) for pattern in pattern_list]
    result = accumulate_hebbian(
        pairs,
        encoding=normalized_encoding,
        trace=trace,
    )
    if result.source_dimension != result.target_dimension:
        raise ValueError(
            "Autoassociative training produced different source and target "
            "dimensions; every pattern must be associated with itself."
        )

    return AutoassociativeMemory(
        weight_matrix=result.weight_matrix,
        pattern_dimension=result.source_dimension,
        encoding=normalized_encoding,
        pattern_count=result.pair_count,
        contributions=result.contributions,
    )


def infer_autoassociative(
    memory: AutoassociativeMemory,
    query: ArrayLike,
) -> InferenceResult:
    """Recall a stored pattern from ``query`` using an autoassociative memory.

    The query must match the stored pattern dimension. Net input is computed
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
    if query_vector.shape[0] != memory.pattern_dimension:
        raise ValueError(
            f"Query length {query_vector.shape[0]} does not match the stored "
            f"pattern dimension {memory.pattern_dimension}."
        )

    expected_weight_shape = (memory.pattern_dimension, memory.pattern_dimension)
    weight_matrix = np.asarray(memory.weight_matrix, dtype=np.float64)
    if weight_matrix.ndim != 2 or weight_matrix.shape != expected_weight_shape:
        raise ValueError(
            f"Stored weight matrix has shape {weight_matrix.shape}, "
            f"expected {expected_weight_shape}."
        )
    if not np.all(np.isfinite(weight_matrix)):
        raise ValueError("Stored weight matrix contains non-finite values.")

    net_input = query_vector @ weight_matrix
    if net_input.shape != (memory.pattern_dimension,):
        raise ValueError(
            f"Net input has shape {net_input.shape}, expected "
            f"({memory.pattern_dimension},)."
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
        source_dimension=memory.pattern_dimension,
        target_dimension=memory.pattern_dimension,
    )
