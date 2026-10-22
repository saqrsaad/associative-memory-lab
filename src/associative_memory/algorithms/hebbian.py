"""Hebbian weight accumulation from source-target training pairs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .encoding import Encoding, coerce_encoding, validate_vector


@dataclass(frozen=True)
class HebbianTrainingResult:
    """Structured result of accumulating Hebbian outer-product contributions.

    Attributes:
        weight_matrix: Accumulated weight matrix with shape ``(n, m)``.
        source_dimension: Source pattern length ``n``.
        target_dimension: Target pattern length ``m``.
        pair_count: Number of accumulated training pairs.
        encoding: Encoding used for validation, or ``None`` when no encoding
            validation was requested.
        contributions: Individual outer-product matrices in training order,
            or ``None`` when tracing was not requested.
    """

    weight_matrix: NDArray[np.float64]
    source_dimension: int
    target_dimension: int
    pair_count: int
    encoding: Encoding | None
    contributions: tuple[NDArray[np.float64], ...] | None = None


def accumulate_hebbian(
    pairs: Iterable[tuple[ArrayLike, ArrayLike]],
    *,
    encoding: Encoding | str | None = None,
    trace: bool = False,
) -> HebbianTrainingResult:
    """Accumulate ``W = sum(outer(source, target))`` over training pairs.

    The weight matrix follows the row-vector convention and has shape
    ``(n, m)``. Source and target dimensions must be consistent across all
    pairs. Bipolar negative contributions are preserved.

    Args:
        pairs: Non-empty sequence of ``(source, target)`` pattern pairs.
        encoding: Optional encoding used to validate both vectors in every
            pair. When omitted, encoding validation is skipped.
        trace: When ``True``, individual outer-product contributions are kept
            in the returned result in their original order.

    Returns:
        A :class:`HebbianTrainingResult` containing the accumulated weight
        matrix and metadata.

    Raises:
        TypeError: If a pattern does not contain real numeric values.
        ValueError: If the dataset is empty, a pair is malformed, dimensions
            are inconsistent, or a vector is invalid.
    """

    pair_list = list(pairs)
    if not pair_list:
        raise ValueError("Training data must contain at least one pair.")

    normalized_encoding = coerce_encoding(encoding) if encoding is not None else None

    contributions: list[NDArray[np.float64]] = []
    source_dimension: int | None = None
    target_dimension: int | None = None

    for index, pair in enumerate(pair_list):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(
                f"Training pair {index} must be a (source, target) pair."
            )
        source, target = pair
        source_vector = validate_vector(
            source,
            name=f"pairs[{index}].source",
            encoding=normalized_encoding,
        )
        target_vector = validate_vector(
            target,
            name=f"pairs[{index}].target",
            encoding=normalized_encoding,
        )

        current_source_dimension = int(source_vector.shape[0])
        current_target_dimension = int(target_vector.shape[0])
        if source_dimension is None:
            source_dimension = current_source_dimension
            target_dimension = current_target_dimension
        else:
            if current_source_dimension != source_dimension:
                raise ValueError(
                    f"pairs[{index}].source has length {current_source_dimension}, "
                    f"but earlier pairs use length {source_dimension}."
                )
            if current_target_dimension != target_dimension:
                raise ValueError(
                    f"pairs[{index}].target has length {current_target_dimension}, "
                    f"but earlier pairs use length {target_dimension}."
                )

        contributions.append(np.outer(source_vector, target_vector))

    stacked = np.stack(contributions, axis=0)
    weight_matrix = np.sum(stacked, axis=0)

    assert source_dimension is not None
    assert target_dimension is not None
    return HebbianTrainingResult(
        weight_matrix=weight_matrix,
        source_dimension=source_dimension,
        target_dimension=target_dimension,
        pair_count=len(pair_list),
        encoding=normalized_encoding,
        contributions=tuple(contributions) if trace else None,
    )


def hebbian_weight_matrix(
    pairs: Iterable[tuple[ArrayLike, ArrayLike]],
    *,
    encoding: Encoding | str | None = None,
) -> NDArray[np.float64]:
    """Return only the accumulated weight matrix for the given training pairs.

    This is a small convenience wrapper around :func:`accumulate_hebbian` for
    callers that do not need the training trace or metadata.
    """

    return accumulate_hebbian(pairs, encoding=encoding).weight_matrix
