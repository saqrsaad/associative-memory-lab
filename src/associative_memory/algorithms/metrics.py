"""Recall comparison metrics for encoded associative-memory patterns.

The formulas follow ``docs/mathematical-conventions.md`` (Section 8). For a
recalled output ``y`` and an expected target ``t`` of the same length
``N > 0`` and the same encoding:

    M = sum(1[y_i != t_i])           (mismatch count)
    match_ratio = (N - M) / N

Exact recall succeeds only when ``M == 0``. The comparison is an exact
elementwise inequality of encoded values, not a fuzzy distance metric. If
dimensions or encoding semantics are incompatible, the comparison raises
instead of manufacturing a score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .encoding import Encoding, validate_vector


@dataclass(frozen=True)
class PatternComparison:
    """Structured comparison of a recalled output against an expected target.

    Attributes:
        element_count: Number of compared elements ``N``.
        mismatch_count: Number of differing elements ``M``.
        match_ratio: ``(N - M) / N``; a genuine ``0.0`` score is distinguishable
            from an unavailable comparison because a comparison object only
            exists when the comparison was actually performed.
        exact_match: ``True`` only when ``M == 0`` (the documented exact
            recall criterion).
    """

    element_count: int
    mismatch_count: int
    match_ratio: float
    exact_match: bool


def compare_vectors(
    output: ArrayLike,
    expected: ArrayLike,
    encoding: Encoding | str,
) -> PatternComparison:
    """Compare a recalled output against an expected encoded pattern.

    Both arguments are validated as one-dimensional, non-empty, finite
    vectors whose values are legal for ``encoding``; they must have the same
    length. The mismatch count and match ratio follow the formulas in
    ``docs/mathematical-conventions.md`` (Section 8).

    Args:
        output: Recalled pattern, shape ``(N,)``.
        expected: Expected pattern, shape ``(N,)``.
        encoding: Encoding both pattern values must satisfy.

    Raises:
        TypeError: If a pattern does not contain real numeric values.
        ValueError: If a pattern is invalid, violates the encoding, or if the
            two patterns have different lengths.
    """

    output_vector = validate_vector(
        output,
        name="output",
        encoding=encoding,
    )
    expected_vector = validate_vector(
        expected,
        name="expected",
        encoding=encoding,
    )

    if output_vector.shape[0] != expected_vector.shape[0]:
        raise ValueError(
            f"Comparison requires equal lengths: output has length "
            f"{output_vector.shape[0]}, expected has length "
            f"{expected_vector.shape[0]}."
        )

    element_count = int(output_vector.shape[0])
    mismatch_count = int(np.count_nonzero(output_vector != expected_vector))
    match_ratio = (element_count - mismatch_count) / element_count

    return PatternComparison(
        element_count=element_count,
        mismatch_count=mismatch_count,
        match_ratio=match_ratio,
        exact_match=mismatch_count == 0,
    )
