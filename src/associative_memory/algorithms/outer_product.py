"""Outer-product calculation for Hebbian associative memory."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .encoding import Encoding, validate_vector


def outer_product(
    source: ArrayLike,
    target: ArrayLike,
    *,
    encoding: Encoding | str | None = None,
) -> NDArray[np.float64]:
    """Return the outer product of two pattern vectors.

    The project uses row vectors: the source has shape ``(n,)``, the target has
    shape ``(m,)``, and the returned matrix has shape ``(n, m)`` with entries
    ``source[i] * target[j]``.

    When ``encoding`` is provided, both vectors must contain only values that
    are legal for that encoding. The returned matrix is always a new array.

    Raises:
        TypeError: If an input does not contain real numeric values.
        ValueError: If an input has the wrong shape, is empty, contains
            non-finite values, or violates the requested encoding.
    """

    source_vector = validate_vector(source, name="source", encoding=encoding)
    target_vector = validate_vector(target, name="target", encoding=encoding)
    return np.outer(source_vector, target_vector)
