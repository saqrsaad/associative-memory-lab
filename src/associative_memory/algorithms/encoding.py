"""Encoding definitions and vector validation for the algorithm package.

The project uses row vectors: a pattern is represented as a one-dimensional
real array of shape ``(n,)``. Encoding is explicit configuration and is never
inferred from the numeric values of a vector.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike, NDArray


class Encoding(StrEnum):
    """Supported pattern encodings."""

    BINARY = "binary"
    BIPOLAR = "bipolar"


ENCODING_VALUES: Mapping[Encoding, tuple[float, ...]] = MappingProxyType(
    {
        Encoding.BINARY: (0.0, 1.0),
        Encoding.BIPOLAR: (-1.0, 1.0),
    }
)


def coerce_encoding(encoding: Encoding | str) -> Encoding:
    """Return the :class:`Encoding` member for ``encoding``.

    Raises:
        ValueError: If ``encoding`` is not a supported encoding name.
    """

    if isinstance(encoding, Encoding):
        return encoding
    try:
        return Encoding(encoding)
    except ValueError as exc:
        supported = ", ".join(member.value for member in Encoding)
        raise ValueError(
            f"Unsupported encoding {encoding!r}; expected one of: {supported}."
        ) from exc


def validate_encoding_values(
    values: ArrayLike,
    encoding: Encoding | str,
    *,
    name: str = "pattern",
) -> None:
    """Validate that every value is legal for the given encoding.

    Binary values must be ``0`` or ``1``; bipolar values must be ``-1`` or
    ``+1``. Values are compared exactly rather than by a tolerance.
    """

    encoded = coerce_encoding(encoding)
    allowed = np.asarray(ENCODING_VALUES[encoded], dtype=np.float64)
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isin(array, allowed)):
        allowed_text = ", ".join(str(value) for value in ENCODING_VALUES[encoded])
        raise ValueError(
            f"{name} contains values that are not valid for "
            f"{encoded.value} encoding; expected values from {{{allowed_text}}}."
        )


def validate_vector(
    values: ArrayLike,
    *,
    name: str = "vector",
    encoding: Encoding | str | None = None,
) -> NDArray[np.float64]:
    """Validate one pattern vector and return it as a ``float64`` copy.

    The returned array is a copy, so callers' arrays are never mutated by the
    algorithm functions. Encoding validation is applied only when ``encoding``
    is provided.

    Raises:
        TypeError: If the input does not contain real numeric values.
        ValueError: If the input is not one-dimensional, is empty, contains
            non-finite values, or violates the requested encoding.
    """

    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional; received shape {array.shape}."
        )
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise TypeError(
            f"{name} must contain real numeric values; received dtype "
            f"{array.dtype}."
        )

    array = array.astype(np.float64, copy=True)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    if encoding is not None:
        validate_encoding_values(array, encoding, name=name)
    return array
