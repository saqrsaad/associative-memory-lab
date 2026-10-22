"""Shared array-ownership support for application result DTOs.

This private module implements the documented ownership policy: every array
exposed by an application DTO is an **independent, read-only, float64 copy**
owned by the DTO itself. Work performed here is data handling (copying and
flag management), not numerical logic; no algorithm formula lives in the
application layer.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from associative_memory.algorithms.encoding import Encoding, coerce_encoding


def owned_readonly_float_array(values: ArrayLike) -> NDArray[np.float64]:
    """Return an independent, read-only ``float64`` copy of ``values``.

    The copy defensively protects stored experiment data: mutating this
    array raises a ``ValueError``, and in-place changes to any caller-owned
    input can never invalidate the DTO.
    """

    copied = np.array(values, dtype=np.float64, copy=True)
    copied.setflags(write=False)
    return copied


def normalized_encoding(encoding: Encoding | str) -> Encoding:
    """Normalize ``encoding`` through the algorithm package's canonical rule."""

    return coerce_encoding(encoding)
