"""Binary and bipolar activation functions.

Both rules use a strict positive test. At exactly zero, binary activation
returns ``0`` and bipolar activation returns ``-1``; ``>= 0`` is not used.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .encoding import Encoding, coerce_encoding

ACTIVATION_THRESHOLD = 0.0


def _validate_net_input(
    net_input: ArrayLike,
    *,
    name: str = "net_input",
) -> NDArray[np.float64]:
    """Validate a scalar or one-dimensional net-input array."""

    array = np.asarray(net_input)
    if array.ndim > 1:
        raise ValueError(
            f"{name} must be a scalar or one-dimensional; received shape "
            f"{array.shape}."
        )
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise TypeError(
            f"{name} must contain real numeric values; received dtype "
            f"{array.dtype}."
        )

    array = array.astype(np.float64, copy=True)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def binary_activation(net_input: ArrayLike) -> NDArray[np.float64]:
    """Apply binary activation elementwise.

    Returns ``1`` where the net input is strictly greater than zero and ``0``
    otherwise. Scalar input produces a zero-dimensional array; vector input
    produces an array of the same shape.
    """

    values = _validate_net_input(net_input)
    return np.where(values > ACTIVATION_THRESHOLD, 1.0, 0.0)


def bipolar_activation(net_input: ArrayLike) -> NDArray[np.float64]:
    """Apply bipolar activation elementwise.

    Returns ``+1`` where the net input is strictly greater than zero and ``-1``
    otherwise. Scalar input produces a zero-dimensional array; vector input
    produces an array of the same shape.
    """

    values = _validate_net_input(net_input)
    return np.where(values > ACTIVATION_THRESHOLD, 1.0, -1.0)


def activate(
    net_input: ArrayLike,
    encoding: Encoding | str,
) -> NDArray[np.float64]:
    """Apply the activation function selected by ``encoding``."""

    encoded = coerce_encoding(encoding)
    if encoded is Encoding.BINARY:
        return binary_activation(net_input)
    return bipolar_activation(net_input)


def activation_name(encoding: Encoding | str) -> str:
    """Return the canonical activation name for an encoding."""

    encoded = coerce_encoding(encoding)
    if encoded is Encoding.BINARY:
        return "binary_activation"
    return "bipolar_activation"
