"""Shared structured result objects for associative-memory inference.

These objects are plain numerical data. They contain no GUI, Qt, or
presentation-specific state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .encoding import Encoding


@dataclass(frozen=True)
class InferenceResult:
    """Numerical result of one inference (recall) operation.

    Attributes:
        input_pattern: Query pattern of shape ``(n,)`` as supplied to recall.
        weight_matrix: Weight matrix used for the multiplication, shape
            ``(n, m)``.
        net_input: Raw ``x @ W`` result before activation, shape ``(m,)``.
        activated_output: Output after applying the configured activation,
            shape ``(m,)``.
        encoding: Encoding used for validation and activation.
        activation_name: Name of the applied activation rule.
        threshold: Activation threshold (``0.0`` for these rules).
        source_dimension: Query/input length ``n``.
        target_dimension: Output length ``m``.
    """

    input_pattern: NDArray[np.float64]
    weight_matrix: NDArray[np.float64]
    net_input: NDArray[np.float64]
    activated_output: NDArray[np.float64]
    encoding: Encoding
    activation_name: str
    threshold: float
    source_dimension: int
    target_dimension: int

    @property
    def output(self) -> NDArray[np.float64]:
        """Alias for :attr:`activated_output`."""

        return self.activated_output

    @property
    def query(self) -> NDArray[np.float64]:
        """Alias for :attr:`input_pattern`."""

        return self.input_pattern
