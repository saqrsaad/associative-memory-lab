"""Recall result DTO for the application layer."""

from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import NDArray

from associative_memory.algorithms.encoding import Encoding
from associative_memory.application.dto._arrays import (
    normalized_encoding,
    owned_readonly_float_array,
)
from associative_memory.application.dto.training_result import MemoryType

#: ``comparison`` basis when the caller supplied an expected pattern.
COMPARISON_BASIS_EXPECTED = "expected_pattern"

#: ``comparison`` basis for autoassociative recall, where the output is
#: compared against the supplied input pattern (the intended original).
COMPARISON_BASIS_INPUT = "input_pattern"


@dataclass(frozen=True)
class RecallComparison:
    """Comparison metrics for one recall operation.

    Values are copied from the algorithm-package comparison so the
    presentation layer never has to import numerical types. The mismatch
    formulas live in ``algorithms.metrics`` per the mathematical
    conventions.
    """

    element_count: int
    mismatch_count: int
    match_ratio: float
    exact_match: bool


@dataclass(frozen=True)
class RecallResult:
    """Structured result of one recall use case.

    Ownership policy
    ----------------
    Every array attribute (``input_pattern``, ``net_input``,
    ``output_pattern``, ``expected_pattern``) is an independent, read-only
    ``float64`` copy owned by the DTO; mutating it raises ``ValueError``.

    ``comparison`` is ``None`` **only** when no comparison data was
    available (e.g. a heteroassociative recall without an expected target).
    A comparison object with ``mismatch_count == 0`` and ``exact_match`` is
    a genuine successful comparison, never a manufactured score.
    """

    memory_type: MemoryType
    encoding: Encoding
    activation_name: str
    threshold: float
    input_dimension: int
    output_dimension: int
    input_pattern: NDArray
    net_input: NDArray
    output_pattern: NDArray
    expected_pattern: NDArray | None
    comparison: RecallComparison | None
    comparison_basis: str | None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.memory_type, MemoryType):
            raise TypeError(
                "memory_type must be a MemoryType member; received "
                f"{type(self.memory_type).__name__}."
            )
        object.__setattr__(self, "encoding", normalized_encoding(self.encoding))
        object.__setattr__(self, "threshold", float(self.threshold))
        object.__setattr__(
            self, "input_pattern", owned_readonly_float_array(self.input_pattern)
        )
        object.__setattr__(
            self, "net_input", owned_readonly_float_array(self.net_input)
        )
        object.__setattr__(
            self, "output_pattern", owned_readonly_float_array(self.output_pattern)
        )
        if self.expected_pattern is None:
            object.__setattr__(self, "comparison_basis", None)
        else:
            object.__setattr__(
                self, "expected_pattern", owned_readonly_float_array(self.expected_pattern)
            )
        if self.warnings is None:
            object.__setattr__(self, "warnings", ())
        else:
            object.__setattr__(self, "warnings", tuple(self.warnings))
