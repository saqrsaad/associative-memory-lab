"""Training result DTO for the application layer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from numpy.typing import NDArray

from associative_memory.algorithms.activation import (
    ACTIVATION_THRESHOLD,
    activation_name,
)
from associative_memory.algorithms.autoassociative import AutoassociativeMemory
from associative_memory.algorithms.encoding import Encoding
from associative_memory.algorithms.heteroassociative import HeteroassociativeMemory
from associative_memory.application.dto._arrays import (
    normalized_encoding,
    owned_readonly_float_array,
)


class MemoryType(str, Enum):
    """The two supported associative-memory modes (kept distinct by design)."""

    AUTOASSOCIATIVE = "autoassociative"
    HETEROASSOCIATIVE = "heteroassociative"


@dataclass(frozen=True)
class TrainingResult:
    """Structured result of one training use case.

    Ownership policy
    ----------------
    ``memory`` is the trained algorithm-package object and retains the
    canonical arrays used for further inference; treat it as immutable
    input. Every array exposed directly on this DTO (``weight_matrix`` and
    the entries of ``contributions``) is an independent, read-only
    ``float64`` copy owned by the DTO; in-place mutation of either the copy
    or the canonical arrays cannot silently invalidate stored experiments.

    All arrays are reported together with the applied configuration
    (``encoding``, ``activation_name``, ``threshold``) and structure
    (``input_dimension``, ``output_dimension``, ``pair_count``) so an
    experiment can be described completely without recomputation.
    """

    memory_type: MemoryType
    encoding: Encoding
    activation_name: str
    threshold: float
    input_dimension: int
    output_dimension: int
    pair_count: int
    memory: AutoassociativeMemory | HeteroassociativeMemory
    weight_matrix: NDArray
    contributions: tuple[NDArray, ...] | None
    dataset_name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.memory_type, MemoryType):
            raise TypeError(
                "memory_type must be a MemoryType member; received "
                f"{type(self.memory_type).__name__}."
            )
        object.__setattr__(self, "encoding", normalized_encoding(self.encoding))
        object.__setattr__(self, "threshold", float(self.threshold))
        object.__setattr__(
            self, "weight_matrix", owned_readonly_float_array(self.weight_matrix)
        )
        if self.contributions is not None:
            object.__setattr__(
                self,
                "contributions",
                tuple(
                    owned_readonly_float_array(item) for item in self.contributions
                ),
            )

    @property
    def recalled_with(self) -> AutoassociativeMemory | HeteroassociativeMemory:
        """Alias for :attr:`memory`, emphasizing its use for recall."""

        return self.memory


def build_training_result(
    memory: AutoassociativeMemory | HeteroassociativeMemory,
    *,
    threshold: float = ACTIVATION_THRESHOLD,
) -> TrainingResult:
    """Build a :class:`TrainingResult` from a trained algorithm memory.

    Reads the attribute names of the two supported memory families and
    performs no numerical work: dimensions, pair count, and encoding are
    copied from the memory object the algorithms produced.
    """

    if isinstance(memory, AutoassociativeMemory):
        memory_type = MemoryType.AUTOASSOCIATIVE
        input_dimension = memory.pattern_dimension
        output_dimension = memory.pattern_dimension
        pair_count = memory.pattern_count
    else:
        memory_type = MemoryType.HETEROASSOCIATIVE
        input_dimension = memory.source_dimension
        output_dimension = memory.target_dimension
        pair_count = memory.pair_count

    return TrainingResult(
        memory_type=memory_type,
        encoding=memory.encoding,
        activation_name=activation_name(memory.encoding),
        threshold=threshold,
        input_dimension=input_dimension,
        output_dimension=output_dimension,
        pair_count=pair_count,
        memory=memory,
        weight_matrix=memory.weight_matrix,
        contributions=memory.contributions,
        dataset_name=None,
    )
