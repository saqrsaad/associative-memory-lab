"""Experiment report DTO summarizing a complete training and recall workflow."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from associative_memory.algorithms.encoding import Encoding
from associative_memory.application.dto._arrays import normalized_encoding
from associative_memory.application.dto.recall_result import RecallResult
from associative_memory.application.dto.training_result import (
    MemoryType,
    TrainingResult,
)

STATUS_COMPLETED = "completed"
DATA_SOURCE_VECTORS = "vectors"
DATA_SOURCE_IMAGES = "images"


@dataclass(frozen=True)
class ExperimentReport:
    """Consistent summary of one complete training-and-recall workflow.

    Ownership policy
    ----------------
    The report owns its DTO children (``TrainingResult``, ``RecallResult``)
    whose arrays are independent read-only copies; mutating any of them
    raises ``ValueError``. ``metadata`` is a defensive string mapping
    snapshot describing how the experiment can be reproduced (data source,
    pair count, and — for image experiments — the full preprocessing
    configuration). It is an immutable mapping view; the report never holds
    a reference to a caller-owned mutable dict.
    """

    name: str | None
    status: str
    memory_type: MemoryType
    encoding: Encoding
    data_source: str
    training: TrainingResult
    recall_results: tuple[RecallResult, ...]
    metadata: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.memory_type, MemoryType):
            raise TypeError(
                "memory_type must be a MemoryType member; received "
                f"{type(self.memory_type).__name__}."
            )
        object.__setattr__(self, "encoding", normalized_encoding(self.encoding))
        object.__setattr__(
            self, "recall_results", tuple(self.recall_results)
        )
        object.__setattr__(
            self, "metadata", MappingProxyType(dict(self.metadata))
        )

    @property
    def total_recall_count(self) -> int:
        """Number of recall queries performed in the workflow."""

        return len(self.recall_results)

    @property
    def compared_recall_count(self) -> int:
        """Number of recalls for which comparison data was available."""

        return sum(
            1 for result in self.recall_results if result.comparison is not None
        )

    @property
    def exact_recall_count(self) -> int:
        """Number of recalls meeting the exact recall criterion (M == 0)."""

        return sum(
            1
            for result in self.recall_results
            if result.comparison is not None and result.comparison.exact_match
        )

    @property
    def mismatched_recall_count(self) -> int:
        """Number of compared recalls that failed the exact criterion."""

        return self.compared_recall_count - self.exact_recall_count
