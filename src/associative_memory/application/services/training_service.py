"""Training use cases for the application layer.

The service validates a training dataset at the use-case boundary and then
invokes the existing algorithm-package implementations. No numerical
formula is implemented here: validation reuses ``algorithms.validate_vector``
and training reuses ``algorithms.train_autoassociative`` /
``algorithms.train_heteroassociative``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from associative_memory.algorithms.autoassociative import (
    AutoassociativeMemory,
    train_autoassociative,
)
from associative_memory.algorithms.encoding import Encoding, coerce_encoding, validate_vector
from associative_memory.algorithms.heteroassociative import (
    HeteroassociativeMemory,
    train_heteroassociative,
)
from associative_memory.application.dto.training_result import (
    MemoryType,
    TrainingResult,
    build_training_result,
)

_MEMORY_TYPE_OPTIONS = ", ".join(member.value for member in MemoryType)


class TrainingError(Exception):
    """Raised when a training use case cannot be performed.

    The message describes the problem at the use-case level; the original
    lower-level exception remains available as ``__cause__`` for logging.
    """


@dataclass(frozen=True)
class TrainingDataset:
    """Raw training data for one experiment, owned by its creator.

    The arrays referenced here are never mutated by the application: they
    are validated and copied by the algorithm package. Construction is
    intentionally unvalidated; incompatible patterns are rejected by the
    training service before any training begins.

    For :attr:`MemoryType.AUTOASSOCIATIVE` every pattern is associated with
    itself (the ``patterns`` classmethod stores each entry twice); for
    :attr:`MemoryType.HETEROASSOCIATIVE` each pair holds a source and a
    distinct target pattern.
    """

    patterns: tuple[tuple[ArrayLike, ArrayLike], ...]
    memory_type: MemoryType
    name: str | None = None

    @classmethod
    def from_patterns(
        cls,
        patterns: Iterable[ArrayLike],
        *,
        name: str | None = None,
    ) -> TrainingDataset:
        """Build an autoassociative dataset from stored patterns."""

        materialized = tuple(patterns)
        return cls(
            patterns=tuple((pattern, pattern) for pattern in materialized),
            memory_type=MemoryType.AUTOASSOCIATIVE,
            name=name,
        )

    @classmethod
    def from_pairs(
        cls,
        pairs: Iterable[tuple[ArrayLike, ArrayLike]],
        *,
        name: str | None = None,
    ) -> TrainingDataset:
        """Build a heteroassociative dataset from source/target pairs."""

        return cls(
            patterns=tuple(pairs),
            memory_type=MemoryType.HETEROASSOCIATIVE,
            name=name,
        )


class TrainingService:
    """Coordinate dataset validation and algorithm training."""

    def train(
        self,
        dataset: TrainingDataset,
        *,
        encoding: Encoding | str,
        trace: bool = False,
        memory_type: MemoryType | str | None = None,
    ) -> TrainingResult:
        """Train one associative memory from ``dataset``.

        Every pattern is validated (dimension consistency, encoding values,
        finiteness) before training begins; no incompatible example is ever
        silently skipped. ``memory_type`` optionally restates the requested
        memory mode; when given, it must match the dataset mode.

        Raises:
            TrainingError: For an empty dataset, an unknown memory type or
                encoding name, a self-association mismatch in an
                autoassociative dataset, or any invalid pattern.
        """

        normalized_memory_type = _coerce_memory_type(
            memory_type if memory_type is not None else dataset.memory_type
        )
        if normalized_memory_type is not dataset.memory_type:
            raise TrainingError(
                f"Requested memory type {normalized_memory_type.value!r} does "
                f"not match the dataset memory type {dataset.memory_type.value!r}."
            )

        try:
            normalized_encoding = coerce_encoding(encoding)
        except ValueError as exc:
            raise TrainingError(str(exc)) from exc

        self._validate_dataset(dataset, encoding=normalized_encoding)

        try:
            if normalized_memory_type is MemoryType.AUTOASSOCIATIVE:
                trained: AutoassociativeMemory | HeteroassociativeMemory = (
                    train_autoassociative(
                        (source for source, _ in dataset.patterns),
                        encoding=normalized_encoding,
                        trace=trace,
                    )
                )
            else:
                trained = train_heteroassociative(
                    dataset.patterns,
                    encoding=normalized_encoding,
                    trace=trace,
                )
        except (TypeError, ValueError) as exc:
            raise TrainingError(str(exc)) from exc

        result = build_training_result(trained)
        object.__setattr__(result, "dataset_name", dataset.name)
        return result

    def _validate_dataset(
        self,
        dataset: TrainingDataset,
        *,
        encoding: Encoding,
    ) -> None:
        """Reject incompatible patterns before training begins.

        Reuses ``algorithms.validate_vector`` on every source and target;
        errors are converted into :class:`TrainingError` while retaining the
        original exception for diagnostic logging.
        """

        if len(dataset.patterns) == 0:
            kind = (
                "pattern" if dataset.memory_type is MemoryType.AUTOASSOCIATIVE else "pair"
            )
            raise TrainingError(
                f"Training dataset must contain at least one {kind}."
            )

        source_dimension: int | None = None
        target_dimension: int | None = None

        for index, pair in enumerate(dataset.patterns):
            try:
                source, target = pair
                source_vector = validate_vector(
                    source,
                    name=f"pairs[{index}].source",
                    encoding=encoding,
                )
                target_vector = validate_vector(
                    target,
                    name=f"pairs[{index}].target",
                    encoding=encoding,
                )
            except (TypeError, ValueError) as exc:
                raise TrainingError(str(exc)) from exc

            if dataset.memory_type is MemoryType.AUTOASSOCIATIVE and not np.array_equal(
                source_vector, target_vector
            ):
                raise TrainingError(
                    f"pairs[{index}] is not self-associated; an autoassociative "
                    f"dataset stores each pattern with itself."
                )

            if source_dimension is None:
                source_dimension = int(source_vector.shape[0])
                target_dimension = int(target_vector.shape[0])
            else:
                if int(source_vector.shape[0]) != source_dimension:
                    raise TrainingError(
                        f"pairs[{index}].source has length "
                        f"{source_vector.shape[0]}, but earlier pairs use "
                        f"length {source_dimension}."
                    )
                if int(target_vector.shape[0]) != target_dimension:
                    raise TrainingError(
                        f"pairs[{index}].target has length "
                        f"{target_vector.shape[0]}, but earlier pairs use "
                        f"length {target_dimension}."
                    )


def _coerce_memory_type(value: MemoryType | str) -> MemoryType:
    """Return the :class:`MemoryType` member for ``value``."""

    if isinstance(value, MemoryType):
        return value
    if isinstance(value, str):
        try:
            return MemoryType(value)
        except ValueError:
            pass
    raise TrainingError(
        f"Unknown memory type {value!r}; expected one of: {_MEMORY_TYPE_OPTIONS}."
    )
