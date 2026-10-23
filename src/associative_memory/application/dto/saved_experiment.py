"""Saved-experiment DTOs for application-layer persistence.

These DTOs describe what an experiment history entry contains (metadata
and numerical arrays) independently of any storage format. The JSON/NumPy
manifest lives in the infrastructure layer; views and view models only
ever see the objects defined here.

Portability policy
------------------
Original images are **not copied** into saved experiments. Only the exact
encoded numerical patterns and the complete preprocessing configuration
are stored; the file *names* of the original sources are recorded as
references. A missing external image file therefore never invalidates a
stored experiment's numerical results — every recalled value can still be
reproduced from the stored numbers, and only the original-image preview
may be unavailable.

Schema
------
``SCHEMA_VERSION`` is the version of the on-disk experiment manifest this
application writes and reads. Loaders must reject documents with unknown
versions instead of guessing. ``APPLICATION_VERSION`` identifies the
application build that produced the file and is kept in sync with the
project version declared in ``pyproject.toml``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from associative_memory.algorithms.encoding import Encoding
from associative_memory.application.dto._arrays import (
    normalized_encoding,
    owned_readonly_float_array,
)
from associative_memory.application.dto.recall_result import (
    COMPARISON_BASIS_EXPECTED,
    COMPARISON_BASIS_INPUT,
)
from associative_memory.application.dto.training_result import (
    MemoryType,
    TrainingResult,
)
from associative_memory.infrastructure.image_preprocessor import (
    ForegroundPolarity,
    FlatteningOrder,
    GrayscaleMethod,
    PreprocessingConfig,
    ResizeMode,
)

#: Schema version this application writes and reads (see module docstring).
SCHEMA_VERSION = 1

#: Application version recorded in every saved experiment. Keep in sync
#: with the ``version`` declared in ``pyproject.toml``.
APPLICATION_VERSION = "0.1.0"

#: Default experiment name when the user chooses not to name one.
DEFAULT_EXPERIMENT_NAME = "Untitled experiment"

#: Maximum stored experiment name length (characters).
MAX_EXPERIMENT_NAME_LENGTH = 120

#: Valid comparison-basis values for a stored recall record.
STORED_COMPARISON_BASES: tuple[str, ...] = (
    COMPARISON_BASIS_EXPECTED,
    COMPARISON_BASIS_INPUT,
)


@dataclass(frozen=True)
class PreprocessingSettings:
    """An exact snapshot of one preprocessing configuration.

    Enum-like fields are stored as their canonical string values and
    :meth:`to_config` rebuilds the infrastructure
    :class:`PreprocessingConfig` from them, so restoring a configuration
    reproduces the identical grid, threshold, polarity, resizing,
    flattening, and encoding settings.
    """

    width: int
    height: int
    grayscale_method: str
    resampling: str
    threshold: int
    polarity: str
    encoding: Encoding
    resize_mode: str
    flattening_order: str

    @classmethod
    def from_config(cls, config: PreprocessingConfig) -> PreprocessingSettings:
        """Capture ``config`` into a serializable settings snapshot."""

        return cls(
            width=int(config.width),
            height=int(config.height),
            grayscale_method=config.grayscale_method.value,
            resampling=config.resampling.name,
            threshold=int(config.threshold),
            polarity=config.polarity.value,
            encoding=normalized_encoding(config.encoding),
            resize_mode=config.resize_mode.value,
            flattening_order=config.flattening_order.value,
        )

    def to_config(self) -> PreprocessingConfig:
        """Rebuild the exact preprocessing configuration from this snapshot.

        Raises:
            TypeError: If the resampling-filter name is unknown to Pillow.
            ValueError: If any stored enum value or dimension is invalid.
        """

        try:
            resampling = getattr(Image.Resampling, self.resampling)
        except AttributeError as exc:
            raise TypeError(
                f"Unknown resampling filter {self.resampling!r} in stored "
                "preprocessing settings."
            ) from exc
        return PreprocessingConfig(
            width=self.width,
            height=self.height,
            grayscale_method=GrayscaleMethod(self.grayscale_method),
            resampling=resampling,
            threshold=self.threshold,
            polarity=ForegroundPolarity(self.polarity),
            encoding=self.encoding,
            resize_mode=ResizeMode(self.resize_mode),
            flattening_order=FlatteningOrder(self.flattening_order),
        )


@dataclass(frozen=True)
class StoredTraining:
    """Training metadata of one saved experiment."""

    memory_type: MemoryType
    encoding: Encoding
    activation_name: str
    threshold: float
    input_dimension: int
    output_dimension: int
    pair_count: int
    dataset_name: str | None = None
    contribution_count: int | None = None


@dataclass(frozen=True)
class StoredComparison:
    """Stored recall-comparison metrics (copied from a recall result)."""

    element_count: int
    mismatch_count: int
    match_ratio: float
    exact_match: bool


@dataclass(frozen=True)
class StoredRecallRecord:
    """Metadata of one stored recall operation (arrays live separately)."""

    comparison_basis: str | None
    warnings: tuple[str, ...] = ()
    input_source_file: str | None = None
    target_source_file: str | None = None
    comparison: StoredComparison | None = None


@dataclass(frozen=True)
class SavedExperiment:
    """The complete, storable numerical content of one experiment.

    Ownership policy
    ----------------
    Every array is an independent, read-only ``float64`` copy owned by the
    DTO, exactly like the other application result DTOs. ``experiment_id``
    is ``None`` until the repository assigns it during a save.
    """

    training: StoredTraining
    name: str
    weight_matrix: NDArray[np.float64]
    preprocessing: PreprocessingSettings | None = None
    target_preprocessing: PreprocessingSettings | None = None
    recall: StoredRecallRecord | None = None
    contributions: tuple[NDArray[np.float64], ...] | None = None
    recall_input: NDArray[np.float64] | None = None
    recall_net_input: NDArray[np.float64] | None = None
    recall_output: NDArray[np.float64] | None = None
    recall_expected: NDArray[np.float64] | None = None
    experiment_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "training", _validated_training(self.training)
        )
        object.__setattr__(self, "name", _validated_name(self.name))
        object.__setattr__(
            self, "weight_matrix", owned_readonly_float_array(self.weight_matrix)
        )
        if self.contributions is not None:
            object.__setattr__(
                self,
                "contributions",
                tuple(owned_readonly_float_array(item) for item in self.contributions),
            )
        for field in (
            "recall_input",
            "recall_net_input",
            "recall_output",
            "recall_expected",
        ):
            value = getattr(self, field)
            object.__setattr__(
                self, field, None if value is None else owned_readonly_float_array(value)
            )
        if self.recall is not None:
            object.__setattr__(self, "recall", _validated_recall_record(self.recall))


@dataclass(frozen=True)
class SavedExperimentSummary:
    """History-list entry built from an experiment manifest (no arrays).

    ``readable`` is ``False`` for on-disk entries whose manifest could not
    be validated; unknown fields are ``None`` and ``unreadable_reason``
    explains the failure. Unreadable entries stay visible in the history
    so they can still be deleted deliberately.
    """

    experiment_id: str
    name: str
    readable: bool = True
    unreadable_reason: str | None = None
    schema_version: int | None = None
    created_at: str | None = None
    application_version: str | None = None
    memory_type: str | None = None
    encoding: str | None = None
    activation_name: str | None = None
    threshold: float | None = None
    input_dimension: int | None = None
    output_dimension: int | None = None
    pair_count: int | None = None
    dataset_name: str | None = None
    weight_shape: tuple[int, int] | None = None
    contribution_count: int | None = None
    preprocessing: PreprocessingSettings | None = None
    recall_present: bool | None = None
    recall_basis: str | None = None
    recall_exact_match: bool | None = None
    recall_mismatch_count: int | None = None
    recall_element_count: int | None = None

    @property
    def display_name(self) -> str:
        """Name shown in the history list, with unreadable fallback."""

        if self.readable:
            return self.name
        return f"(unreadable experiment) {self.experiment_id[:8]}…"


@dataclass(frozen=True)
class LoadedExperiment:
    """A fully loaded experiment, ready for inspection and reuse.

    ``training`` wraps the reconstructed algorithm memory object and the
    stored weight matrix so the loaded model can be registered with the
    recall workspace and run inference again without retraining.
    """

    summary: SavedExperimentSummary
    name: str
    training: TrainingResult
    preprocessing: PreprocessingSettings | None = None
    target_preprocessing: PreprocessingSettings | None = None
    recall_record: StoredRecallRecord | None = None


def _validated_name(name: object) -> str:
    """Validate a saved experiment name."""

    if isinstance(name, bool) or not isinstance(name, str):
        raise TypeError(
            "Experiment name must be a string; received "
            f"{type(name).__name__}."
        )
    stripped = name.strip()
    if not stripped:
        raise ValueError("Experiment name must not be empty.")
    if len(stripped) > MAX_EXPERIMENT_NAME_LENGTH:
        raise ValueError(
            "Experiment name must not exceed "
            f"{MAX_EXPERIMENT_NAME_LENGTH} characters; received length "
            f"{len(stripped)}."
        )
    return stripped


def _validated_training(training: object) -> StoredTraining:
    """Validate the training metadata of a saved experiment."""

    if not isinstance(training, StoredTraining):
        raise TypeError(
            "training must be a StoredTraining; received "
            f"{type(training).__name__}."
        )
    if not isinstance(training.memory_type, MemoryType):
        raise TypeError(
            "memory_type must be a MemoryType member; received "
            f"{type(training.memory_type).__name__}."
        )
    object.__setattr__(
        training, "encoding", normalized_encoding(training.encoding)
    )
    for field_name in (
        "input_dimension",
        "output_dimension",
        "pair_count",
    ):
        value = getattr(training, field_name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"{field_name} must be an integer; received "
                f"{type(value).__name__}."
            )
        if value < 1:
            raise ValueError(f"{field_name} must be at least 1; received {value}.")
    if training.contribution_count is not None:
        count = training.contribution_count
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(
                "contribution_count must be a non-negative integer or None; "
                f"received {count!r}."
            )
    return training


def _validated_recall_record(record: object) -> StoredRecallRecord:
    """Validate the stored-recall metadata for structural consistency."""

    if not isinstance(record, StoredRecallRecord):
        raise TypeError(
            "recall must be a StoredRecallRecord; received "
            f"{type(record).__name__}."
        )
    if record.comparison is None:
        if record.comparison_basis is not None:
            raise ValueError(
                "A stored recall record without comparison metrics must not "
                "declare a comparison basis."
            )
    else:
        if record.comparison_basis not in STORED_COMPARISON_BASES:
            raise ValueError(
                f"Unknown comparison basis {record.comparison_basis!r}; "
                f"expected one of {', '.join(STORED_COMPARISON_BASES)}."
            )
        comparison = record.comparison
        if not isinstance(comparison, StoredComparison):
            raise TypeError(
                "comparison must be a StoredComparison; received "
                f"{type(comparison).__name__}."
            )
        if (
            isinstance(comparison.element_count, bool)
            or not isinstance(comparison.element_count, int)
            or comparison.element_count < 1
        ):
            raise ValueError(
                "comparison.element_count must be a positive integer; "
                f"received {comparison.element_count!r}."
            )
        if (
            isinstance(comparison.mismatch_count, bool)
            or not isinstance(comparison.mismatch_count, int)
            or comparison.mismatch_count < 0
        ):
            raise ValueError(
                "comparison.mismatch_count must be a non-negative integer; "
                f"received {comparison.mismatch_count!r}."
            )
        if comparison.mismatch_count > comparison.element_count:
            raise ValueError(
                "comparison.mismatch_count must not exceed element_count "
                f"({comparison.mismatch_count} > {comparison.element_count})."
            )
        if (
            isinstance(comparison.match_ratio, bool)
            or not isinstance(comparison.match_ratio, (int, float))
            or not np.isfinite(float(comparison.match_ratio))
        ):
            raise ValueError(
                "comparison.match_ratio must be a finite number; received "
                f"{comparison.match_ratio!r}."
            )
        if not 0.0 <= float(comparison.match_ratio) <= 1.0:
            raise ValueError(
                "comparison.match_ratio must lie in [0, 1]; received "
                f"{comparison.match_ratio!r}."
            )
        if not isinstance(comparison.exact_match, bool):
            raise TypeError(
                "comparison.exact_match must be a bool; received "
                f"{type(comparison.exact_match).__name__}."
            )
        expected_exact = comparison.mismatch_count == 0
        if comparison.exact_match is not expected_exact:
            raise ValueError(
                "comparison.exact_match must be True exactly when "
                "mismatch_count is zero."
            )
    return record
