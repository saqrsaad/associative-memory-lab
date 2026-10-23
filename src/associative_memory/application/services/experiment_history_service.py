"""Experiment-history use cases: save, list, load, delete, and compare.

This service is the presentation layer's single access point for
experiment persistence. Views and view models call this service; they
never read or write JSON or NumPy files directly, and no persistence
logic lives in the presentation or algorithm layers.

Comparison policy
-----------------
A comparison is side-by-side with explicit notes. Numbers are only
presented as directly comparable when both experiments share the
relevant structure (same memory type, encoding, dimensions, matrix
shape, and preprocessing settings where needed). Incompatible values are
labeled "(not comparable)" with a note explaining why, never silently
equated.

Image portability
-----------------
Original images are not copied into saved experiments: the exact encoded
numerical patterns, the complete preprocessing configuration, and the
source file *names* are stored instead. A missing external image file
never invalidates stored numerical results.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from associative_memory.application.dto.experiment_comparison import (
    ComparisonRow,
    ExperimentComparison,
    WeightComparison,
)
from associative_memory.application.dto.saved_experiment import (
    DEFAULT_EXPERIMENT_NAME,
    LoadedExperiment,
    PreprocessingSettings,
    SavedExperiment,
    SavedExperimentSummary,
    StoredComparison,
    StoredRecallRecord,
    StoredTraining,
)
from associative_memory.application.dto.recall_result import RecallResult
from associative_memory.application.dto.training_result import TrainingResult
from associative_memory.application.interfaces.model_repository import (
    ModelRepository,
)
from associative_memory.infrastructure.experiment_repository import (
    build_default_repository,
)
from associative_memory.infrastructure.image_preprocessor import (
    PreprocessingConfig,
)

_LOGGER = logging.getLogger(__name__)

_NOT_STORED = "(not stored)"
_NOT_COMPARABLE = "(not comparable)"


class ExperimentHistoryService:
    """Coordinate saved-experiment persistence behind the repository interface."""

    def __init__(self, repository: ModelRepository | None = None) -> None:
        self._repository = (
            repository if repository is not None else build_default_repository()
        )

    @property
    def storage_location(self) -> Path | None:
        """Where saved experiments live, when the repository exposes it."""

        return self._repository.storage_location

    # ------------------------------------------------------------------ save

    def save_experiment(
        self,
        *,
        name: str,
        training_result: TrainingResult,
        preprocessing_config: PreprocessingConfig | None = None,
        target_preprocessing_config: PreprocessingConfig | None = None,
        recall_result: RecallResult | None = None,
        input_source_file: Path | str | None = None,
        target_source_file: Path | str | None = None,
    ) -> SavedExperimentSummary:
        """Compose and store one experiment; return its history summary.

        The record holds the trained weight matrix (and contributions when
        a training trace exists), the exact preprocessing configurations,
        and — when available — the most recent recall operation with its
        input, net input, output, expected pattern, and comparison
        metrics. Only one recall record per experiment is stored: the
        recall performed most recently at save time.

        Raises:
            ValueError: If a supplied name, configuration, or recall
                result does not match the trained model.
        """

        training = self._stored_training_from_result(training_result)
        preprocessing = self._settings_for(
            preprocessing_config,
            expected_dimension=training_result.input_dimension,
            label="preprocessing",
        )
        target_preprocessing = self._settings_for(
            target_preprocessing_config,
            expected_dimension=training_result.output_dimension,
            label="target preprocessing",
        )
        recall_record, recall_arrays = self._stored_recall_from_result(
            recall_result=recall_result,
            training_result=training_result,
            input_source_file=input_source_file,
            target_source_file=target_source_file,
        )

        saved = SavedExperiment(
            training=training,
            name=name if name and name.strip() else DEFAULT_EXPERIMENT_NAME,
            weight_matrix=training_result.weight_matrix,
            preprocessing=preprocessing,
            target_preprocessing=target_preprocessing,
            recall=recall_record,
            contributions=training_result.contributions,
            **recall_arrays,
        )

        summary = self._repository.save(saved)
        _LOGGER.info(
            "Saved experiment %s (%s) to experiment storage.",
            summary.name,
            summary.experiment_id,
        )
        return summary

    # ------------------------------------------------------------------ history

    def list_experiments(self) -> tuple[SavedExperimentSummary, ...]:
        """Return every stored experiment summary (newest first)."""

        return self._repository.list_experiments()

    def load_experiment(self, experiment_id: str) -> LoadedExperiment:
        """Load one experiment fully (arrays plus reconstructed model)."""

        return self._repository.load(experiment_id)

    def delete_experiment(self, experiment_id: str) -> None:
        """Delete one stored experiment (UI confirms before calling)."""

        self._repository.delete(experiment_id)
        _LOGGER.info("Deleted experiment %s.", experiment_id)

    # ------------------------------------------------------------------ comparison

    def compare_experiments(
        self, left_id: str, right_id: str
    ) -> ExperimentComparison:
        """Compare two saved experiments with explicit comparability notes."""

        if left_id == right_id:
            raise ValueError(
                "Select two different experiments to compare them."
            )
        left = self._repository.load(left_id)
        right = self._repository.load(right_id)
        return _build_comparison(left, right)

    # ------------------------------------------------------------------ record composition

    def _stored_training_from_result(
        self, result: TrainingResult
    ) -> StoredTraining:
        """Map a :class:`TrainingResult` into storable training metadata."""

        return StoredTraining(
            memory_type=result.memory_type,
            encoding=result.encoding,
            activation_name=result.activation_name,
            threshold=result.threshold,
            input_dimension=result.input_dimension,
            output_dimension=result.output_dimension,
            pair_count=result.pair_count,
            dataset_name=result.dataset_name,
            contribution_count=(
                None if result.contributions is None else len(result.contributions)
            ),
        )

    def _settings_for(
        self,
        config: PreprocessingConfig | None,
        *,
        expected_dimension: int,
        label: str,
    ) -> PreprocessingSettings | None:
        """Capture a preprocessing config that matches ``expected_dimension``."""

        if config is None:
            return None
        if config.width * config.height != expected_dimension:
            raise ValueError(
                f"The {label} configuration produces "
                f"{config.width * config.height} elements, which does not "
                f"match the trained pattern dimension {expected_dimension}."
            )
        return PreprocessingSettings.from_config(config)

    def _stored_recall_from_result(
        self,
        *,
        recall_result: RecallResult | None,
        training_result: TrainingResult,
        input_source_file: Path | str | None = None,
        target_source_file: Path | str | None = None,
    ) -> tuple[StoredRecallRecord | None, dict[str, object]]:
        """Map the most recent recall result into the storable record.

        Returns the record and a dict of array field names for
        :class:`SavedExperiment` construction. A ``ValueError`` is raised
        when the recall result was produced by a different model, so a
        saved experiment never mixes numbers from two memories.
        """

        if recall_result is None:
            return None, {}

        if recall_result.memory_type is not training_result.memory_type:
            raise ValueError(
                "The recall result does not match the trained model "
                "(memory type differs); refusing to store inconsistent "
                "numbers together."
            )
        if recall_result.encoding is not training_result.encoding:
            raise ValueError(
                "The recall result does not match the trained model "
                "(encoding differs); refusing to store inconsistent numbers "
                "together."
            )
        if (
            recall_result.input_dimension != training_result.input_dimension
            or recall_result.output_dimension != training_result.output_dimension
        ):
            raise ValueError(
                "The recall result does not match the trained model "
                "(dimensions differ); refusing to store inconsistent numbers "
                "together."
            )

        comparison = None
        if recall_result.comparison is not None:
            metrics = recall_result.comparison
            comparison = StoredComparison(
                element_count=metrics.element_count,
                mismatch_count=metrics.mismatch_count,
                match_ratio=metrics.match_ratio,
                exact_match=metrics.exact_match,
            )
        record = StoredRecallRecord(
            comparison_basis=recall_result.comparison_basis,
            warnings=tuple(recall_result.warnings),
            input_source_file=_source_file_name(input_source_file),
            target_source_file=_source_file_name(target_source_file),
            comparison=comparison,
        )
        arrays: dict[str, object] = {
            "recall_input": recall_result.input_pattern,
            "recall_net_input": recall_result.net_input,
            "recall_output": recall_result.output_pattern,
            "recall_expected": recall_result.expected_pattern,
        }
        if recall_result.expected_pattern is None:
            arrays.pop("recall_expected")
        return record, arrays


def _source_file_name(source: Path | str | None) -> str | None:
    """Reduce a source reference to its file name (no private paths)."""

    if source is None:
        return None
    return Path(source).name


def _display(value: object) -> str:
    """Presentation-ready display of one stored value or ``None``."""

    if value is None:
        return _NOT_STORED
    return str(value)


def _comparison_basis_text(basis: str | None) -> str:
    """Human-readable text for a stored comparison basis."""

    if basis is None:
        return _NOT_STORED
    if basis == "expected_pattern":
        return "Expected target"
    if basis == "input_pattern":
        return "Original query pattern"
    return basis


def _build_comparison(
    left: LoadedExperiment, right: LoadedExperiment
) -> ExperimentComparison:
    """Build the structured comparison of two loaded experiments."""

    left_summary = left.summary
    right_summary = right.summary
    rows: list[ComparisonRow] = []
    notes: list[str] = []

    def add_row(label: str, left_value: str, right_value: str, note: str = "") -> None:
        rows.append(
            ComparisonRow(
                label=label, left=left_value, right=right_value, note=note
            )
        )

    def add_note(text: str) -> None:
        if text not in notes:
            notes.append(text)

    add_row("Experiment name", left.name, right.name)
    add_row(
        "Experiment ID",
        left_summary.experiment_id,
        right_summary.experiment_id,
    )
    add_row("Created at", _display(left_summary.created_at),
            _display(right_summary.created_at))
    add_row("Application version", _display(left_summary.application_version),
            _display(right_summary.application_version))

    memory_equal = left_summary.memory_type == right_summary.memory_type
    add_row(
        "Memory type",
        _display(left_summary.memory_type),
        _display(right_summary.memory_type),
        "" if memory_equal else (
            "Memory types differ: the two models associate different kinds "
            "of patterns, and their results are not directly comparable."
        ),
    )
    if not memory_equal:
        add_note(
            "Memory types differ; recall metrics and weight elements of "
            "these two experiments belong to different association spaces."
        )

    encoding_equal = left_summary.encoding == right_summary.encoding
    add_row(
        "Encoding",
        _display(left_summary.encoding),
        _display(right_summary.encoding),
        "" if encoding_equal else (
            "Encodings differ: encoded element values are not directly "
            "comparable."
        ),
    )

    add_row(
        "Activation rule",
        (
            f"{left_summary.activation_name} "
            f"(threshold {_format_number(left_summary.threshold)})"
            if left_summary.activation_name is not None
            else _NOT_STORED
        ),
        (
            f"{right_summary.activation_name} "
            f"(threshold {_format_number(right_summary.threshold)})"
            if right_summary.activation_name is not None
            else _NOT_STORED
        ),
        "" if (
            left_summary.threshold is not None
            and right_summary.threshold is not None
            and float(left_summary.threshold) == float(right_summary.threshold)
        ) else "The stored activation thresholds differ.",
    )
    pairs_note = ""
    if (
        left_summary.pair_count is not None
        and right_summary.pair_count is not None
        and left_summary.pair_count != right_summary.pair_count
    ):
        pairs_note = "Different numbers of stored training pairs."
    add_row(
        "Training pairs",
        _display(left_summary.pair_count),
        _display(right_summary.pair_count),
        pairs_note,
    )
    add_row(
        "Dataset reference",
        _display(left_summary.dataset_name),
        _display(right_summary.dataset_name),
    )

    dims_note = ""
    if (
        left_summary.input_dimension != right_summary.input_dimension
        or left_summary.output_dimension != right_summary.output_dimension
    ):
        dims_note = (
            "Dimensions differ; direct element-by-element comparison is "
            "unavailable."
        )
    add_row("Input dimension (n)", _display(left_summary.input_dimension),
            _display(right_summary.input_dimension), dims_note)
    add_row("Output dimension (m)", _display(left_summary.output_dimension),
            _display(right_summary.output_dimension), dims_note)

    _add_preprocessing_rows(add_row, add_note, left_summary, right_summary)

    left_shape = left_summary.weight_shape
    right_shape = right_summary.weight_shape
    shape_equal = left_shape == right_shape and left_shape is not None
    add_row(
        "Weight-matrix shape",
        _shape_text(left_shape),
        _shape_text(right_shape),
        "" if shape_equal else (
            "Weight-matrix shapes differ; a direct element-by-element "
            "weight comparison is unavailable."
        ),
    )

    weight_comparison: WeightComparison | None = None
    if (
        shape_equal
        and memory_equal
        and encoding_equal
        and left.training.weight_matrix is not None
        and right.training.weight_matrix is not None
    ):
        weight_comparison = _compare_weight_matrices(
            left.training.weight_matrix, right.training.weight_matrix
        )
        add_row(
            "Identical weight matrices",
            "yes" if weight_comparison.identical else "no",
            "yes" if weight_comparison.identical else "no",
            "Both models were built with the same structure; this compares "
            "their stored weights element by element.",
        )
        add_row(
            "Differing weight elements",
            (
                "0"
                if weight_comparison.identical
                else (
                    f"{weight_comparison.differing_count} of "
                    f"{weight_comparison.element_count}"
                )
            ),
            (
                "0"
                if weight_comparison.identical
                else (
                    f"{weight_comparison.differing_count} of "
                    f"{weight_comparison.element_count}"
                )
            ),
        )
        add_row(
            "Largest absolute weight difference",
            _format_number(weight_comparison.max_abs_difference),
            _format_number(weight_comparison.max_abs_difference),
        )
    elif shape_equal and not memory_equal:
        add_note(
            "Weight matrices have the same shape but different memory "
            "types; element-level comparison would equate numbers from "
            "different association spaces and was not performed."
        )

    _add_recall_rows(add_row, add_note, left_summary, right_summary)

    return ExperimentComparison(
        left=left_summary,
        right=right_summary,
        rows=tuple(rows),
        weight_comparison=weight_comparison,
        notes=tuple(notes),
    )


def _add_preprocessing_rows(
    add_row,
    add_note,
    left_summary: SavedExperimentSummary,
    right_summary: SavedExperimentSummary,
) -> None:
    """Add preprocessing rows for two summaries."""

    left_settings = left_summary.preprocessing
    right_settings = right_summary.preprocessing
    if left_settings is None and right_settings is None:
        add_row(
            "Stored preprocessing",
            _NOT_STORED,
            _NOT_STORED,
            "Neither experiment stores preprocessing settings (e.g. "
            "vector-trained data).",
        )
        return

    if (left_settings is None) != (right_settings is None):
        add_row(
            "Stored preprocessing",
            _settings_brief(left_settings),
            _settings_brief(right_settings),
            "One experiment has no stored preprocessing settings; image "
            "inputs for these experiments are not produced the same way.",
        )
        add_note(
            "Preprocessing availability differs; preprocessing-dependent "
            "values are not directly comparable."
        )
        return

    if left_settings is None or right_settings is None:
        return  # unreachable given the guards above; total-function safeguard
    settings_equal = left_settings == right_settings
    add_row(
        "Pattern grid",
        f"{left_settings.width} x {left_settings.height}",
        f"{right_settings.width} x {right_settings.height}",
        "",
    )
    add_row(
        "Preprocessing threshold",
        str(left_settings.threshold),
        str(right_settings.threshold),
    )
    add_row(
        "Foreground polarity",
        left_settings.polarity,
        right_settings.polarity,
    )
    add_row(
        "Grayscale method",
        left_settings.grayscale_method,
        right_settings.grayscale_method,
    )
    add_row(
        "Resampling filter",
        left_settings.resampling,
        right_settings.resampling,
    )
    add_row(
        "Resize mode",
        left_settings.resize_mode,
        right_settings.resize_mode,
    )
    add_row(
        "Flattening order",
        left_settings.flattening_order,
        right_settings.flattening_order,
    )
    if not settings_equal:
        add_note(
            "The two experiments were preprocessed with different settings; "
            "their stored patterns are not produced from images the same "
            "way."
        )


def _add_recall_rows(
    add_row,
    add_note,
    left_summary: SavedExperimentSummary,
    right_summary: SavedExperimentSummary,
) -> None:
    """Add stored-recall rows for two summaries with comparability notes."""

    left_present = left_summary.recall_present
    right_present = right_summary.recall_present
    add_row(
        "Stored recall record",
        "present" if left_present else _NOT_STORED,
        "present" if right_present else _NOT_STORED,
        "" if (left_present and right_present) else (
            "Stored recall metrics are only available when both "
            "experiments have a stored recall record."
        ),
    )
    if not (left_present and right_present):
        return

    add_row(
        "Recall comparison basis",
        _comparison_basis_text(left_summary.recall_basis),
        _comparison_basis_text(right_summary.recall_basis),
    )

    metrics_comparable = (
        left_summary.recall_mismatch_count is not None
        and right_summary.recall_mismatch_count is not None
        and left_summary.recall_element_count
        == right_summary.recall_element_count
        and left_summary.recall_basis == right_summary.recall_basis
    )
    if not metrics_comparable:
        add_row(
            "Stored recall mismatch count",
            _NOT_COMPARABLE,
            _NOT_COMPARABLE,
            (
                "The two stored recalls compared different element counts "
                "or used different comparison bases; the metrics are not "
                "directly equivalent."
            ),
        )
        return

    add_row(
        "Stored recall mismatch count",
        _display(left_summary.recall_mismatch_count),
        _display(right_summary.recall_mismatch_count),
        "The stored recalls may have used different query patterns; the "
        "values are side-by-side reports, not one shared benchmark.",
    )
    left_ratio = _ratio_from_counts(
        left_summary.recall_element_count,
        left_summary.recall_mismatch_count,
    )
    right_ratio = _ratio_from_counts(
        right_summary.recall_element_count,
        right_summary.recall_mismatch_count,
    )
    add_row(
        "Stored recall match ratio",
        _format_number(left_ratio),
        _format_number(right_ratio),
    )


def _ratio_from_counts(
    element_count: int | None, mismatch_count: int | None
) -> float | None:
    """Recompute the stored match ratio from summary counts."""

    if element_count is None or mismatch_count is None:
        return None
    return (element_count - mismatch_count) / element_count


def _compare_weight_matrices(
    left_matrix: NDArray[np.float64],
    right_matrix: NDArray[np.float64],
) -> WeightComparison:
    """Element-by-element comparison of two equal-shape weight matrices."""

    difference = np.subtract(
        np.asarray(left_matrix, dtype=np.float64),
        np.asarray(right_matrix, dtype=np.float64),
    )
    abs_difference = np.abs(difference)
    element_count = int(difference.size)
    differing_count = int(np.count_nonzero(abs_difference > 0.0))
    max_abs = float(abs_difference.max()) if element_count else 0.0
    return WeightComparison(
        element_count=element_count,
        differing_count=differing_count,
        max_abs_difference=max_abs,
        identical=differing_count == 0,
    )


def _shape_text(shape: tuple[int, int] | None) -> str:
    """Presentation-ready shape text."""

    if shape is None:
        return _NOT_STORED
    return f"{shape[0]} x {shape[1]}"


def _settings_brief(settings: PreprocessingSettings | None) -> str:
    """One-line summary of stored preprocessing settings (or absence)."""

    if settings is None:
        return _NOT_STORED
    return f"{settings.width} x {settings.height} grid"


def _format_number(value: float | None) -> str:
    """Presentation-ready numeric formatting with a six-digit bound."""

    if value is None:
        return _NOT_STORED
    return f"{float(value):.6g}"
