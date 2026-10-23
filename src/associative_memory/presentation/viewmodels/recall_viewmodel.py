"""Recall workspace view model.

Coordinates model selection, input selection, preprocessing review, recall
execution, numerical trace construction, and result inspection between the
recall view and the application services.

The view model contains no learning or inference equations. Recall is
delegated to :class:`associative_memory.application.services.recall_service.RecallService`;
this module only resolves compatible inputs and targets, reshapes already
computed vectors for display, and formats a trace from actual result values.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QObject, Signal

from associative_memory.application.dto.recall_result import (
    COMPARISON_BASIS_EXPECTED,
    COMPARISON_BASIS_INPUT,
    RecallResult,
)
from associative_memory.application.dto.saved_experiment import (
    SavedExperimentSummary,
)
from associative_memory.application.dto.training_result import (
    MemoryType,
    TrainingResult,
)
from associative_memory.application.interfaces.model_repository import (
    ExperimentRepositoryError,
)
from associative_memory.application.services.experiment_history_service import (
    ExperimentHistoryService,
)
from associative_memory.application.services.image_service import (
    Encoding,
    ImageLoadError,
    ImageService,
    PreprocessingConfig,
    PreprocessingResult,
)
from associative_memory.application.services.recall_service import (
    RecallError,
    RecallService,
)
from associative_memory.presentation.viewmodels.dataset_viewmodel import (
    DatasetEntry,
    DatasetViewModel,
)
from associative_memory.presentation.viewmodels.training_viewmodel import (
    TrainingViewModel,
)
from associative_memory.presentation.widgets.status import StatusKind
from associative_memory.presentation.widgets.step_navigator import TraceStep

_LOGGER = logging.getLogger(__name__)

#: Number of values shown in compact trace/vector summaries.
SUMMARY_VALUE_LIMIT = 16


class RecallState(StrEnum):
    """Explicit recall lifecycle states."""

    NOT_READY = "not_ready"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class RecallModelOption:
    """One trained model available for recall in this session."""

    model_id: str
    label: str
    result: TrainingResult
    preprocessing_config: PreprocessingConfig | None = None
    target_preprocessing_config: PreprocessingConfig | None = None
    origin_entry_id: str | None = None

    @property
    def memory_type(self) -> MemoryType:
        return self.result.memory_type

    @property
    def encoding(self) -> Encoding:
        return self.result.encoding

    @property
    def input_dimension(self) -> int:
        return self.result.input_dimension

    @property
    def output_dimension(self) -> int:
        return self.result.output_dimension

    @property
    def weight_matrix(self) -> NDArray[np.float64]:
        return self.result.weight_matrix


@dataclass(frozen=True)
class RecallInputOption:
    """A query pattern that is dimensionally and semantically compatible."""

    input_id: str
    label: str
    vector: NDArray[np.float64]
    encoding: Encoding
    preprocessing: PreprocessingResult | None = None
    source_path: Path | None = None
    dataset_entry_id: str | None = None

    def __post_init__(self) -> None:
        vector = np.array(self.vector, dtype=np.float64, copy=True)
        vector.setflags(write=False)
        object.__setattr__(self, "vector", vector)

    @property
    def vector_length(self) -> int:
        return int(self.vector.shape[0])


@dataclass(frozen=True)
class RecallTargetOption:
    """An optional expected target used for a single recall comparison."""

    target_id: str
    label: str
    vector: NDArray[np.float64]
    encoding: Encoding
    preprocessing: PreprocessingResult | None = None
    source_path: Path | None = None
    dataset_entry_id: str | None = None

    def __post_init__(self) -> None:
        vector = np.array(self.vector, dtype=np.float64, copy=True)
        vector.setflags(write=False)
        object.__setattr__(self, "vector", vector)


class RecallViewModel(QObject):
    """Coordinate the recall workflow for one recall workspace."""

    models_changed = Signal()
    inputs_changed = Signal()
    selection_changed = Signal()
    state_changed = Signal(object)  # RecallState
    result_ready = Signal(object)  # RecallResult
    trace_ready = Signal(list)  # list[TraceStep]
    experiment_saved = Signal(object)  # SavedExperimentSummary
    error_occurred = Signal(str)
    status_message = Signal(str, StatusKind)

    def __init__(
        self,
        dataset_view_model: DatasetViewModel | None = None,
        training_view_model: TrainingViewModel | None = None,
        recall_service: RecallService | None = None,
        image_service: ImageService | None = None,
        history_service: ExperimentHistoryService | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)

        self._training_view_model = training_view_model
        if training_view_model is not None:
            resolved_dataset = training_view_model.dataset_view_model
            if (
                dataset_view_model is not None
                and dataset_view_model is not resolved_dataset
            ):
                raise ValueError(
                    "dataset_view_model must be the training view model's "
                    "dataset view model."
                )
            self._dataset_view_model = resolved_dataset
        else:
            self._dataset_view_model = dataset_view_model or DatasetViewModel()

        self._recall_service = recall_service or RecallService()
        self._image_service = image_service or ImageService()
        self._history_service = (
            history_service
            if history_service is not None
            else ExperimentHistoryService()
        )

        self._state = RecallState.NOT_READY
        self._models: list[RecallModelOption] = []
        self._inputs: list[RecallInputOption] = []
        self._targets: list[RecallTargetOption] = []
        self._imported_inputs: dict[str, RecallInputOption] = {}
        self._selected_model_id: str | None = None
        self._selected_input_id: str | None = None
        self._selected_target_id: str | None = None
        self._current_result: RecallResult | None = None
        self._error_message: str | None = None
        self._model_counter = 0

        self._dataset_view_model.entries_changed.connect(self._rebuild_inputs)
        self._dataset_view_model.preview_changed.connect(self._rebuild_inputs)

        if training_view_model is not None:
            training_view_model.result_ready.connect(
                self._on_training_result_ready
            )
            if training_view_model.current_result is not None:
                self.add_training_result(training_view_model.current_result)

    # ------------------------------------------------------------------ properties

    @property
    def state(self) -> RecallState:
        return self._state

    @property
    def models(self) -> tuple[RecallModelOption, ...]:
        return tuple(self._models)

    @property
    def inputs(self) -> tuple[RecallInputOption, ...]:
        return tuple(self._inputs)

    @property
    def targets(self) -> tuple[RecallTargetOption, ...]:
        return tuple(self._targets)

    @property
    def selected_model(self) -> RecallModelOption | None:
        return self._find_model(self._selected_model_id)

    @property
    def selected_input(self) -> RecallInputOption | None:
        return self._find_input(self._selected_input_id)

    @property
    def selected_target(self) -> RecallTargetOption | None:
        return self._find_target(self._selected_target_id)

    @property
    def current_result(self) -> RecallResult | None:
        return self._current_result

    @property
    def current_weight_matrix(self) -> NDArray[np.float64] | None:
        model = self.selected_model
        return None if model is None else model.weight_matrix

    @property
    def error_message(self) -> str | None:
        return self._error_message

    @property
    def has_model(self) -> bool:
        return self.selected_model is not None

    @property
    def has_input(self) -> bool:
        return self.selected_input is not None

    @property
    def can_import_image(self) -> bool:
        """Whether a model-compatible image preprocessing config exists."""

        return (
            self.selected_model is not None
            and self._import_config_for_model() is not None
        )

    @property
    def input_preprocessing_summary(self) -> str | None:
        """Human-readable preprocessing review for the selected input."""

        option = self.selected_input
        if option is None:
            return None
        return self._preprocessing_summary(option)

    @property
    def memory_type(self) -> MemoryType | None:
        model = self.selected_model
        return None if model is None else model.memory_type

    @property
    def is_heteroassociative(self) -> bool:
        return self.memory_type is MemoryType.HETEROASSOCIATIVE

    @property
    def can_recall(self) -> bool:
        return (
            self._state != RecallState.RUNNING
            and self.selected_model is not None
            and self.selected_input is not None
        )

    @property
    def warnings(self) -> tuple[str, ...]:
        result = self._current_result
        return () if result is None else result.warnings

    @property
    def comparison_basis_text(self) -> str | None:
        result = self._current_result
        if result is None or result.comparison is None:
            return None
        if result.comparison_basis == COMPARISON_BASIS_EXPECTED:
            return "Expected target"
        if result.comparison_basis == COMPARISON_BASIS_INPUT:
            return "Original query pattern"
        return result.comparison_basis

    # ------------------------------------------------------------------ experiment storage

    @property
    def can_save_experiment(self) -> bool:
        """Whether a trained model is available to save as an experiment."""

        return self.selected_model is not None

    @property
    def experiment_name_hint(self) -> str:
        """A short default name for the save-experiment dialog."""

        model = self.selected_model
        if model is None:
            return "Untitled experiment"
        return (
            f"{model.memory_type.value} · {model.encoding.value} · "
            f"{model.input_dimension}×{model.output_dimension}"
        )

    def save_current_experiment(
        self, name: str
    ) -> SavedExperimentSummary | None:
        """Save the selected model (and last recall) as an experiment.

        The saving happens through the application history service; the
        presentation never touches storage files. Returns the stored
        summary, or ``None`` when nothing was saved (an error message is
        emitted in that case). Recall state is intentionally left
        unchanged: persistence failures do not invalidate a computed
        result.
        """

        model = self.selected_model
        if model is None:
            message = "Select a trained model before saving an experiment."
            self.status_message.emit(message, StatusKind.WARNING)
            return None

        input_option = self.selected_input
        target_option = self.selected_target
        try:
            summary = self._history_service.save_experiment(
                name=name,
                training_result=model.result,
                preprocessing_config=model.preprocessing_config,
                target_preprocessing_config=model.target_preprocessing_config,
                recall_result=self._current_result,
                input_source_file=(
                    input_option.source_path if input_option else None
                ),
                target_source_file=(
                    target_option.source_path if target_option else None
                ),
            )
        except (ExperimentRepositoryError, ValueError) as exc:
            message = f"Saving the experiment failed: {exc}"
            self.error_occurred.emit(message)
            self.status_message.emit(message, StatusKind.ERROR)
            return None
        except Exception:  # presentation boundary: log, never crash
            _LOGGER.exception("Unexpected experiment saving failure")
            message = (
                "Saving the experiment failed unexpectedly. See the "
                "application log for details."
            )
            self.error_occurred.emit(message)
            self.status_message.emit(message, StatusKind.ERROR)
            return None

        suffix = (
            "; recall result stored"
            if self._current_result is not None
            else ""
        )
        self.experiment_saved.emit(summary)
        self.status_message.emit(
            f"Experiment saved: {summary.name} (id "
            f"{(summary.experiment_id or '')[:8]}…{suffix}).",
            StatusKind.SUCCESS,
        )
        return summary

    # ------------------------------------------------------------------ model catalog

    def add_training_result(
        self,
        result: TrainingResult,
        *,
        label: str | None = None,
        preprocessing_config: PreprocessingConfig | None = None,
        target_preprocessing_config: PreprocessingConfig | None = None,
        origin_entry_id: str | None = None,
        select: bool = True,
    ) -> str:
        """Register a trained model for recall and return its catalog id.

        Registering the same result object twice is idempotent. The model's
        preprocessing configurations are inferred from the shared dataset
        when not supplied, so image-compatible recall can use the exact
        training grid instead of inventing dimensions.
        """

        if not isinstance(result, TrainingResult):
            raise TypeError(
                "result must be a TrainingResult; received "
                f"{type(result).__name__}."
            )

        for existing in self._models:
            if existing.result is result:
                if select:
                    self.select_model(existing.model_id)
                return existing.model_id

        inferred_source, inferred_target = self._infer_model_configs(result)
        source_config = preprocessing_config or inferred_source
        target_config = (
            target_preprocessing_config
            if target_preprocessing_config is not None
            else inferred_target
        )

        self._model_counter += 1
        model_id = uuid4().hex
        option = RecallModelOption(
            model_id=model_id,
            label=label or self._default_model_label(result, self._model_counter),
            result=result,
            preprocessing_config=source_config,
            target_preprocessing_config=target_config,
            origin_entry_id=origin_entry_id,
        )
        self._models.append(option)
        self._selected_model_id = model_id
        self._selected_target_id = None
        self._current_result = None
        self._error_message = None

        self.models_changed.emit()
        self._rebuild_inputs()
        self._sync_state()
        self.status_message.emit(
            f"Recall model selected: {option.label}.", StatusKind.INFO
        )
        return model_id

    #: Convenience aliases used by callers that treat the catalog as a store.
    register_training_result = add_training_result
    set_training_result = add_training_result

    def select_model(self, model_id: str) -> None:
        """Select a previously trained model from the session catalog."""

        if self._find_model(model_id) is None:
            self._report_error(f"Unknown recall model {model_id!r}.")
            return
        if model_id == self._selected_model_id:
            return

        self._selected_model_id = model_id
        self._current_result = None
        self._error_message = None
        self._selected_target_id = None
        self._rebuild_inputs()
        self.selection_changed.emit()
        self._sync_state()
        self.status_message.emit(
            "Recall model changed; run recall to produce a result.", StatusKind.INFO
        )

    # ------------------------------------------------------------------ input selection

    def select_input(self, input_id: str) -> None:
        """Select one compatible query pattern."""

        if self._find_input(input_id) is None:
            self._report_error(f"Unknown recall input {input_id!r}.")
            return
        if input_id == self._selected_input_id:
            return

        self._selected_input_id = input_id
        self._current_result = None
        self._error_message = None
        self._rebuild_targets(preserve_selection=False)
        self.selection_changed.emit()
        self._sync_state()
        self.status_message.emit(
            "Recall input changed; run recall to produce a result.", StatusKind.INFO
        )

    def select_target(self, target_id: str | None) -> None:
        """Select (or clear) the expected target used for comparison."""

        if target_id is None:
            changed = self._selected_target_id is not None
            self._selected_target_id = None
        else:
            if self._find_target(target_id) is None:
                self._report_error(f"Unknown recall target {target_id!r}.")
                return
            changed = target_id != self._selected_target_id
            self._selected_target_id = target_id

        if not changed:
            return
        self._current_result = None
        self._error_message = None
        self.selection_changed.emit()
        self._sync_state()
        self.status_message.emit(
            "Expected target changed; run recall to compare again.",
            StatusKind.INFO,
        )

    # ------------------------------------------------------------------ input import

    def import_image(self, path: str | Path) -> bool:
        """Preprocess an image file for recall with a model-compatible config.

        The configuration is taken from the model's captured training
        configuration, the selected input, or the active dataset settings —
        and only when its dimensions and encoding match the model. The
        configuration is never silently changed to force compatibility.
        """

        model = self.selected_model
        if model is None:
            self._report_error(
                "Select a trained model before importing a recall image."
            )
            return False

        config = self._import_config_for_model()
        if config is None:
            self._report_error(
                "No preprocessing configuration compatible with this model is "
                "available. Add a compatible image entry to the dataset first, "
                "or select a model trained from image patterns."
            )
            return False

        try:
            result = self._image_service.preprocess_file(path, config)
        except ImageLoadError as exc:
            self._report_error(str(exc))
            return False
        except (OSError, TypeError, ValueError) as exc:
            self._report_error(f"Could not preprocess the recall image: {exc}")
            return False
        except Exception as exc:  # presentation boundary: never crash the view
            _LOGGER.exception("Unexpected failure importing a recall image")
            self._report_error(f"Unexpected image import failure: {exc}")
            return False

        vector = np.asarray(result.encoded_vector, dtype=np.float64)
        if vector.shape != (model.input_dimension,):
            self._report_error(
                f"The imported pattern has length {vector.shape[0]}, but the "
                f"selected model expects {model.input_dimension} values."
            )
            return False
        if result.encoding is not model.encoding:
            self._report_error(
                f"The imported pattern uses {result.encoding.value} encoding, "
                f"but the selected model uses {model.encoding.value} encoding."
            )
            return False

        input_id = f"import:{uuid4().hex}"
        option = RecallInputOption(
            input_id=input_id,
            label=f"{Path(path).name} (imported for recall)",
            vector=vector,
            encoding=result.encoding,
            preprocessing=result,
            source_path=result.source_path,
            dataset_entry_id=None,
        )
        self._imported_inputs[input_id] = option
        # The displayed result, if any, belongs to the previous query. Clear
        # it before switching cues so the results workspace can never show
        # output associated with a different input.
        self._current_result = None
        self._error_message = None
        self._selected_target_id = None
        self._selected_input_id = input_id
        self._rebuild_inputs()
        self.selection_changed.emit()
        self._sync_state()
        self.status_message.emit(
            f"Recall image imported: {Path(path).name} "
            f"({result.width}×{result.height}, "
            f"{result.encoding.value} encoding).",
            StatusKind.SUCCESS,
        )
        return True

    # ------------------------------------------------------------------ recall

    def run_recall(self) -> None:
        """Run recall for the selected model, input, and expected target."""

        if self._state == RecallState.RUNNING:
            return

        model = self.selected_model
        if model is None:
            self._report_error(
                "Cannot run recall: no trained model is selected."
            )
            return

        input_option = self.selected_input
        if input_option is None:
            self._report_error(
                "Cannot run recall: no compatible input pattern is selected."
            )
            return

        if input_option.vector.shape != (model.input_dimension,):
            self._report_error(
                f"Input length {input_option.vector.shape[0]} does not match "
                f"the model input dimension {model.input_dimension}."
            )
            return
        if input_option.encoding is not model.encoding:
            self._report_error(
                f"Input encoding {input_option.encoding.value} does not match "
                f"the model encoding {model.encoding.value}."
            )
            return

        expected_vector: NDArray[np.float64] | None = None
        if model.memory_type is MemoryType.HETEROASSOCIATIVE:
            target = self.selected_target
            if target is not None:
                if target.vector.shape != (model.output_dimension,):
                    self._report_error(
                        f"Expected target length {target.vector.shape[0]} does "
                        f"not match the model output dimension "
                        f"{model.output_dimension}."
                    )
                    return
                if target.encoding is not model.encoding:
                    self._report_error(
                        f"Expected target encoding {target.encoding.value} does "
                        f"not match the model encoding {model.encoding.value}."
                    )
                    return
                expected_vector = target.vector

        self._state = RecallState.RUNNING
        self._error_message = None
        self._current_result = None
        self.state_changed.emit(self._state)
        self.status_message.emit("Recall in progress…", StatusKind.IN_PROGRESS)

        try:
            result = self._recall_service.recall(
                model.result.memory,
                input_option.vector,
                expected_pattern=expected_vector,
            )
        except RecallError as exc:
            self._fail_recall(str(exc))
            return
        except Exception as exc:  # presentation boundary: never crash the view
            _LOGGER.exception("Unexpected recall failure")
            self._fail_recall(f"Unexpected recall failure: {exc}")
            return

        self._current_result = result
        self._state = RecallState.COMPLETED
        self.state_changed.emit(self._state)
        self.result_ready.emit(result)
        self.trace_ready.emit(self.trace_steps())
        self.status_message.emit(
            self._completion_message(result),
            self._completion_status(result),
        )

    # ------------------------------------------------------------------ display data

    @property
    def input_config(self) -> PreprocessingConfig | None:
        option = self.selected_input
        if option is None or option.preprocessing is None:
            return None
        config = option.preprocessing.config
        if config.width * config.height != option.vector_length:
            return None
        return config

    @property
    def expected_preprocessing(self) -> PreprocessingResult | None:
        result = self._current_result
        if result is None:
            return None
        if result.comparison_basis == COMPARISON_BASIS_INPUT:
            option = self.selected_input
            return None if option is None else option.preprocessing
        target = self.selected_target
        return None if target is None else target.preprocessing

    @property
    def expected_config(self) -> PreprocessingConfig | None:
        result = self._current_result
        if result is None:
            return None

        preprocessing = self.expected_preprocessing
        if preprocessing is not None:
            config = preprocessing.config
            if config.width * config.height == result.output_dimension:
                return config
            return None

        if result.comparison_basis == COMPARISON_BASIS_INPUT:
            return self.input_config
        model = self.selected_model
        if model is not None and model.target_preprocessing_config is not None:
            config = model.target_preprocessing_config
            if config.width * config.height == result.output_dimension:
                return config
        return None

    @property
    def input_grid_shape(self) -> tuple[int, int] | None:
        return _shape_for(self.selected_input, self.input_config)

    @property
    def output_grid_shape(self) -> tuple[int, int] | None:
        result = self._current_result
        if result is None:
            return None
        return _shape_for_length(result.output_dimension, self.expected_config)

    @property
    def expected_grid_shape(self) -> tuple[int, int] | None:
        result = self._current_result
        if result is None or result.expected_pattern is None:
            return None
        return _shape_for_length(
            int(result.expected_pattern.shape[0]), self.expected_config
        )

    @property
    def input_matrix(self) -> NDArray[np.float64] | None:
        option = self.selected_input
        config = self.input_config
        if option is None or config is None:
            return None
        return _reshape_vector(option.vector, config)

    @property
    def output_matrix(self) -> NDArray[np.float64] | None:
        result = self._current_result
        config = self.expected_config
        if result is None or config is None:
            return None
        return _reshape_vector(result.output_pattern, config)

    @property
    def expected_matrix(self) -> NDArray[np.float64] | None:
        result = self._current_result
        config = self.expected_config
        if result is None or result.expected_pattern is None or config is None:
            return None
        return _reshape_vector(result.expected_pattern, config)

    # ------------------------------------------------------------------ trace

    def trace_steps(self) -> list[TraceStep]:
        """Build the eight-step calculation trace from actual result values."""

        model = self.selected_model
        result = self._current_result
        if model is None or result is None:
            return []

        input_option = self.selected_input
        weight = model.weight_matrix
        if weight.ndim != 2:
            return []
        weight_rows, weight_cols = weight.shape
        input_length = int(result.input_pattern.shape[0])
        output_length = int(result.output_pattern.shape[0])

        steps: list[TraceStep] = []

        steps.append(
            TraceStep(
                title="1 · Preprocessed input",
                summary=self._preprocessing_summary(input_option),
                detail=self._preprocessing_detail(input_option),
            )
        )

        steps.append(
            TraceStep(
                title="2 · Flattened pattern",
                summary=(
                    f"Flattened query x has shape ({input_length},) and "
                    f"{input_length} encoded value(s).\n"
                    f"{self._vector_summary(result.input_pattern, name='x')}"
                ),
                detail=self._vector_detail(result.input_pattern, name="x"),
                equation="x = flattened encoded input pattern",
            )
        )

        steps.append(
            TraceStep(
                title="3 · Weight matrix",
                summary=(
                    f"W has shape ({weight_rows}, {weight_cols}) with "
                    f"{weight_rows * weight_cols} weight(s).\n"
                    f"Memory type: {model.memory_type.value}; encoding: "
                    f"{model.encoding.value}."
                ),
                detail=(
                    f"W shape: ({weight_rows}, {weight_cols})\n"
                    f"Rows are input components; columns are output components."
                ),
                equation=(
                    "W = Σ outer(s, s)"
                    if model.memory_type is MemoryType.AUTOASSOCIATIVE
                    else "W = Σ outer(s, t)"
                ),
            )
        )

        steps.append(
            TraceStep(
                title="4 · Net-input computation",
                summary=(
                    f"x ({input_length},) @ W ({weight_rows}, {weight_cols}) "
                    f"→ y_in ({output_length},).\n"
                    "Each output component is a weighted sum over the input."
                ),
                detail=(
                    f"x shape: ({input_length},)\n"
                    f"W shape: ({weight_rows}, {weight_cols})\n"
                    f"y_in shape: ({output_length},)\n"
                    "y_in[j] = Σ_i x[i] · W[i, j]"
                ),
                equation="y_in = x @ W",
            )
        )

        steps.append(
            TraceStep(
                title="5 · Net-input values",
                summary=self._vector_summary(result.net_input, name="y_in"),
                detail=self._vector_detail(result.net_input, name="y_in"),
                equation="y_in = x @ W  (values before activation)",
            )
        )

        activation = self._activation_text(result)
        steps.append(
            TraceStep(
                title="6 · Activation rule",
                summary=activation,
                detail=(
                    f"{activation}\n"
                    f"Activation: {result.activation_name}\n"
                    f"Threshold: {_format_number(result.threshold)} "
                    "(strictly greater than threshold is ON)."
                ),
                equation="y = f(y_in)",
            )
        )

        steps.append(
            TraceStep(
                title="7 · Final output",
                summary=self._vector_summary(result.output_pattern, name="y"),
                detail=self._vector_detail(result.output_pattern, name="y"),
                equation="y = activated output pattern",
            )
        )

        steps.append(
            TraceStep(
                title="8 · Comparison",
                summary=self._comparison_summary(result),
                detail=self._comparison_detail(result),
                equation=(
                    "M = Σ 1[y_i ≠ t_i] · match ratio = (N − M) / N"
                    if result.comparison is not None
                    else "No valid comparison target"
                ),
            )
        )

        return steps

    # ------------------------------------------------------------------ internal selection

    def _on_training_result_ready(self, result: object) -> None:
        if not isinstance(result, TrainingResult):
            return
        self.add_training_result(result)

    def _rebuild_inputs(self) -> None:
        model = self.selected_model
        previous_id = self._selected_input_id
        previous_option = self.selected_input
        options: list[RecallInputOption] = []

        if model is not None:
            entries = self._dataset_view_model.entries
            for entry in entries:
                option = self._input_from_entry(entry, model)
                if option is not None:
                    options.append(option)

            preview = self._dataset_view_model.current_result
            if preview is not None and not any(
                entry.result is preview for entry in entries
            ):
                option = self._input_from_preprocessing(
                    "preview:working-preview",
                    preview,
                    "Working preview (not added to dataset)",
                    dataset_entry_id=None,
                    model=model,
                )
                if option is not None:
                    options.append(option)

            for option in self._imported_inputs.values():
                if _compatible_with_model(
                    option.vector, option.encoding, model
                ):
                    options.append(option)

        self._inputs = options
        ids = {option.input_id for option in options}
        new_input_id = (
            previous_id
            if previous_id in ids
            else (options[0].input_id if options else None)
        )
        new_option = next(
            (option for option in options if option.input_id == new_input_id),
            None,
        )
        selection_differs = new_input_id != previous_id
        content_differs = (
            previous_option is not None
            and new_option is not None
            and (
                previous_option.vector.shape != new_option.vector.shape
                or previous_option.encoding is not new_option.encoding
                or previous_option.preprocessing is not new_option.preprocessing
                or not np.array_equal(
                    previous_option.vector, new_option.vector
                )
            )
        )
        if selection_differs or content_differs:
            # Never leave a rendered result attached to a different cue.
            self._current_result = None
            self._error_message = None
        self._selected_input_id = new_input_id

        self._rebuild_targets(preserve_selection=True)
        self.inputs_changed.emit()
        self._sync_state()

    def _rebuild_targets(self, *, preserve_selection: bool) -> None:
        model = self.selected_model
        previous_id = self._selected_target_id
        previous_target = self.selected_target
        targets: list[RecallTargetOption] = []

        if model is not None and model.memory_type is MemoryType.HETEROASSOCIATIVE:
            entries = self._dataset_view_model.entries
            for entry in entries:
                option = self._target_from_entry(entry, model)
                if option is not None:
                    targets.append(option)

            preview = self._dataset_view_model.current_result
            if preview is not None and not any(
                entry.result is preview for entry in entries
            ):
                option = self._target_from_preprocessing(
                    "preview:working-preview",
                    preview,
                    "Working preview (not added to dataset)",
                    dataset_entry_id=None,
                    model=model,
                )
                if option is not None:
                    targets.append(option)

            for option in self._imported_inputs.values():
                if (
                    option.vector.shape == (model.output_dimension,)
                    and option.encoding is model.encoding
                ):
                    targets.append(
                        RecallTargetOption(
                            target_id=option.input_id,
                            label=option.label,
                            vector=option.vector,
                            encoding=option.encoding,
                            preprocessing=option.preprocessing,
                            source_path=option.source_path,
                        )
                    )

        self._targets = targets
        ids = {option.target_id for option in targets}

        if model is None or model.memory_type is not MemoryType.HETEROASSOCIATIVE:
            self._selected_target_id = None
            if previous_id is not None:
                self._current_result = None
                self._error_message = None
            return

        if preserve_selection and previous_id in ids:
            new_target_id = previous_id
        else:
            resolved = self._resolved_target_id_for_input()
            new_target_id = resolved if resolved in ids else None

        new_target = next(
            (option for option in targets if option.target_id == new_target_id),
            None,
        )
        selection_differs = new_target_id != previous_id
        content_differs = (
            previous_target is not None
            and new_target is not None
            and (
                previous_target.vector.shape != new_target.vector.shape
                or previous_target.encoding is not new_target.encoding
                or previous_target.preprocessing is not new_target.preprocessing
                or not np.array_equal(
                    previous_target.vector, new_target.vector
                )
            )
        )
        if selection_differs or content_differs:
            # The reference used for the displayed comparison no longer
            # matches the selected target.
            self._current_result = None
            self._error_message = None
        self._selected_target_id = new_target_id

    def _resolved_target_id_for_input(self) -> str | None:
        """Resolve the selected dataset entry's target label, if it exists.

        The training workspace uses target labels as references to dataset
        entry ids. An unknown label is not guessed and simply yields no
        expected target.
        """

        option = self.selected_input
        if option is None or option.dataset_entry_id is None:
            return None
        entry = self._dataset_view_model.entry_for_id(option.dataset_entry_id)
        if entry is None or not entry.target_label:
            return None
        candidate = f"entry:{entry.target_label}"
        return candidate if self._find_target(candidate) is not None else None

    def _input_from_entry(
        self, entry: DatasetEntry, model: RecallModelOption
    ) -> RecallInputOption | None:
        vector = np.asarray(entry.result.encoded_vector, dtype=np.float64)
        if not _compatible_with_model(vector, entry.result.encoding, model):
            return None
        return self._input_from_preprocessing(
            f"entry:{entry.entry_id}",
            entry.result,
            self._entry_label(entry),
            dataset_entry_id=entry.entry_id,
            model=model,
        )

    def _input_from_preprocessing(
        self,
        input_id: str,
        preprocessing: PreprocessingResult,
        label: str,
        *,
        dataset_entry_id: str | None,
        model: RecallModelOption,
    ) -> RecallInputOption | None:
        vector = np.asarray(preprocessing.encoded_vector, dtype=np.float64)
        if not _compatible_with_model(vector, preprocessing.encoding, model):
            return None
        return RecallInputOption(
            input_id=input_id,
            label=label,
            vector=vector,
            encoding=preprocessing.encoding,
            preprocessing=preprocessing,
            source_path=preprocessing.source_path,
            dataset_entry_id=dataset_entry_id,
        )

    def _target_from_entry(
        self, entry: DatasetEntry, model: RecallModelOption
    ) -> RecallTargetOption | None:
        vector = np.asarray(entry.result.encoded_vector, dtype=np.float64)
        if not _target_compatible(vector, entry.result.encoding, model):
            return None
        return self._target_from_preprocessing(
            f"entry:{entry.entry_id}",
            entry.result,
            self._entry_label(entry),
            dataset_entry_id=entry.entry_id,
            model=model,
        )

    def _target_from_preprocessing(
        self,
        target_id: str,
        preprocessing: PreprocessingResult,
        label: str,
        *,
        dataset_entry_id: str | None,
        model: RecallModelOption,
    ) -> RecallTargetOption | None:
        vector = np.asarray(preprocessing.encoded_vector, dtype=np.float64)
        if not _target_compatible(vector, preprocessing.encoding, model):
            return None
        return RecallTargetOption(
            target_id=target_id,
            label=label,
            vector=vector,
            encoding=preprocessing.encoding,
            preprocessing=preprocessing,
            source_path=preprocessing.source_path,
            dataset_entry_id=dataset_entry_id,
        )

    def _entry_label(self, entry: DatasetEntry) -> str:
        name = entry.source_path.name if entry.source_path else "(untitled)"
        if entry.target_label:
            return f"{name} → target {entry.target_label}"
        return name

    def _find_model(self, model_id: str | None) -> RecallModelOption | None:
        if model_id is None:
            return None
        for model in self._models:
            if model.model_id == model_id:
                return model
        return None

    def _find_input(self, input_id: str | None) -> RecallInputOption | None:
        if input_id is None:
            return None
        for option in self._inputs:
            if option.input_id == input_id:
                return option
        return None

    def _find_target(self, target_id: str | None) -> RecallTargetOption | None:
        if target_id is None:
            return None
        for option in self._targets:
            if option.target_id == target_id:
                return option
        return None

    # ------------------------------------------------------------------ internal configuration

    def _infer_model_configs(
        self, result: TrainingResult
    ) -> tuple[PreprocessingConfig | None, PreprocessingConfig | None]:
        entries = self._dataset_view_model.entries
        source_config = _first_matching_config(
            entries, result.input_dimension, result.encoding
        )
        target_config = _first_matching_config(
            entries, result.output_dimension, result.encoding
        )
        return source_config, target_config

    def _import_config_for_model(self) -> PreprocessingConfig | None:
        model = self.selected_model
        if model is None:
            return None

        candidates: list[PreprocessingConfig | None] = [
            model.preprocessing_config,
            self.input_config,
            self._dataset_view_model.config,
        ]
        for config in candidates:
            if config is None:
                continue
            if (
                config.encoding is model.encoding
                and config.width * config.height == model.input_dimension
            ):
                return config
        return None

    @staticmethod
    def _default_model_label(result: TrainingResult, index: int) -> str:
        name = result.dataset_name or "Training workspace"
        return (
            f"Model {index}: {result.memory_type.value} · "
            f"{result.encoding.value} · "
            f"{result.input_dimension}×{result.output_dimension} · "
            f"{result.pair_count} pair(s) · {name}"
        )

    def _sync_state(self) -> None:
        if self._state == RecallState.RUNNING:
            return
        if self.selected_model is None or self.selected_input is None:
            new_state = RecallState.NOT_READY
        elif self._current_result is not None:
            new_state = RecallState.COMPLETED
        else:
            new_state = RecallState.READY

        if new_state is self._state:
            return
        self._state = new_state
        self.state_changed.emit(new_state)

    def _report_error(self, message: str) -> None:
        self._state = RecallState.FAILED
        self._error_message = message
        self.state_changed.emit(self._state)
        self.error_occurred.emit(message)
        self.status_message.emit(message, StatusKind.ERROR)

    def _fail_recall(self, message: str) -> None:
        self._state = RecallState.FAILED
        self._error_message = message
        self._current_result = None
        self.state_changed.emit(self._state)
        self.error_occurred.emit(message)
        self.status_message.emit(f"Recall failed: {message}", StatusKind.ERROR)

    # ------------------------------------------------------------------ trace text

    def _preprocessing_summary(self, option: RecallInputOption | None) -> str:
        if option is None:
            return "No input pattern is selected."
        if option.preprocessing is None:
            return (
                "No preprocessing metadata is available for this numeric "
                "pattern.\n"
                f"Encoded input length: {option.vector_length} · "
                f"encoding: {option.encoding.value}."
            )
        preprocessing = option.preprocessing
        config = preprocessing.config
        return (
            f"Source: {option.label}\n"
            f"Original image: {preprocessing.original_width}×"
            f"{preprocessing.original_height} px · format "
            f"{preprocessing.source_format or 'unknown'}\n"
            f"Target grid: {config.width}×{config.height} = "
            f"{config.width * config.height} values\n"
            f"Threshold: {config.threshold} · polarity: "
            f"{config.polarity.value} · encoding: {config.encoding.value}\n"
            f"Resize: {config.resize_mode.value} · flattening: "
            f"{config.flattening_order.value} · grayscale: "
            f"{config.grayscale_method.value}"
        )

    def _preprocessing_detail(self, option: RecallInputOption | None) -> str:
        if option is None or option.preprocessing is None:
            return self._preprocessing_summary(option)
        preprocessing = option.preprocessing
        config = preprocessing.config
        source = str(preprocessing.source_path) if preprocessing.source_path else "—"
        original_mode = (
            preprocessing.original_image.mode
            if preprocessing.original_image is not None
            else "—"
        )
        return (
            f"Source path: {source}\n"
            f"Source format: {preprocessing.source_format or '—'}\n"
            f"Original dimensions: {preprocessing.original_width}×"
            f"{preprocessing.original_height}\n"
            f"Original mode: {original_mode}\n"
            f"Grayscale method: {config.grayscale_method.value}\n"
            f"Resampling: {config.resampling.name}\n"
            f"Resize mode: {config.resize_mode.value} "
            f"(aspect preserved: {preprocessing.aspect_ratio_preserved})\n"
            f"Threshold: {config.threshold}\n"
            f"Polarity: {config.polarity.value}\n"
            f"Encoding: {config.encoding.value}\n"
            f"Flattening order: {config.flattening_order.value}\n"
            f"Encoded vector length: {preprocessing.vector_length}"
        )

    @staticmethod
    def _activation_text(result: RecallResult) -> str:
        threshold = _format_number(result.threshold)
        if result.encoding is Encoding.BINARY:
            return (
                f"Binary activation: y_i = 1 when y_in[i] > {threshold}; "
                "otherwise 0.\n"
                f"Threshold {threshold} is strict: y_in[i] = {threshold} maps "
                "to 0 (OFF)."
            )
        return (
            f"Bipolar activation: y_i = +1 when y_in[i] > {threshold}; "
            "otherwise −1.\n"
            f"Threshold {threshold} is strict: y_in[i] = {threshold} maps "
            "to −1 (OFF)."
        )

    @staticmethod
    def _vector_summary(values: NDArray[np.float64], *, name: str) -> str:
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0:
            return f"{name} is empty."
        head = [_format_number(value) for value in array[:SUMMARY_VALUE_LIMIT]]
        suffix = (
            ""
            if array.size <= SUMMARY_VALUE_LIMIT
            else f", … ({array.size - SUMMARY_VALUE_LIMIT} more)"
        )
        return (
            f"{name} ({array.size} values) = "
            f"[{', '.join(head)}{suffix}]\n"
            f"min {_format_number(np.min(array))} · "
            f"max {_format_number(np.max(array))} · "
            f"nonzero {int(np.count_nonzero(array))}"
        )

    @staticmethod
    def _vector_detail(values: NDArray[np.float64], *, name: str) -> str:
        array = np.asarray(values, dtype=np.float64)
        rendered = np.array2string(
            array,
            separator=", ",
            threshold=array.size,
            max_line_width=100,
            formatter={"float_kind": lambda value: _format_number(value)},
        )
        return f"{name} = {rendered}\nLength: {array.size}"

    def _comparison_summary(self, result: RecallResult) -> str:
        comparison = result.comparison
        if comparison is None:
            return (
                "Comparison unavailable: no valid expected target was supplied.\n"
                "Exact match, mismatch count, and match ratio are not "
                "calculated. This is not a zero-match result."
            )
        basis = self.comparison_basis_text or "comparison target"
        exact = "yes" if comparison.exact_match else "no"
        return (
            f"Comparison basis: {basis}\n"
            f"Exact match: {exact}\n"
            f"Mismatch count: {comparison.mismatch_count} of "
            f"{comparison.element_count}\n"
            f"Match ratio: {_format_number(comparison.match_ratio)} "
            f"({_format_number(comparison.match_ratio * 100)}%)\n"
            "This is a single-pattern match ratio, not dataset-level "
            "classification accuracy."
        )

    def _comparison_detail(self, result: RecallResult) -> str:
        comparison = result.comparison
        if comparison is None or result.expected_pattern is None:
            return self._comparison_summary(result)
        output = np.asarray(result.output_pattern)
        expected = np.asarray(result.expected_pattern)
        mismatch_indices = np.flatnonzero(output != expected)
        if mismatch_indices.size == 0:
            differences = "All components are equal."
        else:
            lines = []
            for index in mismatch_indices:
                lines.append(
                    f"[{int(index)}] output {_format_number(output[index])} "
                    f"≠ expected {_format_number(expected[index])}"
                )
            differences = "\n".join(lines)
        return (
            f"Output length: {output.size}\n"
            f"Expected length: {expected.size}\n"
            f"Mismatch indices:\n{differences}"
        )

    @staticmethod
    def _completion_message(result: RecallResult) -> str:
        comparison = result.comparison
        if comparison is None:
            return (
                "Recall completed. No valid comparison target was supplied, "
                "so no exact-match or ratio result is available."
            )
        if comparison.exact_match:
            return (
                f"Recall completed: exact match across "
                f"{comparison.element_count} element(s)."
            )
        return (
            f"Recall completed: {comparison.mismatch_count} of "
            f"{comparison.element_count} element(s) differ "
            f"(match ratio {comparison.match_ratio:.6g})."
        )

    @staticmethod
    def _completion_status(result: RecallResult) -> StatusKind:
        comparison = result.comparison
        if comparison is None:
            return StatusKind.INFO
        return StatusKind.SUCCESS if comparison.exact_match else StatusKind.WARNING


def _compatible_with_model(
    vector: NDArray[np.float64],
    encoding: Encoding,
    model: RecallModelOption,
) -> bool:
    return (
        vector.shape == (model.input_dimension,)
        and encoding is model.encoding
    )


def _target_compatible(
    vector: NDArray[np.float64],
    encoding: Encoding,
    model: RecallModelOption,
) -> bool:
    return (
        vector.shape == (model.output_dimension,)
        and encoding is model.encoding
    )


def _first_matching_config(
    entries: Sequence[DatasetEntry],
    dimension: int,
    encoding: Encoding,
) -> PreprocessingConfig | None:
    for entry in entries:
        config = entry.config
        if config.encoding is not encoding:
            continue
        if entry.result.encoding is not encoding:
            continue
        if config.width * config.height != dimension:
            continue
        if entry.result.encoded_vector.shape != (dimension,):
            continue
        return config
    return None


def _shape_for_length(
    length: int, config: PreprocessingConfig | None
) -> tuple[int, int] | None:
    if config is None:
        return None
    if config.width * config.height != length:
        return None
    return (config.height, config.width)


def _shape_for(
    option: RecallInputOption | None,
    config: PreprocessingConfig | None,
) -> tuple[int, int] | None:
    if option is None:
        return None
    return _shape_for_length(option.vector_length, config)


def _reshape_vector(
    vector: NDArray[np.float64],
    config: PreprocessingConfig,
) -> NDArray[np.float64] | None:
    array = np.asarray(vector, dtype=np.float64)
    expected = config.width * config.height
    if array.shape != (expected,):
        return None
    order = "C" if config.flattening_order.value == "row_major" else "F"
    return np.array(array.reshape((config.height, config.width), order=order))


def _format_number(value: object) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return np.format_float_positional(numeric, unique=True, trim="-")
