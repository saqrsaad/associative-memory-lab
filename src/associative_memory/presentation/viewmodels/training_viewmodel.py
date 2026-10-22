"""Training workspace view model.

Coordinates memory configuration, dataset validation, and training
execution between the training view and the application services.
Implements no numerical formulas itself: training delegates to
``TrainingService`` and validation to ``algorithms.validate_vector``.

Ownership policy
----------------
The view model owns the training state and the current result. Result
arrays are read-only copies owned by the :class:`TrainingResult` DTO;
the view model never mutates them.

Training runs synchronously for normal (small) datasets so tests and
simple flows remain straightforward. A background worker is available
via :meth:`train_async` for genuinely long-running operations that must
not block the GUI thread.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

from associative_memory.algorithms.encoding import coerce_encoding
from associative_memory.application.dto.training_result import (
    MemoryType,
    TrainingResult,
)
from associative_memory.application.services.training_service import (
    TrainingDataset,
    TrainingError,
    TrainingService,
)
from associative_memory.presentation.viewmodels.dataset_viewmodel import (
    MEMORY_AUTOASSOCIATIVE,
    MEMORY_HETEROASSOCIATIVE,
    DatasetEntry,
    DatasetViewModel,
)

_LOGGER = logging.getLogger(__name__)


class TrainingState(Enum):
    """Explicit training lifecycle states."""

    NOT_READY = "not_ready"
    READY = "ready"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class TrainingTraceStep:
    """One step of the mathematical training trace.

    Attributes:
        pair_index: Zero-based index of the training pair.
        source_pattern: Source pattern vector (read-only copy).
        target_pattern: Target pattern vector (read-only copy).
        contribution: Individual outer-product matrix for this pair
            (read-only copy).
        accumulated: Accumulated weight matrix after this pair
            (read-only copy).
        equation: Human-readable equation for this step.
    """

    pair_index: int
    source_pattern: np.ndarray
    target_pattern: np.ndarray
    contribution: np.ndarray
    accumulated: np.ndarray
    equation: str


class TrainingViewModel(QObject):
    """Coordinate training configuration and execution."""

    state_changed = Signal(TrainingState)
    result_ready = Signal(object)  # TrainingResult
    error_occurred = Signal(str)
    trace_ready = Signal(list)  # list[TrainingTraceStep]
    status_message = Signal(str)

    def __init__(
        self,
        dataset_view_model: DatasetViewModel | None = None,
        training_service: TrainingService | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._dataset_view_model = dataset_view_model or DatasetViewModel()
        self._training_service = training_service or TrainingService()

        self._memory_type = MemoryType.AUTOASSOCIATIVE
        self._encoding = "binary"
        self._trace_enabled = False
        self._state = TrainingState.NOT_READY
        self._current_result: TrainingResult | None = None
        self._error_message: str | None = None

        # React to dataset changes: invalidate the trained model.
        self._dataset_view_model.entries_changed.connect(
            self._on_dataset_changed
        )
        self._dataset_view_model.memory_type_changed.connect(
            self._on_dataset_memory_type_changed
        )

    # ------------------------------------------------------------------ properties

    @property
    def state(self) -> TrainingState:
        return self._state

    @property
    def memory_type(self) -> MemoryType:
        return self._memory_type

    @property
    def encoding(self) -> str:
        return self._encoding

    @property
    def trace_enabled(self) -> bool:
        return self._trace_enabled

    @property
    def current_result(self) -> TrainingResult | None:
        return self._current_result

    @property
    def error_message(self) -> str | None:
        return self._error_message

    @property
    def is_ready(self) -> bool:
        return self._state == TrainingState.READY

    @property
    def is_training(self) -> bool:
        return self._state == TrainingState.TRAINING

    @property
    def entries(self) -> tuple[DatasetEntry, ...]:
        return self._dataset_view_model.entries

    # ------------------------------------------------------------------ configuration

    def set_memory_type(self, value: str) -> None:
        """Switch the memory type.

        Switching to autoassociative clears heteroassociative target
        assumptions in the dataset view model.
        """

        if value == MEMORY_AUTOASSOCIATIVE:
            new_type = MemoryType.AUTOASSOCIATIVE
        elif value == MEMORY_HETEROASSOCIATIVE:
            new_type = MemoryType.HETEROASSOCIATIVE
        else:
            self._set_error(f"Unknown memory type: {value!r}")
            return

        self._memory_type = new_type
        self._dataset_view_model.set_memory_type(value)
        self._invalidate_model()
        self._update_readiness()
        self.status_message.emit(
            f"Memory type set to {value}. Model requires retraining."
        )

    def set_encoding(self, value: str) -> None:
        """Set the encoding for training and pattern validation.

        Raises via :class:`TrainingError` semantics (emitted through
        ``error_occurred``) if the encoding name is unsupported.
        """

        try:
            coerce_encoding(value)
        except ValueError as exc:
            self._set_error(f"Invalid encoding: {exc}")
            return

        self._encoding = value
        self._invalidate_model()
        self._update_readiness()
        self.status_message.emit(
            f"Encoding set to {value}. Model requires retraining."
        )

    def set_trace_enabled(self, enabled: bool) -> None:
        """Enable or disable the detailed calculation trace."""

        self._trace_enabled = enabled

    # ------------------------------------------------------------------ training

    def train(self) -> None:
        """Train synchronously from the current dataset and configuration.

        Suitable for normal (small) datasets that complete immediately.
        For genuinely long-running operations, use :meth:`train_async`
        to avoid blocking the GUI thread.
        """

        if self._state == TrainingState.TRAINING:
            return

        pairs = self._build_training_pairs()
        if pairs is None:
            return  # error already reported

        dataset = TrainingDataset(
            patterns=tuple(pairs),
            memory_type=self._memory_type,
            name="dataset-workspace",
        )

        self._state = TrainingState.TRAINING
        self._error_message = None
        self._current_result = None
        self.state_changed.emit(self._state)
        self.status_message.emit("Training in progress…")

        try:
            result = self._training_service.train(
                dataset,
                encoding=self._encoding,
                trace=self._trace_enabled,
            )
        except TrainingError as exc:
            self._state = TrainingState.FAILED
            self._error_message = str(exc)
            self._current_result = None
            self.state_changed.emit(self._state)
            self.error_occurred.emit(str(exc))
            self.status_message.emit(f"Training failed: {str(exc)}")
            return

        self._state = TrainingState.COMPLETED
        self._error_message = None
        self._current_result = result
        self.state_changed.emit(self._state)
        self.result_ready.emit(result)

        if self._trace_enabled and result.contributions is not None:
            self.trace_ready.emit(self.trace_steps())

        self.status_message.emit(
            f"Training complete: {result.pair_count} pair(s), "
            f"{result.input_dimension}x{result.output_dimension} matrix."
        )

    def train_async(self) -> None:
        """Train on a background QThread for long-running operations.

        Keeps the GUI thread responsive during slow training. Results are
        delivered via the same signals as :meth:`train`.
        """

        if self._state == TrainingState.TRAINING:
            return

        pairs = self._build_training_pairs()
        if pairs is None:
            return

        dataset = TrainingDataset(
            patterns=tuple(pairs),
            memory_type=self._memory_type,
            name="dataset-workspace",
        )

        self._state = TrainingState.TRAINING
        self._error_message = None
        self._current_result = None
        self.state_changed.emit(self._state)
        self.status_message.emit("Training in progress…")

        self._thread = QThread()
        self._worker = _TrainingWorker(
            self._training_service,
            dataset,
            self._encoding,
            self._trace_enabled,
        )
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_training_finished)
        self._worker.error.connect(self._on_training_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.start()

    def cancel_training(self) -> None:
        """Request cancellation of an in-progress training run.

        Note: :class:`TrainingService` training is not interruptible once
        started; this flag prevents result application if set.
        """

        if self._state != TrainingState.TRAINING:
            return
        self._cancel_requested = True

    # ------------------------------------------------------------------ trace access

    def trace_steps(self) -> list[TrainingTraceStep]:
        """Return the mathematical trace steps from the last training run.

        Returns an empty list when tracing was disabled or no training has
        been performed.
        """

        if self._current_result is None or self._current_result.contributions is None:
            return []

        pairs = self._build_training_pairs()
        if pairs is None:
            return []

        steps: list[TrainingTraceStep] = []
        accumulated = np.zeros_like(self._current_result.weight_matrix)

        for index, (source, target) in enumerate(pairs):
            contribution = self._current_result.contributions[index]
            accumulated = accumulated + contribution

            if self._memory_type is MemoryType.AUTOASSOCIATIVE:
                equation = (
                    f"W[{index}] = W[{index - 1}] + outer(s, s)"
                    if index > 0
                    else "W[0] = outer(s, s)"
                )
            else:
                equation = (
                    f"W[{index}] = W[{index - 1}] + outer(s, t)"
                    if index > 0
                    else "W[0] = outer(s, t)"
                )

            steps.append(
                TrainingTraceStep(
                    pair_index=index,
                    source_pattern=np.array(source, dtype=np.float64, copy=True),
                    target_pattern=np.array(target, dtype=np.float64, copy=True),
                    contribution=np.array(contribution, dtype=np.float64, copy=True),
                    accumulated=np.array(accumulated, dtype=np.float64, copy=True),
                    equation=equation,
                )
            )

        return steps

    # ------------------------------------------------------------------ internal

    def _build_training_pairs(
        self,
    ) -> list[tuple[np.ndarray, np.ndarray]] | None:
        """Build (source, target) pairs from dataset entries.

        Returns ``None`` and reports an error if the dataset is invalid.
        """

        entries = self._dataset_view_model.entries
        if not entries:
            self._set_error("Cannot train: dataset is empty.")
            return None

        # Collect all encoded vectors.
        vectors: dict[str, np.ndarray] = {}
        for entry in entries:
            if entry.result is None:
                self._set_error(
                    f"Entry {entry.entry_id} has no processed pattern."
                )
                return None
            vectors[entry.entry_id] = entry.result.encoded_vector.astype(
                np.float64
            )

        pairs: list[tuple[np.ndarray, np.ndarray]] = []

        if self._memory_type is MemoryType.AUTOASSOCIATIVE:
            for entry in entries:
                vec = vectors[entry.entry_id]
                pairs.append((vec, vec.copy()))
        else:
            # Heteroassociative: each entry's source is its vector; the
            # target is looked up by target_label (entry_id reference).
            label_to_vector: dict[str, np.ndarray] = {}
            for entry in entries:
                if entry.target_label is not None:
                    label_to_vector[entry.target_label] = vectors[entry.entry_id]

            for entry in entries:
                source = vectors[entry.entry_id]
                target_label = entry.target_label
                if target_label is None:
                    self._set_error(
                        f"Heteroassociative entry {entry.entry_id} has no "
                        f"target label assigned."
                    )
                    return None
                if target_label not in label_to_vector:
                    self._set_error(
                        f"Heteroassociative entry {entry.entry_id} references "
                        f"unknown target {target_label!r}."
                    )
                    return None
                pairs.append((source, label_to_vector[target_label].copy()))

        # Validate dimensions are consistent.
        source_lengths = {len(p[0]) for p in pairs}
        target_lengths = {len(p[1]) for p in pairs}
        if len(source_lengths) != 1:
            self._set_error(
                "All source patterns must have the same length for training."
            )
            return None
        if len(target_lengths) != 1:
            self._set_error(
                "All target patterns must have the same length for training."
            )
            return None

        return pairs

    def _on_training_finished(self, result: TrainingResult) -> None:
        """Handle successful training completion (async path)."""

        self._state = TrainingState.COMPLETED
        self._error_message = None
        self._current_result = result
        self.state_changed.emit(self._state)
        self.result_ready.emit(result)

        if self._trace_enabled and result.contributions is not None:
            self.trace_ready.emit(self.trace_steps())

        self.status_message.emit(
            f"Training complete: {result.pair_count} pair(s), "
            f"{result.input_dimension}x{result.output_dimension} matrix."
        )

        self._cleanup_thread()

    def _on_training_error(self, message: str) -> None:
        """Handle training failure (async path)."""

        self._state = TrainingState.FAILED
        self._error_message = message
        self._current_result = None
        self.state_changed.emit(self._state)
        self.error_occurred.emit(message)
        self.status_message.emit(f"Training failed: {message}")
        self._cleanup_thread()

    def _cleanup_thread(self) -> None:
        """Release worker thread resources."""

        if self._worker is not None:
            self._worker = None
        if self._thread is not None:
            self._thread = None

    def _invalidate_model(self) -> None:
        """Mark the current model as stale due to config/dataset changes."""

        if self._state in (TrainingState.COMPLETED, TrainingState.FAILED):
            self._state = TrainingState.NOT_READY
            self._current_result = None
            self.state_changed.emit(self._state)

    def _update_readiness(self) -> None:
        """Update READY/NOT_READY based on dataset availability."""

        if self._state == TrainingState.TRAINING:
            return

        entries = self._dataset_view_model.entries
        if not entries:
            new_state = TrainingState.NOT_READY
        else:
            new_state = TrainingState.READY

        if self._state != new_state:
            self._state = new_state
            self.state_changed.emit(self._state)

    def _on_dataset_changed(self) -> None:
        """React to dataset modifications: invalidate and re-evaluate."""

        self._invalidate_model()
        self._update_readiness()
        self.status_message.emit(
            "Dataset changed: existing model is stale and must be retrained."
        )

    def _on_dataset_memory_type_changed(self, value: str) -> None:
        """Keep the view model's memory type in sync with the dataset."""

        self._memory_type = (
            MemoryType.AUTOASSOCIATIVE
            if value == MEMORY_AUTOASSOCIATIVE
            else MemoryType.HETEROASSOCIATIVE
        )
        self._invalidate_model()
        self._update_readiness()

    def _set_error(self, message: str) -> None:
        self._error_message = message
        self._state = TrainingState.FAILED
        self.state_changed.emit(self._state)
        self.error_occurred.emit(message)
        self.status_message.emit(message)


class _TrainingWorker(QObject):
    """Worker object that runs training on a background QThread."""

    finished = Signal(object)  # TrainingResult
    error = Signal(str)

    def __init__(
        self,
        service: TrainingService,
        dataset: TrainingDataset,
        encoding: str,
        trace: bool,
    ) -> None:
        super().__init__()
        self._service = service
        self._dataset = dataset
        self._encoding = encoding
        self._trace = trace

    def run(self) -> None:
        """Execute training on the worker thread."""

        try:
            result = self._service.train(
                self._dataset,
                encoding=self._encoding,
                trace=self._trace,
            )
            self.finished.emit(result)
        except TrainingError as exc:
            self.error.emit(str(exc))
        except Exception as exc:  # presentation boundary: never crash UI
            _LOGGER.exception("Unexpected training failure")
            self.error.emit(f"Unexpected training failure: {exc}")
