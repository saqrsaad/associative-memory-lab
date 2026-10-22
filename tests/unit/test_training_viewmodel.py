"""Unit tests for the TrainingViewModel."""

from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import QThread

from associative_memory.application.dto.training_result import MemoryType
from associative_memory.infrastructure.image_preprocessor import (
    PreprocessingResult,
)
from associative_memory.presentation.viewmodels.dataset_viewmodel import (
    MEMORY_AUTOASSOCIATIVE,
    MEMORY_HETEROASSOCIATIVE,
    DatasetEntry,
    DatasetViewModel,
)
from associative_memory.presentation.viewmodels.training_viewmodel import (
    TrainingState,
    TrainingViewModel,
    _TrainingWorker,
)


def test_training_viewmodel_defaults() -> None:
    vm = TrainingViewModel()

    assert vm.state == TrainingState.NOT_READY
    assert vm.memory_type == MemoryType.AUTOASSOCIATIVE
    assert vm.encoding == "binary"
    assert not vm.trace_enabled
    assert vm.current_result is None
    assert vm.error_message is None


def test_training_viewmodel_set_memory_type() -> None:
    vm = TrainingViewModel()
    vm.set_memory_type(MEMORY_HETEROASSOCIATIVE)

    assert vm.memory_type == MemoryType.HETEROASSOCIATIVE
    assert vm.state == TrainingState.NOT_READY


def test_training_viewmodel_set_encoding() -> None:
    vm = TrainingViewModel()
    vm.set_encoding("bipolar")

    assert vm.encoding == "bipolar"


def test_training_viewmodel_set_trace_enabled() -> None:
    vm = TrainingViewModel()
    vm.set_trace_enabled(True)

    assert vm.trace_enabled is True


def _make_entry(
    dvm: DatasetViewModel,
    entry_id: str,
    vector: np.ndarray,
    memory_type: str,
    target_label: str | None = None,
) -> DatasetEntry:
    """Build a DatasetEntry with a valid PreprocessingResult for ``vector``."""

    config = dvm.config
    result = PreprocessingResult(
        config=config,
        source_path=None,
        source_format=None,
        original_image=None,
        original_width=config.width,
        original_height=config.height,
        grayscale=np.zeros((config.height, config.width), dtype=np.uint8),
        resized=np.zeros((config.height, config.width), dtype=np.uint8),
        binary_matrix=np.zeros((config.height, config.width), dtype=np.uint8),
        encoded_matrix=vector.reshape(-1, 1).astype(np.int8),
        encoded_vector=vector,
        aspect_ratio_preserved=True,
    )
    return DatasetEntry(
        entry_id=entry_id,
        source_path=None,
        source_format=None,
        original_width=config.width,
        original_height=config.height,
        original_mode="L",
        memory_type=memory_type,
        config=config,
        result=result,
        target_label=target_label,
    )


def test_training_viewmodel_trains_heteroassociative() -> None:
    vm = TrainingViewModel()
    dvm = vm._dataset_view_model

    dvm.set_memory_type(MEMORY_HETEROASSOCIATIVE)

    s1 = np.array([1.0, -1.0], dtype=np.float64)
    s2 = np.array([-1.0, 1.0], dtype=np.float64)

    entry1 = _make_entry(dvm, "e1", s1, MEMORY_HETEROASSOCIATIVE, target_label="e2")
    entry2 = _make_entry(dvm, "e2", s2, MEMORY_HETEROASSOCIATIVE, target_label="e1")

    dvm._entries = [entry1, entry2]
    dvm.entries_changed.emit()

    vm.set_encoding("bipolar")
    vm.train()

    assert vm.state == TrainingState.COMPLETED
    assert vm.current_result is not None
    assert vm.current_result.pair_count == 2
    assert vm.current_result.input_dimension == 2
    assert vm.current_result.output_dimension == 2

    # W = outer(s1, s2) + outer(s2, s1) (hetero target labels swap sources)
    expected = np.outer(s1, s2) + np.outer(s2, s1)
    np.testing.assert_array_equal(vm.current_result.weight_matrix, expected)


def test_training_viewmodel_trains_autoassociative() -> None:
    vm = TrainingViewModel()
    dvm = vm._dataset_view_model

    dvm.set_memory_type(MEMORY_AUTOASSOCIATIVE)

    p1 = np.array([1.0, -1.0, 1.0], dtype=np.float64)
    p2 = np.array([-1.0, 1.0, 1.0], dtype=np.float64)

    entry1 = _make_entry(dvm, "e1", p1, MEMORY_AUTOASSOCIATIVE)
    entry2 = _make_entry(dvm, "e2", p2, MEMORY_AUTOASSOCIATIVE)

    dvm._entries = [entry1, entry2]
    dvm.entries_changed.emit()

    vm.set_encoding("bipolar")
    vm.train()

    assert vm.state == TrainingState.COMPLETED
    assert vm.current_result.pair_count == 2
    assert vm.current_result.input_dimension == 3
    assert vm.current_result.output_dimension == 3

    expected = np.outer(p1, p1) + np.outer(p2, p2)
    np.testing.assert_array_equal(vm.current_result.weight_matrix, expected)


def test_training_viewmodel_rejects_empty_dataset() -> None:
    vm = TrainingViewModel()
    errors = []
    vm.error_occurred.connect(lambda msg: errors.append(msg))

    vm.train()

    assert vm.state == TrainingState.FAILED
    assert "empty" in errors[0].lower()


def test_training_viewmodel_invalid_encoding() -> None:
    vm = TrainingViewModel()
    errors = []
    vm.error_occurred.connect(lambda msg: errors.append(msg))

    # Encoding validation happens at set_encoding time
    vm.set_encoding("ternary")

    assert vm.state == TrainingState.FAILED
    assert "ternary" in errors[0].lower()


def test_training_viewmodel_trace_steps() -> None:
    vm = TrainingViewModel()
    dvm = vm._dataset_view_model

    dvm.set_memory_type(MEMORY_AUTOASSOCIATIVE)

    p1 = np.array([1.0, -1.0], dtype=np.float64)
    p2 = np.array([-1.0, 1.0], dtype=np.float64)

    entry1 = _make_entry(dvm, "e1", p1, MEMORY_AUTOASSOCIATIVE)
    entry2 = _make_entry(dvm, "e2", p2, MEMORY_AUTOASSOCIATIVE)

    dvm._entries = [entry1, entry2]
    dvm.entries_changed.emit()

    vm.set_trace_enabled(True)
    vm.set_encoding("bipolar")
    vm.train()

    assert vm.state == TrainingState.COMPLETED
    steps = vm.trace_steps()
    assert len(steps) == 2
    assert steps[0].pair_index == 0
    assert steps[1].pair_index == 1
    np.testing.assert_array_equal(steps[0].source_pattern, p1)
    np.testing.assert_array_equal(steps[0].target_pattern, p1)
    np.testing.assert_array_equal(steps[0].contribution, np.outer(p1, p1))
    np.testing.assert_array_equal(
        steps[1].accumulated, np.outer(p1, p1) + np.outer(p2, p2)
    )


def test_training_viewmodel_dataset_change_invalidates_model() -> None:
    vm = TrainingViewModel()
    dvm = vm._dataset_view_model

    dvm.set_memory_type(MEMORY_AUTOASSOCIATIVE)

    p = np.array([1.0, -1.0], dtype=np.float64)
    entry = _make_entry(dvm, "e1", p, MEMORY_AUTOASSOCIATIVE)
    dvm._entries = [entry]
    dvm.entries_changed.emit()

    vm.train()
    assert vm.state == TrainingState.COMPLETED

    # Now clear the dataset - model should be invalidated
    dvm.clear_dataset()
    assert vm.state == TrainingState.NOT_READY
    assert vm.current_result is None


def test_training_viewmodel_remains_responsive_during_training() -> None:
    """GUI state transitions to TRAINING immediately (not blocked)."""

    from associative_memory.application.services.training_service import (
        TrainingService,
    )

    class SlowTrainingService(TrainingService):
        """Training service that sleeps to simulate long-running work."""

        def train(self, dataset, *, encoding, trace=False):
            time.sleep(0.2)  # Simulate long-running operation
            return super().train(dataset, encoding=encoding, trace=trace)

    vm = TrainingViewModel(training_service=SlowTrainingService())
    dvm = vm._dataset_view_model

    dvm.set_memory_type(MEMORY_AUTOASSOCIATIVE)
    p = np.array([1.0, -1.0], dtype=np.float64)
    entry = _make_entry(dvm, "e1", p, MEMORY_AUTOASSOCIATIVE)
    dvm._entries = [entry]
    dvm.entries_changed.emit()

    assert vm.state == TrainingState.READY

    # Start training asynchronously - this runs on a background thread
    vm.train_async()

    # Immediately after starting, state should be TRAINING (GUI not blocked)
    assert vm.state == TrainingState.TRAINING

    # Wait for completion
    deadline = time.time() + 5.0
    while vm.state == TrainingState.TRAINING and time.time() < deadline:
        time.sleep(0.01)

    assert vm.state == TrainingState.COMPLETED
    assert vm.current_result is not None


def test_training_worker_runs_on_separate_thread(qt_app) -> None:
    """Verify the training worker moves to a QThread."""

    from associative_memory.application.services.training_service import (
        TrainingService,
        TrainingDataset,
    )

    service = TrainingService()
    dvm = DatasetViewModel()
    dvm.set_memory_type(MEMORY_AUTOASSOCIATIVE)
    p = np.array([1.0, -1.0], dtype=np.float64)
    entry = _make_entry(dvm, "e1", p, MEMORY_AUTOASSOCIATIVE)
    dvm._entries = [entry]
    dvm.entries_changed.emit()

    dataset = TrainingDataset(
        patterns=tuple((e.result.encoded_vector.astype(np.float64),) * 2 for e in dvm._entries),
        memory_type=MemoryType.AUTOASSOCIATIVE,
    )

    worker = _TrainingWorker(
        service,
        dataset,
        "binary",
        False,
    )

    # Before moving to thread, worker has no thread
    assert worker.thread() is QThread.currentThread()

    thread = QThread()
    worker.moveToThread(thread)
    assert worker.thread() is thread
    thread.quit()
    thread.wait()
