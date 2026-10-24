"""UI integration tests for the TrainingView."""

from __future__ import annotations

import time

import numpy as np
from PySide6.QtWidgets import QApplication

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
)
from associative_memory.presentation.views.training_view import TrainingView


def test_training_view_instantiates(qt_app: QApplication) -> None:
    view = TrainingView()
    assert view is not None


def test_training_view_shows_not_ready_state(qt_app: QApplication) -> None:
    view = TrainingView()
    view.show()

    assert view._view_model.state == TrainingState.NOT_READY
    assert not view._train_button.isEnabled()


def test_training_view_heteroassociative_training(qt_app: QApplication) -> None:
    dvm = DatasetViewModel()
    dvm.set_memory_type(MEMORY_HETEROASSOCIATIVE)

    view = TrainingView(dataset_view_model=dvm)
    view.show()

    s1 = np.array([1.0, -1.0], dtype=np.float64)
    s2 = np.array([-1.0, 1.0], dtype=np.float64)

    entry1 = _make_entry(dvm, "e1", s1, MEMORY_HETEROASSOCIATIVE, target_label="e2")
    entry2 = _make_entry(dvm, "e2", s2, MEMORY_HETEROASSOCIATIVE, target_label="e1")

    dvm._entries = [entry1, entry2]
    dvm.entries_changed.emit()

    # The fabricated vectors are bipolar-valued; declare bipolar so the
    # encoding matches the pattern values (as the real preprocessor would).
    view._view_model.set_encoding("bipolar")

    view._refresh_state()
    assert view._view_model.state == TrainingState.READY
    assert view._train_button.isEnabled()

    view._train_button.click()

    # The Train button runs training asynchronously; wait for the queued
    # result to be delivered to this thread.
    _wait_until_not_training(view, qt_app)
    assert view._view_model.state == TrainingState.COMPLETED
    result = view._view_model.current_result
    assert result is not None
    assert result.pair_count == 2

    expected = np.outer(s1, s2) + np.outer(s2, s1)
    np.testing.assert_array_equal(result.weight_matrix, expected)


def test_training_view_rejects_empty_dataset(qt_app: QApplication) -> None:
    view = TrainingView()
    view.show()

    assert not view._train_button.isEnabled()
    view._train_button.click()
    assert view._view_model.state == TrainingState.NOT_READY


def test_training_view_retry_after_failure_is_usable(qt_app: QApplication) -> None:
    view = TrainingView()
    view.show()

    dvm = view._dataset_view_model
    entry = _make_entry(
        dvm, "e1", np.array([1.0, -1.0, 1.0], dtype=np.float64), MEMORY_AUTOASSOCIATIVE
    )
    dvm._entries = [entry]
    dvm.entries_changed.emit()

    # The default training encoding is binary; -1 is not a valid binary
    # value, so training must fail with a clear error.
    view._view_model.set_encoding("binary")
    view._refresh_state()
    assert view._view_model.state == TrainingState.READY

    view._train_button.click()
    _wait_until_not_training(view, qt_app)
    assert view._view_model.state == TrainingState.FAILED

    # Drain the failed worker thread before retrying.
    deadline = time.time() + 5.0
    while view._view_model._thread is not None and time.time() < deadline:
        qt_app.processEvents()
        time.sleep(0.01)

    # The retry path must stay usable after a failed run.
    assert view._train_button.isEnabled()

    # Correct the configuration and retry: training must then complete.
    view._view_model.set_encoding("bipolar")
    view._refresh_state()
    assert view._view_model.state == TrainingState.READY

    view._train_button.click()
    _wait_until_not_training(view, qt_app)
    assert view._view_model.state == TrainingState.COMPLETED
    assert view._view_model.current_result is not None


def test_training_view_memory_type_combo(qt_app: QApplication) -> None:
    view = TrainingView()
    view.show()

    assert view._memory_combo.currentText() == "Autoassociative"
    view._memory_combo.setCurrentIndex(1)
    assert view._view_model.memory_type is not None


def test_training_view_encoding_combo(qt_app: QApplication) -> None:
    view = TrainingView()
    view.show()

    assert view._encoding_combo.currentText() == "Binary (0/1)"
    view._encoding_combo.setCurrentIndex(1)
    assert view._view_model.encoding == "bipolar"


def test_training_view_encoding_combo_tracks_viewmodel(qt_app: QApplication) -> None:
    view = TrainingView()
    view.show()

    assert view._view_model.encoding == "binary"
    assert view._encoding_combo.currentText() == "Binary (0/1)"

    # Viewmodel-driven encoding updates must be reflected in the combo.
    view._view_model.set_encoding("bipolar")

    assert view._view_model.encoding == "bipolar"
    assert view._encoding_combo.currentIndex() == 1
    assert view._encoding_combo.currentText() == "Bipolar (-1/+1)"


def test_training_view_shows_weight_matrix(qt_app: QApplication) -> None:
    dvm = DatasetViewModel()
    dvm.set_memory_type(MEMORY_AUTOASSOCIATIVE)

    view = TrainingView(dataset_view_model=dvm)
    view.show()

    p = np.array([1.0, -1.0, 1.0], dtype=np.float64)
    entry = _make_entry(dvm, "e1", p, MEMORY_AUTOASSOCIATIVE)
    dvm._entries = [entry]
    dvm.entries_changed.emit()

    # The fabricated vector is bipolar-valued; declare bipolar so the
    # encoding matches the pattern values (as the real preprocessor would).
    view._view_model.set_encoding("bipolar")

    view._refresh_state()
    view._train_button.click()

    _wait_until_not_training(view, qt_app)
    assert view._view_model.state == TrainingState.COMPLETED
    matrix_view = view._weight_matrix_view
    assert matrix_view.matrix() is not None
    assert matrix_view.matrix().shape == (3, 3)


def test_training_view_trace_exposes_calculation_details(qt_app: QApplication) -> None:
    dvm = DatasetViewModel()
    dvm.set_memory_type(MEMORY_HETEROASSOCIATIVE)

    view = TrainingView(dataset_view_model=dvm)
    view.show()

    s1 = np.array([1.0, -1.0], dtype=np.float64)
    s2 = np.array([-1.0, 1.0], dtype=np.float64)
    dvm._entries = [
        _make_entry(dvm, "e1", s1, MEMORY_HETEROASSOCIATIVE, target_label="e2"),
        _make_entry(dvm, "e2", s2, MEMORY_HETEROASSOCIATIVE, target_label="e1"),
    ]
    dvm.entries_changed.emit()

    view._view_model.set_encoding("bipolar")
    view._view_model.set_trace_enabled(True)
    view._refresh_state()

    view._train_button.click()
    _wait_until_not_training(view, qt_app)
    assert view._view_model.state == TrainingState.COMPLETED

    # The final weight matrix is exposed through the weight-matrix viewer,
    # with row/column labels enabled for the s/t labeling convention.
    assert view._weight_matrix_view.matrix() is not None
    assert view._weight_matrix_view._matrix_view._show_row_labels is True
    assert view._weight_matrix_view._matrix_view._show_col_labels is True

    # The learning trace exposes source, target, contribution and
    # accumulated values, plus a final weight-matrix summary.
    assert view._trace_list.count() >= 2
    first_text = view._trace_list.item(0).text()
    assert "Pair 0" in first_text
    assert "source:" in first_text
    assert "target:" in first_text
    assert "contribution" in first_text
    assert "accumulated" in first_text

    final_text = view._trace_list.item(view._trace_list.count() - 1).text()
    assert "Final weight matrix" in final_text


def test_training_view_allows_retraining_after_completion(qt_app: QApplication) -> None:
    dvm = DatasetViewModel()
    dvm.set_memory_type(MEMORY_AUTOASSOCIATIVE)

    view = TrainingView(dataset_view_model=dvm)
    view.show()

    entry = _make_entry(
        dvm, "e1", np.array([1.0, -1.0, 1.0], dtype=np.float64), MEMORY_AUTOASSOCIATIVE
    )
    dvm._entries = [entry]
    dvm.entries_changed.emit()

    view._view_model.set_encoding("bipolar")
    view._refresh_state()

    view._train_button.click()
    _wait_until_not_training(view, qt_app)
    assert view._view_model.state == TrainingState.COMPLETED
    assert view._view_model.current_result is not None
    assert view._train_button.isEnabled()

    # Drain worker-thread shutdown before starting the next training run.
    deadline = time.time() + 5.0
    while view._view_model._thread is not None and time.time() < deadline:
        qt_app.processEvents()
        time.sleep(0.01)

    # A completed run must allow starting another training run without an
    # unrelated dataset or configuration change.
    view._train_button.click()
    _wait_until_not_training(view, qt_app)
    assert view._view_model.state == TrainingState.COMPLETED
    assert view._view_model.current_result is not None


def _wait_until_not_training(
    view: TrainingView, qt_app: QApplication, timeout_s: float = 5.0
) -> None:
    """Process Qt events until the asynchronous training state leaves TRAINING."""

    deadline = time.time() + timeout_s
    vm = view._view_model
    while vm.state == TrainingState.TRAINING and time.time() < deadline:
        qt_app.processEvents()
        time.sleep(0.01)


def _make_entry(
    dvm: DatasetViewModel,
    entry_id: str,
    vector: np.ndarray,
    memory_type: str,
    target_label: str | None = None,
) -> DatasetEntry:
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
