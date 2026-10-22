"""Unit tests for the WeightMatrixView widget."""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QApplication

from associative_memory.presentation.widgets.weight_matrix_view import WeightMatrixView


def test_weight_matrix_view_instantiates(qt_app: QApplication) -> None:
    view = WeightMatrixView()
    assert view.objectName() == "WeightMatrixView"
    assert view.matrix() is None


def test_weight_matrix_view_set_and_get_matrix(qt_app: QApplication) -> None:
    view = WeightMatrixView()
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    view.set_matrix(matrix)

    assert view.matrix() is not None
    np.testing.assert_array_equal(view.matrix(), matrix)


def test_weight_matrix_view_dim_label_updates(qt_app: QApplication) -> None:
    view = WeightMatrixView()
    matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    view.set_matrix(matrix)

    assert "2x3" in view._dim_label.text()
    assert "6 elements" in view._dim_label.text()


def test_weight_matrix_view_clears_on_none(qt_app: QApplication) -> None:
    view = WeightMatrixView()
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    view.set_matrix(matrix)
    view.set_matrix(None)

    assert view.matrix() is None
    assert view._dim_label.text() == "No matrix"


def test_weight_matrix_view_row_col_labels(qt_app: QApplication) -> None:
    view = WeightMatrixView()
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    view.set_matrix(matrix, row_labels=["s0", "s1"], col_labels=["t0", "t1"])

    assert view._matrix_view._row_labels == ["s0", "s1"]
    assert view._matrix_view._col_labels == ["t0", "t1"]


def test_weight_matrix_view_paint_does_not_crash(qt_app: QApplication) -> None:
    view = WeightMatrixView()
    matrix = np.random.randn(4, 4)
    view.set_matrix(matrix)
    view.show()

    pixmap = view.grab()
    assert not pixmap.isNull()


def test_weight_matrix_view_size_hint(qt_app: QApplication) -> None:
    view = WeightMatrixView()
    size = view.sizeHint()
    assert size.width() > 0
    assert size.height() > 0
