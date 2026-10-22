"""Unit tests for the MatrixView widget."""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QApplication

from associative_memory.presentation.widgets.matrix_view import MatrixView


def test_matrix_view_instantiates(qt_app: QApplication) -> None:
    view = MatrixView()
    assert view.objectName() == "MatrixView"
    assert view.matrix() is None


def test_matrix_view_set_and_get_matrix(qt_app: QApplication) -> None:
    view = MatrixView()
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    view.set_matrix(matrix)

    assert view.matrix() is not None
    np.testing.assert_array_equal(view.matrix(), matrix)


def test_matrix_view_clears_on_none(qt_app: QApplication) -> None:
    view = MatrixView()
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    view.set_matrix(matrix)
    view.set_matrix(None)

    assert view.matrix() is None


def test_matrix_view_row_col_labels(qt_app: QApplication) -> None:
    view = MatrixView()
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    view.set_matrix(matrix, row_labels=["r0", "r1"], col_labels=["c0", "c1"])

    assert view._row_labels == ["r0", "r1"]
    assert view._col_labels == ["c0", "c1"]



def test_matrix_view_paint_does_not_crash(qt_app: QApplication) -> None:
    view = MatrixView()
    matrix = np.random.randn(4, 4)
    view.set_matrix(matrix)
    view.show()

    # Force a paint event via grab; should not raise.
    pixmap = view.grab()
    assert not pixmap.isNull()


def test_matrix_view_value_color_extremes() -> None:
    from associative_memory.presentation.widgets.matrix_view import MatrixView

    assert MatrixView._value_color(0.0, 1.0) == (255, 255, 255)
    assert MatrixView._value_color(1.0, 1.0)[0] == 255  # red channel max
    assert MatrixView._value_color(-1.0, 1.0)[2] == 255  # blue channel max
    assert MatrixView._value_color(5.0, 0.0) == (255, 255, 255)


def test_matrix_view_format_value() -> None:
    from associative_memory.presentation.widgets.matrix_view import MatrixView

    assert MatrixView._format_value(1.0) == "1"
    assert MatrixView._format_value(1.5) == "1.500"
    assert MatrixView._format_value(0.0) == "0"


def test_matrix_view_size_hint_without_matrix(qt_app: QApplication) -> None:
    view = MatrixView()
    size = view.sizeHint()
    assert size.width() > 0
    assert size.height() > 0
