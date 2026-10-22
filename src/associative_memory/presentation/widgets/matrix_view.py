"""Painted matrix visualization for the training workspace."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import QWidget

from associative_memory.presentation.theme.palette import PALETTE
from associative_memory.presentation.theme.typography import MONO_FONT_FAMILY

_OBJECT_NAME = "MatrixView"

#: Cells smaller than this never show value text.
_VALUE_TEXT_MIN_CELL = 16


class MatrixView(QWidget):
    """A painted matrix viewer that avoids one-widget-per-cell overhead.

    A single widget draws the entire matrix using QPainter, so even large
    matrices stay lightweight. Cell (row ``r``, column ``c``) maps to grid
    position (``r``, ``c``) matching the row-vector convention.

    Colors communicate numeric value only, never status: negative values use
    cool colors, positive values use warm colors, zero is neutral. The exact
    value is printed in each cell when the cell is large enough to read it.

    Attributes:
        matrix: The currently displayed matrix (read-only input).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName(_OBJECT_NAME)
        self._matrix: NDArray[np.float64] | None = None
        self._row_labels: list[str] | None = None
        self._col_labels: list[str] | None = None
        self._show_row_labels = False
        self._show_col_labels = False
        self._label_width = 40
        self._label_height = 24
        self.setMinimumSize(120, 120)

    # ------------------------------------------------------------------ data

    def set_matrix(
        self,
        matrix: NDArray[np.float64] | None,
        *,
        row_labels: list[str] | None = None,
        col_labels: list[str] | None = None,
    ) -> None:
        """Display ``matrix`` with optional row/column labels.

        Clears the view when ``matrix`` is ``None``.
        """

        self._matrix = None if matrix is None else np.asarray(matrix, dtype=np.float64)
        self._row_labels = row_labels
        self._col_labels = col_labels
        self.update()

    def set_row_labels(self, labels: list[str] | None) -> None:
        """Set row labels shown along the left edge."""

        self._row_labels = labels
        self.update()

    def set_col_labels(self, labels: list[str] | None) -> None:
        """Set column labels shown along the top edge."""

        self._col_labels = labels
        self.update()

    def matrix(self) -> NDArray[np.float64] | None:
        """The currently displayed matrix (read-only input)."""

        return self._matrix

    # ------------------------------------------------------------------ layout hints

    def sizeHint(self) -> QSize:
        if self._matrix is None:
            return QSize(200, 200)
        rows, cols = self._matrix.shape
        cell = max(_VALUE_TEXT_MIN_CELL, 24)
        width = cols * cell + 2
        height = rows * cell + 2
        if self._show_row_labels:
            width += self._label_width
        if self._show_col_labels:
            height += self._label_height
        return QSize(width, height)

    # ------------------------------------------------------------------ painting

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        if self._matrix is None:
            painter.setPen(QColor(PALETTE.TEXT_SECONDARY))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "No matrix"
            )
            return

        rows, cols = self._matrix.shape
        cell = max(_VALUE_TEXT_MIN_CELL, min(self.width(), self.height()) // max(cols, rows, 1))
        cell = max(cell, _VALUE_TEXT_MIN_CELL)
        total_w = cols * cell
        total_h = rows * cell

        # Available space after labels.
        avail_w = self.width() - (self._label_width if self._show_row_labels else 0)
        avail_h = self.height() - (self._label_height if self._show_col_labels else 0)

        x_offset = self._label_width if self._show_row_labels else 0
        y_offset = self._label_height if self._show_col_labels else 0
        x0 = x_offset + max(0.0, (avail_w - total_w) / 2.0)
        y0 = y_offset + max(0.0, (avail_h - total_h) / 2.0)

        # Draw labels.
        if self._show_row_labels and self._row_labels is not None:
            font = QFont(MONO_FONT_FAMILY, max(7, int(cell) - 3))
            painter.setFont(font)
            painter.setPen(QColor(PALETTE.TEXT_PRIMARY))
            for row in range(rows):
                label_rect = QRectF(0.0, y0 + row * cell, self._label_width, cell)
                painter.drawText(
                    label_rect,
                    Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                    str(self._row_labels[row]),
                )

        if self._show_col_labels and self._col_labels is not None:
            font = QFont(MONO_FONT_FAMILY, max(7, int(cell) - 3))
            painter.setFont(font)
            painter.setPen(QColor(PALETTE.TEXT_PRIMARY))
            for col in range(cols):
                label_rect = QRectF(x0 + col * cell, 0.0, cell, self._label_height)
                painter.drawText(
                    label_rect,
                    Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
                    str(self._col_labels[col]),
                )

        # Draw cells.
        show_text = cell >= _VALUE_TEXT_MIN_CELL
        if show_text:
            font = QFont(MONO_FONT_FAMILY, max(6, int(cell) - 4))
            painter.setFont(font)

        abs_max = float(np.max(np.abs(self._matrix))) if self._matrix.size else 0.0
        border_pen = painter.pen()
        border_pen.setColor(QColor(PALETTE.BORDER))
        painter.setPen(border_pen)

        text_rect = QRectF()
        for row in range(rows):
            for col in range(cols):
                value = float(self._matrix[row, col])
                x = x0 + col * cell
                y = y0 + row * cell
                text_rect.setRect(x, y, cell, cell)

                fill = self._value_color(value, abs_max)
                painter.fillRect(text_rect, QColor(*fill))

                painter.drawRect(text_rect)

                if show_text:
                    text_color = (
                        QColor(PALETTE.SURFACE)
                        if abs(value) > abs_max * 0.5
                        else QColor(PALETTE.TEXT_PRIMARY)
                    )
                    painter.setPen(text_color)
                    painter.drawText(
                        text_rect,
                        Qt.AlignmentFlag.AlignCenter,
                        self._format_value(value),
                    )
                    painter.setPen(border_pen)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _value_color(value: float, abs_max: float) -> tuple[int, int, int]:
        """Return RGB for a matrix value using a diverging colormap.

        Negative values are blue-tinted, positive values are red-tinted,
        zero is white. Intensity scales with magnitude relative to abs_max.
        """

        if abs_max == 0.0:
            return (255, 255, 255)

        intensity = min(abs(value) / abs_max, 1.0)
        if value > 0:
            # Red channel: white -> strong red
            return (255, int(255 * (1.0 - intensity * 0.6)), int(255 * (1.0 - intensity * 0.6)))
        if value < 0:
            # Blue channel: white -> strong blue
            return (int(255 * (1.0 - intensity * 0.6)), int(255 * (1.0 - intensity * 0.6)), 255)
        return (255, 255, 255)

    @staticmethod
    def _format_value(value: float) -> str:
        """Format a matrix cell value for display."""

        if value == int(value):
            return str(int(value))
        return f"{value:.3f}"
