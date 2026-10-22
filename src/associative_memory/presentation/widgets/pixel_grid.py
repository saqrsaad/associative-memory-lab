"""Encoded pixel grid visualization for the dataset workspace."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtWidgets import QWidget

from associative_memory.presentation.theme.palette import PALETTE
from associative_memory.presentation.theme.typography import MONO_FONT_FAMILY

_OBJECT_NAME = "PixelGrid"

#: Cells smaller than this never show value text (it would be unreadable).
_VALUE_TEXT_MIN_CELL = 18


class PixelGrid(QWidget):
    """Painted grid of an encoded matrix (row-major layout).

    A single widget draws the whole grid instead of one widget per pixel,
    so even large matrices stay lightweight. Cell (row ``r``, column ``c``)
    of the matrix maps to grid row ``r``, column ``c`` — matching the
    algorithm package's row-vector convention and row-major flattening.

    Colors communicate value, never status: positive values are dark,
    negative values are medium gray, and zero values are white.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName(_OBJECT_NAME)
        self._matrix: NDArray[np.integer] | None = None
        self.setMinimumSize(120, 120)

    def set_matrix(self, matrix: NDArray[np.integer] | None) -> None:
        """Show ``matrix`` or clear the grid when ``None``."""

        self._matrix = None if matrix is None else np.asarray(matrix)
        self.update()

    def matrix(self) -> NDArray[np.integer] | None:
        """The currently displayed matrix (read-only input)."""

        return self._matrix

    def sizeHint(self) -> QSize:
        if self._matrix is None:
            return QSize(200, 200)
        rows, cols = self._matrix.shape
        cell = max(_VALUE_TEXT_MIN_CELL, 24)
        return QSize(cols * cell + 2, rows * cell + 2)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        if self._matrix is None:
            painter.setPen(QColor(PALETTE.TEXT_SECONDARY))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "No pattern"
            )
            return

        rows, cols = self._matrix.shape
        cell_w = self.width() / max(cols, 1)
        cell_h = self.height() / max(rows, 1)
        cell = min(cell_w, cell_h)
        total_w = cols * cell
        total_h = rows * cell
        x0 = (self.width() - total_w) / 2.0
        y0 = (self.height() - total_h) / 2.0

        show_text = cell >= _VALUE_TEXT_MIN_CELL
        if show_text:
            font = QFont(MONO_FONT_FAMILY, max(6, int(cell) - 4))
            painter.setFont(font)

        border_pen = painter.pen()
        border_pen.setColor(QColor(PALETTE.BORDER))
        painter.setPen(border_pen)

        text_rect = QRectF()
        for row in range(rows):
            for col in range(cols):
                value = int(self._matrix[row, col])
                x = x0 + col * cell
                y = y0 + row * cell
                text_rect.setRect(x, y, cell, cell)

                if value > 0:
                    painter.fillRect(text_rect, QColor(PALETTE.TEXT_PRIMARY))
                    text_color = QColor(PALETTE.SURFACE)
                elif value < 0:
                    painter.fillRect(text_rect, QColor(PALETTE.TEXT_SECONDARY))
                    text_color = QColor(PALETTE.SURFACE)
                else:
                    painter.fillRect(text_rect, QColor(PALETTE.SURFACE))
                    text_color = QColor(PALETTE.TEXT_PRIMARY)

                painter.drawRect(text_rect)

                if show_text:
                    painter.setPen(text_color)
                    painter.drawText(
                        text_rect,
                        Qt.AlignmentFlag.AlignCenter,
                        str(value),
                    )
                    painter.setPen(border_pen)
