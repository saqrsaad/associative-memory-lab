"""Weight-matrix viewer built on top of MatrixView."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from associative_memory.presentation.theme.spacing import SPACE_XS, SPACE_XXS
from associative_memory.presentation.widgets.matrix_view import MatrixView

_OBJECT_NAME = "WeightMatrixView"


class WeightMatrixView(QWidget):
    """A weight-matrix viewer with metrics, legend, and scrollable matrix.

    Composite widget: a MatrixView inside a QScrollArea, plus dimension
    metrics and a signed-value legend. All arrays are read-only input.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName(_OBJECT_NAME)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACE_XS)

        # Header: dimensions.
        self._dim_label = QLabel(self)
        self._dim_label.setObjectName("MatrixDimension")
        layout.addWidget(self._dim_label)

        # Matrix area inside a scroll area.
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        self._matrix_view = MatrixView(scroll)
        scroll.setWidget(self._matrix_view)
        layout.addWidget(scroll, stretch=1)

        # Legend.
        legend_row = QHBoxLayout()
        legend_row.setContentsMargins(0, 0, 0, 0)
        legend_row.setSpacing(SPACE_XXS)

        neg_swatch = QFrame(self)
        neg_swatch.setObjectName("LegendSwatch")
        neg_swatch.setAutoFillBackground(True)
        neg_palette = neg_swatch.palette()
        neg_palette.setColor(neg_swatch.backgroundRole(), QColor(142, 202, 230))
        neg_swatch.setPalette(neg_palette)

        zero_swatch = QFrame(self)
        zero_swatch.setObjectName("LegendSwatch")
        zero_swatch.setAutoFillBackground(True)
        zero_palette = zero_swatch.palette()
        zero_palette.setColor(zero_swatch.backgroundRole(), QColor(255, 255, 255))
        zero_palette.setColor(zero_swatch.foregroundRole(), QColor(209, 213, 219))
        zero_swatch.setPalette(zero_palette)

        pos_swatch = QFrame(self)
        pos_swatch.setObjectName("LegendSwatch")
        pos_swatch.setAutoFillBackground(True)
        pos_palette = pos_swatch.palette()
        pos_palette.setColor(pos_swatch.backgroundRole(), QColor(255, 154, 162))
        pos_swatch.setPalette(pos_palette)

        legend_row.addWidget(neg_swatch)
        legend_row.addWidget(QLabel("Negative", self))
        legend_row.addWidget(zero_swatch)
        legend_row.addWidget(QLabel("Zero", self))
        legend_row.addWidget(pos_swatch)
        legend_row.addWidget(QLabel("Positive", self))
        legend_row.addStretch(1)

        legend_label = QLabel("Signed weight values", self)
        legend_label.setObjectName("MatrixLegendLabel")
        legend_row.insertWidget(0, legend_label)

        layout.addLayout(legend_row)

        self._matrix: NDArray[np.float64] | None = None

    # ------------------------------------------------------------------ data

    def set_matrix(
        self,
        matrix: NDArray[np.float64] | None,
        *,
        row_labels: list[str] | None = None,
        col_labels: list[str] | None = None,
    ) -> None:
        """Display ``matrix`` with optional labels."""

        self._matrix = None if matrix is None else np.asarray(matrix, dtype=np.float64)
        self._matrix_view.set_matrix(self._matrix, row_labels=row_labels, col_labels=col_labels)

        if self._matrix is None:
            self._dim_label.setText("No matrix")
        else:
            rows, cols = self._matrix.shape
            self._dim_label.setText(f"{rows}×{cols}  ({rows * cols} elements)")

        self.update()

    def matrix(self) -> NDArray[np.float64] | None:
        """The currently displayed matrix (read-only input)."""

        return self._matrix

    def set_row_labels(self, labels: list[str] | None) -> None:
        self._matrix_view.set_row_labels(labels)

    def set_col_labels(self, labels: list[str] | None) -> None:
        self._matrix_view.set_col_labels(labels)
