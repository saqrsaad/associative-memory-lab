"""Flattened encoded vector visualization for the dataset workspace."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

_OBJECT_NAME = "VectorView"
_VALUES_OBJECT_NAME = "VectorValues"
_LENGTH_OBJECT_NAME = "VectorLength"


class VectorView(QWidget):
    """Displays an encoded vector as monospaced, row-wrapped values.

    Row wrapping mirrors the row-major layout of the pixel grid so the
    vector view and the pixel grid agree. Presentation-only: it renders
    caller-provided vectors and performs no numerical work.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName(_OBJECT_NAME)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._values_label = QLabel(self)
        self._values_label.setObjectName(_VALUES_OBJECT_NAME)
        self._values_label.setWordWrap(False)
        self._values_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )

        self._length_label = QLabel(self)
        self._length_label.setObjectName(_LENGTH_OBJECT_NAME)
        self._length_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        layout.addWidget(self._values_label)
        layout.addWidget(self._length_label)

        self._vector: NDArray[np.integer] | None = None
        self._row_width: int | None = None
        self._display_text = ""
        self._length_text = ""

        self._refresh()

    def set_vector(
        self,
        vector: NDArray[np.integer] | None,
        *,
        row_width: int | None = None,
    ) -> None:
        """Show ``vector`` wrapped at ``row_width`` values per row."""

        self._vector = None if vector is None else np.asarray(vector)
        self._row_width = row_width
        self._refresh()

    def vector(self) -> NDArray[np.integer] | None:
        """The currently displayed vector (read-only input)."""

        return self._vector

    def row_width(self) -> int | None:
        """Values per display row, or ``None`` for a single row."""

        return self._row_width

    @property
    def display_text(self) -> str:
        """The rendered vector text (used by focused assertions)."""

        return self._display_text

    def _refresh(self) -> None:
        if self._vector is None:
            self._values_label.setText("")
            self._length_label.setText("No vector")
            self._display_text = ""
            self._length_text = ""
            return

        values = [str(int(value)) for value in self._vector]
        if self._row_width and self._row_width > 0:
            rows = [
                " ".join(values[i : i + self._row_width])
                for i in range(0, len(values), self._row_width)
            ]
        else:
            rows = [" ".join(values)]

        self._values_label.setText("\n".join(rows))
        self._length_label.setText(f"Vector length: {len(values)}")
        self._display_text = "\n".join(rows)
        self._length_text = f"Vector length: {len(values)}"
