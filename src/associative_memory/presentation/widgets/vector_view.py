"""Flattened encoded vector visualization for the dataset workspace."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

_OBJECT_NAME = "VectorView"
_VALUES_OBJECT_NAME = "VectorValues"
_LENGTH_OBJECT_NAME = "VectorLength"
_TOGGLE_OBJECT_NAME = "VectorToggle"


class VectorView(QWidget):
    """Displays an encoded vector as monospaced, row-wrapped values.

    Row wrapping mirrors the row-major layout of the pixel grid so the
    vector view and the pixel grid agree. Presentation-only: it renders
    caller-provided vectors and performs no numerical work.

    For large vectors callers may pass ``max_values`` to show a compact
    head-of-vector summary. The exact values remain available through the
    "Show all values" control, so underlying data is never silently hidden.
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
        self._values_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        self._length_label = QLabel(self)
        self._length_label.setObjectName(_LENGTH_OBJECT_NAME)
        self._length_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self._toggle_button = QPushButton("Show all values", self)
        self._toggle_button.setObjectName(_TOGGLE_OBJECT_NAME)
        self._toggle_button.setToolTip(
            "Reveal every exact vector value instead of the summary."
        )
        self._toggle_button.clicked.connect(self._toggle_truncation)
        self._toggle_button.hide()

        layout.addWidget(self._values_label)
        layout.addWidget(self._length_label)
        layout.addWidget(self._toggle_button)

        self._vector: NDArray[np.integer] | None = None
        self._row_width: int | None = None
        self._max_values: int | None = None
        self._show_all = False
        self._display_text = ""
        self._length_text = ""

        self._refresh()

    def set_vector(
        self,
        vector: NDArray[np.integer] | None,
        *,
        row_width: int | None = None,
        max_values: int | None = None,
    ) -> None:
        """Show ``vector`` wrapped at ``row_width`` values per row.

        Args:
            vector: Values to display, or ``None`` to clear the view.
            row_width: Values per display row, or ``None`` for a single row.
            max_values: Optional maximum number of values shown in the
                compact summary. The user can reveal all exact values.
        """

        if max_values is not None and max_values < 1:
            raise ValueError("max_values must be a positive integer or None.")

        self._vector = None if vector is None else np.asarray(vector)
        self._row_width = row_width
        self._max_values = max_values
        self._show_all = False
        self._refresh()

    def vector(self) -> NDArray[np.integer] | None:
        """The currently displayed vector (read-only input)."""

        return self._vector

    def row_width(self) -> int | None:
        """Values per display row, or ``None`` for a single row."""

        return self._row_width

    @property
    def max_values(self) -> int | None:
        """The summary limit, or ``None`` when every value is shown."""

        return self._max_values

    @property
    def display_text(self) -> str:
        """The rendered vector text (used by focused assertions)."""

        return self._display_text

    @property
    def is_truncated(self) -> bool:
        """Whether the current display hides values behind the toggle."""

        return (
            self._vector is not None
            and self._max_values is not None
            and len(self._vector) > self._max_values
            and not self._show_all
        )

    def _toggle_truncation(self) -> None:
        self._show_all = not self._show_all
        self._refresh()

    def _refresh(self) -> None:
        if self._vector is None:
            self._values_label.setText("")
            self._length_label.setText("No vector")
            self._display_text = ""
            self._length_text = ""
            self._toggle_button.hide()
            return

        all_values = [_format_value(value) for value in self._vector]
        total = len(all_values)
        exhausted = (
            self._max_values is None
            or total <= self._max_values
            or self._show_all
        )
        values = all_values if exhausted else all_values[: self._max_values]

        if self._row_width and self._row_width > 0:
            rows = [
                " ".join(values[i : i + self._row_width])
                for i in range(0, len(values), self._row_width)
            ]
        else:
            rows = [" ".join(values)]

        if not exhausted:
            hidden = total - len(values)
            rows.append(
                f"… {hidden} more value(s) hidden; use the button below to "
                "inspect exact values."
            )

        display_text = "\n".join(rows)
        self._values_label.setText(display_text)
        self._length_label.setText(f"Vector length: {total}")
        self._display_text = display_text
        self._length_text = f"Vector length: {total}"

        needs_toggle = self._max_values is not None and total > self._max_values
        self._toggle_button.setVisible(needs_toggle)
        self._toggle_button.setText(
            "Show summary" if self._show_all else f"Show all {total} values"
        )


def _format_value(value: object) -> str:
    """Render a value without hiding or truncating the underlying number."""

    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return np.format_float_positional(numeric, unique=True, trim="-")
