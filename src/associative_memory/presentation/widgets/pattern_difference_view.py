"""Pixel-level comparison view for recalled patterns.

The view renders a category grid derived from the actual recalled and
expected encoded vectors. Categories are computed here only for display
(matching ON/OFF, missing foreground, extra foreground); the underlying
mismatch metrics always come from the application service. Colours are
paired with a textual legend so interpretation never depends on colour
alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from associative_memory.presentation.theme.palette import PALETTE
from associative_memory.presentation.theme.spacing import SPACE_XS, SPACE_XXS

_OBJECT_NAME = "PatternDifferenceView"
_GRID_OBJECT_NAME = "DifferenceGrid"
_LEGEND_OBJECT_NAME = "DifferenceLegend"

#: Difference categories. The values are stored on the grid and exposed for
#: focused assertions; they are not algorithm output.
CATEGORY_MATCH_ON = 0
CATEGORY_MATCH_OFF = 1
CATEGORY_MISSING = 2
CATEGORY_EXTRA = 3
CATEGORY_UNAVAILABLE = 4

_CATEGORY_COLORS: dict[int, str] = {
    CATEGORY_MATCH_ON: PALETTE.TEXT_PRIMARY,
    CATEGORY_MATCH_OFF: PALETTE.SURFACE,
    CATEGORY_MISSING: PALETTE.ERROR_TEXT,
    CATEGORY_EXTRA: PALETTE.WARNING_TEXT,
    CATEGORY_UNAVAILABLE: PALETTE.NEUTRAL_SOFT,
}


@dataclass(frozen=True)
class DifferenceCounts:
    """Category counts derived from the displayed pattern pair."""

    matching_on: int
    matching_off: int
    missing: int
    extra: int

    @property
    def match_count(self) -> int:
        """Number of matching positions, both ON and OFF."""

        return self.matching_on + self.matching_off

    @property
    def mismatch_count(self) -> int:
        """Number of mismatching positions."""

        return self.missing + self.extra

    @property
    def element_count(self) -> int:
        """Total compared positions."""

        return self.match_count + self.mismatch_count


class _DifferenceGrid(QWidget):
    """Painted category grid; one QImage keeps large patterns cheap."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName(_GRID_OBJECT_NAME)
        self.setMinimumSize(160, 160)
        self._categories: NDArray[np.int8] | None = None
        self._message = "Comparison unavailable"

    def set_categories(self, categories: NDArray[np.int8] | None) -> None:
        self._categories = categories
        self.update()

    def set_message(self, message: str) -> None:
        self._message = message
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        frame = self.rect().adjusted(1, 1, -1, -1)
        painter.fillRect(frame, QColor(PALETTE.SURFACE))

        if self._categories is None:
            painter.fillRect(frame, QColor(PALETTE.NEUTRAL_SOFT))
            painter.setPen(QColor(PALETTE.TEXT_SECONDARY))
            painter.drawText(
                frame,
                Qt.AlignmentFlag.AlignCenter,
                self._message,
            )
            painter.setPen(QColor(PALETTE.BORDER))
            painter.drawRect(frame)
            return

        rows, cols = self._categories.shape
        image = self._build_image(self._categories)
        if image.isNull():
            return

        available = frame.adjusted(4, 4, -4, -4)
        scale = min(
            available.width() / max(cols, 1),
            available.height() / max(rows, 1),
        )
        width = max(1, int(cols * scale))
        height = max(1, int(rows * scale))
        x = available.x() + (available.width() - width) // 2
        y = available.y() + (available.height() - height) // 2
        target = QRectF(float(x), float(y), float(width), float(height))

        painter.drawImage(
            target,
            image,
            QRectF(0.0, 0.0, float(cols), float(rows)),
        )

        border_pen = QPen(QColor(PALETTE.BORDER))
        border_pen.setWidth(1)
        painter.setPen(border_pen)
        painter.drawRect(target)

        # Show a subtle cell grid only when it is readable and inexpensive.
        cell = min(width / max(cols, 1), height / max(rows, 1))
        if cell >= 12 and rows * cols <= 4096:
            grid_pen = QPen(QColor(PALETTE.BORDER))
            grid_pen.setWidth(1)
            painter.setPen(grid_pen)
            for col in range(1, cols):
                line_x = target.x() + col * target.width() / cols
                painter.drawLine(
                    int(line_x),
                    int(target.top()),
                    int(line_x),
                    int(target.bottom()),
                )
            for row in range(1, rows):
                line_y = target.y() + row * target.height() / rows
                painter.drawLine(
                    int(target.left()),
                    int(line_y),
                    int(target.right()),
                    int(line_y),
                )

    @staticmethod
    def _build_image(categories: NDArray[np.int8]) -> QImage:
        """Build an RGB image whose pixels match the category legend."""

        rgb = np.empty((*categories.shape, 3), dtype=np.uint8)
        for category, color in _CATEGORY_COLORS.items():
            rgb[categories == category] = _hex_to_rgb(color)
        rows, cols = categories.shape
        data = np.ascontiguousarray(rgb).tobytes()
        return QImage(
            data,
            cols,
            rows,
            cols * 3,
            QImage.Format.Format_RGB888,
        ).copy()


class PatternDifferenceView(QFrame):
    """Difference view with a painted grid and a textual legend/counts.

    Attribute policy: the matrices are read-only display input. The view
    derives categories only when the two matrices share a shape and contain
    values legal for the supplied encoding. Any other situation is reported
    as an unavailable comparison, never as manufactured agreement.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName(_OBJECT_NAME)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACE_XXS, SPACE_XXS, SPACE_XXS, SPACE_XXS)
        layout.setSpacing(SPACE_XS)

        self._grid = _DifferenceGrid(self)
        self._grid.setAccessibleName("Pattern difference grid")
        layout.addWidget(self._grid, stretch=1)

        self._legend_label = QLabel(self)
        self._legend_label.setObjectName(_LEGEND_OBJECT_NAME)
        self._legend_label.setWordWrap(True)
        self._legend_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self._legend_label)

        self._counts: DifferenceCounts | None = None
        self._difference_matrix: NDArray[np.int8] | None = None
        self._output_matrix: NDArray[np.float64] | None = None
        self._expected_matrix: NDArray[np.float64] | None = None
        self._reason: str | None = "No comparison has been run."

    # ------------------------------------------------------------------ data

    @property
    def counts(self) -> DifferenceCounts | None:
        """Category counts, or ``None`` while no valid comparison is shown."""

        return self._counts

    @property
    def is_available(self) -> bool:
        """Whether a valid comparison is currently displayed."""

        return self._counts is not None

    @property
    def reason(self) -> str | None:
        """Why the comparison is unavailable, or ``None`` when available."""

        return self._reason

    def difference_matrix(self) -> NDArray[np.int8] | None:
        """The displayed category matrix (read-only)."""

        return self._difference_matrix

    def output_matrix(self) -> NDArray[np.float64] | None:
        """The recalled matrix currently displayed."""

        return self._output_matrix

    def expected_matrix(self) -> NDArray[np.float64] | None:
        """The expected matrix currently displayed."""

        return self._expected_matrix

    # ------------------------------------------------------------------ content

    def set_matrices(
        self,
        output: NDArray | None,
        expected: NDArray | None,
        encoding: object,
        *,
        unavailable_reason: str | None = None,
    ) -> None:
        """Compare two equally shaped encoded matrices and display the result.

        ``encoding`` may be an ``Encoding`` member or its string value.
        Invalid shapes, values, or missing matrices make the comparison
        unavailable and never raise from the presentation layer.
        """

        try:
            output_array = _coerce_matrix(output, "output")
            expected_array = _coerce_matrix(expected, "expected")
            encoding_name = _encoding_name(encoding)
            _validate_encoded_values(output_array, encoding_name)
            _validate_encoded_values(expected_array, encoding_name)
        except (TypeError, ValueError) as exc:
            self.set_unavailable(str(exc))
            return

        if unavailable_reason:
            self.set_unavailable(unavailable_reason)
            return

        if output_array.shape != expected_array.shape:
            self.set_unavailable(
                "Output and expected pattern have incompatible matrix shapes "
                f"({output_array.shape} and {expected_array.shape})."
            )
            return

        output_on = output_array > 0
        expected_on = expected_array > 0
        categories = np.full(output_array.shape, CATEGORY_MATCH_OFF, dtype=np.int8)
        categories[output_on & expected_on] = CATEGORY_MATCH_ON
        categories[~output_on & expected_on] = CATEGORY_MISSING
        categories[output_on & ~expected_on] = CATEGORY_EXTRA

        counts = DifferenceCounts(
            matching_on=int(np.count_nonzero(output_on & expected_on)),
            matching_off=int(np.count_nonzero(~output_on & ~expected_on)),
            missing=int(np.count_nonzero(~output_on & expected_on)),
            extra=int(np.count_nonzero(output_on & ~expected_on)),
        )

        self._output_matrix = output_array
        self._expected_matrix = expected_array
        self._difference_matrix = categories
        self._counts = counts
        self._reason = None

        self._grid.set_categories(categories)
        self._legend_label.setText(self._legend_text(counts))
        self.setAccessibleDescription(self._legend_label.text())
        self._grid.setAccessibleDescription(self._legend_label.text())

    def set_unavailable(self, reason: str) -> None:
        """Display an explicit unavailable comparison state."""

        self._output_matrix = None
        self._expected_matrix = None
        self._difference_matrix = None
        self._counts = None
        self._reason = reason
        self._grid.set_categories(None)
        self._grid.set_message("Comparison unavailable")
        self._legend_label.setText(
            "Comparison unavailable: "
            f"{reason} No pixel agreement metrics are reported."
        )
        self.setAccessibleDescription(self._legend_label.text())
        self._grid.setAccessibleDescription(self._legend_label.text())

    def clear(self) -> None:
        """Clear the view to the unavailable state."""

        self.set_unavailable("No recall result is displayed.")

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _legend_text(counts: DifferenceCounts) -> str:
        total = max(counts.element_count, 1)
        ratio = counts.match_count / total
        return (
            "Legend: dark = matching ON · light = matching OFF · red = missing "
            "(expected ON, output OFF) · amber = extra (output ON, expected OFF).\n"
            f"Counts: matched ON {counts.matching_on} · matched OFF "
            f"{counts.matching_off} · missing {counts.missing} · extra "
            f"{counts.extra} · mismatches {counts.mismatch_count} of "
            f"{counts.element_count} · match ratio {_format_ratio(ratio)}."
        )


def _coerce_matrix(values: NDArray | None, name: str) -> NDArray[np.float64]:
    if values is None:
        raise ValueError(f"{name} matrix is not available.")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} matrix must be two-dimensional.")
    if array.size == 0:
        raise ValueError(f"{name} matrix must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} matrix contains non-finite values.")
    array.setflags(write=False)
    return array


def _encoding_name(encoding: object) -> str:
    value = getattr(encoding, "value", encoding)
    name = str(value)
    if name not in {"binary", "bipolar"}:
        raise ValueError(f"Unsupported encoding {value!r} for comparison.")
    return name


def _validate_encoded_values(array: NDArray[np.float64], encoding: str) -> None:
    allowed = (0.0, 1.0) if encoding == "binary" else (-1.0, 1.0)
    if not np.all(np.isin(array, allowed)):
        rendered = ", ".join(str(value) for value in allowed)
        raise ValueError(
            f"Pattern values are not valid for {encoding} encoding "
            f"(expected {{{rendered}}})."
        )


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def _format_ratio(value: float) -> str:
    """Render a ratio without unnecessary trailing zeros or rounding."""

    return str(int(value)) if value.is_integer() else repr(value)
