"""Metric card: a reusable result-summary tile for workspace pages."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from associative_memory.presentation.theme.spacing import (
    SPACE_SM,
    SPACE_XS,
    SPACE_XXXS,
)
from associative_memory.presentation.widgets.status import StatusBadge, StatusKind

_FRAME_OBJECT_NAME = "MetricCard"
_TITLE_OBJECT_NAME = "MetricCardTitle"
_VALUE_OBJECT_NAME = "MetricCardValue"
_DESCRIPTION_OBJECT_NAME = "MetricCardDescription"

#: Shown instead of a value when none is available (design rule: a
#: readable placeholder instead of a fabricated zero).
NOT_CALCULATED_TEXT = "Not calculated"

#: Significant digits used when a ``float`` value is rendered; callers
#: needing different precision should pass a pre-formatted ``str``.
_FLOAT_DISPLAY_DIGITS = 4


class MetricCard(QFrame):
    """A titled card showing one value with optional description/status.

    Presentation-only: values, descriptions, and status text come from
    the caller (typically a viewmodel formatting a service result); the
    card performs no application logic.

    Value policy: ``None`` renders the documented ``"Not calculated"``
    placeholder (never a fabricated zero), ``int`` renders as-is, and
    ``float`` renders with four significant digits.
    """

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName(_FRAME_OBJECT_NAME)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self._title = title

        self._title_label = QLabel(title, self)
        self._title_label.setObjectName(_TITLE_OBJECT_NAME)
        self._title_label.setWordWrap(True)

        self._value_label = QLabel(self)
        self._value_label.setObjectName(_VALUE_OBJECT_NAME)
        self._value_label.setWordWrap(True)

        self._description_label = QLabel(self)
        self._description_label.setObjectName(_DESCRIPTION_OBJECT_NAME)
        self._description_label.setWordWrap(True)
        self._description_label.hide()

        self._status_badge: StatusBadge | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACE_SM, SPACE_XS, SPACE_SM, SPACE_XS)
        layout.setSpacing(SPACE_XXXS)
        layout.addWidget(self._title_label)
        layout.addWidget(self._value_label)
        layout.addWidget(self._description_label)

        self.set_value(None)

    @property
    def title(self) -> str:
        """The card title (never changes after construction)."""

        return self._title

    @property
    def value_label(self) -> QLabel:
        """The value label (exposed for focused unit assertions)."""

        return self._value_label

    @property
    def description_label(self) -> QLabel:
        """The description label (exposed for focused unit assertions)."""

        return self._description_label

    @property
    def status_badge(self) -> StatusBadge | None:
        """The embedded status badge, or ``None`` while no status is set."""

        return self._status_badge

    def set_value(self, value: str | int | float | None) -> None:
        """Display ``value``; ``None`` shows the unavailable placeholder."""

        missing = value is None
        self._value_label.setText(
            NOT_CALCULATED_TEXT if missing else self._format_value(value)
        )
        state = "missing" if missing else ""
        for widget in (self, self._value_label):
            widget.setProperty("state", state)
            self._repolish(widget)
        self._apply_accessibility()

    def set_description(self, text: str | None) -> None:
        """Show ``text`` under the value, or hide the row when ``None``."""

        if text is None:
            self._description_label.clear()
            self._description_label.hide()
        else:
            self._description_label.setText(text)
            self._description_label.show()
        self._apply_accessibility()

    def set_status(self, kind: StatusKind | str, text: str | None = None) -> None:
        """Attach (or replace) the card's status badge.

        Args:
            kind: One of the :class:`StatusKind` values.
            text: Optional status text; the badge default label applies
                when omitted.
        """

        if self._status_badge is None:
            self._status_badge = StatusBadge(kind, text, self)
            self.layout().addWidget(self._status_badge)
        else:
            self._status_badge.set_status(kind, text)
        self._status_badge.show()
        self._apply_accessibility()

    def clear_status(self) -> None:
        """Remove the status badge from the card."""

        if self._status_badge is not None:
            self._status_badge.hide()
            self._status_badge.deleteLater()
            self._status_badge = None
        self._apply_accessibility()

    @staticmethod
    def _format_value(value: str | int | float) -> str:
        """Render ``value`` per the documented display policy."""

        if isinstance(value, str):
            return value
        if isinstance(value, float):
            return format(value, f".{_FLOAT_DISPLAY_DIGITS}g")
        return str(value)

    @staticmethod
    def _repolish(widget: QWidget) -> None:
        """Re-apply centralized styles after a dynamic property change."""

        widget.style().unpolish(widget)
        widget.style().polish(widget)

    def _apply_accessibility(self) -> None:
        """Expose title, value, and status to assistive technology."""

        self.setAccessibleName(f"Metric card: {self._title}")
        self._value_label.setAccessibleName(
            f"{self._title}: {self._value_label.text()}"
        )
