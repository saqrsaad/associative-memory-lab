"""Status presentation for the Associative Memory Lab interface.

Five status kinds are supported — success, information, warning, error,
and in-progress — each with a dedicated background/text color pair, a
Qt standard icon, and a default label. Status is never communicated
through color alone: :class:`StatusBadge` always renders an icon and
text next to the colored background (Task 05 accessibility rule).

The badge is presentation-only: it displays caller-provided text and
contains no application or numerical logic.
"""

from __future__ import annotations

from enum import StrEnum

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QWidget

from associative_memory.presentation.theme.icons import status_icon
from associative_memory.presentation.theme.palette import PALETTE
from associative_memory.presentation.theme.spacing import (
    ICON_STATUS_PX,
    SPACE_XXS,
    SPACE_XXXS,
)

_FRAME_OBJECT_NAME = "StatusBadge"
_ICON_OBJECT_NAME = "StatusBadgeIcon"
_TEXT_OBJECT_NAME = "StatusBadgeText"


class StatusKind(StrEnum):
    """The five distinguishable presentation statuses."""

    SUCCESS = "success"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    IN_PROGRESS = "in_progress"


#: Default label per status; text keeps status readable without color.
STATUS_DEFAULT_TEXT: dict[StatusKind, str] = {
    StatusKind.SUCCESS: "Success",
    StatusKind.INFO: "Information",
    StatusKind.WARNING: "Warning",
    StatusKind.ERROR: "Error",
    StatusKind.IN_PROGRESS: "In progress",
}


def status_colors(kind: StatusKind) -> tuple[str, str]:
    """Return ``(background_hex, text_hex)`` for ``kind``.

    The pairs follow the WCAG contrast policy documented in
    :mod:`associative_memory.presentation.theme.palette`.
    """

    mapping: dict[StatusKind, tuple[str, str]] = {
        StatusKind.SUCCESS: (PALETTE.SUCCESS_SOFT, PALETTE.SUCCESS_TEXT),
        StatusKind.INFO: (PALETTE.INFO_SOFT, PALETTE.INFO_TEXT),
        StatusKind.WARNING: (PALETTE.WARNING_SOFT, PALETTE.WARNING_TEXT),
        StatusKind.ERROR: (PALETTE.ERROR_SOFT, PALETTE.ERROR_TEXT),
        StatusKind.IN_PROGRESS: (PALETTE.IN_PROGRESS_SOFT, PALETTE.IN_PROGRESS_TEXT),
    }
    return mapping[kind]


class StatusBadge(QFrame):
    """A small status indicator combining color, icon, and text.

    The frame (and its text label) carry a ``status`` dynamic property so
    the centralized stylesheet restyles them without any per-widget
    stylesheet strings. Dynamic property changes require a style
    re-polish, which this widget performs itself.
    """

    def __init__(
        self,
        kind: StatusKind | str = StatusKind.INFO,
        text: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName(_FRAME_OBJECT_NAME)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self._icon_label = QLabel(self)
        self._icon_label.setObjectName(_ICON_OBJECT_NAME)
        self._text_label = QLabel(self)
        self._text_label.setObjectName(_TEXT_OBJECT_NAME)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(SPACE_XXS, SPACE_XXXS, SPACE_XXS, SPACE_XXXS)
        layout.setSpacing(SPACE_XXXS)
        layout.addWidget(self._icon_label)
        layout.addWidget(self._text_label)
        layout.addStretch(1)

        self._kind = StatusKind.INFO
        self._text_label.setText(STATUS_DEFAULT_TEXT[self._kind])
        self.set_status(kind, text)

    @property
    def status_kind(self) -> StatusKind:
        """The currently displayed status kind."""

        return self._kind

    @property
    def text_label(self) -> QLabel:
        """The text label (exposed for focused unit assertions)."""

        return self._text_label

    @property
    def icon_label(self) -> QLabel:
        """The status icon label (exposed for focused unit assertions)."""

        return self._icon_label

    def set_status(self, kind: StatusKind | str, text: str | None = None) -> None:
        """Switch the displayed status and optionally replace the text.

        Text rule: an explicit ``text`` always wins. Without one, previously
        customized text is preserved; otherwise the new kind's default
        label is applied.

        Args:
            kind: New status kind; plain strings are coerced through
                :class:`StatusKind` values (unknown values raise
                ``ValueError``).
            text: Optional replacement label.
        """

        new_kind = kind if isinstance(kind, StatusKind) else StatusKind(kind)
        previous_was_default = self._text_label.text() == STATUS_DEFAULT_TEXT[self._kind]
        self._kind = new_kind

        if text is None:
            replacement = (
                previous_was_default
                or not self._text_label.text()
            )
            self._text_label.setText(
                STATUS_DEFAULT_TEXT[new_kind] if replacement else self._text_label.text()
            )
        else:
            self._text_label.setText(text)

        self._icon_label.setPixmap(
            status_icon(new_kind.value).pixmap(ICON_STATUS_PX, ICON_STATUS_PX)
        )

        # Dynamic-property selectors require a style re-polish to apply.
        for widget in (self, self._text_label):
            widget.setProperty("status", new_kind.value)
            widget.style().unpolish(widget)
            widget.style().polish(widget)

        self._update_accessibility()

    def set_text(self, text: str) -> None:
        """Replace the text label without changing the status kind."""

        self._text_label.setText(text)
        self._update_accessibility()

    def _update_accessibility(self) -> None:
        """Expose status and message through the accessibility interfaces."""

        self.setAccessibleName(f"{self._kind.value} status")
        self.setAccessibleDescription(
            f"{STATUS_DEFAULT_TEXT[self._kind]}: {self._text_label.text()}"
        )
