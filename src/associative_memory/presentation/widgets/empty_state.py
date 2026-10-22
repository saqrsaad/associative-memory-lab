"""Empty state: deliberate placeholder presentation for absent content."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from associative_memory.presentation.theme.spacing import (
    ICON_EMPTY_PX,
    SPACE_LG,
    SPACE_MD,
    SPACE_XS,
)

_FRAME_OBJECT_NAME = "EmptyState"
_TITLE_OBJECT_NAME = "EmptyStateTitle"
_DESCRIPTION_OBJECT_NAME = "EmptyStateDescription"
_ICON_OBJECT_NAME = "EmptyStateIcon"

_ACTION_TEXT_SUFFIX = "…"

#: Presentation variants: neutral absence or an explicit error blocking
#: the section content.
VALID_VARIANTS = ("info", "error")


class EmptyState(QFrame):
    """A centered placeholder explaining why a section has no content.

    Presentation-only: it renders caller-provided explanation text and an
    optional next-action button; the callback is a plain Python callable
    connected from the owning view.

    Variants (documented error/warning presentation pattern):
    * ``"info"`` — dashed neutral border, regular content absence.
    * ``"error"`` — red-tinted surface and border for sections blocked
      by a persistent error (paired with a ``StatusBadge(error)`` next
      to it by the calling page).
    """

    def __init__(
        self,
        title: str,
        description: str | None = None,
        *,
        variant: str = "info",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName(_FRAME_OBJECT_NAME)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self._title = title
        self._description = description

        self._icon_label = QLabel(self)
        self._icon_label.setObjectName(_ICON_OBJECT_NAME)
        self._icon_label.hide()

        self._title_label = QLabel(title, self)
        self._title_label.setObjectName(_TITLE_OBJECT_NAME)
        self._title_label.setWordWrap(True)
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        self._description_label = QLabel(self)
        self._description_label.setObjectName(_DESCRIPTION_OBJECT_NAME)
        self._description_label.setWordWrap(True)
        self._description_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        self._action_button: QPushButton | None = None
        self._callback: Callable[[], None] | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACE_LG, SPACE_MD, SPACE_LG, SPACE_MD)
        layout.setSpacing(SPACE_XS)
        layout.addStretch(1)
        layout.addWidget(self._icon_label, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self._title_label)
        layout.addWidget(self._description_label)
        layout.addStretch(1)

        self.set_variant(variant)
        self.set_description(description)
        self._apply_accessibility()

    @property
    def title(self) -> str:
        """The primary heading text."""

        return self._title

    @property
    def description_label(self) -> QLabel:
        """The description label (exposed for focused unit assertions)."""

        return self._description_label

    @property
    def icon_label(self) -> QLabel:
        """The optional leading icon label (exposed for unit assertions)."""

        return self._icon_label

    @property
    def title_label(self) -> QLabel:
        """The title label (exposed for focused unit assertions)."""

        return self._title_label

    @property
    def action_button(self) -> QPushButton | None:
        """The optional action button, or ``None`` when unset."""

        return self._action_button

    def set_variant(self, variant: str) -> None:
        """Switch between the ``"info"`` and ``"error"`` variants."""

        if variant not in VALID_VARIANTS:
            raise ValueError(
                f"Unknown empty-state variant {variant!r}; "
                f"expected one of: {', '.join(VALID_VARIANTS)}."
            )
        self.setProperty("variant", variant)
        self.style().unpolish(self)
        self.style().polish(self)

    def set_icon(self, icon: QIcon | None) -> None:
        """Show an optional leading icon above the title."""

        if icon is None or icon.isNull():
            self._icon_label.clear()
            self._icon_label.hide()
            return
        self._icon_label.setPixmap(icon.pixmap(ICON_EMPTY_PX, ICON_EMPTY_PX))
        self._icon_label.show()

    def set_description(self, text: str | None) -> None:
        """Replace the explanation text, hiding the label when ``None``."""

        self._description = text
        if text is None:
            self._description_label.clear()
            self._description_label.hide()
        else:
            self._description_label.setText(text)
            self._description_label.show()
        self._apply_accessibility()

    def set_action(self, text: str, callback: Callable[[], None] | None) -> None:
        """Show the optional next-action button.

        Args:
            text: Button label; an accessible name is derived from it.
            callback: Plain callable invoked on click (ownership stays
                with the owning view).
        """

        if self._action_button is None:
            self._action_button = QPushButton(self)
            self.layout().addWidget(
                self._action_button, alignment=Qt.AlignmentFlag.AlignHCenter
            )
            self._action_button.clicked.connect(self._on_action_clicked)
        self._action_button.setText(text)
        self._action_button.setObjectName("EmptyStateAction")
        self._action_button.setProperty("variant", "primary")
        self._action_button.style().unpolish(self._action_button)
        self._action_button.style().polish(self._action_button)
        self._action_button.show()
        self._callback = callback
        self._apply_accessibility()

    def clear_action(self) -> None:
        """Hide and disconnect the action button."""

        if self._action_button is not None:
            self._action_button.hide()
            self._callback = None
        self._apply_accessibility()

    def _on_action_clicked(self) -> None:
        """Invoke the caller-owned callback; never holds business logic."""

        if self._callback is not None:
            self._callback()

    def _apply_accessibility(self) -> None:
        """Expose the explanation through the accessibility interfaces."""

        self.setAccessibleName(self._title)
        if self._description:
            self.setAccessibleDescription(f"{self._title}: {self._description}")
        else:
            self.setAccessibleDescription(self._title)
