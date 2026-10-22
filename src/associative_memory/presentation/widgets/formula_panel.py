"""Formula/explanation panel for the training workspace."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from associative_memory.presentation.theme.spacing import SPACE_XS, SPACE_XXS

_OBJECT_NAME = "FormulaPanel"


class FormulaPanel(QWidget):
    """Displays the training equation and configuration context.

    Presentation-only: renders caller-provided text and performs no
    numerical work. Shows the Hebbian learning equation, encoding, and
    dimension information.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName(_OBJECT_NAME)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACE_XS, SPACE_XS, SPACE_XS, SPACE_XS)
        layout.setSpacing(SPACE_XXS)

        self._equation_label = QLabel(self)
        self._equation_label.setObjectName("FormulaEquation")
        self._equation_label.setWordWrap(True)
        self._equation_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        self._caption_label = QLabel(self)
        self._caption_label.setObjectName("FormulaCaption")
        self._caption_label.setWordWrap(True)

        layout.addWidget(self._equation_label)
        layout.addWidget(self._caption_label)

        self._equation = ""
        self._caption = ""

    # ------------------------------------------------------------------ content

    def set_equation(self, equation: str) -> None:
        """Set the mathematical equation text."""

        self._equation = equation
        self._equation_label.setText(equation)
        self._equation_label.setVisible(bool(equation))

    def set_caption(self, caption: str) -> None:
        """Set the descriptive caption."""

        self._caption = caption
        self._caption_label.setText(caption)
        self._caption_label.setVisible(bool(caption))

    def set_learning_rule(
        self,
        *,
        memory_type: str,
        encoding: str,
        source_dim: int,
        target_dim: int,
        pair_count: int,
    ) -> None:
        """Build a standard learning-rule caption from training metadata."""

        rule = (
            "outer(s, s)" if memory_type == "autoassociative" else "outer(s, t)"
        )
        self.set_caption(
            f"{memory_type} · {encoding} encoding · "
            f"{pair_count} pair(s) · {source_dim}×{target_dim} matrix · "
            f"W = Σ {rule}"
        )

    def clear(self) -> None:
        """Clear all displayed content."""

        self.set_equation("")
        self.set_caption("")
