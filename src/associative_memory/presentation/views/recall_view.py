"""Recall and results view — placeholder for the not-yet-implemented destination."""

from __future__ import annotations

from PySide6.QtWidgets import QVBoxLayout, QWidget

from associative_memory.presentation.views import placeholder_page


class RecallView(QWidget):
    """Placeholder page for the Recall and Results destination.

    Query cues, net-input traces, recalled outputs, and comparison
    metrics are implemented in a later task.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(
            placeholder_page(
                "Recall and Results",
                "Query trained memories and inspect recall results here.",
                self,
            )
        )
