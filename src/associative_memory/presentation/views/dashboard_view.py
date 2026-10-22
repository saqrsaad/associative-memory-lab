"""Dashboard view — placeholder for the not-yet-implemented destination."""

from __future__ import annotations

from PySide6.QtWidgets import QVBoxLayout, QWidget

from associative_memory.presentation.views import placeholder_page


class DashboardView(QWidget):
    """Placeholder page for the Dashboard destination.

    This page is deliberately empty of business logic; the dataset,
    model-status, and experiment-overview widgets are implemented in a
    later task.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(
            placeholder_page(
                "Dashboard",
                "Dataset overview, model status, and recent experiments "
                "will appear here.",
                self,
            )
        )
