"""Image dataset view — placeholder for the not-yet-implemented destination."""

from __future__ import annotations

from PySide6.QtWidgets import QVBoxLayout, QWidget

from associative_memory.presentation.views import placeholder_page


class ImageDatasetView(QWidget):
    """Placeholder page for the Image Dataset destination.

    Image import, thumbnail preview, and pair organization are implemented
    in a later task.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(
            placeholder_page(
                "Image Dataset",
                "Import, organize, and preview image datasets here.",
                self,
            )
        )
