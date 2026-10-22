"""Image preview canvas for the dataset workspace."""

from __future__ import annotations

from PIL import Image
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QWidget

_OBJECT_NAME = "ImageCanvas"


class ImageCanvas(QLabel):
    """Displays a PIL image scaled to fit while preserving aspect ratio.

    Presentation-only: it renders caller-provided images and performs no
    image processing. Arrays and images it displays are treated as
    read-only input.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName(_OBJECT_NAME)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setScaledContents(False)
        self.setMinimumSize(120, 120)
        self._pixmap: QPixmap | None = None

    def set_image(self, image: Image.Image | None) -> None:
        """Render ``image`` or clear the canvas when ``None``."""

        if image is None:
            self._pixmap = None
            self.clear()
            return
        self._pixmap = QPixmap.fromImage(self._to_qimage(image))
        self._rescale()

    def clear(self) -> None:
        """Clear the canvas."""

        self._pixmap = None
        super().clear()

    def sizeHint(self) -> QSize:
        if self._pixmap is not None:
            return self._pixmap.size()
        return QSize(160, 160)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._pixmap is None:
            return
        self.setPixmap(
            self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    @staticmethod
    def _to_qimage(image: Image.Image) -> QImage:
        """Convert a PIL image to a QImage whose pixels are copied."""

        mode = image.mode
        if mode == "L":
            qformat = QImage.Format.Format_Grayscale8
        elif mode == "RGB":
            qformat = QImage.Format.Format_RGB888
        elif mode == "RGBA":
            qformat = QImage.Format.Format_RGBA8888
        else:
            image = image.convert("RGB")
            qformat = QImage.Format.Format_RGB888

        width, height = image.size
        return QImage(image.tobytes(), width, height, qformat)
