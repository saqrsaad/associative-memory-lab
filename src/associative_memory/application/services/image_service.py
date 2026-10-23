"""Application service for image loading and preprocessing.

This service coordinates the infrastructure image loader and preprocessor
for the application layer. It returns structured result objects and never
implements image or algorithm mathematics itself, and it has no GUI
dependencies, so workflows can be exercised independently of the interface.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from associative_memory.algorithms.encoding import Encoding
from associative_memory.infrastructure.image_loader import (
    SUPPORTED_IMAGE_FORMATS,
    ImageLoadError,
    LoadedImage,
    load_image,
)
from associative_memory.infrastructure.image_preprocessor import (
    DEFAULT_PREPROCESSING_CONFIG,
    FlatteningOrder,
    ForegroundPolarity,
    GrayscaleMethod,
    PreprocessingConfig,
    PreprocessingResult,
    ResizeMode,
    preprocess_image,
)

__all__ = [
    "DEFAULT_PREPROCESSING_CONFIG",
    "Encoding",
    "FlatteningOrder",
    "ForegroundPolarity",
    "GrayscaleMethod",
    "ImageLoadError",
    "ImageService",
    "PreprocessingConfig",
    "PreprocessingResult",
    "ResizeMode",
    "SUPPORTED_IMAGE_FORMATS",
    "load_image",
    "preprocess_image",
]

_SupportedPath = str | Path


class ImageService:
    """Coordinate image file loading and preprocessing for the application."""

    def preprocess_file(
        self,
        path: _SupportedPath,
        config: PreprocessingConfig | None = None,
    ) -> PreprocessingResult:
        """Load an image file and preprocess it into a pattern result.

        Args:
            path: Path to a supported image file (see
                :data:`SUPPORTED_IMAGE_FORMATS`).
            config: Explicit preprocessing configuration. When ``None``, the
                documented default configuration
                (:data:`DEFAULT_PREPROCESSING_CONFIG`: a 16x16 grid with
                luminance grayscale, NEAREST resampling, threshold 128,
                ``DARK_IS_ON`` polarity, binary encoding,
                aspect-preserving resize, row-major flattening) is used.

        Returns:
            The structured preprocessing result, including source metadata
            and every intermediate representation.

        Raises:
            The exceptions documented by
            :func:`associative_memory.infrastructure.image_loader.load_image`
            and :func:`associative_memory.infrastructure.image_preprocessor.preprocess_image`.
        """

        loaded: LoadedImage = load_image(path)
        effective_config = (
            config if config is not None else DEFAULT_PREPROCESSING_CONFIG
        )
        return preprocess_image(
            loaded.image,
            effective_config,
            source_path=loaded.source_path,
            source_format=loaded.source_format,
        )

    def preprocess_image(
        self,
        image: Image.Image,
        config: PreprocessingConfig | None = None,
        *,
        source_path: Path | None = None,
        source_format: str | None = None,
    ) -> PreprocessingResult:
        """Preprocess an already-loaded image into a pattern result.

        This is the application-layer wrapper around
        :func:`associative_memory.infrastructure.image_preprocessor.preprocess_image`
        so that presentation code never reaches into infrastructure directly.

        Args:
            image: The original Pillow image.
            config: Explicit preprocessing configuration. When ``None``, the
                documented default configuration is used.
            source_path: Optional path the image was loaded from (metadata).
            source_format: Optional detected format of the source (metadata).

        Returns:
            The structured preprocessing result.

        Raises:
            The exceptions documented by
            :func:`associative_memory.infrastructure.image_preprocessor.preprocess_image`.
        """

        effective_config = (
            config if config is not None else DEFAULT_PREPROCESSING_CONFIG
        )
        return preprocess_image(
            image,
            effective_config,
            source_path=source_path,
            source_format=source_format,
        )
