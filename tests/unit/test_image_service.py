"""Unit tests for the application image service workflow (GUI-free)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from associative_memory.algorithms.encoding import Encoding
from associative_memory.application.services.image_service import (
    DEFAULT_PREPROCESSING_CONFIG,
    SUPPORTED_IMAGE_FORMATS,
    ImageLoadError,
    ImageService,
)
from associative_memory.infrastructure.image_preprocessor import (
    ForegroundPolarity,
    PreprocessingConfig,
)


def _solid_image(width: int, height: int, value: int) -> Image.Image:
    """Create a small grayscale image for file round-trips."""

    return Image.new("L", (width, height), value)


def test_supported_formats_reexported_for_callers():
    assert SUPPORTED_IMAGE_FORMATS == frozenset({"PNG", "JPEG", "BMP"})


def test_default_config_is_reexported_for_callers():
    assert DEFAULT_PREPROCESSING_CONFIG == PreprocessingConfig(16, 16)


def test_preprocess_file_uses_default_config(tmp_path: Path):
    path = tmp_path / "pattern.png"
    _solid_image(20, 12, 255).save(path)

    result = ImageService().preprocess_file(path)

    assert result.width == 16
    assert result.height == 16
    assert result.threshold == 128
    assert result.polarity is ForegroundPolarity.DARK_IS_ON
    assert result.encoding is Encoding.BINARY
    assert result.encoded_vector.shape == (16 * 16,)
    # Solid white image: every pixel (and padded background) is >= the
    # threshold, i.e. OFF under DARK_IS_ON.
    assert (result.encoded_vector == 0).sum() == 256


def test_preprocess_file_returns_source_metadata(tmp_path: Path):
    path = tmp_path / "pattern.png"
    _solid_image(4, 4, 0).save(path)

    result = ImageService().preprocess_file(path)

    assert result.source_path == path
    assert result.source_format == "PNG"
    assert result.original_width == 4
    assert result.original_height == 4


def test_preprocess_file_uses_provided_config(tmp_path: Path):
    path = tmp_path / "pattern.bmp"
    _solid_image(3, 3, 10).save(path)
    # STRETCH maps the solid 3x3 content onto the full 6x5 target, so every
    # vector position represents the same dark value.
    config = PreprocessingConfig(
        6, 5, encoding="bipolar", resize_mode="stretch"
    )

    result = ImageService().preprocess_file(path, config)

    assert result.width == 6
    assert result.height == 5
    assert result.encoding is Encoding.BIPOLAR
    assert result.encoded_vector.shape == (30,)
    # Solid dark image below the threshold -> the whole vector is ON (+1).
    assert (result.encoded_vector == 1).sum() == 30


def test_preprocess_file_propagates_missing_file_error(tmp_path: Path):
    missing = tmp_path / "missing.png"

    with pytest.raises(FileNotFoundError):
        ImageService().preprocess_file(missing)


def test_preprocess_file_propagates_corrupt_file_error(tmp_path: Path):
    path = tmp_path / "corrupt.png"
    path.write_bytes(b"garbage bytes")

    with pytest.raises(ImageLoadError):
        ImageService().preprocess_file(path)


def test_preprocess_file_rejects_unsupported_arguments():
    service = ImageService()

    with pytest.raises(TypeError, match="str or pathlib.Path"):
        service.preprocess_file(123)  # type: ignore[arg-type]


def test_preprocessed_vector_matches_expected_pattern(tmp_path: Path):
    """Verify the full path: file -> vector pixel mapping (row-major)."""

    image = Image.new("L", (2, 3), 255)
    image.putpixel((0, 1), 0)  # col 0, row 1 -> dark/ON
    path = tmp_path / "pattern.png"
    image.save(path)

    result = ImageService().preprocess_file(path, PreprocessingConfig(2, 3))

    assert result.encoded_matrix.shape == (3, 2)
    assert result.encoded_vector.tolist() == [0, 0, 1, 0, 0, 0]
