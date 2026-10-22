"""Unit tests for the image preprocessing pipeline and its configuration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from associative_memory.algorithms.encoding import Encoding
from associative_memory.infrastructure.image_loader import load_image
from associative_memory.infrastructure.image_preprocessor import (
    DEFAULT_PREPROCESSING_CONFIG,
    FlatteningOrder,
    ForegroundPolarity,
    GrayscaleMethod,
    PreprocessingConfig,
    ResizeMode,
    preprocess_image,
)

THRESHOLD = 128


def _l_image(
    width: int,
    height: int,
    values: dict[tuple[int, int], int],
) -> Image.Image:
    """Create a tiny grayscale image and set explicit pixel values."""

    image = Image.new("L", (width, height))
    for (x, y), value in values.items():
        image.putpixel((x, y), value)
    return image


def _rgb_image(
    width: int,
    height: int,
    values: dict[tuple[int, int], tuple[int, int, int]],
) -> Image.Image:
    """Create a tiny RGB image and set explicit pixel values."""

    image = Image.new("RGB", (width, height))
    for (x, y), value in values.items():
        image.putpixel((x, y), value)
    return image


def _edge_image_and_configs() -> tuple[Image.Image, PreprocessingConfig, PreprocessingConfig]:
    """A 4x1 grayscale image with exact boundary values, plus both
    polarity configurations using ``THRESHOLD``."""

    boundary = _l_image(
        4, 1, {(0, 0): 127, (1, 0): 128, (2, 0): 0, (3, 0): 255}
    )
    dark_config = PreprocessingConfig(
        4, 1, threshold=THRESHOLD, polarity=ForegroundPolarity.DARK_IS_ON
    )
    light_config = PreprocessingConfig(
        4, 1, threshold=THRESHOLD, polarity=ForegroundPolarity.LIGHT_IS_ON
    )
    return boundary, dark_config, light_config


def _flat_pattern_image() -> Image.Image:
    """2 rows x 3 columns pattern with distinct ON/OFF cells (no resize)."""

    return _l_image(
        3,
        2,
        {
            (0, 0): 10, (1, 0): 200, (2, 0): 10,
            (0, 1): 10, (1, 1): 10, (2, 1): 200,
        },
    )


def _wide_pattern_image() -> Image.Image:
    """An 8x4 grayscale image: left half dark (ON), right half light."""

    return _l_image(
        8,
        4,
        {(x, y): 40 if x < 4 else 200 for y in range(4) for x in range(8)},
    )


# --- Configuration validation -------------------------------------------------


@pytest.mark.parametrize("field", ["width", "height"])
@pytest.mark.parametrize(
    "invalid_value",
    [0, -1, "8", 8.0, 3.5, True, False, None],
)
def test_invalid_dimension_values_rejected(invalid_value, field: str):
    config = {"width": 8, "height": 8}
    config[field] = invalid_value

    with pytest.raises((TypeError, ValueError), match=f"{field} must be"):
        PreprocessingConfig(**config)


@pytest.mark.parametrize(
    ("invalid_threshold", "expected_error"),
    [
        (-1, ValueError),
        (256, ValueError),
        (128.5, TypeError),
        ("64", TypeError),
        (True, TypeError),
        (None, TypeError),
    ],
)
def test_invalid_threshold_values_rejected(invalid_threshold, expected_error):
    with pytest.raises(expected_error, match="threshold must be"):
        PreprocessingConfig(8, 8, threshold=invalid_threshold)


@pytest.mark.parametrize(
    ("name", "invalid_value"),
    [
        ("polarity", "foreground_first"),
        ("polarity", 128),
        ("encoding", "ternary"),
        ("encoding", 3),
        ("flattening_order", "diagonal"),
        ("flattening_order", None),
        ("resize_mode", "shrink_to_fit"),
        ("resize_mode", 1),
        ("grayscale_method", "pcm"),
    ],
)
def test_unknown_enum_values_rejected(name: str, invalid_value):
    config = {"width": 8, "height": 8}
    config[name] = invalid_value
    # ``encoding`` is delegated to the algorithm package's stable
    # ``coerce_encoding``, whose message wording differs from the other
    # configuration fields.
    expected_match = (
        "Unsupported encoding" if name == "encoding" else f"{name} must be one of"
    )

    with pytest.raises(ValueError, match=expected_match):
        PreprocessingConfig(**config)


def test_invalid_resampling_name_rejected():
    with pytest.raises(ValueError, match="resampling must be"):
        PreprocessingConfig(8, 8, resampling="sharp")


@pytest.mark.parametrize(
    ("field", "text_value", "expected_member"),
    [
        ("polarity", "light_is_on", ForegroundPolarity.LIGHT_IS_ON),
        ("encoding", "bipolar", Encoding.BIPOLAR),
        ("flattening_order", "column_major", FlatteningOrder.COLUMN_MAJOR),
        ("resize_mode", "stretch", ResizeMode.STRETCH),
        ("grayscale_method", "average", GrayscaleMethod.AVERAGE),
    ],
)
def test_string_values_are_coerced_to_enum_members(
    field: str, text_value: str, expected_member
):
    config_kwargs = {"width": 8, "height": 8, field: text_value}

    config = PreprocessingConfig(**config_kwargs)

    assert getattr(config, field) is expected_member


def test_resampling_name_is_coerced():
    config = PreprocessingConfig(8, 8, resampling="lanczos")

    assert config.resampling is Image.Resampling.LANCZOS


# --- Grayscale conversion -----------------------------------------------------


def test_luminance_grayscale_conversion_matches_expected_values():
    image = _rgb_image(
        3,
        1,
        {
            (0, 0): (0, 0, 0),
            (1, 0): (255, 255, 255),
            (2, 0): (30, 90, 150),
        },
    )
    config = PreprocessingConfig(
        3, 1, grayscale_method=GrayscaleMethod.LUMINANCE_601
    )

    result = preprocess_image(image, config)

    assert np.array_equal(result.grayscale, np.array([[0, 255, 79]]))


def test_average_grayscale_conversion_rounds_half_up():
    image = _rgb_image(
        3,
        1,
        {
            (0, 0): (0, 0, 0),
            (1, 0): (255, 255, 255),
            (2, 0): (30, 90, 150),
        },
    )
    config = PreprocessingConfig(3, 1, grayscale_method=GrayscaleMethod.AVERAGE)

    result = preprocess_image(image, config)

    assert np.array_equal(result.grayscale, np.array([[0, 255, 90]]))


def test_grayscale_representation_keeps_original_size():
    image = _l_image(5, 3, {})

    result = preprocess_image(image, PreprocessingConfig(8, 8))

    assert result.grayscale.shape == (3, 5)


# --- Threshold boundary behavior and polarity ---------------------------------


def test_dark_is_on_polarity_boundary_semantics():
    boundary, dark_config, _ = _edge_image_and_configs()

    result = preprocess_image(boundary, dark_config)

    # 127 (< threshold) -> ON, 128 (== threshold) -> OFF, 0 -> ON, 255 -> OFF.
    assert np.array_equal(result.binary_matrix, np.array([[1, 0, 1, 0]]))


def test_light_is_on_polarity_boundary_semantics():
    boundary, _, light_config = _edge_image_and_configs()

    result = preprocess_image(boundary, light_config)

    # 127 (< threshold) -> OFF, 128 (>= threshold) -> ON, 0 -> OFF, 255 -> ON.
    assert np.array_equal(result.binary_matrix, np.array([[0, 1, 0, 1]]))


def test_polarities_are_exact_complements():
    boundary, dark_config, light_config = _edge_image_and_configs()

    dark = preprocess_image(boundary, dark_config)
    light = preprocess_image(boundary, light_config)

    assert np.array_equal(
        dark.binary_matrix + light.binary_matrix, np.ones((1, 4))
    )


# --- Encoding -----------------------------------------------------------------


def test_binary_encoding_uses_zero_and_one_values():
    boundary, dark_config, _ = _edge_image_and_configs()

    result = preprocess_image(boundary, dark_config)

    assert result.encoding is Encoding.BINARY
    assert np.array_equal(result.encoded_matrix, result.binary_matrix)
    assert result.encoded_matrix.dtype == np.int8
    assert result.encoded_vector.dtype == np.int8


def test_bipolar_encoding_maps_on_to_plus_one_and_off_to_minus_one():
    boundary, _, _ = _edge_image_and_configs()
    config = PreprocessingConfig(
        4, 1, threshold=THRESHOLD, encoding=Encoding.BIPOLAR
    )

    result = preprocess_image(boundary, config)

    assert result.encoding is Encoding.BIPOLAR
    assert np.array_equal(result.encoded_matrix, np.array([[1, -1, 1, -1]]))
    assert result.encoded_vector.tolist() == [1, -1, 1, -1]


# --- Flattening order ---------------------------------------------------------


def test_row_major_flatten_matches_matrix_rows_in_order():
    image = _flat_pattern_image()
    config = PreprocessingConfig(
        3, 2, threshold=THRESHOLD, flattening_order=FlatteningOrder.ROW_MAJOR
    )

    result = preprocess_image(image, config)

    assert np.array_equal(result.binary_matrix, np.array([[1, 0, 1], [1, 1, 0]]))
    assert result.encoded_vector.tolist() == [1, 0, 1, 1, 1, 0]
    assert np.array_equal(result.encoded_matrix[0], result.encoded_vector[:3])
    assert np.array_equal(result.encoded_matrix[1], result.encoded_vector[3:])


def test_column_major_flatten_matches_matrix_columns_in_order():
    image = _flat_pattern_image()
    config = PreprocessingConfig(
        3, 2, threshold=THRESHOLD, flattening_order=FlatteningOrder.COLUMN_MAJOR
    )

    result = preprocess_image(image, config)

    assert result.encoded_vector.tolist() == [1, 1, 0, 1, 1, 0]


# --- Vector dimensions --------------------------------------------------------


@pytest.mark.parametrize("width", [1, 4, 16])
@pytest.mark.parametrize("height", [1, 3, 16])
def test_vector_length_equals_width_times_height(width: int, height: int):
    image = _l_image(2, 2, {})

    result = preprocess_image(image, PreprocessingConfig(width, height))

    assert result.vector_length == width * height
    assert result.encoded_vector.shape == (width * height,)


def test_encoded_matrix_shape_matches_target_size():
    image = _l_image(2, 2, {})

    result = preprocess_image(image, PreprocessingConfig(5, 4))

    assert result.encoded_matrix.shape == (4, 5)


# --- Resize behavior ----------------------------------------------------------


def test_preserve_aspect_mode_pads_with_background_intensity():
    image = _wide_pattern_image()
    config = PreprocessingConfig(
        8, 8, threshold=THRESHOLD, polarity=ForegroundPolarity.DARK_IS_ON
    )

    result = preprocess_image(image, config)

    # The 8x4 content is scaled by 1.0 and padded 2 rows on top and on the
    # bottom with 255, the light background intensity for DARK_IS_ON.
    assert result.resized.shape == (8, 8)
    assert result.resized[0].tolist() == [255] * 8
    assert result.resized[7].tolist() == [255] * 8
    assert result.resized[2].tolist() == [40, 40, 40, 40, 200, 200, 200, 200]
    assert result.encoded_matrix[0].tolist() == [0] * 8
    assert result.aspect_ratio_preserved is True


def test_preserve_aspect_mode_pads_for_light_is_on_polarity():
    image = _wide_pattern_image()
    config = PreprocessingConfig(
        8, 8, threshold=THRESHOLD, polarity=ForegroundPolarity.LIGHT_IS_ON
    )

    result = preprocess_image(image, config)

    # For LIGHT_IS_ON the background is dark (0), so padding uses 0.
    assert result.resized[0].tolist() == [0] * 8
    assert result.encoded_matrix[0].tolist() == [0] * 8


def test_stretch_mode_distorts_the_image_and_reports_it():
    image = _wide_pattern_image()
    config = PreprocessingConfig(
        4, 4, threshold=THRESHOLD, resize_mode=ResizeMode.STRETCH
    )

    result = preprocess_image(image, config)

    assert result.resized.shape == (4, 4)
    assert result.aspect_ratio_preserved is False


# --- Safety of returned data --------------------------------------------------


def test_repeated_processing_is_reproducible(tmp_path: Path):
    path = tmp_path / "repeated.png"
    _wide_pattern_image().save(path)
    config = PreprocessingConfig(16, 16)

    first = preprocess_image(load_image(path).image, config)
    second = preprocess_image(load_image(path).image, config)

    assert np.array_equal(first.grayscale, second.grayscale)
    assert np.array_equal(first.resized, second.resized)
    assert np.array_equal(first.binary_matrix, second.binary_matrix)
    assert np.array_equal(first.encoded_matrix, second.encoded_matrix)
    assert np.array_equal(first.encoded_vector, second.encoded_vector)


def test_returned_arrays_are_read_only():
    image = _l_image(4, 4, {})

    result = preprocess_image(image, PreprocessingConfig(4, 4))

    for array in (
        result.grayscale,
        result.resized,
        result.binary_matrix,
        result.encoded_matrix,
        result.encoded_vector,
    ):
        assert array.flags.writeable is False


def test_array_mutation_raises_value_error():
    image = _l_image(4, 4, {})

    result = preprocess_image(image, PreprocessingConfig(4, 4))

    with pytest.raises(ValueError):
        result.encoded_vector[0] = 5
    with pytest.raises(ValueError):
        result.binary_matrix[0, 0] = 1


def test_original_image_copy_is_independent_of_the_input():
    image = _l_image(4, 4, {(0, 0): 10})

    result = preprocess_image(image, PreprocessingConfig(4, 4))

    stored_pixels = result.original_image.tobytes()
    image.putpixel((0, 0), 250)

    assert result.original_image.tobytes() == stored_pixels


def test_non_pillow_image_input_rejected():
    with pytest.raises(TypeError, match="PIL.Image.Image"):
        preprocess_image(np.zeros((4, 4)), PreprocessingConfig(4, 4))


# --- Result metadata ----------------------------------------------------------


def test_result_exposes_effective_config_and_metadata():
    image = _l_image(6, 5, {})
    config = PreprocessingConfig(
        8,
        7,
        threshold=64,
        polarity=ForegroundPolarity.LIGHT_IS_ON,
        encoding=Encoding.BIPOLAR,
        flattening_order=FlatteningOrder.COLUMN_MAJOR,
        resize_mode=ResizeMode.STRETCH,
    )

    result = preprocess_image(image, config)

    assert result.config is config
    assert result.width == 8
    assert result.height == 7
    assert result.threshold == 64
    assert result.polarity is ForegroundPolarity.LIGHT_IS_ON
    assert result.encoding is Encoding.BIPOLAR
    assert result.flattening_order is FlatteningOrder.COLUMN_MAJOR
    assert result.resize_mode is ResizeMode.STRETCH
    assert result.grayscale_method is GrayscaleMethod.LUMINANCE_601
    assert result.resampling is Image.Resampling.NEAREST
    assert result.original_width == 6
    assert result.original_height == 5
    assert result.source_path is None
    assert result.source_format is None


def test_default_config_values_are_as_documented():
    config = DEFAULT_PREPROCESSING_CONFIG

    assert config.width == 16
    assert config.height == 16
    assert config.threshold == 128
    assert config.grayscale_method is GrayscaleMethod.LUMINANCE_601
    assert config.resampling is Image.Resampling.NEAREST
    assert config.polarity is ForegroundPolarity.DARK_IS_ON
    assert config.encoding is Encoding.BINARY
    assert config.resize_mode is ResizeMode.PRESERVE_ASPECT
    assert config.flattening_order is FlatteningOrder.ROW_MAJOR
