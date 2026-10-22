"""Preprocessing pipeline that turns an image into a pattern vector.

Pipeline order (fixed and documented; changing the order is a numerical
change that would alter stored patterns and would require a new decision):

1. **Grayscale conversion** — the original image is first converted to a
   single-channel grayscale representation (per the configured
   :class:`GrayscaleMethod`).
2. **Resizing** — the grayscale image is resized to the configured target
   ``(width, height)`` using the configured resampling filter. Two resize
   modes are supported:

   * ``PRESERVE_ASPECT`` (default): the image is scaled to fit inside the
     target rectangle while preserving its aspect ratio, then padded with
     the background intensity (derived from the foreground polarity, see
     below) to reach the exact target size. Padding is centered.
   * ``STRETCH``: the image is resized directly to the target rectangle,
     which distorts the aspect ratio. The result exposes
     ``aspect_ratio_preserved=False`` so callers can clearly indicate the
     distortion to the user.

3. **Thresholding** — each resized grayscale value is classified as
   ON (foreground) or OFF (background) using the configured threshold and
   foreground polarity:

   * ``DARK_IS_ON`` (default): values **below** the threshold become ON;
     values greater than or equal to the threshold become OFF.
   * ``LIGHT_IS_ON``: values **greater than or equal to** the threshold
     become ON; values below the threshold become OFF.

   The two polarities partition the grayscale range exactly; the boundary
   pixel equal to the threshold is ON for ``LIGHT_IS_ON`` and OFF for
   ``DARK_IS_ON``. Black never silently means ON and white never silently
   means OFF: the polarity is explicit configuration.
4. **Encoding** — the ON/OFF matrix is encoded as ``0/1`` values
   (:attr:`Encoding.BINARY`) or ``-1/+1`` values
   (:attr:`Encoding.BIPOLAR`), preserving the row/column layout. The
   encoding values follow the algorithm package conventions
   (``algorithms.encoding`` is the single source of truth).
5. **Flattening** — the encoded matrix is flattened to a one-dimensional
   vector, row-major by default. The same pixel always maps to the same
   vector index for a given target size and configuration, so training and
   recall must reuse the same configuration.

Default configuration (:data:`DEFAULT_PREPROCESSING_CONFIG`, documented here
so every run is reproducible):

* ``width=16``, ``height=16`` — classic 16x16 pattern grid.
* ``grayscale_method=GrayscaleMethod.LUMINANCE_601`` — Pillow's ``"L"``
  conversion, the ITU-R 601-2 luma transform
  (``R*299/1000 + G*587/1000 + B*114/1000``). The alternative
  ``GrayscaleMethod.AVERAGE`` is the arithmetic mean of the RGB channels,
  rounded half up.
* ``resampling=Image.Resampling.NEAREST`` — no invented gray values at
  edges, which keeps threshold behavior transparent for education.
* ``threshold=128`` on the 0–255 grayscale scale.
* ``polarity=ForegroundPolarity.DARK_IS_ON`` — dark pixels below 128 are
  foreground.
* ``encoding=Encoding.BINARY``.
* ``resize_mode=ResizeMode.PRESERVE_ASPECT``.
* ``flattening_order=FlatteningOrder.ROW_MAJOR``.

All arrays returned in :class:`PreprocessingResult` are read-only so that
stored experiment data cannot be mutated accidentally. The stored original
image is a private copy created before any conversion, so external mutation
of the caller's image cannot invalidate a stored result.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps

from associative_memory.algorithms.encoding import (
    Encoding,
    coerce_encoding,
    validate_encoding_values,
)


class ForegroundPolarity(str, Enum):
    """Which grayscale side of the threshold is the foreground (ON)."""

    DARK_IS_ON = "dark_is_on"
    LIGHT_IS_ON = "light_is_on"


class GrayscaleMethod(str, Enum):
    """Supported grayscale conversion methods."""

    LUMINANCE_601 = "luminance_601"
    AVERAGE = "average"


class ResizeMode(str, Enum):
    """How the image is resized to the target rectangle."""

    PRESERVE_ASPECT = "preserve_aspect"
    STRETCH = "stretch"


class FlatteningOrder(str, Enum):
    """Vector flattening order for the encoded matrix."""

    ROW_MAJOR = "row_major"
    COLUMN_MAJOR = "column_major"


@dataclass(frozen=True)
class PreprocessingConfig:
    """Explicit, reproducible preprocessing configuration.

    Raises:
        TypeError: If a numeric field is not an integer or a enum-like field
            is not an enum member or supported string value.
        ValueError: If a value is out of range or an unknown enum value is
            supplied.
    """

    width: int
    height: int
    grayscale_method: GrayscaleMethod = GrayscaleMethod.LUMINANCE_601
    resampling: Image.Resampling = Image.Resampling.NEAREST
    threshold: int = 128
    polarity: ForegroundPolarity = ForegroundPolarity.DARK_IS_ON
    encoding: Encoding = Encoding.BINARY
    resize_mode: ResizeMode = ResizeMode.PRESERVE_ASPECT
    flattening_order: FlatteningOrder = FlatteningOrder.ROW_MAJOR

    def __post_init__(self) -> None:
        _validate_positive_int(self.width, "width")
        _validate_positive_int(self.height, "height")
        _validate_threshold_int(self.threshold)
        object.__setattr__(
            self, "grayscale_method",
            _coerce_enum_member(
                self.grayscale_method, GrayscaleMethod, "grayscale_method",
            ),
        )
        object.__setattr__(
            self, "polarity",
            _coerce_enum_member(self.polarity, ForegroundPolarity, "polarity"),
        )
        object.__setattr__(
            self, "resize_mode",
            _coerce_enum_member(self.resize_mode, ResizeMode, "resize_mode"),
        )
        object.__setattr__(
            self, "flattening_order",
            _coerce_enum_member(
                self.flattening_order, FlatteningOrder, "flattening_order",
            ),
        )
        object.__setattr__(self, "resampling", _coerce_resampling(self.resampling))
        object.__setattr__(
            self, "encoding",
            coerce_encoding(self.encoding),
        )


@dataclass(frozen=True)
class PreprocessingResult:
    """Structured result of preprocessing one image.

    Contains the intermediate representations required for educational
    visualization plus the full effective configuration. Every array is
    read-only.
    """

    config: PreprocessingConfig
    source_path: Path | None
    source_format: str | None
    original_image: Image.Image
    original_width: int
    original_height: int
    grayscale: NDArray[np.uint8]
    resized: NDArray[np.uint8]
    binary_matrix: NDArray[np.uint8]
    encoded_matrix: NDArray[np.int8]
    encoded_vector: NDArray[np.int8]
    aspect_ratio_preserved: bool

    @property
    def width(self) -> int:
        """Target width of the preprocessed pattern."""

        return self.config.width

    @property
    def height(self) -> int:
        """Target height of the preprocessed pattern."""

        return self.config.height

    @property
    def threshold(self) -> int:
        """Grayscale threshold applied to the resized image."""

        return self.config.threshold

    @property
    def polarity(self) -> ForegroundPolarity:
        """Foreground polarity used for thresholding."""

        return self.config.polarity

    @property
    def encoding(self) -> Encoding:
        """Encoding of the encoded matrix and vector."""

        return self.config.encoding

    @property
    def flattening_order(self) -> FlatteningOrder:
        """Flattening order used to build the encoded vector."""

        return self.config.flattening_order

    @property
    def resize_mode(self) -> ResizeMode:
        """Resize mode used for the resized representation."""

        return self.config.resize_mode

    @property
    def resampling(self) -> Image.Resampling:
        """Resampling filter used while resizing."""

        return self.config.resampling

    @property
    def grayscale_method(self) -> GrayscaleMethod:
        """Grayscale conversion method used for the grayscale data."""

        return self.config.grayscale_method

    @property
    def vector_length(self) -> int:
        """Expected vector length, equal to ``width * height``."""

        return self.config.width * self.config.height


def preprocess_image(
    image: Image.Image,
    config: PreprocessingConfig,
    *,
    source_path: Path | None = None,
    source_format: str | None = None,
) -> PreprocessingResult:
    """Preprocess a Pillow image into a structured pattern result.

    The pipeline order and every intermediate representation are documented
    in the module docstring.

    Args:
        image: The original image. It is copied immediately; the argument
            itself is never modified and cannot affect the result.
        config: Explicit preprocessing configuration.
        source_path: Optional path the image was loaded from (metadata).
        source_format: Optional detected format of the source (metadata).

    Raises:
        TypeError: If ``image`` is not a ``PIL.Image.Image`` instance.
    """

    if not isinstance(image, Image.Image):
        raise TypeError(
            f"image must be a PIL.Image.Image; received {type(image).__name__}."
        )

    original_image = image.copy()
    gray_image = _to_grayscale(original_image, config.grayscale_method)
    resized_image, aspect_ratio_preserved = _resize(
        gray_image, config.width, config.height, config.resize_mode,
        config.resampling, config.polarity,
    )

    grayscale = np.asarray(gray_image, dtype=np.uint8)
    resized = np.asarray(resized_image, dtype=np.uint8)
    binary_matrix = _threshold_to_binary(resized, config.threshold, config.polarity)
    encoded_matrix = _encode(binary_matrix, config.encoding)
    encoded_vector = encoded_matrix.flatten(order=_ravel_order(config))

    result = PreprocessingResult(
        config=config,
        source_path=source_path,
        source_format=source_format,
        original_image=original_image,
        original_width=image.width,
        original_height=image.height,
        grayscale=grayscale,
        resized=resized,
        binary_matrix=binary_matrix,
        encoded_matrix=encoded_matrix,
        encoded_vector=encoded_vector,
        aspect_ratio_preserved=aspect_ratio_preserved,
    )

    for array in (
        grayscale, resized, binary_matrix, encoded_matrix, encoded_vector,
    ):
        array.setflags(write=False)

    # Tie the produced vector to the algorithm package's encoding rules as
    # a cheap self-check of the pipeline output.
    validate_encoding_values(encoded_vector, config.encoding, name="encoded vector")
    return result


def _to_grayscale(
    image: Image.Image,
    method: GrayscaleMethod,
) -> Image.Image:
    """Convert ``image`` to an 8-bit grayscale Pillow image."""

    if method is GrayscaleMethod.LUMINANCE_601:
        return image.convert("L")

    rgb = np.asarray(image.convert("RGB"), dtype=np.float64)
    averaged = np.floor(rgb.mean(axis=2) + 0.5).astype(np.uint8)
    return Image.fromarray(averaged, mode="L")


def _resize(
    gray_image: Image.Image,
    width: int,
    height: int,
    resize_mode: ResizeMode,
    resampling: Image.Resampling,
    polarity: ForegroundPolarity,
) -> tuple[Image.Image, bool]:
    """Resize ``gray_image`` to ``(width, height)`` and report aspect ratio.

    Returns the resized image and whether the aspect ratio was preserved.
    Padded regions use the background intensity: 255 for ``DARK_IS_ON``
    (light background) and 0 for ``LIGHT_IS_ON`` (dark background). Both
    values map to OFF under the documented threshold rules for any
    threshold in ``[0, 255]``.
    """

    if resize_mode is ResizeMode.PRESERVE_ASPECT:
        pad_intensity = 255 if polarity is ForegroundPolarity.DARK_IS_ON else 0
        padded = ImageOps.pad(
            gray_image,
            (width, height),
            method=resampling,
            color=pad_intensity,
            centering=(0.5, 0.5),
        )
        return padded, True
    stretched = gray_image.resize((width, height), resample=resampling)
    return stretched, False


def _threshold_to_binary(
    resized: NDArray[np.uint8],
    threshold: int,
    polarity: ForegroundPolarity,
) -> NDArray[np.uint8]:
    """Classify the resized grayscale values into an ON/OFF 0/1 matrix."""

    if polarity is ForegroundPolarity.DARK_IS_ON:
        on_mask = resized < threshold
    else:
        on_mask = resized >= threshold
    return np.where(on_mask, np.uint8(1), np.uint8(0)).astype(np.uint8)


def _encode(
    binary_matrix: NDArray[np.uint8],
    encoding: Encoding,
) -> NDArray[np.int8]:
    """Encode an ON/OFF matrix as binary ``0/1`` or bipolar ``-1/+1`` values."""

    if encoding is Encoding.BIPOLAR:
        return (binary_matrix.astype(np.int16) * 2 - 1).astype(np.int8)
    return binary_matrix.astype(np.int8)


def _ravel_order(config: PreprocessingConfig) -> str:
    """Return the NumPy flatten order string for the configured flattening."""

    if config.flattening_order is FlatteningOrder.ROW_MAJOR:
        return "C"
    return "F"


def _validate_positive_int(value: object, name: str) -> None:
    """Validate that ``value`` is a positive integer (bools rejected)."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer; received {value!r}.")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer; received {value!r}.")


def _validate_threshold_int(value: object) -> None:
    """Validate that ``threshold`` is an integer in ``[0, 255]``."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"threshold must be an integer; received {value!r}.")
    if not 0 <= value <= 255:
        raise ValueError(
            f"threshold must be in the range [0, 255]; received {value!r}."
        )


def _coerce_enum_member(
    value: object,
    enum_cls: type[Enum],
    field_name: str,
) -> Enum:
    """Coerce a string or enum member to the enum, with a clear error."""

    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls(value)
        except ValueError:
            pass
    options_text = ", ".join(member.value for member in enum_cls)  # type: ignore[attr-defined]
    raise ValueError(
        f"{field_name} must be one of: {options_text}; received {value!r}."
    )


def _coerce_resampling(value: object) -> Image.Resampling:
    """Coerce a resampling filter name or member to ``Image.Resampling``."""

    if isinstance(value, Image.Resampling):
        return value
    if isinstance(value, str):
        try:
            return Image.Resampling[value.upper()]
        except KeyError:
            pass
    options_text = ", ".join(member.name for member in Image.Resampling)
    raise ValueError(
        f"resampling must be an Image.Resampling filter "
        f"({options_text}); received {value!r}."
    )


DEFAULT_PREPROCESSING_CONFIG = PreprocessingConfig(width=16, height=16)
"""The documented default preprocessing configuration (a 16x16 grid)."""
