"""Image file loading and validation for the preprocessing pipeline.

This module uses **Pillow** (declared as ``Pillow>=10.0`` in
``pyproject.toml``) as the project's image-loading library. Pillow decodes
every consumer format required here (PNG, JPEG/JPG, BMP) and exposes the
format it detected after decoding, which allows validation to rely on the
actual image content instead of the file extension.

Supported input formats:

* PNG (single-frame; animated PNG content is rejected).
* JPEG/JPG.
* BMP.

Policies enforced in this module:

* The file must exist, be a readable regular file, and contain decodable
  image data; a missing file, a directory, or unreadable content is
  reported with a specific exception.
* The format detected by Pillow must be one of the supported formats;
  the file extension is never used for format detection.
* Zero-sized images cannot occur through Pillow, but dimensions are
  validated explicitly and rejected with a clear error.
* Animated or multi-frame content (e.g. APNG) is rejected clearly instead
  of being silently reduced to its first frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

from PIL import Image

SUPPORTED_IMAGE_FORMATS = frozenset({"PNG", "JPEG", "BMP"})

_SupportedPath = Union[str, Path]

# Error types raised by Pillow while opening or decoding an image file.
# ``UnidentifiedImageError`` is a subclass of ``OSError``, so it is handled
# before ``OSError`` in the loader's ``except`` clauses.
_DECODE_ERRORS: tuple[type[Exception], ...] = (OSError,)
if hasattr(Image, "DecompressionBombError"):
    _DECODE_ERRORS = (OSError, Image.DecompressionBombError)


class ImageLoadError(Exception):
    """Raised when an image file cannot be loaded as a supported image."""


@dataclass(frozen=True)
class LoadedImage:
    """A validated, fully decoded image, detached from its file handle.

    ``image`` is a private copy of the decoded pixels; mutating the image
    they were loaded from cannot affect it.
    """

    image: Image.Image
    source_path: Path | None
    source_format: str

    @property
    def width(self) -> int:
        """Width of the original image in pixels."""

        return self.image.width

    @property
    def height(self) -> int:
        """Height of the original image in pixels."""

        return self.image.height

    @property
    def mode(self) -> str:
        """Pillow color mode of the original image (e.g. ``"RGB"``)."""

        return self.image.mode


def load_image(path: _SupportedPath) -> LoadedImage:
    """Load and validate an image file, returning a detached copy.

    The image is fully decoded before the file handle is released, and the
    format reported by Pillow is validated against
    :data:`SUPPORTED_IMAGE_FORMATS`.

    Raises:
        TypeError: If ``path`` is neither a ``str`` nor a ``pathlib.Path``.
        FileNotFoundError: If the file does not exist.
        IsADirectoryError: If the path names a directory.
        OSError: If the path exists but is not a regular file.
        PermissionError: If the file cannot be opened for reading.
        ImageLoadError: If the content is corrupt, uses an unsupported
            format, has invalid dimensions, or is animated/multi-frame.
    """

    file_path = _coerce_path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")
    if file_path.is_dir():
        raise IsADirectoryError(
            f"Expected an image file, but {file_path} is a directory."
        )
    if not file_path.is_file():
        raise OSError(
            f"Expected a regular image file, but {file_path} is neither a "
            f"file nor a directory."
        )

    try:
        with Image.open(file_path) as source_image:
            source_image.load()
            # ``format`` and frame attributes belong to the opened image
            # object; a copy carries the pixels but not these attributes.
            detected_format = source_image.format
            is_animated = bool(getattr(source_image, "is_animated", False))
            frame_count = int(getattr(source_image, "n_frames", 1) or 1)
            decoded = source_image.copy()
    except PermissionError:
        raise
    except Image.UnidentifiedImageError as exc:
        raise ImageLoadError(
            f"{file_path} is not a readable image file, or its content is "
            f"corrupt."
        ) from exc
    except _DECODE_ERRORS as exc:
        raise ImageLoadError(
            f"{file_path} could not be decoded as image data: {exc}"
        ) from exc

    if detected_format is None or detected_format not in SUPPORTED_IMAGE_FORMATS:
        supported_text = ", ".join(sorted(SUPPORTED_IMAGE_FORMATS))
        if detected_format is None:
            raise ImageLoadError(
                f"{file_path} has no identifiable image format; supported "
                f"formats are: {supported_text}."
            )
        raise ImageLoadError(
            f"Unsupported image format {detected_format!r} in {file_path}; "
            f"supported formats are: {supported_text}."
        )

    if decoded.width <= 0 or decoded.height <= 0:
        raise ImageLoadError(
            f"{file_path} has invalid image dimensions {decoded.size}; "
            f"width and height must be positive."
        )

    if is_animated or frame_count > 1:
        raise ImageLoadError(
            f"{file_path} contains animated or multi-frame image content "
            f"({frame_count} frames), which is not supported; provide a "
            f"static single-frame {detected_format} image."
        )

    return LoadedImage(
        image=decoded,
        source_path=file_path,
        source_format=detected_format,
    )


def _coerce_path(path: _SupportedPath) -> Path:
    """Return ``path`` as a :class:`pathlib.Path` instance."""

    if isinstance(path, Path):
        return path
    if isinstance(path, str):
        return Path(path)
    raise TypeError(
        f"Image path must be a str or pathlib.Path; received "
        f"{type(path).__name__}."
    )
