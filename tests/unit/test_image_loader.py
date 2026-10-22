"""Unit tests for image file loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from associative_memory.infrastructure.image_loader import (
    SUPPORTED_IMAGE_FORMATS,
    ImageLoadError,
    load_image,
)

SUPPORTED_SAVE_FORMATS = ("PNG", "JPEG", "BMP")


def _solid_color_image(
    width: int,
    height: int,
    color: tuple[int, int, int],
    mode: str = "RGB",
) -> Image.Image:
    """Create a small solid-color image for file round-trips."""

    image = Image.new(mode, (width, height))
    image.putpixel((0, 0), color)
    return image


def test_supported_formats_constant():
    assert SUPPORTED_IMAGE_FORMATS == frozenset({"PNG", "JPEG", "BMP"})


@pytest.mark.parametrize("file_format", SUPPORTED_SAVE_FORMATS)
def test_load_valid_supported_files(tmp_path: Path, file_format: str):
    path = tmp_path / f"valid_{file_format.lower()}.{file_format.lower()}"
    _solid_color_image(6, 4, (30, 90, 150)).save(path)

    loaded = load_image(path)

    assert loaded.width == 6
    assert loaded.height == 4
    assert loaded.source_format == file_format
    assert loaded.source_path == path
    assert isinstance(loaded.image, Image.Image)


def test_load_jpeg_content_with_wrong_extension(tmp_path: Path):
    path = tmp_path / "mislabeled.png"
    _solid_color_image(5, 5, (10, 10, 10)).save(path, format="JPEG")

    loaded = load_image(path)

    assert loaded.source_format == "JPEG"


def test_missing_file_raises_file_not_found(tmp_path: Path):
    path = tmp_path / "missing.png"

    with pytest.raises(FileNotFoundError, match="missing.png"):
        load_image(path)


def test_directory_path_rejected(tmp_path: Path):
    target = tmp_path / "folder.png"
    target.mkdir()

    with pytest.raises(IsADirectoryError):
        load_image(target)


def test_non_path_non_string_argument_rejected():
    with pytest.raises(TypeError, match="str or pathlib.Path"):
        load_image(123)  # type: ignore[arg-type]


def test_unreadable_file_raises_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    path = tmp_path / "locked.png"
    _solid_color_image(2, 2, (0, 0, 0)).save(path)

    def fake_open(path_argument, *args, **kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(
        "associative_memory.infrastructure.image_loader.Image.open", fake_open
    )

    with pytest.raises(PermissionError):
        load_image(path)


def test_corrupt_file_content_rejected(tmp_path: Path):
    path = tmp_path / "corrupt.png"
    path.write_bytes(b"this is not an image at all")

    with pytest.raises(ImageLoadError, match="not a readable image file"):
        load_image(path)


def test_truncated_valid_file_rejected(tmp_path: Path):
    path = tmp_path / "truncated.png"
    _solid_color_image(8, 6, (0, 0, 0)).save(path)
    raw_bytes = path.read_bytes()
    path.write_bytes(raw_bytes[: len(raw_bytes) // 2])

    with pytest.raises(ImageLoadError):
        load_image(path)


def test_unsupported_format_rejected(tmp_path: Path):
    path = tmp_path / "unsupported.gif"
    _solid_color_image(3, 3, (0, 0, 0)).save(path, format="GIF")

    with pytest.raises(
        ImageLoadError, match="Unsupported image format 'GIF'"
    ):
        load_image(path)


def test_multi_frame_png_rejected(tmp_path: Path):
    path = tmp_path / "animated.png"
    base = _solid_color_image(4, 4, (255, 0, 0))
    second = _solid_color_image(4, 4, (0, 0, 255))
    base.save(
        path, format="PNG", save_all=True, append_images=[second],
        duration=100, loop=0,
    )

    with pytest.raises(
        ImageLoadError, match="animated or multi-frame image content"
    ):
        load_image(path)


def test_loaded_image_is_detached_from_file_handle(tmp_path: Path):
    path = tmp_path / "detach.png"
    _solid_color_image(3, 3, (10, 20, 30)).save(path)

    first = load_image(path)
    second = load_image(path)

    assert first.image is not second.image
    assert first.image.tobytes() == second.image.tobytes()
