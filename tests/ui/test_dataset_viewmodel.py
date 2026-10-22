"""Unit tests for the dataset view model (GUI-free where possible)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from associative_memory.algorithms.encoding import Encoding
from associative_memory.application.services.image_service import (
    DEFAULT_PREPROCESSING_CONFIG,
)
from associative_memory.infrastructure.image_preprocessor import (
    ForegroundPolarity,
    PreprocessingConfig,
    preprocess_image,
)
from associative_memory.presentation.viewmodels.dataset_viewmodel import (
    MEMORY_AUTOASSOCIATIVE,
    MEMORY_HETEROASSOCIATIVE,
    DatasetEntry,
    DatasetViewModel,
)


def _make_result(
    width: int = 4,
    height: int = 4,
    encoding: Encoding = Encoding.BINARY,
    polarity: ForegroundPolarity = ForegroundPolarity.DARK_IS_ON,
) -> PreprocessingConfig:
    return PreprocessingConfig(
        width=width,
        height=height,
        encoding=encoding,
        polarity=polarity,
    )


def _vector(length: int, value: int) -> np.ndarray:
    return np.full(length, value, dtype=np.int8)


# -------------------------------------------------------------- properties


def test_view_model_defaults():
    vm = DatasetViewModel()

    assert vm.memory_type == MEMORY_AUTOASSOCIATIVE
    assert vm.config == DEFAULT_PREPROCESSING_CONFIG
    assert vm.entries == ()
    assert vm.selected_entry is None
    assert not vm.has_pattern


def test_config_update_and_reset():
    vm = DatasetViewModel()
    assert vm.update_config(width=8, height=8)

    assert vm.config.width == 8
    assert vm.config.height == 8


def test_invalid_config_is_rejected():
    vm = DatasetViewModel()
    assert not vm.update_config(width=0)
    assert vm.config == DEFAULT_PREPROCESSING_CONFIG  # unchanged


# -------------------------------------------------------------- memory type


def test_switch_memory_type_clears_hetero_targets(tmp_path: Path):
    vm = DatasetViewModel()
    vm.set_memory_type(MEMORY_HETEROASSOCIATIVE)

    image = Image.new("L", (4, 4), 0)
    result = preprocess_image(image, _make_result(4, 4))

    entry = DatasetEntry(
        entry_id="e1",
        source_path=tmp_path / "a.png",
        source_format="PNG",
        original_width=4,
        original_height=4,
        original_mode="L",
        memory_type=MEMORY_HETEROASSOCIATIVE,
        config=_make_result(4, 4),
        result=result,
        target_label="cat",
    )
    vm._entries.append(entry)
    vm.entries_changed.emit()

    vm.set_memory_type(MEMORY_AUTOASSOCIATIVE)

    assert vm.memory_type == MEMORY_AUTOASSOCIATIVE
    assert vm.entries[0].target_label is None


# -------------------------------------------------------------- import


def test_import_valid_image(tmp_path: Path):
    vm = DatasetViewModel()
    path = tmp_path / "pattern.png"
    Image.new("L", (8, 8), 10).save(path)

    messages: list[tuple[str, object]] = []
    vm.status_message.connect(lambda message, kind: messages.append((message, kind)))

    vm.import_image(path)

    assert vm.has_pattern
    assert vm.current_result is not None
    assert vm.current_result.width == 16
    assert vm.current_result.height == 16
    assert any("Imported" in m for m, _ in messages)


def test_import_invalid_image_reports_error_and_keeps_preview(tmp_path: Path):
    vm = DatasetViewModel()
    path = tmp_path / "corrupt.png"
    path.write_bytes(b"not an image")

    errors: list[str] = []
    vm.status_message.connect(lambda message, kind: errors.append(message))

    vm.import_image(path)

    assert not vm.has_pattern
    assert any("corrupt" in m.lower() or "image" in m.lower() for m in errors)


def test_import_missing_file_reports_error(tmp_path: Path):
    vm = DatasetViewModel()
    missing = tmp_path / "missing.png"

    errors: list[str] = []
    vm.status_message.connect(lambda message, kind: errors.append(message))

    vm.import_image(missing)

    assert not vm.has_pattern
    assert len(errors) >= 1


# -------------------------------------------------------------- entries


def test_add_entry_compatibility_and_length_mismatch(tmp_path: Path):
    vm = DatasetViewModel()

    # First entry
    image_a = Image.new("L", (4, 4), 0)
    result_a = preprocess_image(image_a, _make_result(4, 4))
    vm._current_image = image_a
    vm._current_result = result_a
    vm.add_entry()
    assert len(vm.entries) == 1

    # Different length cannot be added
    image_b = Image.new("L", (8, 8), 0)
    result_b = preprocess_image(image_b, _make_result(8, 8))
    vm._current_image = image_b
    vm._current_result = result_b

    warnings: list[str] = []
    vm.status_message.connect(lambda message, kind: warnings.append(message))

    vm.add_entry()
    assert len(vm.entries) == 1
    assert any("length" in w.lower() for w in warnings)


def test_add_entry_encoding_mismatch_rejected(tmp_path: Path):
    vm = DatasetViewModel()

    image_a = Image.new("L", (4, 4), 0)
    result_a = preprocess_image(image_a, _make_result(4, 4, encoding=Encoding.BINARY))
    vm._current_image = image_a
    vm._current_result = result_a
    vm.add_entry()
    assert len(vm.entries) == 1

    image_b = Image.new("L", (4, 4), 0)
    result_b = preprocess_image(
        image_b, _make_result(4, 4, encoding=Encoding.BIPOLAR)
    )
    vm._current_image = image_b
    vm._current_result = result_b

    vm.add_entry()
    assert len(vm.entries) == 1


def test_add_entry_keeps_config_snapshot(tmp_path: Path):
    vm = DatasetViewModel()
    vm.update_config(width=8, height=8)

    image = Image.new("L", (8, 8), 0)
    result = preprocess_image(image, vm.config)
    vm._current_image = image
    vm._current_result = result
    vm.add_entry()

    entry = vm.entries[0]
    assert entry.config.width == 8
    assert entry.config.height == 8
    # Changing the active config later must not change the stored entry.
    vm.update_config(width=16)
    assert vm.entries[0].config.width == 8


def test_remove_entry_and_select():
    vm = DatasetViewModel()
    result = preprocess_image(Image.new("L", (4, 4), 0), _make_result(4, 4))

    entry = DatasetEntry(
        entry_id="e1",
        source_path=None,
        source_format=None,
        original_width=4,
        original_height=4,
        original_mode="L",
        memory_type=MEMORY_AUTOASSOCIATIVE,
        config=_make_result(4, 4),
        result=result,
    )
    vm._entries.append(entry)

    vm.select_entry("e1")
    assert vm.selected_entry is entry

    vm.remove_entry("e1")
    assert len(vm.entries) == 0
    assert vm.selected_entry is None


def test_clear_dataset():
    vm = DatasetViewModel()
    result = preprocess_image(Image.new("L", (4, 4), 0), _make_result(4, 4))

    entry = DatasetEntry(
        entry_id="e1",
        source_path=None,
        source_format=None,
        original_width=4,
        original_height=4,
        original_mode="L",
        memory_type=MEMORY_AUTOASSOCIATIVE,
        config=_make_result(4, 4),
        result=result,
    )
    vm._entries.append(entry)
    vm.select_entry("e1")

    vm.clear_dataset()
    assert len(vm.entries) == 0
    assert vm.selected_entry is None


# -------------------------------------------------------------- target labels


def test_set_target_label_heteroassociative():
    vm = DatasetViewModel()
    vm.set_memory_type(MEMORY_HETEROASSOCIATIVE)

    result = preprocess_image(Image.new("L", (4, 4), 0), _make_result(4, 4))
    entry = DatasetEntry(
        entry_id="e1",
        source_path=None,
        source_format=None,
        original_width=4,
        original_height=4,
        original_mode="L",
        memory_type=MEMORY_HETEROASSOCIATIVE,
        config=_make_result(4, 4),
        result=result,
    )
    vm._entries.append(entry)
    vm.select_entry("e1")

    vm.set_selected_target("cat")
    assert vm.entries[0].target_label == "cat"

    vm.set_selected_target("")
    assert vm.entries[0].target_label is None


def test_target_label_ignored_autoassociative():
    vm = DatasetViewModel()
    vm.set_memory_type(MEMORY_AUTOASSOCIATIVE)

    result = preprocess_image(Image.new("L", (4, 4), 0), _make_result(4, 4))
    entry = DatasetEntry(
        entry_id="e1",
        source_path=None,
        source_format=None,
        original_width=4,
        original_height=4,
        original_mode="L",
        memory_type=MEMORY_AUTOASSOCIATIVE,
        config=_make_result(4, 4),
        result=result,
    )
    vm._entries.append(entry)
    vm.select_entry("e1")

    vm.set_selected_target("should-be-ignored")
    assert vm.entries[0].target_label is None


# -------------------------------------------------------------- compatibility


def test_current_is_compatible():
    vm = DatasetViewModel()
    assert not vm.current_is_compatible()

    image = Image.new("L", (4, 4), 0)
    result = preprocess_image(image, _make_result(4, 4))
    vm._current_image = image
    vm._current_result = result
    assert vm.current_is_compatible()
