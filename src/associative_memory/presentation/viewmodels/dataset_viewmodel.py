"""Dataset workspace view model.

Coordinates image import, preprocessing, and training-pair organization
between the dataset view and the application services. Implements no
image-processing or algorithm formulas itself: loading and preprocessing
are delegated to :class:`associative_memory.application.services.image_service.ImageService`,
and encoding rules come from the algorithm package.

Ownership policy
----------------
The view model owns the dataset entries and the working preview. Entries
are immutable snapshots: when a target label changes, the view model
replaces the entry rather than mutating it. Views receive entries and
preprocessing results as read-only input and must not mutate them
(PreprocessingResult arrays are already read-only by construction).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from uuid import uuid4

from PIL import Image
from PySide6.QtCore import QObject, Signal

from associative_memory.application.services.image_service import (
    DEFAULT_PREPROCESSING_CONFIG,
    ImageLoadError,
    ImageService,
    PreprocessingConfig,
    PreprocessingResult,
)
from associative_memory.presentation.widgets.status import StatusKind

_LOGGER = logging.getLogger(__name__)

#: Memory type values used within the view model; kept as plain strings to
#: avoid importing application-layer DTOs into the presentation layer.
MEMORY_AUTOASSOCIATIVE = "autoassociative"
MEMORY_HETEROASSOCIATIVE = "heteroassociative"


@dataclass
class DatasetEntry:
    """One organized dataset entry, owned by :class:`DatasetViewModel`.

    Attributes:
        entry_id: Stable identifier for this entry within one session.
        source_path: Path the pattern was imported from, if any.
        source_format: Image format detected by Pillow (content-based).
        original_width: Original image width in pixels.
        original_height: Original image height in pixels.
        original_mode: Original Pillow color mode.
        memory_type: Memory type the entry was added under.
        config: Preprocessing configuration snapshot used to encode the
            pattern. Entries keep their snapshot even when the active
            preprocessing settings change.
        result: Structured preprocessing result, including the encoded
            vector and every intermediate representation.
        target_label: Heteroassociative target label (class label) for
            this entry, or ``None`` when unassigned or when the dataset
            uses autoassociative memory.
    """

    entry_id: str
    source_path: Path | None
    source_format: str | None
    original_width: int
    original_height: int
    original_mode: str
    memory_type: str
    config: PreprocessingConfig
    result: PreprocessingResult
    target_label: str | None = None


class DatasetViewModel(QObject):
    """Coordinate dataset import, preprocessing, and pair organization."""

    config_changed = Signal(PreprocessingConfig)
    preview_changed = Signal(object)  # PreprocessingResult | None
    entries_changed = Signal()
    entry_selected = Signal(object)  # DatasetEntry | None
    status_message = Signal(str, StatusKind)
    memory_type_changed = Signal(str)

    def __init__(
        self,
        image_service: ImageService | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._image_service = image_service or ImageService()
        self._memory_type = MEMORY_AUTOASSOCIATIVE
        self._config = DEFAULT_PREPROCESSING_CONFIG
        self._entries: list[DatasetEntry] = []
        self._selected_id: str | None = None
        self._current_image: Image.Image | None = None
        self._current_source_path: Path | None = None
        self._current_source_format: str | None = None
        self._current_result: PreprocessingResult | None = None

    # ------------------------------------------------------------------ properties

    @property
    def image_service(self) -> ImageService:
        return self._image_service

    @property
    def memory_type(self) -> str:
        return self._memory_type

    @property
    def config(self) -> PreprocessingConfig:
        return self._config

    @property
    def entries(self) -> tuple[DatasetEntry, ...]:
        return tuple(self._entries)

    @property
    def selected_entry(self) -> DatasetEntry | None:
        if self._selected_id is None:
            return None
        for entry in self._entries:
            if entry.entry_id == self._selected_id:
                return entry
        return None

    @property
    def current_result(self) -> PreprocessingResult | None:
        return self._current_result

    @property
    def has_pattern(self) -> bool:
        return self._current_result is not None

    # ------------------------------------------------------------------ memory type

    def set_memory_type(self, value: str) -> None:
        """Switch the dataset memory type.

        Switching heteroassociative to autoassociative clears all target
        labels, since autoassociative targets equal the source patterns.
        """

        normalized = value
        if normalized not in (MEMORY_AUTOASSOCIATIVE, MEMORY_HETEROASSOCIATIVE):
            raise ValueError(
                f"Unknown memory type {value!r}; expected "
                f"{MEMORY_AUTOASSOCIATIVE!r} or {MEMORY_HETEROASSOCIATIVE!r}."
            )
        if normalized is self._memory_type:
            return

        cleared_targets = False
        if (
            normalized == MEMORY_AUTOASSOCIATIVE
            and self._memory_type == MEMORY_HETEROASSOCIATIVE
        ):
            if any(entry.target_label is not None for entry in self._entries):
                self._entries = [
                    replace(entry, target_label=None) for entry in self._entries
                ]
                cleared_targets = True

        self._memory_type = normalized
        self.memory_type_changed.emit(normalized)
        self.entries_changed.emit()
        if cleared_targets:
            self.status_message.emit(
                "Autoassociative mode: each pattern is its own target.",
                StatusKind.INFO,
            )
        else:
            self.status_message.emit(
                "Heteroassociative mode: assign a target label to each entry.",
                StatusKind.INFO,
            )

    # ------------------------------------------------------------------ preprocessing

    def update_config(self, **fields: object) -> bool:
        """Update the active preprocessing configuration.

        Returns ``False`` and reports a status message when the requested
        configuration is invalid; the previous configuration is kept.
        """

        try:
            new_config = replace(self._config, **fields)
        except (TypeError, ValueError) as exc:
            self._config_validation_error(str(exc))
            return False

        self._config = new_config
        self.config_changed.emit(new_config)

        if self._current_image is not None:
            self._reprocess_current_image()

        self.status_message.emit(
            "Preprocessing updated. Stored entries keep their configuration "
            "snapshots; mixed settings may be incompatible in one set.",
            StatusKind.INFO,
        )
        return True

    def _reprocess_current_image(self) -> None:
        assert self._current_image is not None
        try:
            result = self._image_service.preprocess_image(
                self._current_image,
                self._config,
                source_path=self._current_source_path,
                source_format=self._current_source_format,
            )
        except Exception as exc:  # presentation boundary: never crash on preview
            _LOGGER.exception("Failed to re-preprocess current image")
            self.status_message.emit(
                f"Preprocessing failed: {exc}", StatusKind.ERROR
            )
            return
        self._current_result = result
        self.preview_changed.emit(result)

    def _config_validation_error(self, message: str) -> None:
        self.status_message.emit(
            f"Invalid preprocessing configuration: {message}", StatusKind.ERROR
        )

    # ------------------------------------------------------------------ import

    def import_image(self, path: str | Path) -> None:
        """Import and preprocess an image file as the working pattern.

        On failure the last valid preview is left untouched and an error
        status is reported instead of crashing.
        """

        try:
            result = self._image_service.preprocess_file(path, self._config)
        except ImageLoadError as exc:
            _LOGGER.debug("Image import failed for %s: %s", path, exc)
            self.status_message.emit(str(exc), StatusKind.ERROR)
            return
        except OSError as exc:
            _LOGGER.debug("Image import failed for %s: %s", path, exc)
            self.status_message.emit(
                f"Could not read image file: {exc}", StatusKind.ERROR
            )
            return
        except Exception as exc:  # presentation boundary: never crash on import
            _LOGGER.exception("Unexpected failure importing %s", path)
            self.status_message.emit(
                f"Could not import image: {exc}", StatusKind.ERROR
            )
            return

        self._current_image = result.original_image
        self._current_source_path = result.source_path
        self._current_source_format = result.source_format
        self._current_result = result
        self._selected_id = None
        self.entry_selected.emit(None)
        self.preview_changed.emit(result)
        source_name = Path(path).name if not isinstance(path, Path) else path.name
        self.status_message.emit(
            f"Imported {source_name} ({result.original_width}×{result.original_height}).",
            StatusKind.SUCCESS,
        )

    # ------------------------------------------------------------------ dataset entries

    def add_entry(self) -> None:
        """Add the working pattern to the dataset when compatible."""

        if self._current_result is None:
            self.status_message.emit(
                "No processed pattern to add. Import an image first.",
                StatusKind.WARNING,
            )
            return

        for entry in self._entries:
            if entry.result.vector_length != self._current_result.vector_length:
                self.status_message.emit(
                    "Cannot add pattern: its vector length "
                    f"({self._current_result.vector_length}) does not match "
                    f"the dataset length ({entry.result.vector_length}).",
                    StatusKind.WARNING,
                )
                return
            if entry.config.encoding is not self._current_result.encoding:
                self.status_message.emit(
                    "Cannot add pattern: its encoding "
                    f"({self._current_result.encoding.value}) differs from "
                    f"the dataset encoding ({entry.config.encoding.value}).",
                    StatusKind.WARNING,
                )
                return

        entry = DatasetEntry(
            entry_id=uuid4().hex,
            source_path=self._current_source_path,
            source_format=self._current_source_format,
            original_width=self._current_result.original_width,
            original_height=self._current_result.original_height,
            original_mode=self._current_image.mode
            if self._current_image is not None
            else "",
            memory_type=self._memory_type,
            config=self._config,
            result=self._current_result,
        )
        self._entries.append(entry)
        self.entries_changed.emit()
        self.status_message.emit(
            f"Pattern added to dataset ({entry.result.vector_length} elements).",
            StatusKind.SUCCESS,
        )

    def remove_entry(self, entry_id: str) -> None:
        """Remove one dataset entry by identifier."""

        self._entries = [
            entry for entry in self._entries if entry.entry_id != entry_id
        ]
        if self._selected_id == entry_id:
            self._selected_id = None
            self.entry_selected.emit(None)
        self.entries_changed.emit()
        self.status_message.emit("Entry removed.", StatusKind.INFO)

    def select_entry(self, entry_id: str | None) -> None:
        """Select one dataset entry for review and target assignment."""

        self._selected_id = entry_id
        entry = self.selected_entry
        self.entry_selected.emit(entry)

    def set_entry_target(self, entry_id: str, label: str | None) -> None:
        """Assign (or clear) the target label of one specific entry.

        Only meaningful for heteroassociative memory; autoassociative
        targets always equal the source pattern.
        """

        if self._memory_type == MEMORY_AUTOASSOCIATIVE:
            return
        entry = self.entry_for_id(entry_id)
        if entry is None:
            return
        cleaned = (
            None
            if label is None or not str(label).strip()
            else str(label).strip()
        )
        if cleaned == entry.target_label:
            return
        index = next(
            i
            for i, other in enumerate(self._entries)
            if other.entry_id == entry.entry_id
        )
        self._entries[index] = replace(entry, target_label=cleaned)
        self.entries_changed.emit()
        self.status_message.emit(
            f"Target label set to {cleaned!r}." if cleaned else "Target label cleared.",
            StatusKind.SUCCESS,
        )

    def set_selected_target(self, label: str | None) -> None:
        """Assign (or clear) the target label of the selected entry."""

        entry = self.selected_entry
        if entry is None:
            return
        self.set_entry_target(entry.entry_id, label)

    def clear_dataset(self) -> None:
        """Remove every dataset entry."""

        self._entries.clear()
        if self._selected_id is not None:
            self._selected_id = None
            self.entry_selected.emit(None)
        self.entries_changed.emit()
        self.status_message.emit("Dataset cleared.", StatusKind.INFO)

    # ------------------------------------------------------------------ helpers

    def entry_for_id(self, entry_id: str) -> DatasetEntry | None:
        for entry in self._entries:
            if entry.entry_id == entry_id:
                return entry
        return None

    def current_is_compatible(self) -> bool:
        """Whether the working pattern may be added to the dataset."""

        if self._current_result is None:
            return False
        for entry in self._entries:
            if entry.result.vector_length != self._current_result.vector_length:
                return False
            if entry.config.encoding is not self._current_result.encoding:
                return False
        return True
