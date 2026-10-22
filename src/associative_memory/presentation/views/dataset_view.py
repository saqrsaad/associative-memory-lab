"""Dataset workspace view.

The view owns only presentation state: it routes user actions to the
:class:`DatasetViewModel` and renders what the view model exposes. All
image processing and validation live in the application services; the
view never computes vectors, encodings, or compatibility itself.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from associative_memory.application.dto.training_result import MemoryType
from associative_memory.application.services.image_service import (
    DEFAULT_PREPROCESSING_CONFIG,
    Encoding,
    FlatteningOrder,
    ForegroundPolarity,
    GrayscaleMethod,
    PreprocessingConfig,
    ResizeMode,
)
from associative_memory.presentation.theme.spacing import (
    SPACE_MD,
    SPACE_SM,
    SPACE_XS,
    SPACE_XXS,
)
from associative_memory.presentation.viewmodels.dataset_viewmodel import (
    DatasetEntry,
    DatasetViewModel,
)
from associative_memory.presentation.widgets import (
    ImageCanvas,
    MetricCard,
    PixelGrid,
    StatusBadge,
    StatusKind,
    VectorView,
)

_CONFIG_PANEL_WIDTH = 260
_ENTRY_LIST_WIDTH = 260
_PREVIEW_SIZE = 220


class EntryRow(QFrame):
    """One dataset entry row with its target-label assignment."""

    def __init__(
        self,
        entry: DatasetEntry,
        is_selected: bool,
        on_select: Callable[[str], None],
        on_remove: Callable[[str], None],
        on_target_changed: Callable[[str, str], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("EntryRow")
        self._entry = entry
        self._on_target_changed = on_target_changed

        layout = QHBoxLayout(self)
        layout.setContentsMargins(SPACE_XS, SPACE_XXS, SPACE_XS, SPACE_XXS)
        layout.setSpacing(SPACE_XS)

        badge_text = "source" if entry.memory_type is MemoryType.AUTOASSOCIATIVE else "input"
        if entry.memory_type is MemoryType.HETEROASSOCIATIVE and entry.target_label:
            badge_text = f"target: {entry.target_label}"

        self._role_label = QLabel(badge_text, self)
        self._role_label.setMinimumWidth(95)

        self._name_label = QLabel(
            Path(entry.source_path).name if entry.source_path else "(untitled)",
            self,
        )
        self._name_label.setToolTip(str(entry.source_path) if entry.source_path else "")
        self._name_label.setMinimumWidth(120)

        self._target_input = QComboBox(self)
        self._target_input.setEditable(True)
        self._target_input.setMinimumWidth(110)
        self._target_input.addItem(entry.target_label or "")
        self._target_input.setCurrentText(entry.target_label or "")
        self._target_input.setEnabled(entry.memory_type is MemoryType.HETEROASSOCIATIVE)
        self._target_input.currentTextChanged.connect(self._on_text_changed)

        select_button = QPushButton("Select", self)
        select_button.setMinimumWidth(64)
        select_button.clicked.connect(lambda: on_select(entry.entry_id))

        remove_button = QPushButton("Remove", self)
        remove_button.setMinimumWidth(64)
        remove_button.clicked.connect(lambda: on_remove(entry.entry_id))

        layout.addWidget(self._role_label)
        layout.addWidget(self._name_label)
        layout.addWidget(self._target_input)
        layout.addStretch(1)
        layout.addWidget(select_button)
        layout.addWidget(remove_button)

        self.set_selected(is_selected)

    def set_selected(self, selected: bool) -> None:
        self.setProperty("selected", "true" if selected else "false")
        self.style().unpolish(self)
        self.style().polish(self)

    def update_entry(self, entry: DatasetEntry, is_selected: bool) -> None:
        self._entry = entry
        self._role_label.setText(
            "source"
            if entry.memory_type is MemoryType.AUTOASSOCIATIVE
            else f"target: {entry.target_label or '—'}"
        )
        self._target_input.setEnabled(entry.memory_type is MemoryType.HETEROASSOCIATIVE)
        self._target_input.blockSignals(True)
        self._target_input.setCurrentText(entry.target_label or "")
        self._target_input.blockSignals(False)
        self._name_label.setText(
            Path(entry.source_path).name if entry.source_path else "(untitled)"
        )
        self._name_label.setToolTip(str(entry.source_path) if entry.source_path else "")
        self.set_selected(is_selected)

    def _on_text_changed(self, text: str) -> None:
        self._on_target_changed(self._entry.entry_id, text)


class DatasetView(QWidget):
    """Dataset workspace view: import, preprocess, and organize patterns."""

    def __init__(
        self,
        view_model: DatasetViewModel | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._view_model = view_model or DatasetViewModel()
        self._rows: dict[str, EntryRow] = {}

        self._build_layout()
        self._wire_signals()
        self._refresh_state()

    # ------------------------------------------------------------------ layout

    def _build_layout(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(SPACE_SM, SPACE_SM, SPACE_SM, SPACE_SM)
        root.setSpacing(SPACE_XS)

        # Import toolbar.
        toolbar = QHBoxLayout()
        toolbar.setSpacing(SPACE_XS)

        self._import_button = QPushButton("Import Image…", self)
        self._import_button.setMinimumWidth(130)
        self._import_button.setToolTip("Load a PNG, JPEG, or BMP image file.")
        toolbar.addWidget(self._import_button)

        self._add_button = QPushButton("Add to Dataset", self)
        self._add_button.setMinimumWidth(120)
        self._add_button.setToolTip("Add the working pattern to the dataset.")
        self._add_button.setEnabled(False)
        self._add_button.clicked.connect(self._view_model.add_entry)
        toolbar.addWidget(self._add_button)

        self._memory_combo = QComboBox(self)
        self._memory_combo.addItem("Autoassociative", MemoryType.AUTOASSOCIATIVE)
        self._memory_combo.addItem("Heteroassociative", MemoryType.HETEROASSOCIATIVE)
        self._memory_combo.setToolTip(
            "Autoassociative: each pattern is its own target. "
            "Heteroassociative: assign a target label per entry."
        )
        toolbar.addWidget(self._memory_combo)

        toolbar.addStretch(1)

        self._file_label = QLabel("No image imported", self)
        self._file_label.setObjectName("FileLabel")
        toolbar.addWidget(self._file_label)

        root.addLayout(toolbar)

        # Central splitter: config panel left, preview right.
        splitter = self._build_splitter()
        root.addWidget(splitter, stretch=1)

        # Dataset entries.
        root.addWidget(self._build_entries_panel())

        self._status_badge = StatusBadge(StatusKind.INFO, "Ready")
        root.addWidget(self._status_badge)

    def _build_splitter(self) -> QSplitter:
        splitter = QSplitter(Qt.Orientation.Horizontal, self)

        # Left: preprocessing controls.
        config_panel = QFrame(self)
        config_panel.setObjectName("ConfigPanel")
        config_panel.setMinimumWidth(_CONFIG_PANEL_WIDTH)
        config_panel.setMaximumWidth(_CONFIG_PANEL_WIDTH)
        config_layout = QVBoxLayout(config_panel)
        config_layout.setContentsMargins(SPACE_SM, SPACE_SM, SPACE_SM, SPACE_SM)
        config_layout.setSpacing(SPACE_XS)

        controls_title = QLabel("Preprocessing", config_panel)
        _title_font = controls_title.font()
        _title_font.setWeight(QFont.Weight.Bold)
        controls_title.setFont(_title_font)
        config_layout.addWidget(controls_title)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(SPACE_XS)
        grid.setVerticalSpacing(SPACE_XXS)

        self._width_spin = QSpinBox(config_panel)
        self._width_spin.setRange(1, 512)
        self._width_spin.setValue(DEFAULT_PREPROCESSING_CONFIG.width)
        self._width_spin.setToolTip("Target width in pixels.")

        self._height_spin = QSpinBox(config_panel)
        self._height_spin.setRange(1, 512)
        self._height_spin.setValue(DEFAULT_PREPROCESSING_CONFIG.height)
        self._height_spin.setToolTip("Target height in pixels.")

        self._grayscale_combo = QComboBox(config_panel)
        self._grayscale_combo.addItem("Luminance 601", GrayscaleMethod.LUMINANCE_601)
        self._grayscale_combo.addItem("Average", GrayscaleMethod.AVERAGE)
        self._grayscale_combo.setToolTip("Grayscale conversion method.")

        self._threshold_spin = QSpinBox(config_panel)
        self._threshold_spin.setRange(0, 255)
        self._threshold_spin.setValue(DEFAULT_PREPROCESSING_CONFIG.threshold)
        self._threshold_spin.setToolTip("Grayscale threshold (0-255).")

        self._polarity_combo = QComboBox(config_panel)
        self._polarity_combo.addItem("Dark is ON", ForegroundPolarity.DARK_IS_ON)
        self._polarity_combo.addItem("Light is ON", ForegroundPolarity.LIGHT_IS_ON)
        self._polarity_combo.setToolTip("Which grayscale side is foreground.")

        self._encoding_combo = QComboBox(config_panel)
        self._encoding_combo.addItem("Binary (0/1)", Encoding.BINARY)
        self._encoding_combo.addItem("Bipolar (-1/+1)", Encoding.BIPOLAR)
        self._encoding_combo.setToolTip("Encoded pattern values.")

        self._resize_combo = QComboBox(config_panel)
        self._resize_combo.addItem("Preserve aspect", ResizeMode.PRESERVE_ASPECT)
        self._resize_combo.addItem("Stretch", ResizeMode.STRETCH)
        self._resize_combo.setToolTip("Resizing strategy.")

        self._flattening_combo = QComboBox(config_panel)
        self._flattening_combo.addItem("Row-major", FlatteningOrder.ROW_MAJOR)
        self._flattening_combo.addItem("Column-major", FlatteningOrder.COLUMN_MAJOR)
        self._flattening_combo.setToolTip("Vector flattening order.")

        grid.addWidget(QLabel("Width"), 0, 0)
        grid.addWidget(self._width_spin, 0, 1)
        grid.addWidget(QLabel("Height"), 1, 0)
        grid.addWidget(self._height_spin, 1, 1)
        grid.addWidget(QLabel("Grayscale"), 2, 0)
        grid.addWidget(self._grayscale_combo, 2, 1)
        grid.addWidget(QLabel("Threshold"), 3, 0)
        grid.addWidget(self._threshold_spin, 3, 1)
        grid.addWidget(QLabel("Polarity"), 4, 0)
        grid.addWidget(self._polarity_combo, 4, 1)
        grid.addWidget(QLabel("Encoding"), 5, 0)
        grid.addWidget(self._encoding_combo, 5, 1)
        grid.addWidget(QLabel("Resize mode"), 6, 0)
        grid.addWidget(self._resize_combo, 6, 1)
        grid.addWidget(QLabel("Flattening"), 7, 0)
        grid.addWidget(self._flattening_combo, 7, 1)

        config_layout.addLayout(grid)

        self._preview_button = QPushButton("Update Preview", config_panel)
        self._preview_button.setMinimumHeight(40)
        self._preview_button.setToolTip(
            "Re-preprocess the imported image with these settings."
        )
        config_layout.addWidget(self._preview_button)

        self._config_note = QLabel(
            "Stored entries keep their own configuration snapshots. "
            "Changing settings updates only the working preview.",
            config_panel,
        )
        self._config_note.setWordWrap(True)
        self._config_note.setObjectName("Caption")
        config_layout.addWidget(self._config_note)
        config_layout.addStretch(1)

        splitter.addWidget(config_panel)

        # Right: preview area.
        preview_panel = QWidget(self)
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(SPACE_SM)

        preview_row = QHBoxLayout()
        preview_row.setSpacing(SPACE_MD)

        # Original image.
        original_group = QVBoxLayout()
        original_group.setSpacing(SPACE_XXS)
        original_group.addWidget(QLabel("Original Image", self))
        self._original_canvas = ImageCanvas(self)
        self._original_canvas.setMinimumSize(_PREVIEW_SIZE, _PREVIEW_SIZE)
        original_group.addWidget(self._original_canvas)
        preview_row.addLayout(original_group, stretch=1)

        # Processed preview.
        processed_group = QVBoxLayout()
        processed_group.setSpacing(SPACE_XXS)
        processed_group.addWidget(QLabel("Processed Image", self))
        self._processed_canvas = ImageCanvas(self)
        self._processed_canvas.setMinimumSize(_PREVIEW_SIZE, _PREVIEW_SIZE)
        processed_group.addWidget(self._processed_canvas)
        preview_row.addLayout(processed_group, stretch=1)

        preview_layout.addLayout(preview_row)

        # Pixel grid and vector view side by side.
        grid_row = QHBoxLayout()
        grid_row.setSpacing(SPACE_MD)

        grid_group = QVBoxLayout()
        grid_group.setSpacing(SPACE_XXS)
        grid_group.addWidget(QLabel("Encoded Pixel Grid", self))
        self._pixel_grid = PixelGrid(self)
        self._pixel_grid.setMinimumSize(_PREVIEW_SIZE, _PREVIEW_SIZE)
        grid_group.addWidget(self._pixel_grid)
        grid_row.addLayout(grid_group, stretch=1)

        vector_group = QVBoxLayout()
        vector_group.setSpacing(SPACE_XXS)
        vector_group.addWidget(QLabel("Encoded Vector", self))
        self._vector_view = VectorView(self)
        self._vector_view.setMinimumSize(_PREVIEW_SIZE, _PREVIEW_SIZE)
        vector_group.addWidget(self._vector_view)
        grid_row.addLayout(vector_group, stretch=1)

        preview_layout.addLayout(grid_row)

        # Metrics.
        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(SPACE_MD)

        self._width_metric = MetricCard("Pattern Width", self)
        self._height_metric = MetricCard("Pattern Height", self)
        self._vector_length_metric = MetricCard("Vector Length", self)
        self._encoding_metric = MetricCard("Encoding", self)
        self._pixels_on_metric = MetricCard("Pixels ON", self)

        for card in (
            self._width_metric,
            self._height_metric,
            self._vector_length_metric,
            self._encoding_metric,
            self._pixels_on_metric,
        ):
            metrics_row.addWidget(card, stretch=1)

        preview_layout.addLayout(metrics_row)
        splitter.addWidget(preview_panel)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        return splitter

    def _build_entries_panel(self) -> QFrame:
        panel = QFrame(self)
        panel.setObjectName("EntriesPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACE_SM)

        header = QHBoxLayout()
        header.setContentsMargins(SPACE_XS, SPACE_XXS, SPACE_XS, SPACE_XXS)
        header.setSpacing(SPACE_XS)
        entries_title = QLabel("Dataset Entries", self)
        _title_font = entries_title.font()
        _title_font.setWeight(QFont.Weight.Bold)
        entries_title.setFont(_title_font)
        header.addWidget(entries_title)
        header.addStretch(1)

        self._clear_button = QPushButton("Clear All", self)
        self._clear_button.setMinimumWidth(72)
        self._clear_button.setToolTip("Remove every dataset entry.")
        self._clear_button.setEnabled(False)
        header.addWidget(self._clear_button)

        layout.addLayout(header)

        self._entries_list = QListWidget(self)
        self._entries_list.setMinimumHeight(160)
        self._entries_list.setAlternatingRowColors(True)
        self._entries_list.currentRowChanged.connect(self._on_entry_row_changed)
        layout.addWidget(self._entries_list, stretch=1)

        actions = QHBoxLayout()
        actions.setSpacing(SPACE_XS)

        self._remove_button = QPushButton("Remove Selected", self)
        self._remove_button.setMinimumWidth(110)
        self._remove_button.setEnabled(False)
        self._remove_button.clicked.connect(self._remove_selected_entry)
        actions.addWidget(self._remove_button)

        actions.addStretch(1)
        layout.addLayout(actions)

        return panel

    # ------------------------------------------------------------------ wiring

    def _wire_signals(self) -> None:
        vm = self._view_model

        self._import_button.clicked.connect(self._import_image)
        self._preview_button.clicked.connect(self._update_preview)
        self._memory_combo.currentIndexChanged.connect(self._on_memory_changed)

        self._width_spin.valueChanged.connect(
            lambda value: vm.update_config(width=value)
        )
        self._height_spin.valueChanged.connect(
            lambda value: vm.update_config(height=value)
        )
        self._grayscale_combo.currentIndexChanged.connect(
            lambda index: vm.update_config(
                grayscale_method=self._grayscale_combo.itemData(index)
            )
        )
        self._threshold_spin.valueChanged.connect(
            lambda value: vm.update_config(threshold=value)
        )
        self._polarity_combo.currentIndexChanged.connect(
            lambda index: vm.update_config(
                polarity=self._polarity_combo.itemData(index)
            )
        )
        self._encoding_combo.currentIndexChanged.connect(
            lambda index: vm.update_config(encoding=self._encoding_combo.itemData(index))
        )
        self._resize_combo.currentIndexChanged.connect(
            lambda index: vm.update_config(
                resize_mode=self._resize_combo.itemData(index)
            )
        )
        self._flattening_combo.currentIndexChanged.connect(
            lambda index: vm.update_config(
                flattening_order=self._flattening_combo.itemData(index)
            )
        )

        vm.config_changed.connect(self._on_config_changed)
        vm.preview_changed.connect(self._on_preview_changed)
        vm.entries_changed.connect(self._on_entries_changed)
        vm.entry_selected.connect(self._on_entry_selected)
        vm.status_message.connect(self._on_status_message)
        vm.memory_type_changed.connect(self._on_memory_type_changed)

    # ------------------------------------------------------------------ actions

    def _import_image(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Import Image")
        file_dialog.setNameFilters(
            ["Image files (*.png *.jpg *.jpeg *.bmp)", "All files (*.*)"]
        )
        file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        if not file_dialog.exec():
            return

        selected = file_dialog.selectedFiles()
        if not selected:
            return
        vm = self._view_model
        vm.import_image(selected[0])
        path = Path(selected[0])
        self._file_label.setText(path.name)
        self._file_label.setToolTip(str(path))

    def _update_preview(self) -> None:
        vm = self._view_model
        fields = {
            "width": self._width_spin.value(),
            "height": self._height_spin.value(),
            "grayscale_method": self._grayscale_combo.currentData(),
            "threshold": self._threshold_spin.value(),
            "polarity": self._polarity_combo.currentData(),
            "encoding": self._encoding_combo.currentData(),
            "resize_mode": self._resize_combo.currentData(),
            "flattening_order": self._flattening_combo.currentData(),
        }
        ok = vm.update_config(**fields)
        if not ok:
            self._refresh_controls_from_config()
        else:
            self._status_badge.set_status(StatusKind.INFO, "Preview updated")

    # ------------------------------------------------------------------ refresh

    def _refresh_state(self) -> None:
        vm = self._view_model
        self._file_label.setText("No image imported")
        self._file_label.setToolTip("")
        self._memory_combo.blockSignals(True)
        self._memory_combo.setCurrentIndex(
            0 if vm.memory_type is MemoryType.AUTOASSOCIATIVE else 1
        )
        self._memory_combo.blockSignals(False)
        self._refresh_controls_from_config()
        self._on_entries_changed()
        self._on_preview_changed(vm.current_result)
        self._on_status_message("Ready", StatusKind.INFO)

    def _refresh_controls_from_config(self) -> None:
        config = self._view_model.config

        self._width_spin.blockSignals(True)
        self._height_spin.blockSignals(True)
        self._threshold_spin.blockSignals(True)
        self._width_spin.setValue(config.width)
        self._height_spin.setValue(config.height)
        self._threshold_spin.setValue(config.threshold)
        self._width_spin.blockSignals(False)
        self._height_spin.blockSignals(False)
        self._threshold_spin.blockSignals(False)

        self._set_combo(self._grayscale_combo, config.grayscale_method)
        self._set_combo(self._polarity_combo, config.polarity)
        self._set_combo(self._encoding_combo, config.encoding)
        self._set_combo(self._resize_combo, config.resize_mode)
        self._set_combo(self._flattening_combo, config.flattening_order)

    def _set_combo(self, combo: QComboBox, value: object) -> None:
        index = combo.findData(value)
        if index == -1:
            return
        combo.blockSignals(True)
        combo.setCurrentIndex(index)
        combo.blockSignals(False)

    # ------------------------------------------------------------------ event handlers

    def _on_memory_changed(self, index: int) -> None:
        value = self._memory_combo.itemData(index)
        self._view_model.set_memory_type(value)

    def _on_memory_type_changed(self, memory_type: MemoryType) -> None:
        self._on_entries_changed()

    def _on_config_changed(self, config: PreprocessingConfig) -> None:
        self._refresh_controls_from_config()
        self._update_preview_metrics(config)

    def _on_preview_changed(self, result) -> None:
        if result is None:
            self._original_canvas.clear()
            self._processed_canvas.clear()
            self._pixel_grid.set_matrix(None)
            self._vector_view.set_vector(None)
            self._update_preview_metrics(None)
            self._add_button.setEnabled(False)
            return

        self._original_canvas.set_image(result.original_image)
        self._processed_canvas.set_image(None)  # placeholder: processed image TBD
        self._pixel_grid.set_matrix(result.encoded_matrix)
        self._vector_view.set_vector(
            result.encoded_vector, row_width=result.width
        )
        self._update_preview_metrics(result)
        self._add_button.setEnabled(self._view_model.current_is_compatible())

    def _update_preview_metrics(self, result) -> None:
        if result is None:
            for card in (
                self._width_metric,
                self._height_metric,
                self._vector_length_metric,
                self._encoding_metric,
                self._pixels_on_metric,
            ):
                card.set_value(None)
            return

        self._width_metric.set_value(result.width)
        self._height_metric.set_value(result.height)
        self._vector_length_metric.set_value(result.vector_length)
        self._encoding_metric.set_value(result.encoding.value)
        if result.encoded_matrix is not None:
            self._pixels_on_metric.set_value(
                int((result.encoded_matrix != 0).sum())
            )
        else:
            self._pixels_on_metric.set_value(None)

    def _on_entries_changed(self) -> None:
        vm = self._view_model
        selected_id = vm.selected_entry.entry_id if vm.selected_entry else None

        self._entries_list.blockSignals(True)
        self._entries_list.clear()
        self._rows = {}

        for entry in vm.entries:
            item = QListWidgetItem(self._entries_list)
            row = EntryRow(
                entry,
                is_selected=(entry.entry_id == selected_id),
                on_select=self._view_model.select_entry,
                on_remove=self._view_model.remove_entry,
                on_target_changed=self._view_model.set_selected_target,
            )
            item.setSizeHint(row.sizeHint())
            self._entries_list.addItem(item)
            self._entries_list.setItemWidget(item, row)
            self._rows[entry.entry_id] = row

        self._entries_list.blockSignals(False)

        self._clear_button.setEnabled(len(vm.entries) > 0)
        self._remove_button.setEnabled(selected_id is not None)
        self._add_button.setEnabled(
            self._view_model.current_is_compatible()
        )

    def _on_entry_row_changed(self, row: int) -> None:
        if row < 0:
            self._view_model.select_entry(None)
            return
        item = self._entries_list.item(row)
        if item is None:
            return
        widget = self._entries_list.itemWidget(item)
        if isinstance(widget, EntryRow):
            self._view_model.select_entry(widget._entry.entry_id)

    def _on_entry_selected(self, entry: DatasetEntry | None) -> None:
        for entry_id, row in self._rows.items():
            row.set_selected(entry is not None and entry.entry_id == entry_id)
        if entry is None:
            self._remove_button.setEnabled(False)
        else:
            self._remove_button.setEnabled(True)

    def _on_status_message(self, message: str, kind: StatusKind) -> None:
        self._status_badge.set_status(kind, message)

    def _remove_selected_entry(self) -> None:
        entry = self._view_model.selected_entry
        if entry is None:
            return
        self._view_model.remove_entry(entry.entry_id)
