"""Training workspace view.

Wires together the training view model, shared dataset view model, and
Task 05 design-system widgets to provide a complete training workflow:
configure memory type and encoding, review the dataset, train, and
inspect the learning trace and weight matrix.

Presentation-only: all numerical work is delegated to the view model and
application services.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from associative_memory.application.dto.training_result import MemoryType
from associative_memory.presentation.theme.spacing import (
    SPACE_MD,
    SPACE_SM,
    SPACE_XS,
    SPACE_XXS,
)
from associative_memory.presentation.viewmodels.dataset_viewmodel import (
    MEMORY_AUTOASSOCIATIVE,
    MEMORY_HETEROASSOCIATIVE,
    DatasetViewModel,
)
from associative_memory.presentation.viewmodels.training_viewmodel import (
    TrainingState,
    TrainingViewModel,
)
from associative_memory.presentation.widgets import (
    FormulaPanel,
    MetricCard,
    StatusBadge,
    StatusKind,
    WeightMatrixView,
)

_CONFIG_PANEL_WIDTH = 280


class TrainingView(QWidget):
    """Training workspace view."""

    def __init__(
        self,
        dataset_view_model: DatasetViewModel | None = None,
        training_view_model: TrainingViewModel | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        if training_view_model is not None:
            shared_dataset = training_view_model.dataset_view_model
            if (
                dataset_view_model is not None
                and dataset_view_model is not shared_dataset
            ):
                raise ValueError(
                    "dataset_view_model must be the training view model's "
                    "dataset view model."
                )
            self._dataset_view_model = shared_dataset
            self._view_model = training_view_model
        else:
            self._dataset_view_model = dataset_view_model or DatasetViewModel()
            self._view_model = TrainingViewModel(
                dataset_view_model=self._dataset_view_model
            )

        self._build_layout()
        self._wire_signals()
        self._refresh_state()

    # ------------------------------------------------------------------ layout

    def _build_layout(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(SPACE_SM, SPACE_SM, SPACE_SM, SPACE_SM)
        root.setSpacing(SPACE_XS)

        # Configuration panel.
        config_panel = self._build_config_panel()
        root.addWidget(config_panel)

        # Dataset review section.
        dataset_panel = self._build_dataset_panel()
        root.addWidget(dataset_panel)

        # Train button and status.
        action_row = QHBoxLayout()
        action_row.setSpacing(SPACE_XS)

        self._train_button = QPushButton("Train Memory", self)
        self._train_button.setMinimumWidth(140)
        self._train_button.setEnabled(False)
        self._train_button.clicked.connect(self._view_model.train)
        action_row.addWidget(self._train_button)

        self._cancel_button = QPushButton("Cancel", self)
        self._cancel_button.setMinimumWidth(100)
        self._cancel_button.setEnabled(False)
        self._cancel_button.clicked.connect(self._view_model.cancel_training)
        action_row.addWidget(self._cancel_button)

        action_row.addStretch(1)

        self._train_status = StatusBadge(StatusKind.INFO, "Ready")
        action_row.addWidget(self._train_status)

        root.addLayout(action_row)

        # Results area (scrollable).
        results_scroll = QScrollArea(self)
        results_scroll.setWidgetResizable(True)
        results_scroll.setFrameShape(QFrame.Shape.NoFrame)

        results_content = QWidget()
        results_layout = QVBoxLayout(results_content)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(SPACE_MD)

        # Formula panel.
        self._formula_panel = FormulaPanel()
        results_layout.addWidget(self._formula_panel)

        # Weight matrix view.
        self._weight_matrix_view = WeightMatrixView()
        results_layout.addWidget(self._weight_matrix_view)

        # Metrics row.
        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(SPACE_MD)

        self._pairs_metric = MetricCard("Training Pairs", self)
        self._source_dim_metric = MetricCard("Source Dimension", self)
        self._target_dim_metric = MetricCard("Target Dimension", self)
        self._encoding_metric = MetricCard("Encoding", self)

        for card in (
            self._pairs_metric,
            self._source_dim_metric,
            self._target_dim_metric,
            self._encoding_metric,
        ):
            metrics_row.addWidget(card, stretch=1)

        results_layout.addLayout(metrics_row)

        # Trace section.
        trace_group = QGroupBox("Learning Trace", self)
        trace_layout = QVBoxLayout(trace_group)
        trace_layout.setContentsMargins(SPACE_XS, SPACE_XS, SPACE_XS, SPACE_XS)
        trace_layout.setSpacing(SPACE_XXS)

        self._trace_toggle = QCheckBox("Show detailed calculation trace", self)
        self._trace_toggle.setChecked(False)
        self._trace_toggle.toggled.connect(self._view_model.set_trace_enabled)
        trace_layout.addWidget(self._trace_toggle)

        self._trace_list = QListWidget(self)
        self._trace_list.setMaximumHeight(200)
        self._trace_list.hide()
        trace_layout.addWidget(self._trace_list)

        results_layout.addWidget(trace_group)

        results_scroll.setWidget(results_content)
        root.addWidget(results_scroll, stretch=1)

    def _build_config_panel(self) -> QFrame:
        panel = QFrame(self)
        panel.setObjectName("ConfigPanel")

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(SPACE_XS, SPACE_XS, SPACE_XS, SPACE_XS)
        layout.setSpacing(SPACE_SM)

        title = QLabel("Training Configuration", panel)
        title_font = title.font()
        title_font.setWeight(QFont.Weight.Bold)
        title.setFont(title_font)
        layout.addWidget(title)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(SPACE_XS)
        grid.setVerticalSpacing(SPACE_XXS)

        self._memory_combo = QComboBox(panel)
        self._memory_combo.addItem("Autoassociative", MEMORY_AUTOASSOCIATIVE)
        self._memory_combo.addItem("Heteroassociative", MEMORY_HETEROASSOCIATIVE)
        self._memory_combo.setToolTip(
            "Autoassociative: each pattern is its own target. "
            "Heteroassociative: assign a target label per entry."
        )

        self._encoding_combo = QComboBox(panel)
        self._encoding_combo.addItem("Binary (0/1)", "binary")
        self._encoding_combo.addItem("Bipolar (-1/+1)", "bipolar")
        self._encoding_combo.setToolTip("Encoded pattern values.")

        grid.addWidget(QLabel("Memory Type"), 0, 0)
        grid.addWidget(self._memory_combo, 0, 1)
        grid.addWidget(QLabel("Encoding"), 1, 0)
        grid.addWidget(self._encoding_combo, 1, 1)

        layout.addLayout(grid)

        self._trace_toggle_config = QCheckBox("Detailed calculation trace", panel)
        self._trace_toggle_config.setChecked(False)
        self._trace_toggle_config.toggled.connect(
            self._view_model.set_trace_enabled
        )
        layout.addWidget(self._trace_toggle_config)

        return panel

    def _build_dataset_panel(self) -> QFrame:
        panel = QFrame(self)
        panel.setObjectName("EntriesPanel")

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACE_SM)

        header = QHBoxLayout()
        header.setContentsMargins(SPACE_XS, SPACE_XXS, SPACE_XS, SPACE_XXS)
        header.setSpacing(SPACE_XS)
        entries_title = QLabel("Training Dataset", panel)
        entries_font = entries_title.font()
        entries_font.setWeight(QFont.Weight.Bold)
        entries_title.setFont(entries_font)
        header.addWidget(entries_title)
        header.addStretch(1)

        self._clear_dataset_button = QPushButton("Clear All", panel)
        self._clear_dataset_button.setMinimumWidth(72)
        self._clear_dataset_button.setToolTip("Remove every dataset entry.")
        self._clear_dataset_button.setEnabled(False)
        self._clear_dataset_button.clicked.connect(
            self._dataset_view_model.clear_dataset
        )
        header.addWidget(self._clear_dataset_button)

        layout.addLayout(header)

        self._entries_list = QListWidget(panel)
        self._entries_list.setMinimumHeight(120)
        self._entries_list.setAlternatingRowColors(True)
        layout.addWidget(self._entries_list, stretch=1)

        # Target assignment for heteroassociative.
        self._target_group = QGroupBox("Target Assignment", panel)
        target_layout = QVBoxLayout(self._target_group)
        target_layout.setContentsMargins(SPACE_XS, SPACE_XS, SPACE_XS, SPACE_XS)
        target_layout.setSpacing(SPACE_XXS)

        self._target_label_input = QComboBox(panel)
        self._target_label_input.setEditable(True)
        self._target_label_input.setToolTip(
            "Enter a target label or select an existing entry ID."
        )
        target_layout.addWidget(QLabel("Target label for selected entry:"))
        target_layout.addWidget(self._target_label_input)

        target_buttons = QHBoxLayout()
        target_buttons.setSpacing(SPACE_XS)
        self._set_target_button = QPushButton("Set Target", panel)
        self._set_target_button.clicked.connect(self._set_selected_target)
        target_buttons.addWidget(self._set_target_button)
        self._clear_target_button = QPushButton("Clear Target", panel)
        self._clear_target_button.clicked.connect(self._clear_selected_target)
        target_buttons.addWidget(self._clear_target_button)
        target_buttons.addStretch(1)
        target_layout.addLayout(target_buttons)

        self._target_group.setVisible(False)
        layout.addWidget(self._target_group)

        return panel

    # ------------------------------------------------------------------ wiring

    def _wire_signals(self) -> None:
        vm = self._view_model
        dvm = self._dataset_view_model

        self._memory_combo.currentIndexChanged.connect(self._on_memory_changed)
        self._encoding_combo.currentIndexChanged.connect(self._on_encoding_changed)

        vm.state_changed.connect(self._on_state_changed)
        vm.result_ready.connect(self._on_result_ready)
        vm.error_occurred.connect(self._on_error)
        vm.trace_ready.connect(self._on_trace_ready)
        vm.status_message.connect(self._on_status_message)

        dvm.entries_changed.connect(self._on_dataset_entries_changed)
        dvm.entry_selected.connect(self._on_dataset_entry_selected)
        dvm.memory_type_changed.connect(self._on_dataset_memory_type_changed)

    # ------------------------------------------------------------------ refresh

    def _refresh_state(self) -> None:
        vm = self._view_model
        dvm = self._dataset_view_model

        self._memory_combo.blockSignals(True)
        self._memory_combo.setCurrentIndex(
            0 if dvm.memory_type == MEMORY_AUTOASSOCIATIVE else 1
        )
        self._memory_combo.blockSignals(False)

        self._encoding_combo.blockSignals(True)
        self._encoding_combo.setCurrentText(vm.encoding)
        self._encoding_combo.blockSignals(False)

        self._trace_toggle_config.blockSignals(True)
        self._trace_toggle_config.setChecked(vm.trace_enabled)
        self._trace_toggle_config.blockSignals(False)

        self._target_group.setVisible(
            dvm.memory_type == MEMORY_HETEROASSOCIATIVE
        )

        self._on_dataset_entries_changed()
        self._on_state_changed(vm.state)

    def _on_state_changed(self, state: TrainingState) -> None:
        self._train_button.setEnabled(state == TrainingState.READY)
        self._cancel_button.setEnabled(state == TrainingState.TRAINING)

        if state == TrainingState.COMPLETED:
            self._train_status.set_status(StatusKind.SUCCESS, "Training complete")
        elif state == TrainingState.FAILED:
            self._train_status.set_status(StatusKind.ERROR, "Training failed")
        elif state == TrainingState.TRAINING:
            self._train_status.set_status(
                StatusKind.IN_PROGRESS, "Training in progress…"
            )
        else:
            self._train_status.set_status(StatusKind.INFO, "Ready")

    def _on_dataset_entries_changed(self) -> None:
        dvm = self._dataset_view_model
        self._clear_dataset_button.setEnabled(len(dvm.entries) > 0)

        self._entries_list.blockSignals(True)
        self._entries_list.clear()

        for entry in dvm.entries:
            item = QListWidgetItem(self._entries_list)
            label = (
                f"{entry.source_path.name if entry.source_path else '(untitled)'} "
                f"({entry.result.vector_length} elements)"
            )
            if entry.target_label:
                label += f" -> target: {entry.target_label}"
            item.setText(label)
            item.setData(Qt.ItemDataRole.UserRole, entry.entry_id)
            self._entries_list.addItem(item)

        self._entries_list.blockSignals(False)

    def _on_dataset_entry_selected(self, entry) -> None:
        if entry is None:
            self._target_label_input.blockSignals(True)
            self._target_label_input.setCurrentText("")
            self._target_label_input.blockSignals(False)
            self._set_target_button.setEnabled(False)
            self._clear_target_button.setEnabled(False)
            return

        self._target_label_input.blockSignals(True)
        self._target_label_input.setCurrentText(entry.target_label or "")
        self._target_label_input.blockSignals(False)
        self._set_target_button.setEnabled(True)
        self._clear_target_button.setEnabled(True)

    def _on_dataset_memory_type_changed(self, value: str) -> None:
        self._target_group.setVisible(value == MEMORY_HETEROASSOCIATIVE)
        self._refresh_state()

    # ------------------------------------------------------------------ configuration handlers

    def _on_memory_changed(self, index: int) -> None:
        value = self._memory_combo.itemData(index)
        self._view_model.set_memory_type(value)

    def _on_encoding_changed(self, index: int) -> None:
        value = self._encoding_combo.itemData(index)
        self._view_model.set_encoding(value)

    # ------------------------------------------------------------------ target assignment

    def _set_selected_target(self) -> None:
        entry = self._dataset_view_model.selected_entry
        if entry is None:
            return
        label = self._target_label_input.currentText().strip()
        self._dataset_view_model.set_selected_target(
            label if label else None
        )

    def _clear_selected_target(self) -> None:
        self._dataset_view_model.set_selected_target(None)

    # ------------------------------------------------------------------ results

    def _on_result_ready(self, result) -> None:
        self._weight_matrix_view.set_matrix(
            result.weight_matrix,
            row_labels=[f"s[{i}]" for i in range(result.input_dimension)],
            col_labels=[f"t[{j}]" for j in range(result.output_dimension)],
        )

        self._pairs_metric.set_value(result.pair_count)
        self._source_dim_metric.set_value(result.input_dimension)
        self._target_dim_metric.set_value(result.output_dimension)
        self._encoding_metric.set_value(result.encoding.value)

        self._formula_panel.set_learning_rule(
            memory_type=result.memory_type.value,
            encoding=result.encoding.value,
            source_dim=result.input_dimension,
            target_dim=result.output_dimension,
            pair_count=result.pair_count,
        )
        self._formula_panel.set_equation(
            f"W = Σ outer(s, {'s' if result.memory_type is MemoryType.AUTOASSOCIATIVE else 't'})"
        )

        self._trace_list.clear()
        self._trace_list.hide()

    def _on_trace_ready(self, steps) -> None:
        self._trace_list.clear()
        for step in steps:
            item = QListWidgetItem(
                f"Pair {step.pair_index}: {step.equation} "
                f"(contribution {step.contribution.shape[0]}×{step.contribution.shape[1]})"
            )
            item.setData(Qt.ItemDataRole.UserRole, step)
            self._trace_list.addItem(item)
        self._trace_list.show()

    def _on_error(self, message: str) -> None:
        pass  # already shown via status_message

    def _on_status_message(self, message: str, kind: StatusKind) -> None:
        self._train_status.set_status(kind, message)
