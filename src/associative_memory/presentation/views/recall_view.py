"""Recall workspace view.

Coordinates the recall workflow between the view model and the existing
design-system widgets: select a trained model and compatible cue, review
preprocessing, run recall, inspect the input/output representations,
compare against a reference, inspect the numerical trace, and review
evidence-based warnings.

Presentation-only: all inference, comparison, and formatting of the
calculation trace are delegated to :class:`RecallViewModel` and the
application services.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from associative_memory.application.dto.recall_result import (
    COMPARISON_BASIS_INPUT,
    RecallResult,
)
from associative_memory.application.services.image_service import (
    PreprocessingConfig,
)
from associative_memory.presentation.theme.spacing import (
    SPACE_MD,
    SPACE_SM,
    SPACE_XS,
    SPACE_XXS,
)
from associative_memory.presentation.viewmodels.dataset_viewmodel import (
    DatasetViewModel,
)
from associative_memory.presentation.viewmodels.recall_viewmodel import (
    RecallState,
    RecallViewModel,
)
from associative_memory.presentation.viewmodels.training_viewmodel import (
    TrainingViewModel,
)
from associative_memory.presentation.widgets import (
    EmptyState,
    FormulaPanel,
    ImageCanvas,
    MetricCard,
    PatternDifferenceView,
    PixelGrid,
    StatusBadge,
    StatusKind,
    StepNavigator,
    TraceStep,
    VectorView,
    WeightMatrixView,
)

_PREVIEW_SIZE = 180
_VECTOR_SUMMARY_LIMIT = 32
_TRACE_MIN_HEIGHT = 320
_WEIGHT_MIN_HEIGHT = 260


class RecallView(QWidget):
    """The recall and results workspace."""

    def __init__(
        self,
        dataset_view_model: DatasetViewModel | None = None,
        training_view_model: TrainingViewModel | None = None,
        view_model: RecallViewModel | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        if view_model is not None:
            self._view_model = view_model
        else:
            self._view_model = RecallViewModel(
                dataset_view_model=dataset_view_model,
                training_view_model=training_view_model,
            )

        self._build_layout()
        self._wire_signals()
        self._refresh_all()

    # ------------------------------------------------------------------ layout

    def _build_layout(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(SPACE_SM, SPACE_SM, SPACE_SM, SPACE_SM)
        root.setSpacing(SPACE_XS)

        root.addWidget(self._build_control_panel())

        self._state_stack = QStackedWidget(self)
        self._state_stack.setObjectName("RecallStateStack")

        self._placeholder_page = self._build_placeholder_page()
        self._results_scroll = self._build_results_page()
        self._state_stack.addWidget(self._placeholder_page)
        self._state_stack.addWidget(self._results_scroll)

        root.addWidget(self._state_stack, stretch=1)

    def _build_control_panel(self) -> QFrame:
        panel = QFrame(self)
        panel.setObjectName("ConfigPanel")

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(SPACE_SM, SPACE_SM, SPACE_SM, SPACE_SM)
        layout.setSpacing(SPACE_XS)

        title = QLabel("Recall Setup", panel)
        title_font = title.font()
        title_font.setWeight(QFont.Weight.Bold)
        title.setFont(title_font)

        self._model_combo = self._make_combo(panel, "Trained recall model")
        self._model_combo.setToolTip(
            "Models trained during this session. Each new training run adds "
            "another model to the catalog."
        )
        self._input_combo = self._make_combo(panel, "Recall input pattern")
        self._input_combo.setToolTip(
            "Dataset entries, the current working preview, and imported "
            "images that match the selected model's dimensions and encoding."
        )
        self._target_combo = self._make_combo(panel, "Expected recall target")
        self._target_combo.setToolTip(
            "Expected target for heteroassociative comparison. Without one, "
            "no exact-match status or ratio is available."
        )

        self._import_button = QPushButton("Import Image…", panel)
        self._import_button.setMinimumWidth(120)
        self._import_button.setToolTip(
            "Select a trained model before importing a recall image."
        )
        self._import_button.clicked.connect(self._choose_import_image)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(SPACE_XS)
        grid.setVerticalSpacing(SPACE_XXS)
        grid.setColumnStretch(1, 1)

        grid.addWidget(QLabel("Trained model", panel), 0, 0)
        grid.addWidget(self._model_combo, 0, 1)
        grid.addWidget(self._import_button, 0, 2)

        grid.addWidget(QLabel("Input pattern", panel), 1, 0)
        grid.addWidget(self._input_combo, 1, 1, 1, 2)

        self._target_label = QLabel("Expected target", panel)
        grid.addWidget(self._target_label, 2, 0)
        grid.addWidget(self._target_combo, 2, 1, 1, 2)

        self._preprocessing_label = QLabel(panel)
        self._preprocessing_label.setObjectName("RecallPreprocessingSummary")
        self._preprocessing_label.setWordWrap(True)
        self._preprocessing_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        grid.addWidget(self._preprocessing_label, 3, 0, 1, 3)

        layout.addWidget(title)
        layout.addLayout(grid)

        actions = QHBoxLayout()
        actions.setSpacing(SPACE_XS)

        self._run_button = QPushButton("Run Recall", panel)
        self._run_button.setProperty("variant", "primary")
        self._run_button.setMinimumWidth(140)
        self._run_button.setToolTip(
            "Run recall for the selected model and query pattern."
        )
        self._run_button.clicked.connect(self._view_model.run_recall)
        actions.addWidget(self._run_button)

        self._save_button = QPushButton("Save Experiment…", panel)
        self._save_button.setMinimumWidth(140)
        self._save_button.setToolTip(
            "Store the selected model (and the latest recall result) in the "
            "experiment history."
        )
        self._save_button.clicked.connect(self._choose_save_experiment)
        actions.addWidget(self._save_button)

        actions.addStretch(1)

        self._status_badge = StatusBadge(StatusKind.INFO, "Ready", panel)
        self._status_badge.text_label.setWordWrap(True)
        self._status_badge.setMaximumWidth(620)
        actions.addWidget(self._status_badge)

        layout.addLayout(actions)
        return panel

    def _build_placeholder_page(self) -> QWidget:
        page = QWidget(self)
        page.setObjectName("RecallPlaceholderPage")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._empty_state = EmptyState(
            "Ready to run recall",
            "Select a trained model and a compatible input pattern.",
            parent=page,
        )
        layout.addWidget(self._empty_state)
        return page

    def _build_results_page(self) -> QScrollArea:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACE_MD)

        self._formula_panel = FormulaPanel(content)
        self._formula_panel.set_equation("y_in = x @ W")
        layout.addWidget(self._formula_panel)

        layout.addWidget(self._section_title("Result Summary", content))

        self._comparison_note = QLabel(content)
        self._comparison_note.setObjectName("RecallComparisonNote")
        self._comparison_note.setWordWrap(True)
        self._comparison_note.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self._comparison_note)

        metrics = QGridLayout()
        metrics.setContentsMargins(0, 0, 0, 0)
        metrics.setHorizontalSpacing(SPACE_SM)
        metrics.setVerticalSpacing(SPACE_SM)

        self._memory_metric = MetricCard("Memory Type", content)
        self._encoding_metric = MetricCard("Encoding", content)
        self._input_dim_metric = MetricCard("Input Dimension", content)
        self._weight_dim_metric = MetricCard("Weight Matrix", content)
        self._output_dim_metric = MetricCard("Output Dimension", content)
        self._expected_metric = MetricCard("Expected Target", content)
        self._exact_metric = MetricCard("Exact Match", content)
        self._mismatch_metric = MetricCard("Mismatch Count", content)
        self._ratio_metric = MetricCard("Match Ratio", content)

        self._input_dim_metric.set_description("Query vector length (n).")
        self._weight_dim_metric.set_description(
            "W shape: input rows × output columns."
        )
        self._output_dim_metric.set_description("Recalled vector length (m).")
        self._exact_metric.set_description(
            "Every component equals the comparison reference."
        )
        self._mismatch_metric.set_description(
            "Unequal components in the comparison."
        )
        self._ratio_metric.set_description(
            "Single-pattern agreement, not dataset-level accuracy."
        )

        metric_cards = (
            self._memory_metric,
            self._encoding_metric,
            self._input_dim_metric,
            self._weight_dim_metric,
            self._output_dim_metric,
            self._expected_metric,
            self._exact_metric,
            self._mismatch_metric,
            self._ratio_metric,
        )
        for index, card in enumerate(metric_cards):
            metrics.addWidget(card, index // 3, index % 3)

        layout.addLayout(metrics)

        layout.addWidget(self._section_title("Image Comparison", content))

        previews = QGridLayout()
        previews.setContentsMargins(0, 0, 0, 0)
        previews.setSpacing(SPACE_MD)

        self._input_image_canvas = ImageCanvas(content)
        self._input_image_canvas.setAccessibleName("Original input image")
        input_image_cell, self._input_image_title = self._preview_cell(
            "Original input image", self._input_image_canvas, content
        )

        self._input_pattern_grid = PixelGrid(content)
        self._input_pattern_grid.setAccessibleName("Processed input pattern")
        input_pattern_cell, _ = self._preview_cell(
            "Processed input pattern", self._input_pattern_grid, content
        )

        self._expected_pattern_grid = PixelGrid(content)
        self._expected_pattern_grid.setAccessibleName(
            "Expected reference pattern"
        )
        expected_pattern_cell, _ = self._preview_cell(
            "Expected reference pattern", self._expected_pattern_grid, content
        )

        self._output_pattern_grid = PixelGrid(content)
        self._output_pattern_grid.setAccessibleName("Recalled output pattern")
        output_pattern_cell, _ = self._preview_cell(
            "Recalled output pattern", self._output_pattern_grid, content
        )

        previews.addWidget(input_image_cell, 0, 0)
        previews.addWidget(input_pattern_cell, 0, 1)
        previews.addWidget(expected_pattern_cell, 1, 0)
        previews.addWidget(output_pattern_cell, 1, 1)
        for column in range(2):
            previews.setColumnStretch(column, 1)
            previews.setRowStretch(column, 1)
        layout.addLayout(previews)

        layout.addWidget(self._section_title("Difference View", content))
        self._difference_view = PatternDifferenceView(content)
        self._difference_view.setMinimumHeight(260)
        self._difference_view.setAccessibleName("Pattern difference view")
        layout.addWidget(self._difference_view)

        layout.addWidget(self._section_title("Pattern Vectors", content))

        vectors = QGridLayout()
        vectors.setContentsMargins(0, 0, 0, 0)
        vectors.setSpacing(SPACE_MD)

        self._input_vector_view = VectorView(content)
        self._input_vector_view.setAccessibleName("Input pattern vector")
        self._net_vector_view = VectorView(content)
        self._net_vector_view.setAccessibleName("Net input vector")
        self._output_vector_view = VectorView(content)
        self._output_vector_view.setAccessibleName("Output pattern vector")
        self._expected_vector_view = VectorView(content)
        self._expected_vector_view.setAccessibleName(
            "Expected reference vector"
        )

        vectors.addWidget(
            self._vector_panel("Input pattern x", self._input_vector_view, content),
            0,
            0,
        )
        vectors.addWidget(
            self._vector_panel(
                "Net input y_in (before activation)", self._net_vector_view, content
            ),
            0,
            1,
        )
        vectors.addWidget(
            self._vector_panel(
                "Output y (after activation)", self._output_vector_view, content
            ),
            1,
            0,
        )
        vectors.addWidget(
            self._vector_panel(
                "Expected reference pattern", self._expected_vector_view, content
            ),
            1,
            1,
        )
        for column in range(2):
            vectors.setColumnStretch(column, 1)
        layout.addLayout(vectors)

        layout.addWidget(self._section_title("Calculation Trace", content))
        trace_hint = QLabel(
            "Step through the exact values that produced the recalled output. "
            "Summaries are shown by default; use “Show exact values” to inspect "
            "complete vectors without flooding the page.",
            content,
        )
        trace_hint.setObjectName("Caption")
        trace_hint.setWordWrap(True)
        layout.addWidget(trace_hint)

        self._trace_navigator = StepNavigator(content)
        self._trace_navigator.setMinimumHeight(_TRACE_MIN_HEIGHT)
        layout.addWidget(self._trace_navigator)

        layout.addWidget(self._section_title("Weight Matrix", content))
        self._weight_matrix_view = WeightMatrixView(content)
        self._weight_matrix_view.setMinimumHeight(_WEIGHT_MIN_HEIGHT)
        self._weight_matrix_view.setAccessibleName("Recall weight matrix")
        layout.addWidget(self._weight_matrix_view)

        layout.addWidget(
            self._section_title("Warnings and Interpretation", content)
        )

        self._warning_panel = QFrame(content)
        self._warning_panel.setObjectName("RecallWarningsPanel")
        self._warning_panel.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground, True
        )
        warning_layout = QVBoxLayout(self._warning_panel)
        warning_layout.setContentsMargins(SPACE_SM, SPACE_XS, SPACE_SM, SPACE_XS)
        warning_layout.setSpacing(SPACE_XXS)

        self._warning_title = QLabel(self._warning_panel)
        self._warning_title.setObjectName("RecallWarningsTitle")
        self._warning_title.setWordWrap(True)

        self._warning_body = QLabel(self._warning_panel)
        self._warning_body.setObjectName("RecallWarningsText")
        self._warning_body.setWordWrap(True)
        self._warning_body.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        warning_layout.addWidget(self._warning_title)
        warning_layout.addWidget(self._warning_body)
        layout.addWidget(self._warning_panel)

        self._set_warning_panel(
            "info",
            "No recall has been run",
            "Run recall to see comparison results, interference warnings, "
            "and interpretation notes.",
        )

        layout.addStretch(1)

        scroll.setWidget(content)
        return scroll

    @staticmethod
    def _make_combo(parent: QWidget, accessible_name: str) -> QComboBox:
        combo = QComboBox(parent)
        combo.setAccessibleName(accessible_name)
        combo.setMinimumContentsLength(28)
        combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        return combo

    @staticmethod
    def _section_title(text: str, parent: QWidget) -> QLabel:
        label = QLabel(text, parent)
        label.setObjectName("SectionTitle")
        label.setWordWrap(True)
        return label

    @staticmethod
    def _preview_cell(
        title: str, widget: QWidget, parent: QWidget
    ) -> tuple[QWidget, QLabel]:
        cell = QWidget(parent)
        layout = QVBoxLayout(cell)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACE_XXS)

        label = QLabel(title, cell)
        label.setObjectName("SubsectionLabel")
        label.setWordWrap(True)

        widget.setMinimumSize(_PREVIEW_SIZE, _PREVIEW_SIZE)
        layout.addWidget(label)
        layout.addWidget(widget, stretch=1)
        return cell, label

    @staticmethod
    def _vector_panel(
        title: str, view: VectorView, parent: QWidget
    ) -> QFrame:
        panel = QFrame(parent)
        panel.setObjectName("VectorPanel")

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(SPACE_XS, SPACE_XS, SPACE_XS, SPACE_XS)
        layout.setSpacing(SPACE_XXS)

        label = QLabel(title, panel)
        label.setObjectName("SubsectionLabel")
        label.setWordWrap(True)

        layout.addWidget(label)
        layout.addWidget(view)
        return panel

    # ------------------------------------------------------------------ wiring

    def _wire_signals(self) -> None:
        vm = self._view_model

        vm.models_changed.connect(self._refresh_models)
        vm.inputs_changed.connect(self._refresh_inputs)
        vm.selection_changed.connect(self._refresh_selection)
        vm.state_changed.connect(self._refresh_controls)
        vm.result_ready.connect(self._render_result)
        vm.trace_ready.connect(self._render_trace)
        vm.status_message.connect(self._on_status_message)

        self._model_combo.currentIndexChanged.connect(self._on_model_changed)
        self._input_combo.currentIndexChanged.connect(self._on_input_changed)
        self._target_combo.currentIndexChanged.connect(self._on_target_changed)

    # ------------------------------------------------------------------ refresh

    def _refresh_all(self) -> None:
        self._refresh_models()

        result = self._view_model.current_result
        if result is not None:
            # The view model can already hold a result when this view is
            # constructed later with a shared view model.
            self._render_result(result)
            self._on_status_message(
                "Showing the current recall result.", StatusKind.INFO
            )
            return

        if not self._view_model.models:
            self._on_status_message(
                "No trained model available. Train a model on the Training page.",
                StatusKind.WARNING,
            )
        elif not self._view_model.inputs:
            self._on_status_message(
                "No compatible input pattern is available for this model.",
                StatusKind.WARNING,
            )
        else:
            self._on_status_message("Ready to run recall.", StatusKind.INFO)

    def _refresh_models(self) -> None:
        vm = self._view_model

        self._model_combo.blockSignals(True)
        self._model_combo.clear()
        if not vm.models:
            self._model_combo.addItem("No trained model available", "")
        else:
            for model in vm.models:
                self._model_combo.addItem(model.label, model.model_id)
                self._model_combo.setItemData(
                    self._model_combo.count() - 1,
                    model.label,
                    Qt.ItemDataRole.ToolTipRole,
                )
        self._model_combo.blockSignals(False)

        self._refresh_inputs()

    def _refresh_inputs(self) -> None:
        vm = self._view_model

        self._input_combo.blockSignals(True)
        self._input_combo.clear()
        if not vm.inputs:
            self._input_combo.addItem("No compatible input pattern", "")
        else:
            for option in vm.inputs:
                self._input_combo.addItem(option.label, option.input_id)
                tooltip = (
                    str(option.source_path)
                    if option.source_path is not None
                    else f"{option.vector_length} values · "
                    f"{option.encoding.value} encoding"
                )
                self._input_combo.setItemData(
                    self._input_combo.count() - 1,
                    tooltip,
                    Qt.ItemDataRole.ToolTipRole,
                )
        self._input_combo.blockSignals(False)

        self._refresh_targets()

    def _refresh_targets(self) -> None:
        vm = self._view_model

        self._target_combo.blockSignals(True)
        self._target_combo.clear()
        self._target_combo.addItem(
            "No expected target (comparison unavailable)", ""
        )
        for target in vm.targets:
            self._target_combo.addItem(target.label, target.target_id)
        self._target_combo.blockSignals(False)

        self._refresh_selection()

    def _refresh_selection(self) -> None:
        vm = self._view_model
        model = vm.selected_model
        option = vm.selected_input
        target = vm.selected_target

        self._set_combo_current(
            self._model_combo, None if model is None else model.model_id
        )
        self._set_combo_current(
            self._input_combo, None if option is None else option.input_id
        )
        self._set_target_current(None if target is None else target.target_id)

        self._update_preprocessing_summary()
        self._update_target_visibility()
        self._update_comparison_note()
        self._refresh_controls()

    def _refresh_controls(self, _state: object | None = None) -> None:
        vm = self._view_model
        has_model = vm.has_model

        self._model_combo.setEnabled(bool(vm.models))
        self._input_combo.setEnabled(has_model and bool(vm.inputs))
        self._target_combo.setEnabled(
            has_model and vm.is_heteroassociative and bool(vm.targets)
        )

        can_import = vm.can_import_image
        self._import_button.setEnabled(can_import)
        if can_import:
            self._import_button.setToolTip(
                "Preprocess a PNG, JPEG, or BMP image with model-compatible "
                "dimensions and encoding."
            )
        elif has_model:
            self._import_button.setToolTip(
                "No preprocessing configuration compatible with this model is "
                "available. Add a compatible image entry to the dataset first."
            )
        else:
            self._import_button.setToolTip(
                "Select a trained model before importing a recall image."
            )

        self._run_button.setEnabled(vm.can_recall)
        self._refresh_save_button()

        self._update_empty_state()
        if vm.current_result is not None:
            self._state_stack.setCurrentWidget(self._results_scroll)
        else:
            self._state_stack.setCurrentWidget(self._placeholder_page)

    def _update_empty_state(self) -> None:
        vm = self._view_model
        if vm.current_result is not None:
            return

        self._empty_state.set_variant("info")

        if vm.state is RecallState.RUNNING:
            self._empty_state.set_title("Running recall")
            self._empty_state.set_description(
                "Computing y_in = x @ W and applying the activation rule. "
                "The results workspace will appear when the calculation "
                "finishes."
            )
            return

        if vm.state is RecallState.FAILED:
            self._empty_state.set_variant("error")
            self._empty_state.set_title("Recall could not be completed")
            self._empty_state.set_description(
                vm.error_message
                or "The recall request failed. Review the selected model and "
                "input, then try again."
            )
            return

        if not vm.models:
            self._empty_state.set_title("No trained model available")
            self._empty_state.set_description(
                "Recall uses models trained during this session. Open the "
                "Training page, add compatible dataset entries, and train a "
                "memory first."
            )
            return

        model = vm.selected_model
        if model is None:
            self._empty_state.set_title("Select a trained model")
            self._empty_state.set_description(
                "Choose one of the session's trained models to start a recall."
            )
            return

        if not vm.inputs:
            self._empty_state.set_title("No compatible input pattern")
            self._empty_state.set_description(
                "The selected model expects "
                f"{model.input_dimension} values with "
                f"{model.encoding.value} encoding. Add a matching dataset "
                "entry, use a compatible working preview, or import an image "
                "with the matching preprocessing configuration."
            )
            return

        option = vm.selected_input
        if option is None:
            self._empty_state.set_title("Select an input pattern")
            self._empty_state.set_description(
                "Choose a dataset entry, working preview, or imported image "
                "to use as the query cue."
            )
            return

        target_note = ""
        if vm.is_heteroassociative and vm.selected_target is None:
            target_note = (
                " No expected target is selected, so comparison metrics will "
                "be unavailable."
            )

        self._empty_state.set_title("Ready to run recall")
        self._empty_state.set_description(
            f"Cue: {option.label} ({option.vector_length} values, "
            f"{option.encoding.value} encoding). Review the preprocessing "
            "summary above, then choose Run Recall to compute y_in = x @ W "
            "and inspect the result." + target_note
        )

    def _update_preprocessing_summary(self) -> None:
        option = self._view_model.selected_input
        if option is None:
            self._preprocessing_label.setText(
                "No compatible input pattern is selected. Add a dataset entry "
                "whose dimensions and encoding match the trained model."
            )
            self._preprocessing_label.setProperty("state", "missing")
        else:
            self._preprocessing_label.setText(
                self._view_model.input_preprocessing_summary or ""
            )
            self._preprocessing_label.setProperty("state", "")

        self._preprocessing_label.style().unpolish(self._preprocessing_label)
        self._preprocessing_label.style().polish(self._preprocessing_label)

    def _update_target_visibility(self) -> None:
        visible = self._view_model.is_heteroassociative
        self._target_label.setVisible(visible)
        self._target_combo.setVisible(visible)

    def _update_comparison_note(self) -> None:
        vm = self._view_model
        result = vm.current_result

        if result is not None:
            self._comparison_note.setText(self._result_comparison_note(result))
            return

        model = vm.selected_model
        if model is None:
            self._comparison_note.setText(
                "Recall requires a trained model and a compatible input "
                "pattern."
            )
            return

        if vm.is_heteroassociative:
            target = vm.selected_target
            if target is None:
                self._comparison_note.setText(
                    "Heteroassociative recall compares against an expected "
                    "target. No target is selected, so exact-match status, "
                    "mismatch count, and match ratio will be unavailable."
                )
            else:
                self._comparison_note.setText(
                    f"Heteroassociative comparison target: {target.label}"
                )
            return

        self._comparison_note.setText(
            "Autoassociative recall compares the output with the original "
            "query pattern."
        )

    def _result_comparison_note(self, result: RecallResult) -> str:
        comparison = result.comparison
        if comparison is None:
            return (
                "No comparison reference was available for this recall. "
                "Exact-match status, mismatch count, and match ratio are not "
                "calculated; this is not a zero-match result."
            )

        if result.comparison_basis == COMPARISON_BASIS_INPUT:
            reference = "the original query pattern (autoassociative recall)"
        else:
            target = self._view_model.selected_target
            reference = (
                target.label
                if target is not None
                else "the selected expected target"
            )

        return (
            f"Compared with {reference}. "
            f"{comparison.mismatch_count} of {comparison.element_count} "
            "components differ."
        )

    @staticmethod
    def _set_combo_current(combo: QComboBox, value: str | None) -> None:
        combo.blockSignals(True)
        index = combo.findData(value) if value is not None else -1
        if index < 0 and combo.count() == 1:
            index = 0
        combo.setCurrentIndex(index)
        combo.blockSignals(False)

    def _set_target_current(self, value: str | None) -> None:
        self._target_combo.blockSignals(True)
        index = self._target_combo.findData(value or "")
        self._target_combo.setCurrentIndex(index if index >= 0 else 0)
        self._target_combo.blockSignals(False)

    # ------------------------------------------------------------------ actions

    def _on_model_changed(self, index: int) -> None:
        if index < 0:
            return
        model_id = self._model_combo.itemData(index)
        if model_id:
            self._view_model.select_model(str(model_id))

    def _on_input_changed(self, index: int) -> None:
        if index < 0:
            return
        input_id = self._input_combo.itemData(index)
        if input_id:
            self._view_model.select_input(str(input_id))

    def _on_target_changed(self, index: int) -> None:
        if index < 0:
            return
        target_id = self._target_combo.itemData(index)
        self._view_model.select_target(None if not target_id else str(target_id))

    def _choose_import_image(self) -> None:
        dialog = QFileDialog(self)
        dialog.setWindowTitle("Import Recall Image")
        dialog.setNameFilters(
            ["Image files (*.png *.jpg *.jpeg *.bmp)", "All files (*.*)"]
        )
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        if not dialog.exec():
            return

        selected = dialog.selectedFiles()
        if selected:
            self._view_model.import_image(selected[0])

    def _refresh_save_button(self) -> None:
        vm = self._view_model

        can_save = vm.can_save_experiment
        self._save_button.setEnabled(can_save)
        if can_save:
            self._save_button.setToolTip(
                "Store the selected model (and the latest recall result, if "
                "any) in the experiment history for later browsing, "
                "opening, and comparison."
            )
        else:
            self._save_button.setToolTip(
                "Select a trained model before saving an experiment."
            )

    def _choose_save_experiment(self) -> None:
        vm = self._view_model
        if not vm.can_save_experiment:
            return

        name, confirmed = QInputDialog.getText(
            self,
            "Save Experiment",
            "Experiment name:",
            text=vm.experiment_name_hint,
        )
        if not confirmed:
            return
        if not name.strip():
            self._on_status_message(
                "No experiment name entered; nothing was saved.",
                StatusKind.WARNING,
            )
            return
        vm.save_current_experiment(name)

    # ------------------------------------------------------------------ result rendering

    def _render_result(self, result: object) -> None:
        if not isinstance(result, RecallResult):
            return

        vm = self._view_model
        comparison = result.comparison

        self._formula_panel.set_equation("y_in = x @ W")
        self._formula_panel.set_caption(
            f"{result.memory_type.value} · {result.encoding.value} encoding · "
            f"{result.input_dimension}×{result.output_dimension} · "
            f"activation {result.activation_name} · threshold "
            f"{_format_number(result.threshold)}"
        )

        self._memory_metric.set_value(result.memory_type.value.capitalize())
        self._encoding_metric.set_value(result.encoding.value)
        self._input_dim_metric.set_value(result.input_dimension)

        weight = vm.current_weight_matrix
        if weight is not None:
            self._weight_dim_metric.set_value(
                f"{weight.shape[0]} × {weight.shape[1]}"
            )
        else:
            self._weight_dim_metric.set_value(None)

        self._output_dim_metric.set_value(result.output_dimension)
        self._expected_metric.set_value(self._reference_label(result))
        self._expected_metric.set_description(
            self._reference_description(result)
        )

        if comparison is None:
            self._exact_metric.set_value(None)
            self._exact_metric.set_description(
                "Unavailable: no comparison reference."
            )
            self._mismatch_metric.set_value(None)
            self._mismatch_metric.set_description(
                "Unavailable: no comparison reference."
            )
            self._ratio_metric.set_value(None)
            self._ratio_metric.set_description(
                "Unavailable: no comparison reference; this is not a zero "
                "score. Single-pattern ratio, not dataset accuracy."
            )
        else:
            matching = comparison.element_count - comparison.mismatch_count
            self._exact_metric.set_value(
                "Yes" if comparison.exact_match else "No"
            )
            self._exact_metric.set_description(
                f"All {comparison.element_count} components compared."
            )
            self._mismatch_metric.set_value(comparison.mismatch_count)
            self._mismatch_metric.set_description(
                f"of {comparison.element_count} compared components."
            )
            self._ratio_metric.set_value(
                _format_number(comparison.match_ratio)
            )
            self._ratio_metric.set_description(
                f"{matching} of {comparison.element_count} components match "
                f"({_format_number(comparison.match_ratio * 100)}%) · "
                "single-pattern ratio, not dataset accuracy."
            )

        self._render_images(result)
        self._render_vectors(result)
        self._render_trace(vm.trace_steps())
        self._render_weight_matrix()
        self._update_comparison_note()
        self._update_warning_panel(result)

        self._state_stack.setCurrentWidget(self._results_scroll)
        self._refresh_controls()

    def _reference_label(self, result: RecallResult) -> str | None:
        if result.comparison is None:
            return None
        if result.comparison_basis == COMPARISON_BASIS_INPUT:
            return "Original query"
        target = self._view_model.selected_target
        return target.label if target is not None else "Expected target"

    def _reference_description(self, result: RecallResult) -> str:
        if result.comparison is None:
            return "No valid comparison reference was available."

        if result.comparison_basis == COMPARISON_BASIS_INPUT:
            option = self._view_model.selected_input
            source = (
                option.label if option is not None else "the original query"
            )
            return f"Comparison basis: original query pattern ({source})."

        target = self._view_model.selected_target
        if target is None:
            return "Comparison basis: selected expected target."
        return f"Comparison basis: expected target ({target.label})."

    def _render_images(self, result: RecallResult) -> None:
        vm = self._view_model
        option = vm.selected_input
        preprocessing = None if option is None else option.preprocessing

        if (
            preprocessing is not None
            and preprocessing.original_image is not None
        ):
            self._input_image_canvas.set_image(preprocessing.original_image)
            self._input_image_title.setText("Original input image")
            self._input_image_canvas.setToolTip(
                f"{preprocessing.original_width}×"
                f"{preprocessing.original_height} px"
            )
        else:
            self._input_image_canvas.clear()
            self._input_image_title.setText(
                "Original input image (not available)"
            )
            self._input_image_canvas.setToolTip(
                "This recall input is a stored numeric pattern, so no "
                "original image is available."
            )

        self._input_pattern_grid.set_matrix(vm.input_matrix)
        self._expected_pattern_grid.set_matrix(vm.expected_matrix)
        self._output_pattern_grid.set_matrix(vm.output_matrix)

        if result.comparison is None:
            self._difference_view.set_unavailable(
                "No valid comparison reference was supplied, so no pixel "
                "difference can be computed."
            )
            return

        output = vm.output_matrix
        expected = vm.expected_matrix
        if output is None or expected is None:
            self._difference_view.set_unavailable(
                "The recalled output and reference cannot both be reshaped "
                "to the configured image dimensions. Inspect the vectors "
                "instead."
            )
            return

        self._difference_view.set_matrices(
            output, expected, result.encoding
        )

    def _render_vectors(self, result: RecallResult) -> None:
        vm = self._view_model
        input_width = self._row_width(vm.input_config)
        output_width = self._row_width(vm.expected_config)
        net_width = min(input_width, 8)

        self._input_vector_view.set_vector(
            result.input_pattern,
            row_width=input_width,
            max_values=_VECTOR_SUMMARY_LIMIT,
        )
        self._net_vector_view.set_vector(
            result.net_input,
            row_width=net_width,
            max_values=_VECTOR_SUMMARY_LIMIT,
        )
        self._output_vector_view.set_vector(
            result.output_pattern,
            row_width=output_width,
            max_values=_VECTOR_SUMMARY_LIMIT,
        )
        self._expected_vector_view.set_vector(
            result.expected_pattern,
            row_width=output_width,
            max_values=_VECTOR_SUMMARY_LIMIT,
        )

    def _render_trace(self, steps: object) -> None:
        if not isinstance(steps, (list, tuple)):
            return
        trace_steps = [
            step for step in steps if isinstance(step, TraceStep)
        ]
        self._trace_navigator.set_steps(trace_steps)

    def _render_weight_matrix(self) -> None:
        weight = self._view_model.current_weight_matrix
        self._weight_matrix_view.set_matrix(weight)

    def _update_warning_panel(self, result: RecallResult) -> None:
        comparison = result.comparison

        if comparison is None:
            body = (
                "\n\n".join(result.warnings)
                if result.warnings
                else "No comparison reference was supplied, so recall "
                "success could not be evaluated. This is not a zero-match "
                "result."
            )
            self._set_warning_panel(
                "info",
                "Comparison unavailable",
                body
                + "\n\nNo exact-match status or match ratio is reported, and "
                "no successful classification is claimed.",
            )
            return

        if comparison.mismatch_count:
            body = (
                "\n\n".join(result.warnings)
                if result.warnings
                else "The output differs from the comparison reference in "
                f"{comparison.mismatch_count} of "
                f"{comparison.element_count} components."
            )
            self._set_warning_panel(
                "warning",
                "Mismatch warning",
                body
                + "\n\nInterference between stored patterns is a possible "
                "explanation, not a confirmed cause. Inspect the patterns, "
                "encoding, and training set before treating the result as "
                "correct.",
            )
            return

        self._set_warning_panel(
            "success",
            "Exact match for this recall",
            "The recalled vector equals the comparison reference across all "
            f"{comparison.element_count} components. This is a single-pattern "
            "match, not dataset-level classification accuracy, and it does "
            "not guarantee that other cues will be recalled correctly.",
        )

    def _set_warning_panel(self, state: str, title: str, body: str) -> None:
        self._warning_panel.setProperty("state", state)
        self._warning_title.setProperty("state", state)
        for widget in (self._warning_panel, self._warning_title):
            widget.style().unpolish(widget)
            widget.style().polish(widget)

        self._warning_title.setText(title)
        self._warning_body.setText(body)
        self._warning_panel.setAccessibleName(title)
        self._warning_panel.setAccessibleDescription(body)

    @staticmethod
    def _row_width(config: PreprocessingConfig | None) -> int:
        if config is not None and config.width > 0:
            return config.width
        return 16

    # ------------------------------------------------------------------ status

    def _on_status_message(self, message: str, kind: StatusKind) -> None:
        self._status_badge.set_status(kind, message)


def _format_number(value: object) -> str:
    """Render a scalar without unnecessary trailing zeros.

    ``repr`` supplies the shortest string that round-trips to the same
    binary float, so displayed values are not silently rounded.
    """

    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return repr(numeric)
