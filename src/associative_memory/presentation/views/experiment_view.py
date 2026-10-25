"""Experiment history workspace: listing, opening, deletion, comparison.

The view renders data produced by
:class:`associative_memory.presentation.viewmodels.experiment_history_viewmodel.ExperimentHistoryViewModel`.
It never reads storage files or implements persistence: saving happens in
the recall workspace, opening and deletion happen through the history
view model's application service, and the comparison view displays rows
and notes that the service already annotated for compatibility.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from associative_memory.application.dto.experiment_comparison import (
    ExperimentComparison,
)
from associative_memory.application.dto.saved_experiment import (
    LoadedExperiment,
    PreprocessingSettings,
    SavedExperimentSummary,
)
from associative_memory.presentation.theme.spacing import (
    SPACE_MD,
    SPACE_SM,
    SPACE_XS,
    SPACE_XXS,
)
from associative_memory.presentation.viewmodels.experiment_history_viewmodel import (
    ExperimentHistoryViewModel,
)
from associative_memory.presentation.widgets import (
    EmptyState,
    StatusBadge,
    StatusKind,
    WeightMatrixView,
)

_PAGE_EMPTY = 0
_PAGE_DETAILS = 1
_PAGE_COMPARISON = 2


class ExperimentView(QWidget):
    """The Experiment History page."""

    #: Emitted after the explicit "Open in Recall" action loads an
    #: experiment. Double-click inspection deliberately does not emit it.
    open_in_recall_requested = Signal()

    def __init__(
        self,
        view_model: ExperimentHistoryViewModel | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        if view_model is not None:
            self._view_model = view_model
        else:
            self._view_model = ExperimentHistoryViewModel(parent=self)

        self._build_layout()
        self._wire_signals()
        self._refresh_entries()
        self._view_model.refresh()
        # Track the last-rendered comparison so we can skip a full grid
        # rebuild when the comparison data has not actually changed.
        self._last_shown_comparison: object | None = None

    # ------------------------------------------------------------------ layout

    def _build_layout(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(SPACE_SM, SPACE_SM, SPACE_SM, SPACE_SM)
        root.setSpacing(SPACE_XS)

        root.addWidget(self._build_control_panel())

        self._workspace_stack = QStackedWidget(self)
        self._workspace_stack.setObjectName("HistoryWorkspaceStack")

        self._empty_page = EmptyState(
            "No saved experiments yet",
            "Train a model and run recall, then use “Save Experiment…” on "
            "the Recall page. Saved experiments appear here for opening, "
            "comparison, and deletion.",
            parent=self,
        )

        self._details_scroll = self._build_details_page()
        self._comparison_scroll = self._build_comparison_page()

        self._workspace_stack.addWidget(self._empty_page)
        self._workspace_stack.addWidget(self._details_scroll)
        self._workspace_stack.addWidget(self._comparison_scroll)

        root.addWidget(self._workspace_stack, stretch=1)
        self._show_workspace_page()

    def _build_control_panel(self) -> QFrame:
        panel = QFrame(self)
        panel.setObjectName("ConfigPanel")

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(SPACE_SM, SPACE_SM, SPACE_SM, SPACE_SM)
        layout.setSpacing(SPACE_XS)

        title = QLabel("Saved Experiments", panel)
        title_font = title.font()
        title_font.setWeight(QFont.Weight.Bold)
        title.setFont(title_font)

        self._storage_label = QLabel(panel)
        self._storage_label.setObjectName("HistoryStorageLabel")
        self._storage_label.setWordWrap(True)
        self._storage_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        location_text = self._view_model.storage_location_text
        if location_text:
            self._storage_label.setText(
                "Storage folder (JSON metadata plus NumPy arrays; original "
                "images are not copied):"
                f" {location_text}"
            )

        self._history_list = QListWidget(panel)
        self._history_list.setObjectName("HistoryList")
        self._history_list.setToolTip(
            "Double-click opens an experiment for inspection. Check two "
            "experiments to compare them side by side."
        )
        self._history_list.setMinimumHeight(180)
        self._history_list.setSelectionMode(
            QListWidget.SelectionMode.SingleSelection
        )
        self._history_list.itemSelectionChanged.connect(
            self._on_selection_changed
        )
        self._history_list.itemChanged.connect(self._on_item_changed)
        self._history_list.itemDoubleClicked.connect(
            self._on_item_double_clicked
        )

        actions = QHBoxLayout()
        actions.setSpacing(SPACE_XS)

        self._open_button = QPushButton("Open in Recall", panel)
        self._open_button.setProperty("variant", "primary")
        self._open_button.setToolTip(
            "Load the selected experiment and open it in the Recall "
            "workspace."
        )
        self._open_button.clicked.connect(self._on_open_clicked)
        actions.addWidget(self._open_button)

        self._compare_button = QPushButton("Compare (2 selected)", panel)
        self._compare_button.setToolTip(
            "Compare the two checked experiments, with notes explaining "
            "anything that is not directly comparable."
        )
        self._compare_button.clicked.connect(self._on_compare_clicked)
        actions.addWidget(self._compare_button)

        actions.addStretch(1)

        self._delete_button = QPushButton("Delete…", panel)
        self._delete_button.setToolTip(
            "Remove the selected experiment from storage after "
            "confirmation. This cannot be undone."
        )
        self._delete_button.clicked.connect(self._on_delete_clicked)
        actions.addWidget(self._delete_button)

        self._status_badge = StatusBadge(StatusKind.INFO, "History loaded", panel)
        self._status_badge.text_label.setWordWrap(True)
        self._status_badge.setMaximumWidth(620)
        actions.addWidget(self._status_badge)

        layout.addWidget(title)
        layout.addWidget(self._storage_label)
        layout.addWidget(self._history_list)
        layout.addLayout(actions)
        return panel

    def _build_details_page(self) -> QScrollArea:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        card = QFrame(scroll)
        card.setObjectName("HistoryDetailsCard")

        layout = QVBoxLayout(card)
        layout.setContentsMargins(SPACE_MD, SPACE_MD, SPACE_MD, SPACE_MD)
        layout.setSpacing(SPACE_XS)

        title = QLabel("Saved Experiment Details", card)
        title.setObjectName("HistoryDetailTitle")
        title_font = title.font()
        title_font.setWeight(QFont.Weight.Bold)
        title.setFont(title_font)
        layout.addWidget(title)

        self._details_summary = QLabel(card)
        self._details_summary.setObjectName("HistoryDetails")
        self._details_summary.setWordWrap(True)
        self._details_summary.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self._details_summary)

        layout.addWidget(self._section_title("Stored Weight Matrix", card))
        self._weight_matrix_view = WeightMatrixView(card)
        self._weight_matrix_view.set_matrix(None)
        layout.addWidget(self._weight_matrix_view)

        layout.addWidget(self._section_title("Stored Recall Record", card))
        self._details_recall = QLabel(card)
        self._details_recall.setObjectName("HistoryDetails")
        self._details_recall.setWordWrap(True)
        self._details_recall.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._details_recall.setText("No experiment loaded.")
        layout.addWidget(self._details_recall)

        layout.addStretch(1)
        scroll.setWidget(card)
        return scroll

    def _build_comparison_page(self) -> QScrollArea:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        card = QFrame(scroll)
        card.setObjectName("CompareCard")

        layout = QVBoxLayout(card)
        layout.setContentsMargins(SPACE_MD, SPACE_MD, SPACE_MD, SPACE_MD)
        layout.setSpacing(SPACE_XS)

        title = QLabel("Experiment Comparison", card)
        title.setObjectName("HistoryDetailTitle")
        title_font = title.font()
        title_font.setWeight(QFont.Weight.Bold)
        title.setFont(title_font)
        layout.addWidget(title)

        self._compare_subtitle = QLabel(card)
        self._compare_subtitle.setObjectName("CompareSubtitle")
        self._compare_subtitle.setWordWrap(True)
        layout.addWidget(self._compare_subtitle)

        self._compare_grid_host = QWidget(card)
        self._compare_grid = QGridLayout(self._compare_grid_host)
        self._compare_grid.setContentsMargins(0, 0, 0, 0)
        self._compare_grid.setHorizontalSpacing(SPACE_XS)
        self._compare_grid.setVerticalSpacing(SPACE_XXS)
        layout.addWidget(self._compare_grid_host)

        self._compare_notes = QLabel(card)
        self._compare_notes.setObjectName("CompareNotes")
        self._compare_notes.setWordWrap(True)
        self._compare_notes.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self._compare_notes)

        actions = QHBoxLayout()
        actions.setSpacing(SPACE_XS)
        back_button = QPushButton("Back to Details", card)
        back_button.clicked.connect(self._on_back_to_details)
        actions.addWidget(back_button)
        actions.addStretch(1)
        layout.addLayout(actions)

        scroll.setWidget(card)
        return scroll

    def _section_title(self, text: str, parent: QWidget) -> QLabel:
        label = QLabel(text, parent)
        label.setObjectName("HistorySectionTitle")
        return label

    # ------------------------------------------------------------------ wiring

    def _wire_signals(self) -> None:
        vm = self._view_model

        vm.history_changed.connect(self._refresh_entries)
        vm.loaded_experiment_changed.connect(self._on_loaded_changed)
        vm.comparison_changed.connect(self._on_comparison_changed)
        vm.status_message.connect(self._on_status_message)

    # ------------------------------------------------------------------ history list

    def _selected_summary(self) -> SavedExperimentSummary | None:
        items = self._history_list.selectedItems()
        if not items:
            return None
        experiment_id = items[0].data(Qt.ItemDataRole.UserRole)
        for summary in self._view_model.summaries:
            if summary.experiment_id == experiment_id:
                return summary
        return None

    def _checked_experiment_ids(self) -> list[str]:
        """Experiment ids of checked list items, in display order."""

        ids: list[str] = []
        for index in range(self._history_list.count()):
            item = self._history_list.item(index)
            if item is None or item.checkState() != Qt.CheckState.Checked:
                continue
            experiment_id = item.data(Qt.ItemDataRole.UserRole)
            if experiment_id:
                ids.append(str(experiment_id))
        return ids

    def _refresh_entries(self) -> None:
        summaries = self._view_model.summaries

        previously_checked = set(self._checked_experiment_ids())
        selected_id = None
        selected_items = self._history_list.selectedItems()
        if selected_items:
            selected_id = selected_items[0].data(Qt.ItemDataRole.UserRole)

        self._history_list.blockSignals(True)
        self._history_list.clear()
        for summary in summaries:
            item = QListWidgetItem(summary.display_name, self._history_list)
            item.setData(Qt.ItemDataRole.UserRole, summary.experiment_id)
            item.setFlags(
                item.flags() | Qt.ItemFlag.ItemIsUserCheckable
            )
            item.setCheckState(
                Qt.CheckState.Checked
                if summary.experiment_id in previously_checked
                else Qt.CheckState.Unchecked
            )
            item.setToolTip(self._entry_tooltip(summary))
            item.setData(
                Qt.ItemDataRole.AccessibleTextRole,
                f"Saved experiment {summary.display_name}",
            )
            item.setData(
                Qt.ItemDataRole.AccessibleDescriptionRole,
                self._entry_tooltip(summary),
            )
            if summary.experiment_id == selected_id:
                item.setSelected(True)
        self._history_list.blockSignals(False)

        self._update_action_buttons()
        self._show_workspace_page()

    def _entry_tooltip(self, summary: SavedExperimentSummary) -> str:
        """Accessible tooltip text for one history entry."""

        if not summary.readable:
            reason = summary.unreadable_reason or "unknown error"
            return f"Not readable: {reason}. The entry can still be deleted."

        created = summary.created_at or "unknown time"
        details = (
            f"{summary.memory_type} · {summary.encoding} · "
            f"{summary.input_dimension}×{summary.output_dimension} · "
            f"created {created}"
        )
        recall_text = (
            (f" · stored recall: {summary.recall_mismatch_count} mismatch(es)")
            if summary.recall_present
            else " · no stored recall record"
        )
        return f"{details}{recall_text}"

    def _update_action_buttons(self) -> None:
        vm = self._view_model
        summary = self._selected_summary()
        has_selection = summary is not None

        self._open_button.setEnabled(bool(has_selection and summary.readable))
        self._delete_button.setEnabled(has_selection)
        self._compare_button.setEnabled(
            len(self._checked_experiment_ids()) == 2
        )

        if not vm.has_entries:
            self._open_button.setToolTip("No saved experiments yet.")
        elif has_selection and summary is not None and not summary.readable:
            self._open_button.setToolTip(
                "This experiment is not readable and cannot be opened."
            )
        else:
            self._open_button.setToolTip(
                "Load the selected experiment; its stored model becomes "
                "available in the recall workspace."
            )

        if has_selection and summary is not None:
            reason = "" if summary.readable else (
                f" This entry is not readable ({summary.unreadable_reason}); "
                "it can still be deleted."
            )
            self._delete_button.setToolTip(
                f"Permanently delete {summary.display_name!r}.{reason}"
            )
        else:
            self._delete_button.setToolTip(
                "Select an experiment before deleting it."
            )

        ok_text = (
            "Compare the two checked experiments, with notes explaining "
            "anything that is not directly comparable."
        )
        self._compare_button.setToolTip(
            ok_text
            if self._compare_button.isEnabled()
            else "Check exactly two experiments to enable comparison."
        )

    # ------------------------------------------------------------------ slots

    def _on_selection_changed(self) -> None:
        """Refresh the details page for the current selection.

        Only the loaded experiment can show arrays; any other selection
        displays a prompt instead of stale numbers from a different
        experiment.
        """

        self._update_action_buttons()
        loaded = self._view_model.loaded_experiment
        summary = self._selected_summary()
        if loaded is not None and summary is not None and (
            loaded.summary.experiment_id == summary.experiment_id
        ):
            self._details_summary.setText(self._details_text(loaded))
            self._details_recall.setText(self._recall_record_text(loaded))
            self._weight_matrix_view.set_matrix(loaded.training.weight_matrix)
        elif loaded is not None:
            self._details_summary.setText(
                "Select an experiment, then use “Open in Recall” or "
                "double-click to load it fully."
            )
            self._details_recall.setText("No experiment loaded.")
            self._weight_matrix_view.set_matrix(None)
        else:
            self._details_summary.setText(
                "Select an experiment in the list, then use “Open in Recall” "
                "or double-click to load it fully."
            )
            self._details_recall.setText("No experiment loaded.")
            self._weight_matrix_view.set_matrix(None)
        self._show_workspace_page()

    def _on_item_changed(self, _item: object) -> None:
        self._update_action_buttons()

    def _on_item_double_clicked(
        self, item: QListWidgetItem
    ) -> None:
        experiment_id = item.data(Qt.ItemDataRole.UserRole)
        if experiment_id:
            self._open_experiment(str(experiment_id))

    def _on_open_clicked(self) -> None:
        summary = self._selected_summary()
        if summary is None:
            return
        if self._open_experiment(summary.experiment_id) is not None:
            self.open_in_recall_requested.emit()

    def _open_experiment(self, experiment_id: str) -> LoadedExperiment | None:
        loaded = self._view_model.load_experiment(experiment_id)
        if loaded is not None:
            self._show_workspace_page()
        return loaded

    def _on_delete_clicked(self) -> None:
        summary = self._selected_summary()
        if summary is None:
            return

        answer = QMessageBox.question(
            self,
            "Delete Experiment",
            (
                f"Delete {summary.display_name!r} permanently?\n\n"
                "Its stored metadata and numerical arrays are removed from "
                "the storage folder. This cannot be undone."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            self._on_status_message(
                "Deletion cancelled; nothing was removed.",
                StatusKind.INFO,
            )
            return
        self._view_model.delete_experiment(summary.experiment_id)

    def _on_compare_clicked(self) -> None:
        checked = self._checked_experiment_ids()
        if len(checked) != 2:
            return
        left, right = checked
        if self._view_model.compare(left, right) is not None:
            self._show_workspace_page()

    def _on_back_to_details(self) -> None:
        self._view_model.show_details()

    # ------------------------------------------------------------------ rendering

    def _on_status_message(self, message: str, kind: object) -> None:
        if not isinstance(kind, StatusKind):
            kind = StatusKind.INFO
        self._status_badge.set_status(kind, message)

    def _on_loaded_changed(self, loaded: object) -> None:
        if isinstance(loaded, LoadedExperiment):
            self._refresh_details_text()
            self._weight_matrix_view.set_matrix(loaded.training.weight_matrix)
            self._details_recall.setText(self._recall_record_text(loaded))
        else:
            self._weight_matrix_view.set_matrix(None)
            self._details_recall.setText("No experiment loaded.")
        self._show_workspace_page()

    def _on_comparison_changed(self, comparison: object) -> None:
        if isinstance(comparison, ExperimentComparison):
            self._render_comparison(comparison)
        self._show_workspace_page()

    def _show_workspace_page(self) -> None:
        """Switch the workspace page: comparison, details, else empty."""

        if self._view_model.comparison is not None:
            self._workspace_stack.setCurrentIndex(_PAGE_COMPARISON)
            return

        if self._view_model.has_entries:
            self._workspace_stack.setCurrentIndex(_PAGE_DETAILS)
            return

        self._workspace_stack.setCurrentIndex(_PAGE_EMPTY)

    def _refresh_details_text(self) -> None:
        """Compose the details text from the loaded experiment or selection."""

        loaded = self._view_model.loaded_experiment
        if loaded is None:
            self._details_summary.setText(
                "Select an experiment in the list, then use “Open in Recall” "
                "or double-click to load it fully."
            )
        else:
            self._details_summary.setText(self._details_text(loaded))

    def _details_text(self, loaded: LoadedExperiment) -> str:
        """Build the metadata text of a fully loaded experiment."""

        summary = loaded.summary
        lines = [
            f"Name: {loaded.name}",
            f"Created: {summary.created_at or 'unknown'}",
            f"Experiment ID: {summary.experiment_id}",
            (
                "Saved with application version "
                f"{summary.application_version or 'unknown'}; manifest "
                f"schema version {summary.schema_version or '?'}."
            ),
            (
                f"Memory type: {summary.memory_type} · Encoding: "
                f"{summary.encoding}"
            ),
            (
                f"Activation: {summary.activation_name} (threshold "
                f"{self._format_threshold(summary.threshold)})"
            ),
            (
                f"Dimensions: n = {summary.input_dimension}, m = "
                f"{summary.output_dimension}; weight matrix "
                f"{self._shape_text(summary.weight_shape)}"
            ),
            (
                f"Training pairs: {summary.pair_count} · Dataset reference: "
                f"{summary.dataset_name or '(not stored)'}"
            ),
            (
                "Stored outer-product contributions: "
                + (
                    str(summary.contribution_count)
                    if summary.contribution_count is not None
                    else "not stored (no training trace saved)"
                )
            ),
        ]
        lines.extend(self._preprocessing_lines(loaded))
        lines.append(
            "Original images are not copied into saved experiments; only "
            "the encoded numbers and the preprocessing settings above are "
            "stored. A missing source file does not invalidate any stored "
            "result."
        )
        return "\n".join(lines)

    def _preprocessing_lines(self, loaded: LoadedExperiment) -> list[str]:
        """Preprocessing summary lines for a loaded experiment."""

        if loaded.preprocessing is None and loaded.target_preprocessing is None:
            return [
                "Preprocessing settings: not stored (patterns were defined "
                "numerically)."
            ]

        lines = []
        if loaded.preprocessing is not None:
            lines.append(
                "Preprocessing: "
                + self._settings_text(loaded.preprocessing)
            )
        if loaded.target_preprocessing is not None:
            lines.append(
                "Target preprocessing: "
                + self._settings_text(loaded.target_preprocessing)
            )
        return lines

    def _settings_text(self, settings: PreprocessingSettings) -> str:
        """One summary line describing stored preprocessing settings."""

        return (
            f"{settings.width}×{settings.height} grid · grayscale "
            f"{settings.grayscale_method} · resampling "
            f"{settings.resampling} · threshold {settings.threshold} · "
            f"polarity {settings.polarity} · encoding "
            f"{settings.encoding.value} · resize {settings.resize_mode} · "
            f"flattening {settings.flattening_order}"
        )

    def _recall_record_text(self, loaded: LoadedExperiment) -> str:
        """Describe the stored recall record of a loaded experiment."""

        record = loaded.recall_record
        if record is None:
            return (
                "No recall result was stored with this experiment. Open it "
                "in the recall workspace and run recall to produce a new "
                "result."
            )

        lines = ["Stored recall record: yes"]
        if record.comparison_basis is not None:
            lines.append(
                "Comparison basis: "
                + (
                    "Expected target"
                    if record.comparison_basis == "expected_pattern"
                    else "Original query pattern"
                )
            )
        if record.comparison is not None:
            comparison = record.comparison
            ratio_text = f"{float(comparison.match_ratio):.6g}"
            lines.append(
                f"Compared {comparison.element_count} elements: "
                f"{comparison.mismatch_count} mismatch(es); match ratio "
                f"{ratio_text}; exact match: "
                f"{'yes' if comparison.exact_match else 'no'}."
            )
        else:
            lines.append(
                "No comparison was stored (recall ran without an expected "
                "target)."
            )
        lines.append(
            "Input source file: "
            + (record.input_source_file or "(not stored)")
        )
        lines.append(
            "Expected-target source file: "
            + (record.target_source_file or "(not stored)")
        )
        if record.warnings:
            lines.append("Stored warnings:")
            lines.extend(f"  •  {warning}" for warning in record.warnings)
        return "\n".join(lines)

    def _render_comparison(self, comparison: ExperimentComparison) -> None:
        """Lay out the comparison rows produced by the service."""

        # Skip full grid rebuild when the comparison object is the same
        # reference that was already rendered (e.g. rapid toggling between
        # two experiments). Only the notes/subtitle may need updating.
        if self._last_shown_comparison is comparison:
            # Still update subtitle and notes since those can change
     # independently of the row data object identity.
            self._compare_subtitle.setText(
                "Left: "
                f"{comparison.left.display_name}   ·   Right: "
                f"{comparison.right.display_name}"
                "   —   incompatible values are annotated instead of compared."
            )
            if comparison.notes:
                self._compare_notes.setText(
                    "Notes:\n"
                    + "\n".join(f"  •  {note}" for note in comparison.notes)
                )
            else:
                self._compare_notes.setText(
                    "No comparability warnings: the shared structure of the two "
                    "experiments makes their aligned values comparable."
                )
            return

        self._last_shown_comparison = comparison

        self._compare_subtitle.setText(
            "Left: "
            f"{comparison.left.display_name}   ·   Right: "
            f"{comparison.right.display_name}"
            "   —   incompatible values are annotated instead of compared."
        )

        # Clear previous rows.
        while self._compare_grid.count():
            item = self._compare_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        header = QFrame(self._compare_grid_host)
        header.setObjectName("CompareHeaderRow")
        header_grid = QGridLayout(header)
        header_grid.setContentsMargins(SPACE_XXS, SPACE_XXS, SPACE_XXS, SPACE_XXS)
        header_grid.setHorizontalSpacing(SPACE_XS)
        header_grid.addWidget(self._header_label("Attribute"), 0, 0)
        header_grid.addWidget(
            self._header_label(comparison.left.display_name), 0, 1
        )
        header_grid.addWidget(
            self._header_label(comparison.right.display_name), 0, 2
        )
        header_grid.setColumnStretch(1, 1)
        header_grid.setColumnStretch(2, 1)
        self._compare_grid.addWidget(header, 0, 0, 1, 3)

        row_index = 1
        for entry in comparison.rows:
            row = QFrame(self._compare_grid_host)
            row.setObjectName("CompareRow")
            row_grid = QGridLayout(row)
            row_grid.setContentsMargins(SPACE_XXS, SPACE_XXS, SPACE_XXS, SPACE_XXS)
            row_grid.setHorizontalSpacing(SPACE_XS)
            row_grid.addWidget(self._attribute_label(entry.label), 0, 0)
            row_grid.addWidget(self._value_label(entry.left), 0, 1)
            row_grid.addWidget(self._value_label(entry.right), 0, 2)
            if entry.note:
                note = QLabel(entry.note, row)
                note.setObjectName("CompareNote")
                note.setWordWrap(True)
                row_grid.addWidget(note, 1, 0, 1, 3)
            row_grid.setColumnStretch(1, 1)
            row_grid.setColumnStretch(2, 1)
            self._compare_grid.addWidget(row, row_index, 0, 1, 3)
            row_index += 1

        self._compare_grid.setColumnStretch(0, 0)
        self._compare_grid.setColumnStretch(1, 1)
        self._compare_grid.setColumnStretch(2, 1)

        if comparison.notes:
            self._compare_notes.setText(
                "Notes:\n"
                + "\n".join(f"  •  {note}" for note in comparison.notes)
            )
        else:
            self._compare_notes.setText(
                "No comparability warnings: the shared structure of the two "
                "experiments makes their aligned values comparable."
            )

    def _header_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("CompareHeaderLabel")
        label.setWordWrap(True)
        return label

    def _attribute_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("CompareLabel")
        label.setWordWrap(True)
        return label

    def _value_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("CompareValue")
        label.setWordWrap(True)
        return label

    # ------------------------------------------------------------------ formatting

    @staticmethod
    def _shape_text(shape: tuple[int, int] | None) -> str:
        if shape is None:
            return "unknown"
        return f"{shape[0]}×{shape[1]}"

    @staticmethod
    def _format_threshold(threshold: float | None) -> str:
        if threshold is None:
            return "?"
        return f"{float(threshold):g}"
