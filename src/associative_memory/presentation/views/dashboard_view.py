"""Dashboard view: live dataset, model, and recent-experiment overview.

The view renders only values exposed by
:class:`associative_memory.presentation.viewmodels.dashboard_viewmodel.\
DashboardViewModel`. It performs no storage or numerical work and does not
create sample content: each section switches between an explicit empty state
and live information from the shared workspace view models.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from associative_memory.presentation.theme.spacing import (
    SPACE_MD,
    SPACE_SM,
    SPACE_XS,
)
from associative_memory.presentation.viewmodels.dashboard_viewmodel import (
    DashboardViewModel,
)
from associative_memory.presentation.widgets import (
    EmptyState,
    MetricCard,
    StatusBadge,
    StatusKind,
)

_PAGE_EMPTY = 0
_PAGE_CONTENT = 1


class DashboardView(QWidget):
    """The Dashboard page."""

    #: Requests to navigate to the page that creates the missing content.
    open_dataset_requested = Signal()
    open_training_requested = Signal()
    open_recall_requested = Signal()

    def __init__(
        self,
        view_model: DashboardViewModel | None = None,
        parent: QWidget | None = None,
    ) -> None:
        # Preserve the placeholder page's former positional-parent usage.
        if isinstance(view_model, QWidget):
            if parent is not None:
                raise TypeError("parent was provided twice.")
            parent = view_model
            view_model = None

        super().__init__(parent)

        self._view_model = (
            view_model
            if view_model is not None
            else DashboardViewModel(parent=self)
        )

        self._build_layout()
        self._view_model.changed.connect(self._refresh)
        self._refresh()
        self._view_model.refresh_history()

    # ------------------------------------------------------------------ layout

    def _build_layout(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget(scroll)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(SPACE_MD, SPACE_MD, SPACE_MD, SPACE_MD)
        layout.setSpacing(SPACE_SM)

        intro = QLabel(
            "Live overview of this session and the saved experiment library. "
            "Values update as you import, train, select models, and save.",
            content,
        )
        intro.setObjectName("PageSubtitle")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        layout.addWidget(self._build_dataset_section(content))
        layout.addWidget(self._build_model_section(content))
        layout.addWidget(self._build_experiment_section(content))
        layout.addStretch(1)

        scroll.setWidget(content)
        root.addWidget(scroll)

    def _section_card(
        self,
        title: str,
        description: str,
        parent: QWidget,
    ) -> tuple[QFrame, QVBoxLayout]:
        """Build one bordered section and return its body layout."""

        card = QFrame(parent)
        card.setObjectName("ConfigPanel")

        layout = QVBoxLayout(card)
        layout.setContentsMargins(SPACE_SM, SPACE_SM, SPACE_SM, SPACE_SM)
        layout.setSpacing(SPACE_XS)

        title_label = QLabel(title, card)
        title_label.setObjectName("SectionTitle")
        title_label.setWordWrap(True)
        layout.addWidget(title_label)

        description_label = QLabel(description, card)
        description_label.setObjectName("Caption")
        description_label.setWordWrap(True)
        layout.addWidget(description_label)

        return card, layout

    def _build_dataset_section(self, parent: QWidget) -> QFrame:
        card, layout = self._section_card(
            "Dataset",
            "Patterns currently available for training.",
            parent,
        )

        self._dataset_stack = QStackedWidget(card)
        self._dataset_stack.setObjectName("DashboardDatasetStack")

        empty = EmptyState(
            "No dataset entries yet",
            "Import an image on the Image Dataset page and add a processed "
            "pattern. Entries appear here as soon as they are added.",
            parent=card,
        )
        empty.set_action("Open Image Dataset", self.open_dataset_requested.emit)

        content = QWidget(card)
        content.setObjectName("DashboardDatasetContent")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(SPACE_XS)

        metrics = QHBoxLayout()
        metrics.setSpacing(SPACE_XS)

        self._dataset_entries_card = MetricCard("Entries", content)
        self._dataset_entries_card.set_description("Added patterns")
        metrics.addWidget(self._dataset_entries_card, 1)

        self._dataset_memory_card = MetricCard("Memory type", content)
        self._dataset_memory_card.set_description("Dataset mode")
        metrics.addWidget(self._dataset_memory_card, 1)

        self._dataset_pattern_card = MetricCard("Pattern size", content)
        self._dataset_pattern_card.set_description("Stored entry size")
        metrics.addWidget(self._dataset_pattern_card, 1)

        self._dataset_targets_card = MetricCard("Targets", content)
        self._dataset_targets_card.set_description("Target labels")
        metrics.addWidget(self._dataset_targets_card, 1)

        content_layout.addLayout(metrics)

        self._dataset_detail = QLabel(content)
        self._dataset_detail.setObjectName("StatusMessage")
        self._dataset_detail.setWordWrap(True)
        self._dataset_detail.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        content_layout.addWidget(self._dataset_detail)

        self._dataset_stack.addWidget(empty)
        self._dataset_stack.addWidget(content)
        layout.addWidget(self._dataset_stack)
        return card

    def _build_model_section(self, parent: QWidget) -> QFrame:
        card, layout = self._section_card(
            "Model status",
            "Current training result and models available to the recall "
            "workspace.",
            parent,
        )

        self._model_status_badge = StatusBadge(
            StatusKind.INFO, "Waiting for dataset", card
        )
        layout.addWidget(
            self._model_status_badge,
            alignment=Qt.AlignmentFlag.AlignLeft,
        )

        self._model_stack = QStackedWidget(card)
        self._model_stack.setObjectName("DashboardModelStack")

        empty = EmptyState(
            "No trained model yet",
            "Train a model on the Training page or open a saved experiment "
            "from Experiment History. Available recall models appear here.",
            parent=card,
        )
        empty.set_action("Open Training", self.open_training_requested.emit)

        content = QWidget(card)
        content.setObjectName("DashboardModelContent")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(SPACE_XS)

        metrics = QHBoxLayout()
        metrics.setSpacing(SPACE_XS)

        self._model_available_card = MetricCard("Available models", content)
        self._model_available_card.set_description("Recall catalog")
        metrics.addWidget(self._model_available_card, 1)

        self._model_training_card = MetricCard("Current training", content)
        self._model_training_card.set_description("Shared training state")
        metrics.addWidget(self._model_training_card, 1)

        self._model_selected_card = MetricCard("Selected model", content)
        self._model_selected_card.set_description("Recall workspace")
        metrics.addWidget(self._model_selected_card, 1)

        content_layout.addLayout(metrics)

        self._model_detail = QLabel(content)
        self._model_detail.setObjectName("StatusMessage")
        self._model_detail.setWordWrap(True)
        self._model_detail.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        content_layout.addWidget(self._model_detail)

        self._model_stack.addWidget(empty)
        self._model_stack.addWidget(content)
        layout.addWidget(self._model_stack)
        return card

    def _build_experiment_section(self, parent: QWidget) -> QFrame:
        card, layout = self._section_card(
            "Recent experiments",
            "Newest saved experiments from the shared history.",
            parent,
        )

        self._experiment_stack = QStackedWidget(card)
        self._experiment_stack.setObjectName("DashboardExperimentStack")

        empty = EmptyState(
            "No saved experiments yet",
            "Save a trained model from the Recall page. Saved experiments "
            "appear here and in Experiment History.",
            parent=card,
        )
        empty.set_action("Open Recall", self.open_recall_requested.emit)

        content = QWidget(card)
        content.setObjectName("DashboardExperimentContent")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(SPACE_XS)

        self._experiment_list = QListWidget(content)
        self._experiment_list.setObjectName("HistoryList")
        self._experiment_list.setAccessibleName("Recent saved experiments")
        self._experiment_list.setAlternatingRowColors(True)
        self._experiment_list.setWordWrap(True)
        self._experiment_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._experiment_list.setSelectionMode(
            QListWidget.SelectionMode.NoSelection
        )
        self._experiment_list.setMinimumHeight(150)
        content_layout.addWidget(self._experiment_list)

        self._experiment_caption = QLabel(content)
        self._experiment_caption.setObjectName("Caption")
        self._experiment_caption.setWordWrap(True)
        content_layout.addWidget(self._experiment_caption)

        self._experiment_stack.addWidget(empty)
        self._experiment_stack.addWidget(content)
        layout.addWidget(self._experiment_stack)
        return card

    # --------------------------------------------------------------- rendering

    def _refresh(self) -> None:
        """Render all sections from the current view-model values."""

        self._render_dataset()
        self._render_model()
        self._render_experiments()

    def _render_dataset(self) -> None:
        view_model = self._view_model
        has_entries = view_model.has_dataset_entries
        self._dataset_stack.setCurrentIndex(
            _PAGE_CONTENT if has_entries else _PAGE_EMPTY
        )
        if not has_entries:
            return

        self._dataset_entries_card.set_value(view_model.dataset_entry_count)
        self._dataset_memory_card.set_value(
            view_model.dataset_memory_type_label
        )
        self._dataset_pattern_card.set_value(
            view_model.dataset_pattern_summary
        )
        self._dataset_targets_card.set_value(view_model.dataset_target_summary)
        self._dataset_detail.setText(view_model.dataset_detail_text)

    def _render_model(self) -> None:
        view_model = self._view_model
        self._model_status_badge.set_status(
            view_model.model_status_kind,
            view_model.model_status_text,
        )

        has_content = view_model.has_model_content
        self._model_stack.setCurrentIndex(
            _PAGE_CONTENT if has_content else _PAGE_EMPTY
        )
        if not has_content:
            return

        self._model_available_card.set_value(view_model.model_count)
        self._model_training_card.set_value(
            view_model.current_training_summary
        )
        self._model_selected_card.set_value(view_model.selected_model_label)
        self._model_detail.setText(view_model.model_detail_text)

    def _render_experiments(self) -> None:
        view_model = self._view_model
        rows = view_model.recent_experiment_rows
        self._experiment_stack.setCurrentIndex(
            _PAGE_CONTENT if rows else _PAGE_EMPTY
        )
        if not rows:
            return

        self._experiment_list.clear()
        for row in rows:
            item = QListWidgetItem(
                f"{row.title}\n{row.detail}",
                self._experiment_list,
            )
            item.setData(Qt.ItemDataRole.UserRole, row.experiment_id)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            item.setToolTip(row.tooltip)
            item.setData(
                Qt.ItemDataRole.AccessibleTextRole,
                f"Saved experiment {row.title}",
            )
            item.setData(
                Qt.ItemDataRole.AccessibleDescriptionRole,
                row.tooltip,
            )

        self._experiment_caption.setText(view_model.experiment_caption)
