"""Dashboard view model aggregating live workspace state.

The dashboard owns no application state of its own. It observes the same
shared dataset, training, recall, and experiment-history view models used by
the workspace pages and exposes read-only summaries for the Dashboard view.
This keeps the overview synchronized with imports, training, recall-model
changes, and experiment saves or deletions without duplicating state.

No sample data or fabricated metrics are produced here: every value is read
from an existing view model or DTO.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from PySide6.QtCore import QObject, Signal

from associative_memory.application.dto.saved_experiment import (
    SavedExperimentSummary,
)
from associative_memory.application.dto.training_result import (
    MemoryType,
    TrainingResult,
)
from associative_memory.application.services.experiment_history_service import (
    ExperimentHistoryService,
)
from associative_memory.presentation.viewmodels.dataset_viewmodel import (
    MEMORY_AUTOASSOCIATIVE,
    MEMORY_HETEROASSOCIATIVE,
    DatasetViewModel,
)
from associative_memory.presentation.viewmodels.experiment_history_viewmodel import (
    ExperimentHistoryViewModel,
)
from associative_memory.presentation.viewmodels.recall_viewmodel import (
    RecallViewModel,
)
from associative_memory.presentation.viewmodels.training_viewmodel import (
    TrainingState,
    TrainingViewModel,
)
from associative_memory.presentation.widgets.status import StatusKind

#: Number of most-recent saved experiments shown on the dashboard.
RECENT_EXPERIMENT_LIMIT = 5


@dataclass(frozen=True)
class DashboardExperimentRow:
    """One formatted recent-experiment row for the dashboard."""

    experiment_id: str
    title: str
    detail: str
    tooltip: str


class DashboardViewModel(QObject):
    """Expose live dataset, model, and experiment summaries."""

    changed = Signal()

    def __init__(
        self,
        dataset_view_model: DatasetViewModel | None = None,
        training_view_model: TrainingViewModel | None = None,
        recall_view_model: RecallViewModel | None = None,
        history_view_model: ExperimentHistoryViewModel | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)

        if training_view_model is not None and dataset_view_model is None:
            dataset_view_model = training_view_model.dataset_view_model
        if dataset_view_model is None:
            dataset_view_model = DatasetViewModel(parent=self)
        if (
            training_view_model is not None
            and training_view_model.dataset_view_model is not dataset_view_model
        ):
            raise ValueError(
                "dataset_view_model must be the training view model's "
                "dataset view model."
            )
        if training_view_model is None:
            training_view_model = TrainingViewModel(
                dataset_view_model=dataset_view_model, parent=self
            )

        if recall_view_model is None and history_view_model is None:
            history_service = ExperimentHistoryService()
            history_view_model = ExperimentHistoryViewModel(
                history_service=history_service, parent=self
            )
            recall_view_model = RecallViewModel(
                training_view_model=training_view_model,
                history_service=history_service,
                parent=self,
            )
        else:
            if history_view_model is None:
                history_view_model = ExperimentHistoryViewModel(parent=self)
            if recall_view_model is None:
                recall_view_model = RecallViewModel(
                    training_view_model=training_view_model, parent=self
                )

        self._dataset_view_model = dataset_view_model
        self._training_view_model = training_view_model
        self._recall_view_model = recall_view_model
        self._history_view_model = history_view_model

        self._dataset_view_model.entries_changed.connect(self._emit_changed)
        self._dataset_view_model.memory_type_changed.connect(self._emit_changed)
        self._dataset_view_model.config_changed.connect(self._emit_changed)
        self._dataset_view_model.preview_changed.connect(self._emit_changed)
        self._training_view_model.state_changed.connect(self._emit_changed)
        self._training_view_model.result_ready.connect(self._emit_changed)
        self._recall_view_model.models_changed.connect(self._emit_changed)
        self._recall_view_model.selection_changed.connect(self._emit_changed)
        self._history_view_model.history_changed.connect(self._emit_changed)

    # ------------------------------------------------------------------ dataset

    @property
    def dataset_entry_count(self) -> int:
        """Number of patterns currently added to the dataset."""

        return len(self._dataset_view_model.entries)

    @property
    def has_dataset_entries(self) -> bool:
        """Whether the dataset contains at least one entry."""

        return bool(self._dataset_view_model.entries)

    @property
    def dataset_memory_type_label(self) -> str:
        """Human-readable current dataset memory type."""

        return _memory_type_label(self._dataset_view_model.memory_type)

    @property
    def dataset_pattern_summary(self) -> str | None:
        """Stored pattern size, or a mixed-size summary.

        Entries keep their own preprocessing snapshots, so the dashboard
        reports multiple sizes rather than pretending they are uniform.
        """

        entries = self._dataset_view_model.entries
        if not entries:
            return None
        sizes = {(entry.result.width, entry.result.height) for entry in entries}
        if len(sizes) == 1:
            width, height = next(iter(sizes))
            return f"{width} × {height}"
        return f"Mixed ({len(sizes)} sizes)"

    @property
    def dataset_target_summary(self) -> str:
        """Target-label status for the current dataset memory type."""

        entries = self._dataset_view_model.entries
        if not entries:
            return "No entries"
        if self._dataset_view_model.memory_type == MEMORY_AUTOASSOCIATIVE:
            return "Self-targeted"
        assigned = sum(
            1
            for entry in entries
            if entry.target_label is not None and entry.target_label.strip()
        )
        unassigned = len(entries) - assigned
        if unassigned == 0:
            return "All assigned"
        return f"{unassigned} unassigned"

    @property
    def dataset_detail_text(self) -> str:
        """Supporting dataset details derived from the active configuration."""

        config = self._dataset_view_model.config
        lines = [
            "Active preprocessing: "
            f"{config.width} × {config.height} grid · "
            f"threshold {config.threshold} · "
            f"{config.encoding.value} encoding."
        ]

        current = self._dataset_view_model.current_result
        if current is None:
            lines.append("No working pattern is currently loaded.")
        else:
            lines.append(
                f"Working pattern: {current.width} × {current.height} "
                f"({current.encoded_vector.size} values)."
            )

        if self._dataset_view_model.memory_type == MEMORY_HETEROASSOCIATIVE:
            entries = self._dataset_view_model.entries
            unassigned = sum(
                1
                for entry in entries
                if entry.target_label is None or not entry.target_label.strip()
            )
            if unassigned:
                lines.append(
                    f"{unassigned} heteroassociative target label"
                    f"{'s' if unassigned != 1 else ''} still need assignment."
                )
            else:
                lines.append(
                    "All heteroassociative target labels are assigned."
                )

        return "\n".join(lines)

    # ------------------------------------------------------------------ model

    @property
    def training_state(self) -> TrainingState:
        """Current training lifecycle state."""

        return self._training_view_model.state

    @property
    def has_current_training_result(self) -> bool:
        """Whether the shared training view model has a current model."""

        return self._training_view_model.current_result is not None

    @property
    def has_models(self) -> bool:
        """Whether at least one model is available in the recall catalog."""

        return bool(self._recall_view_model.models)

    @property
    def model_count(self) -> int:
        """Number of models available in the recall catalog."""

        return len(self._recall_view_model.models)

    @property
    def has_model_content(self) -> bool:
        """Whether the model section should show details instead of empty."""

        return (
            self.has_current_training_result
            or self.has_models
            or self.training_state in (
                TrainingState.TRAINING,
                TrainingState.FAILED,
            )
        )

    @property
    def model_status_text(self) -> str:
        """Short status label for the model section badge."""

        state = self.training_state
        if state is TrainingState.TRAINING:
            return "Training in progress"
        if state is TrainingState.FAILED:
            return "Training failed"
        if self.has_current_training_result:
            return "Model trained"
        if self.has_models:
            return "Models available"
        if state is TrainingState.READY:
            return "Ready to train"
        return "Waiting for dataset"

    @property
    def model_status_kind(self) -> StatusKind:
        """Status-badge kind matching :attr:`model_status_text`."""

        state = self.training_state
        if state is TrainingState.TRAINING:
            return StatusKind.IN_PROGRESS
        if state is TrainingState.FAILED:
            return StatusKind.ERROR
        if self.has_current_training_result:
            return StatusKind.SUCCESS
        return StatusKind.INFO

    @property
    def current_training_summary(self) -> str:
        """Pair count of the current training result, or a clear absence."""

        result = self._training_view_model.current_result
        if result is None:
            return "No current result"
        return f"{result.pair_count} pair{'s' if result.pair_count != 1 else ''}"

    @property
    def selected_model_label(self) -> str:
        """Label of the recall model currently selected, if any."""

        selected = self._recall_view_model.selected_model
        return "None selected" if selected is None else selected.label

    @property
    def model_detail_text(self) -> str:
        """Supporting details for the current training and recall model."""

        lines: list[str] = []
        current = self._training_view_model.current_result
        if current is not None:
            lines.append(
                f"Current training result: {_training_summary(current)}"
            )

        selected = self._recall_view_model.selected_model
        if selected is not None:
            lines.append(
                f"Selected recall model: {selected.label} · "
                f"{_training_summary(selected.result)}"
            )
        elif self.has_models:
            lines.append(
                f"{self.model_count} model"
                f"{'s' if self.model_count != 1 else ''} available for recall; "
                "none selected."
            )

        if self.training_state is TrainingState.TRAINING:
            lines.append(
                "Training is running in the background; this page updates "
                "when the run finishes."
            )
        elif self.training_state is TrainingState.FAILED:
            error = self._training_view_model.error_message
            lines.append(
                f"Training error: {error or 'the last training run failed.'}"
            )

        if not lines:
            lines.append("No trained model is available.")
        return "\n".join(lines)

    # -------------------------------------------------------------- experiments

    @property
    def experiment_count(self) -> int:
        """Total number of saved experiments in the history."""

        return len(self._history_view_model.summaries)

    @property
    def recent_experiment_rows(self) -> tuple[DashboardExperimentRow, ...]:
        """The newest saved experiments, formatted for display."""

        summaries = self._history_view_model.summaries[:RECENT_EXPERIMENT_LIMIT]
        return tuple(_experiment_row(summary) for summary in summaries)

    @property
    def experiment_caption(self) -> str:
        """Summary of how many experiments are shown versus stored."""

        count = self.experiment_count
        if count == 0:
            return "No saved experiments."
        shown = min(count, RECENT_EXPERIMENT_LIMIT)
        if shown == count:
            return f"{count} saved experiment{'s' if count != 1 else ''}."
        return (
            f"Showing the latest {shown} of {count} saved experiments."
        )

    # ------------------------------------------------------------------ actions

    def refresh_history(self) -> None:
        """Reload the shared experiment history."""

        self._history_view_model.refresh()

    def _emit_changed(self, *_args: object) -> None:
        """Forward any observed workspace change as one dashboard update."""

        self.changed.emit()


def _memory_type_label(value: str | MemoryType | None) -> str:
    """Return a display label for a memory-type value."""

    if value == MemoryType.AUTOASSOCIATIVE or value == MEMORY_AUTOASSOCIATIVE:
        return "Autoassociative"
    if value == MemoryType.HETEROASSOCIATIVE or value == MEMORY_HETEROASSOCIATIVE:
        return "Heteroassociative"
    return "Unknown memory type"


def _training_summary(result: TrainingResult) -> str:
    """One-line description of a training result."""

    memory_type = _memory_type_label(result.memory_type)
    encoding = result.encoding.value.title()
    if result.memory_type is MemoryType.AUTOASSOCIATIVE:
        dimensions = f"{result.input_dimension} × {result.output_dimension}"
    else:
        dimensions = f"{result.input_dimension} → {result.output_dimension}"
    pairs = f"{result.pair_count} pair{'s' if result.pair_count != 1 else ''}"
    return f"{memory_type} · {encoding} · {dimensions} · {pairs}"


def _experiment_row(summary: SavedExperimentSummary) -> DashboardExperimentRow:
    """Format one saved-experiment summary for the dashboard list."""

    if not summary.readable:
        reason = summary.unreadable_reason or "unknown validation error"
        return DashboardExperimentRow(
            experiment_id=summary.experiment_id,
            title=summary.display_name,
            detail="Unreadable experiment",
            tooltip=(
                f"{summary.display_name}\nNot readable: {reason}\n"
                "Open Experiment History to delete it."
            ),
        )

    memory_type = _memory_type_label(summary.memory_type)
    encoding = _encoding_label(summary.encoding)
    dimensions = _dimensions_text(summary)
    pairs = (
        "pair count unavailable"
        if summary.pair_count is None
        else f"{summary.pair_count} pair"
        f"{'s' if summary.pair_count != 1 else ''}"
    )
    created = _format_timestamp(summary.created_at)
    detail = f"{memory_type} · {encoding} · {dimensions} · {pairs} · {created}"

    tooltip_lines = [summary.display_name, detail]
    if summary.recall_present is True:
        if summary.recall_exact_match is True:
            tooltip_lines.append("Stored recall: exact match")
        elif summary.recall_mismatch_count is None:
            tooltip_lines.append(
                "Stored recall: mismatch count unavailable"
            )
        else:
            mismatches = summary.recall_mismatch_count
            tooltip_lines.append(
                f"Stored recall: {mismatches} mismatch"
                f"{'es' if mismatches != 1 else ''}"
            )
    elif summary.recall_present is False:
        tooltip_lines.append("No stored recall record")
    else:
        tooltip_lines.append("Stored recall record unavailable")

    return DashboardExperimentRow(
        experiment_id=summary.experiment_id,
        title=summary.display_name,
        detail=detail,
        tooltip="\n".join(tooltip_lines),
    )


def _encoding_label(value: str | None) -> str:
    """Return a display label for a stored encoding value."""

    if not value:
        return "Encoding unavailable"
    return value.replace("_", " ").title()


def _dimensions_text(summary: SavedExperimentSummary) -> str:
    """Return a display label for stored input/output dimensions."""

    if (
        summary.input_dimension is None
        or summary.output_dimension is None
    ):
        return "dimensions unavailable"
    if (
        summary.memory_type == MEMORY_AUTOASSOCIATIVE
        or summary.input_dimension == summary.output_dimension
    ):
        return f"{summary.input_dimension} × {summary.output_dimension}"
    return f"{summary.input_dimension} → {summary.output_dimension}"


def _format_timestamp(value: str | None) -> str:
    """Format a stored ISO timestamp, falling back to the stored text."""

    if not value:
        return "date unavailable"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone()
    return parsed.strftime("%Y-%m-%d %H:%M")
