"""Experiment-history view model.

Coordinates the saved-experiment list, opening, deletion, and comparison
through :class:`associative_memory.application.services.\
experiment_history_service.ExperimentHistoryService`. The view model
performs no file or numerical work: storage stays behind the application
repository interface, and comparison rows arrive ready formatted from the
service with explicit availability notes.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Signal

from associative_memory.application.dto.experiment_comparison import (
    ExperimentComparison,
)
from associative_memory.application.dto.saved_experiment import (
    LoadedExperiment,
    SavedExperimentSummary,
)
from associative_memory.application.interfaces.model_repository import (
    ExperimentRepositoryError,
)
from associative_memory.application.services.experiment_history_service import (
    ExperimentHistoryService,
)
from associative_memory.presentation.widgets.status import StatusKind

_LOGGER = logging.getLogger(__name__)


class ExperimentHistoryViewModel(QObject):
    """Coordinate the experiment-history workspace."""

    history_changed = Signal()
    loaded_experiment_changed = Signal(object)  # LoadedExperiment | None
    comparison_changed = Signal(object)  # ExperimentComparison | None
    error_occurred = Signal(str)
    status_message = Signal(str, StatusKind)

    def __init__(
        self,
        history_service: ExperimentHistoryService | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)

        self._history_service = (
            history_service if history_service is not None
            else ExperimentHistoryService()
        )
        self._summaries: tuple[SavedExperimentSummary, ...] = ()
        self._loaded: LoadedExperiment | None = None
        self._comparison: ExperimentComparison | None = None

    # ------------------------------------------------------------------ properties

    @property
    def summaries(self) -> tuple[SavedExperimentSummary, ...]:
        """The current history-list entries (newest first)."""

        return self._summaries

    @property
    def loaded_experiment(self) -> LoadedExperiment | None:
        """The most recently opened experiment, if any."""

        return self._loaded

    @property
    def comparison(self) -> ExperimentComparison | None:
        """The most recent comparison result, if any."""

        return self._comparison

    @property
    def storage_location_text(self) -> str | None:
        """Where saved experiments are stored, for display only."""

        location = self._history_service.storage_location
        return None if location is None else str(location)

    @property
    def has_entries(self) -> bool:
        """Whether the history currently shows any entries."""

        return bool(self._summaries)

    # ------------------------------------------------------------------ operations

    def refresh(self) -> None:
        """Reload the history list from the repository."""

        try:
            self._summaries = self._history_service.list_experiments()
        except ExperimentRepositoryError as exc:
            self._summaries = ()
            message = f"Loading the experiment history failed: {exc}"
            self.error_occurred.emit(message)
            self.status_message.emit(message, StatusKind.ERROR)
        except Exception as exc:  # presentation boundary: log, never crash
            _LOGGER.exception("Unexpected history listing failure")
            message = (
                "Loading the experiment history failed unexpectedly. "
                "See the application log for details."
            )
            self.error_occurred.emit(f"{message} ({exc})")
            self.status_message.emit(message, StatusKind.ERROR)
        else:
            count = len(self._summaries)
            self.status_message.emit(
                f"Experiment history refreshed ({count} saved experiment"
                f"{'s' if count != 1 else ''}).",
                StatusKind.INFO,
            )
        self.history_changed.emit()

    def load_experiment(self, experiment_id: str) -> LoadedExperiment | None:
        """Open one experiment and expose it for recall registration.

        Returns the loaded experiment, or ``None`` when loading failed
        (an error message is emitted in that case).
        """

        selected = next(
            (
                summary
                for summary in self._summaries
                if summary.experiment_id == experiment_id
            ),
            None,
        )
        if selected is None:
            message = (
                "Select a history entry before opening an experiment."
            )
            self.status_message.emit(message, StatusKind.WARNING)
            return None

        try:
            loaded = self._history_service.load_experiment(experiment_id)
        except ExperimentRepositoryError as exc:
            message = f"Opening experiment {selected.name!r} failed: {exc}"
            self.error_occurred.emit(message)
            self.status_message.emit(message, StatusKind.ERROR)
            return None
        except Exception as exc:  # presentation boundary: log, never crash
            _LOGGER.exception("Unexpected experiment loading failure")
            message = (
                f"Opening experiment {selected.name!r} failed unexpectedly. "
                "See the application log for details."
            )
            self.error_occurred.emit(f"{message} ({exc})")
            self.status_message.emit(message, StatusKind.ERROR)
            return None

        self._loaded = loaded
        self.loaded_experiment_changed.emit(loaded)
        self.status_message.emit(
            f"Opened experiment {loaded.name!r}; its stored model is now "
            "available in the recall workspace.",
            StatusKind.SUCCESS,
        )
        return loaded

    def delete_experiment(self, experiment_id: str) -> None:
        """Delete one experiment and refresh the list.

        The view is responsible for asking the user to confirm before
        calling this method.
        """

        previous_id = (
            None if self._loaded is None else self._loaded.summary.experiment_id
        )
        comparison_ids = (
            None
            if self._comparison is None
            else (
                self._comparison.left.experiment_id,
                self._comparison.right.experiment_id,
            )
        )
        try:
            self._history_service.delete_experiment(experiment_id)
        except ExperimentRepositoryError as exc:
            message = f"Deleting the experiment failed: {exc}"
            self.error_occurred.emit(message)
            self.status_message.emit(message, StatusKind.ERROR)
            return
        except Exception as exc:  # presentation boundary: log, never crash
            _LOGGER.exception("Unexpected experiment deletion failure")
            message = (
                "Deleting the experiment failed unexpectedly. See the "
                "application log for details."
            )
            self.error_occurred.emit(f"{message} ({exc})")
            self.status_message.emit(message, StatusKind.ERROR)
            return

        if previous_id == experiment_id:
            self._loaded = None
            self.loaded_experiment_changed.emit(None)
        if comparison_ids and experiment_id in comparison_ids:
            self._comparison = None
            self.comparison_changed.emit(None)
        self.status_message.emit(
            "Experiment deleted and removed from the history.",
            StatusKind.SUCCESS,
        )
        self.refresh()

    def compare(
        self, left_id: str, right_id: str
    ) -> ExperimentComparison | None:
        """Compare the two selected experiments; ``None`` when it failed."""

        try:
            comparison = self._history_service.compare_experiments(
                left_id, right_id
            )
        except ExperimentRepositoryError as exc:
            message = f"Comparing the experiments failed: {exc}"
            self.error_occurred.emit(message)
            self.status_message.emit(message, StatusKind.ERROR)
            return None
        except ValueError as exc:
            message = str(exc)
            self.error_occurred.emit(message)
            self.status_message.emit(message, StatusKind.WARNING)
            return None
        except Exception as exc:  # presentation boundary: log, never crash
            _LOGGER.exception("Unexpected experiment comparison failure")
            message = (
                "Comparing the experiments failed unexpectedly. See the "
                "application log for details."
            )
            self.error_occurred.emit(f"{message} ({exc})")
            self.status_message.emit(message, StatusKind.ERROR)
            return None

        self._comparison = comparison
        self.comparison_changed.emit(comparison)
        self.status_message.emit(
            "Comparison ready; incompatible values are annotated.",
            StatusKind.INFO,
        )
        return comparison

    def show_details(self) -> None:
        """Return the workspace to the details page from a comparison."""

        if self._comparison is None:
            return
        self._comparison = None
        self.comparison_changed.emit(None)

    def clear_loaded(self) -> None:
        """Forget the currently opened experiment (no storage change)."""

        if self._loaded is None:
            return
        self._loaded = None
        self.loaded_experiment_changed.emit(None)
