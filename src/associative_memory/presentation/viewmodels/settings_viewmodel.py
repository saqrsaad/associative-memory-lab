"""Settings view model for implemented preferences and actual configuration.

Only preferences with real persistence and behavior are exposed as
editable settings. The remaining values are read-only snapshots of the
running configuration so the page does not imply that unsupported
preferences (theme selection, configurable storage, startup-page choice)
are implemented.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from associative_memory.application.dto.saved_experiment import (
    APPLICATION_VERSION,
)
from associative_memory.application.services.image_service import (
    DEFAULT_PREPROCESSING_CONFIG,
)
from associative_memory.presentation import APPLICATION_NAME
from associative_memory.presentation.preferences import ApplicationPreferences
from associative_memory.presentation.viewmodels.experiment_history_viewmodel import (
    ExperimentHistoryViewModel,
)


class SettingsViewModel(QObject):
    """Coordinate the supported window preference and read-only facts."""

    changed = Signal()
    reset_window_layout_requested = Signal()

    def __init__(
        self,
        preferences: ApplicationPreferences | None = None,
        history_view_model: ExperimentHistoryViewModel | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._preferences = (
            preferences if preferences is not None else ApplicationPreferences()
        )
        self._history_view_model = (
            history_view_model
            if history_view_model is not None
            else ExperimentHistoryViewModel(parent=self)
        )

    # ------------------------------------------------------------- preferences

    @property
    def remember_window_geometry(self) -> bool:
        """Whether the main window restores its saved size and position."""

        return self._preferences.remember_window_geometry

    def set_remember_window_geometry(self, enabled: bool) -> None:
        """Persist the only currently implemented preference."""

        if enabled == self._preferences.remember_window_geometry:
            return
        self._preferences.set_remember_window_geometry(enabled)
        self.changed.emit()

    def request_reset_window_layout(self) -> None:
        """Discard saved geometry and ask the window to reset its layout."""

        self._preferences.clear_window_geometry()
        self.reset_window_layout_requested.emit()

    # --------------------------------------------------------- actual settings

    @property
    def application_name(self) -> str:
        """Application identity shown in the About section."""

        return APPLICATION_NAME

    @property
    def application_version(self) -> str:
        """Application version recorded by the current build."""

        return APPLICATION_VERSION

    @property
    def storage_location_text(self) -> str:
        """Actual experiment storage location, or a clear absence."""

        return self._history_view_model.storage_location_text or "Not available"

    @property
    def theme_name(self) -> str:
        """The only theme currently implemented by the application."""

        return "Light"

    @property
    def preprocessing_defaults_text(self) -> str:
        """Read-only summary of the default preprocessing configuration."""

        config = DEFAULT_PREPROCESSING_CONFIG
        return "\n".join(
            (
                f"Pattern grid: {config.width} × {config.height}",
                f"Threshold: {config.threshold} · "
                f"polarity: {config.polarity.value}",
                f"Encoding: {config.encoding.value} · "
                f"grayscale: {config.grayscale_method.value}",
                f"Resize: {config.resize_mode.value} · "
                f"flattening: {config.flattening_order.value}",
            )
        )
