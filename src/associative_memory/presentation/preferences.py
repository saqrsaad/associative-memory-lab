"""Minimal persisted presentation preferences backed by Qt ``QSettings``.

The project had no preference-persistence layer before this module. Only
preferences with implemented behavior belong here: storing a control that
has no effect would present an unsupported setting as functional.

The first supported preference is whether the main window should restore
its size and position on the next launch. The value is written through
``QSettings`` using the application identity configured in
:mod:`associative_memory.presentation.app`.
"""

from __future__ import annotations

from PySide6.QtCore import QByteArray, QSettings

_REMEMBER_GEOMETRY_KEY = "window/remember_geometry"
_WINDOW_GEOMETRY_KEY = "window/geometry"


class ApplicationPreferences:
    """Read and write the small set of implemented UI preferences."""

    def __init__(self, settings: QSettings | None = None) -> None:
        """Create a preference store.

        Args:
            settings: Optional ``QSettings`` instance for tests or explicit
                isolation. The default uses the application's configured
                organization and application names.
        """

        self._settings = settings if settings is not None else QSettings()

    @property
    def remember_window_geometry(self) -> bool:
        """Whether the main window should restore its saved geometry."""

        value = self._settings.value(_REMEMBER_GEOMETRY_KEY, False)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def set_remember_window_geometry(self, enabled: bool) -> None:
        """Persist the geometry preference and discard stale geometry.

        Disabling the preference removes any previously saved geometry so a
        later re-enable does not unexpectedly restore an old window shape.
        """

        self._settings.setValue(_REMEMBER_GEOMETRY_KEY, bool(enabled))
        if not enabled:
            self._settings.remove(_WINDOW_GEOMETRY_KEY)
        self._settings.sync()

    @property
    def window_geometry(self) -> QByteArray | None:
        """The saved window geometry, if one exists."""

        value = self._settings.value(_WINDOW_GEOMETRY_KEY)
        if isinstance(value, QByteArray):
            return value
        if isinstance(value, (bytes, bytearray)):
            return QByteArray(value)
        return None

    def set_window_geometry(self, geometry: QByteArray) -> None:
        """Persist the main window's current geometry."""

        self._settings.setValue(_WINDOW_GEOMETRY_KEY, geometry)
        self._settings.sync()

    def clear_window_geometry(self) -> None:
        """Remove the saved geometry without changing the preference."""

        self._settings.remove(_WINDOW_GEOMETRY_KEY)
        self._settings.sync()
