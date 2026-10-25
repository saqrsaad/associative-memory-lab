"""Main application window with sidebar navigation and page switching.

This module is the application shell into which the feature pages are
integrated. It is intentionally free of algorithm or preprocessing
logic: it wires together Task 05's design-system widgets, hosts the
page stack, and delegates page contents to the view modules.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from PySide6.QtCore import QSize
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from associative_memory.application.dto.saved_experiment import (
    LoadedExperiment,
    PreprocessingSettings,
)
from associative_memory.application.services.experiment_history_service import (
    ExperimentHistoryService,
)
from associative_memory.application.services.image_service import (
    PreprocessingConfig,
)
from associative_memory.presentation import APPLICATION_NAME
from associative_memory.presentation.navigation import (
    ALL_DESTINATIONS,
    NavigationDestination,
    item_for_destination,
)
from associative_memory.presentation.preferences import ApplicationPreferences
from associative_memory.presentation.theme.spacing import SPACE_MD, SPACE_SM
from associative_memory.presentation.viewmodels.dashboard_viewmodel import (
    DashboardViewModel,
)
from associative_memory.presentation.viewmodels.dataset_viewmodel import (
    DatasetViewModel,
)
from associative_memory.presentation.viewmodels.experiment_history_viewmodel import (
    ExperimentHistoryViewModel,
)
from associative_memory.presentation.viewmodels.recall_viewmodel import (
    RecallViewModel,
)
from associative_memory.presentation.viewmodels.settings_viewmodel import (
    SettingsViewModel,
)
from associative_memory.presentation.viewmodels.training_viewmodel import (
    TrainingViewModel,
)
from associative_memory.presentation.widgets.empty_state import EmptyState

_LOGGER = logging.getLogger(__name__)

_SIDEBAR_WIDTH = 220
_WINDOW_MIN_SIZE = QSize(900, 600)
_WINDOW_DEFAULT_SIZE = QSize(1200, 800)

#: Destination -> page factory callable used to build each page lazily.
#: Factories are called with ``(parent)`` and must return a QWidget.
_PAGES: dict[NavigationDestination, Callable[[QWidget | None], QWidget]] = {}


def register_page(
    destination: NavigationDestination, page_factory: Callable[..., QWidget]
) -> None:
    """Register the page factory responsible for ``destination``.

    The factory is called with ``(parent)`` when the page is first
    navigated to. Calling this more than once for the same destination
    replaces the registration; subsequent navigations create fresh page
    instances.
    """

    _PAGES[destination] = page_factory


def page_for_destination(
    destination: NavigationDestination, parent: QWidget | None = None
) -> QWidget:
    """Build the page widget for ``destination``.

    Raises:
        KeyError: If no page factory has been registered for the
            destination.
        RuntimeError: If instantiating the view fails; the original
            exception is chained for diagnostic logging.
    """

    if destination not in _PAGES:
        raise KeyError(
            f"No page view registered for destination {destination.value!r}."
        )

    try:
        return _PAGES[destination](parent)
    except Exception as exc:  # presentation boundary: never crash on a page
        raise RuntimeError(
            f"Could not initialize the {destination.value} page."
        ) from exc


class MainWindow(QMainWindow):
    """The application's single main window.

    Layout
    ------
    Sidebar navigation rail on the left, top bar above the central
    content area, and a page stack occupying the remaining space. All
    sizing uses layout policies so the shell responds to resizing.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setWindowTitle(APPLICATION_NAME)
        self.setMinimumSize(_WINDOW_MIN_SIZE)
        self.resize(_WINDOW_DEFAULT_SIZE)
        self._preferences = ApplicationPreferences()
        self._restore_window_geometry()

        central = QWidget(self)
        central.setObjectName("AppCanvas")
        self.setCentralWidget(central)

        shell = QHBoxLayout(central)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)

        self._sidebar = self._build_sidebar()
        shell.addWidget(self._sidebar)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        self._top_bar = self._build_top_bar()
        content_layout.addWidget(self._top_bar)

        self._stack = QStackedWidget()
        self._stack.setObjectName("PageStack")
        content_layout.addWidget(self._stack, stretch=1)

        shell.addWidget(content, stretch=1)

        self._page_widgets: dict[NavigationDestination, QWidget] = {}
        self._current_destination: NavigationDestination | None = None

        # Shared dataset and training view models across the dataset,
        # training, and recall pages so recall can reuse trained models.
        self._dataset_view_model = DatasetViewModel()
        self._training_view_model = TrainingViewModel(
            dataset_view_model=self._dataset_view_model
        )
        # Both workspaces use the same persistence service, so saving and
        # listing always target the same experiment repository.
        experiment_history_service = ExperimentHistoryService()
        # Create the recall view model eagerly so every training result in
        # this session is added to the recall model catalog, even if the
        # recall page is opened only later.
        self._recall_view_model = RecallViewModel(
            training_view_model=self._training_view_model,
            history_service=experiment_history_service,
        )
        # The experiment history shares this window so that recalls saved
        # from the recall page refresh the listing, and experiments opened
        # from history register their stored model with the recall catalog.
        self._experiment_history_view_model = ExperimentHistoryViewModel(
            history_service=experiment_history_service
        )
        self._recall_view_model.experiment_saved.connect(
            self._experiment_history_view_model.refresh
        )
        self._experiment_history_view_model.loaded_experiment_changed.connect(
            self._on_experiment_loaded
        )

        # The dashboard observes the shared workspace view models rather
        # than maintaining a second copy of dataset, model, or history
        # state. The settings view model shares the same preference store
        # used for window restoration in this window.
        self._dashboard_view_model = DashboardViewModel(
            dataset_view_model=self._dataset_view_model,
            training_view_model=self._training_view_model,
            recall_view_model=self._recall_view_model,
            history_view_model=self._experiment_history_view_model,
            parent=self,
        )
        self._settings_view_model = SettingsViewModel(
            preferences=self._preferences,
            history_view_model=self._experiment_history_view_model,
            parent=self,
        )
        self._settings_view_model.reset_window_layout_requested.connect(
            self._reset_window_layout
        )

        # Re-register pages with the shared view models. The page registry
        # calls factories with a single parent argument, so these wrappers
        # bind dependencies explicitly and keep the parent as a keyword
        # argument. Dashboard and Settings are rebound only when the
        # registry still points at the default or an earlier MainWindow,
        # preserving an explicitly overridden factory.
        if _is_rebindable_page_factory(
            _PAGES.get(NavigationDestination.DASHBOARD),
            _default_dashboard_page,
            "_build_dashboard_page",
        ):
            register_page(
                NavigationDestination.DASHBOARD,
                self._build_dashboard_page,
            )
        register_page(
            NavigationDestination.IMAGE_DATASET,
            lambda parent=None: DatasetView(
                view_model=self._dataset_view_model, parent=parent
            ),
        )
        register_page(
            NavigationDestination.TRAINING,
            lambda parent=None: TrainingView(
                training_view_model=self._training_view_model, parent=parent
            ),
        )
        register_page(
            NavigationDestination.RECALL_RESULTS,
            lambda parent=None: RecallView(
                view_model=self._recall_view_model, parent=parent
            ),
        )
        register_page(
            NavigationDestination.EXPERIMENT_HISTORY,
            self._build_experiment_page,
        )
        if _is_rebindable_page_factory(
            _PAGES.get(NavigationDestination.SETTINGS),
            _default_settings_page,
            "_build_settings_page",
        ):
            register_page(
                NavigationDestination.SETTINGS,
                self._build_settings_page,
            )

        self._open_page(ALL_DESTINATIONS[0])

    # ------------------------------------------------------------------ shell

    def _build_sidebar(self) -> QFrame:
        sidebar = QFrame(self)
        sidebar.setObjectName("NavSidebar")
        sidebar.setFixedWidth(_SIDEBAR_WIDTH)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, SPACE_MD, 0, SPACE_MD)
        layout.setSpacing(SPACE_SM)

        app_label = QLabel(APPLICATION_NAME, sidebar)
        app_label.setObjectName("AppTitle")
        app_label.setWordWrap(True)
        layout.addWidget(app_label)

        self._nav_group = QButtonGroup(self)
        self._nav_group.setExclusive(True)
        self._nav_group.buttonToggled.connect(self._on_nav_toggled)

        for destination in ALL_DESTINATIONS:
            item = item_for_destination(destination)
            button = QPushButton(item.label, sidebar)
            button.setObjectName("NavButton")
            button.setCheckable(True)
            button.setAutoExclusive(True)
            button.setToolTip(item.tooltip)
            button.setProperty("destination", destination.value)
            button.setAccessibleName(item.label)
            button.setAccessibleDescription(item.tooltip)
            layout.addWidget(button)
            self._nav_group.addButton(button)
            self._nav_group.setId(button, ALL_DESTINATIONS.index(destination))

        layout.addStretch(1)
        return sidebar

    def _build_top_bar(self) -> QFrame:
        top_bar = QFrame(self)
        top_bar.setObjectName("TopBar")

        layout = QHBoxLayout(top_bar)
        layout.setContentsMargins(SPACE_MD, SPACE_SM, SPACE_MD, SPACE_SM)
        layout.setSpacing(SPACE_MD)

        self._page_title = QLabel(self)
        self._page_title.setObjectName("PageTitle")
        self._page_title.setWordWrap(True)
        layout.addWidget(self._page_title, stretch=1)

        return top_bar

    def _build_dashboard_page(
        self, parent: QWidget | None = None
    ) -> QWidget:
        """Build the live dashboard and route its empty-state actions."""

        page = DashboardView(
            view_model=self._dashboard_view_model, parent=parent
        )
        page.open_dataset_requested.connect(
            lambda: self._open_page(NavigationDestination.IMAGE_DATASET)
        )
        page.open_training_requested.connect(
            lambda: self._open_page(NavigationDestination.TRAINING)
        )
        page.open_recall_requested.connect(
            lambda: self._open_page(NavigationDestination.RECALL_RESULTS)
        )
        return page

    def _build_settings_page(
        self, parent: QWidget | None = None
    ) -> QWidget:
        """Build Settings with the window's shared preference store."""

        return SettingsView(
            view_model=self._settings_view_model, parent=parent
        )

    def _build_experiment_page(
        self, parent: QWidget | None = None
    ) -> QWidget:
        """Build Experiment History and route its explicit recall action."""

        page = ExperimentView(
            view_model=self._experiment_history_view_model, parent=parent
        )
        page.open_in_recall_requested.connect(
            lambda: self._open_page(NavigationDestination.RECALL_RESULTS)
        )
        return page

    # ------------------------------------------------------------------ pages

    @property
    def current_destination(self) -> NavigationDestination | None:
        """The currently selected navigation destination, if any."""

        return self._current_destination

    def _open_page(self, destination: NavigationDestination) -> None:
        """Switch the central content to ``destination``'s page.

        Pages are instantiated lazily and reused once created. If the
        page cannot be initialized, an error state is shown instead and
        the diagnostic is logged.
        """

        self._current_destination = destination
        item = item_for_destination(destination)
        self._page_title.setText(item.label)
        self._select_nav_button(destination)
        self.setWindowTitle(f"{APPLICATION_NAME} — {item.label}")

        page = self._page_widgets.get(destination)
        if page is None:
            try:
                page = page_for_destination(destination, self)
            except Exception as exc:
                _LOGGER.exception(
                    "Failed to initialize page %s", destination.value
                )
                page = EmptyState(
                    f"{item.label} unavailable",
                    str(exc),
                    variant="error",
                    parent=self,
                )
            self._page_widgets[destination] = page

        if self._stack.indexOf(page) == -1:
            self._stack.addWidget(page)
        self._stack.setCurrentWidget(page)

    def _restore_window_geometry(self) -> None:
        """Restore saved window geometry when the preference is enabled."""

        if not self._preferences.remember_window_geometry:
            return
        geometry = self._preferences.window_geometry
        if geometry is not None:
            self.restoreGeometry(geometry)

    def _reset_window_layout(self) -> None:
        """Return the window to its default size and a centered position."""

        if self.isMaximized() or self.isFullScreen():
            self.showNormal()
        self.resize(_WINDOW_DEFAULT_SIZE)
        screen = self.screen()
        if screen is not None:
            frame = self.frameGeometry()
            frame.moveCenter(screen.availableGeometry().center())
            self.move(frame.topLeft())
        self._preferences.clear_window_geometry()

    def closeEvent(self, event: QCloseEvent) -> None:
        """Persist window geometry when the user enabled that preference."""

        if self._preferences.remember_window_geometry:
            self._preferences.set_window_geometry(self.saveGeometry())
        super().closeEvent(event)

    def _select_nav_button(self, destination: NavigationDestination) -> None:
        """Update the sidebar to reflect ``destination`` as active."""

        for button in self._nav_group.buttons():
            button_destination = ALL_DESTINATIONS[
                self._nav_group.id(button)
            ]
            button.setChecked(button_destination is destination)

    def _on_experiment_loaded(self, loaded: object) -> None:
        """Register a saved experiment's model with the recall workspace.

        The stored preprocessing settings are rebuilt only when the stored
        grid dimension matches the model; otherwise the recall view model
        falls back to its own inference (or to manual image import).
        """

        if not isinstance(loaded, LoadedExperiment):
            return

        training = loaded.training
        source_config = self._config_for_dimension(
            loaded.preprocessing, training.input_dimension
        )
        target_config = self._config_for_dimension(
            loaded.target_preprocessing, training.output_dimension
        )
        self._recall_view_model.add_training_result(
            training,
            label=f"Loaded: {loaded.name}",
            preprocessing_config=source_config,
            target_preprocessing_config=target_config,
        )

    @staticmethod
    def _config_for_dimension(
        loaded_settings: object, dimension: int
    ) -> PreprocessingConfig | None:
        """Rebuild stored preprocessing settings when the grid matches."""

        if not isinstance(loaded_settings, PreprocessingSettings):
            return None
        if loaded_settings.width * loaded_settings.height != dimension:
            return None
        try:
            return loaded_settings.to_config()
        except (TypeError, ValueError):
            return None

    def _on_nav_toggled(self, button: QPushButton, checked: bool) -> None:
        """React to navigation button toggles."""

        if not checked:
            return
        destination = ALL_DESTINATIONS[self._nav_group.id(button)]
        self._open_page(destination)


def _is_rebindable_page_factory(
    factory: Callable[..., QWidget] | None,
    default_factory: Callable[..., QWidget],
    method_name: str,
) -> bool:
    """Whether a page factory may be rebound to the current window.

    The default factory is rebound once per window so every window gets its
    own shared view models. A factory that was explicitly overridden (for
    example by a test or an embedding application) is preserved. Bound
    factories from an earlier ``MainWindow`` are rebound for the new window.
    """

    if factory is default_factory:
        return True
    owner = getattr(factory, "__self__", None)
    factory_function = getattr(factory, "__func__", None)
    return (
        isinstance(owner, MainWindow)
        and factory_function is getattr(MainWindow, method_name, None)
    )


def default_page_destination() -> NavigationDestination:
    """The destination shown when the application launches."""

    return ALL_DESTINATIONS[0]


# Register the page views with the shell. Views are imported here so the
# page registry is populated when the main window is imported. The dataset,
# training, recall, dashboard, and settings pages share the view models
# created by MainWindow; their registrations are updated in
# MainWindow.__init__ to bind those instances.
from associative_memory.presentation.views.dashboard_view import DashboardView  # noqa: E402
from associative_memory.presentation.views.dataset_view import DatasetView  # noqa: E402
from associative_memory.presentation.views.experiment_view import (  # noqa: E402
    ExperimentView,
)
from associative_memory.presentation.views.recall_view import RecallView  # noqa: E402
from associative_memory.presentation.views.settings_view import (  # noqa: E402
    SettingsView,
)
from associative_memory.presentation.views.training_view import (  # noqa: E402
    TrainingView,
)


def _default_dashboard_page(parent: QWidget | None = None) -> QWidget:
    """Build the Dashboard page with its own view model."""

    return DashboardView(parent=parent)


def _default_settings_page(parent: QWidget | None = None) -> QWidget:
    """Build the Settings page with its own view model."""

    return SettingsView(parent=parent)


register_page(NavigationDestination.DASHBOARD, _default_dashboard_page)
register_page(
    NavigationDestination.IMAGE_DATASET,
    lambda parent=None: DatasetView(parent=parent),
)
register_page(
    NavigationDestination.TRAINING,
    lambda parent=None: TrainingView(parent=parent),
)
register_page(
    NavigationDestination.RECALL_RESULTS,
    lambda parent=None: RecallView(parent=parent),
)
register_page(
    NavigationDestination.EXPERIMENT_HISTORY,
    lambda parent=None: ExperimentView(parent=parent),
)
register_page(NavigationDestination.SETTINGS, _default_settings_page)
