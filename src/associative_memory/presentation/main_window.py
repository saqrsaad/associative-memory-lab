"""Main application window with sidebar navigation and page switching.

This module is the application shell into which the feature pages are
integrated. It is intentionally free of algorithm or preprocessing
logic: it wires together Task 05's design-system widgets, hosts the
page stack, and delegates page contents to the view modules.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import partial

from PySide6.QtCore import QSize
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

from associative_memory.presentation import APPLICATION_NAME
from associative_memory.presentation.navigation import (
    ALL_DESTINATIONS,
    NavigationDestination,
    item_for_destination,
)
from associative_memory.presentation.theme.spacing import SPACE_MD, SPACE_SM
from associative_memory.presentation.viewmodels.dataset_viewmodel import (
    DatasetViewModel,
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

        # Shared dataset view model across the dataset and training pages.
        self._dataset_view_model = DatasetViewModel()

        # Re-register dataset and training pages with the shared view model.
        register_page(
            NavigationDestination.IMAGE_DATASET,
            partial(DatasetView, dataset_view_model=self._dataset_view_model),
        )
        register_page(
            NavigationDestination.TRAINING,
            partial(TrainingView, dataset_view_model=self._dataset_view_model),
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

    def _select_nav_button(self, destination: NavigationDestination) -> None:
        """Update the sidebar to reflect ``destination`` as active."""

        for button in self._nav_group.buttons():
            button_destination = ALL_DESTINATIONS[
                self._nav_group.id(button)
            ]
            button.setChecked(button_destination is destination)

    def _on_nav_toggled(self, button: QPushButton, checked: bool) -> None:
        """React to navigation button toggles."""

        if not checked:
            return
        destination = ALL_DESTINATIONS[self._nav_group.id(button)]
        self._open_page(destination)


def default_page_destination() -> NavigationDestination:
    """The destination shown when the application launches."""

    return ALL_DESTINATIONS[0]


# Register the page views with the shell. Views are imported here so the
# page registry is populated when the main window is imported. The dataset
# and training pages share a DatasetViewModel created by MainWindow; their
# registrations are updated in MainWindow.__init__ to bind that instance.
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

register_page(NavigationDestination.DASHBOARD, DashboardView)
register_page(NavigationDestination.IMAGE_DATASET, DatasetView)
register_page(NavigationDestination.TRAINING, TrainingView)
register_page(NavigationDestination.RECALL_RESULTS, RecallView)
register_page(NavigationDestination.EXPERIMENT_HISTORY, ExperimentView)
register_page(NavigationDestination.SETTINGS, SettingsView)
