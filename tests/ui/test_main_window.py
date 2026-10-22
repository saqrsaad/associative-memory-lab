"""Integration tests for the application entry point and main window shell."""

from __future__ import annotations

import ast
import logging
from pathlib import Path

from PySide6.QtWidgets import QApplication, QFrame, QPushButton, QStackedWidget

from associative_memory.presentation import APPLICATION_NAME
from associative_memory.presentation.app import create_application
from associative_memory.presentation.main_window import (
    ALL_DESTINATIONS,
    MainWindow,
    NavigationDestination,
)
from associative_memory.presentation.theme import load_stylesheet
from associative_memory.presentation.widgets.empty_state import EmptyState

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ----------------------------------------------------------------- entry point


def test_create_application_applies_name_and_stylesheet(qt_app):
    app = create_application()

    assert isinstance(app, QApplication)
    assert app.applicationName() == APPLICATION_NAME == "Associative Memory Lab"
    assert app.organizationName() == APPLICATION_NAME


def test_stylesheet_loading_works_from_the_entry_point():
    """The entry point installs exactly the loader's centralized stylesheet."""

    app = create_application()

    expected = load_stylesheet()
    assert expected.strip() != ""
    assert "QFrame#NavSidebar" in expected
    assert "QPushButton#NavButton" in expected
    assert "QFrame#TopBar" in expected
    assert app.styleSheet() == expected


def test_create_application_is_idempotent(qt_app):
    first = create_application()
    second = create_application()

    assert first is second  # Qt allows exactly one application instance


# ------------------------------------------------------------- main window


def test_main_window_initializes_into_a_consistent_shell(qt_app):
    window = MainWindow()

    window.show()
    assert window.isVisible() is True
    assert window.centralWidget() is not None

    sidebar = window.findChildren(QFrame, "NavSidebar")
    top_bar = window.findChildren(QFrame, "TopBar")
    stack = window.findChildren(QStackedWidget, "PageStack")

    assert len(sidebar) == 1
    assert len(top_bar) == 1
    assert len(stack) == 1
    assert window.current_destination == NavigationDestination.DASHBOARD
    assert "Dashboard" in window.windowTitle()


def test_all_planned_navigation_destinations_are_reachable(qt_app):
    window = MainWindow()

    buttons = window.findChildren(QPushButton, "NavButton")
    assert len(buttons) == len(ALL_DESTINATIONS)

    for destination in ALL_DESTINATIONS:
        item_label = item_for_label(destination)

        matching = [
            button
            for button in buttons
            if button.text() == item_label
        ]
        assert len(matching) == 1, f"no nav button for {destination.value}"
        matching[0].click()

        assert window.current_destination is destination
        assert item_label in window.windowTitle()
        assert window.findChildren(QStackedWidget, "PageStack")[0].currentWidget() is not None


def test_active_page_changes_correctly(qt_app):
    window = MainWindow()

    first = ALL_DESTINATIONS[0]
    second = ALL_DESTINATIONS[1]

    button_by_label = {
        button.text(): button
        for button in window.findChildren(QPushButton, "NavButton")
    }

    button_by_label[item_for_label(second)].click()
    assert window.current_destination is second
    assert button_by_label[item_for_label(second)].isChecked()
    assert not button_by_label[item_for_label(first)].isChecked()

    button_by_label[item_for_label(first)].click()
    assert window.current_destination is first
    assert button_by_label[item_for_label(first)].isChecked()


def test_page_instances_are_reused_across_switches(qt_app):
    window = MainWindow()
    stack = window.findChildren(QStackedWidget, "PageStack")[0]

    first_widget = stack.currentWidget()

    second = ALL_DESTINATIONS[1]
    button_by_label = {
        button.text(): button
        for button in window.findChildren(QPushButton, "NavButton")
    }
    button_by_label[item_for_label(second)].click()
    second_widget = stack.currentWidget()
    assert second_widget is not first_widget

    button_by_label[item_for_label(ALL_DESTINATIONS[0])].click()
    assert stack.currentWidget() is first_widget  # reused, not recreated


def test_window_remains_usable_when_resized(qt_app):
    window = MainWindow()
    window.resize(640, 480)

    pixmap = window.grab()
    assert not pixmap.isNull()
    assert pixmap.width() > 0
    assert pixmap.height() > 0

    window.resize(1600, 900)
    pixmap = window.grab()
    assert not pixmap.isNull()
    assert pixmap.size().width() == 1600
    assert pixmap.size().height() == 900


def test_page_initialization_error_shows_error_state_without_crashing(
    qt_app, monkeypatch, caplog
):
    """A failing page shows an error state instead of crashing the shell."""

    def failing_view(parent):  # noqa: ARG001
        raise RuntimeError("boom in page init")

    monkeypatch.setitem(
        __import__(
            "associative_memory.presentation.main_window", fromlist=["_PAGES"]
        )._PAGES,
        NavigationDestination.SETTINGS,
        failing_view,
    )

    with caplog.at_level(logging.ERROR):
        window = MainWindow()
        button_by_label = {
            button.text(): button
            for button in window.findChildren(QPushButton, "NavButton")
        }
        button_by_label["Settings"].click()

    assert window.current_destination is NavigationDestination.SETTINGS
    current = window.findChildren(QStackedWidget, "PageStack")[0].currentWidget()
    assert isinstance(current, EmptyState)
    assert current.property("variant") == "error"
    assert "Could not initialize the settings page." in current.description_label.text()
    assert "boom in page init" in caplog.text

    # The shell remains responsive: another destination still opens.
    button_by_label["Dashboard"].click()
    assert window.current_destination is NavigationDestination.DASHBOARD


# ------------------------------------------------------- navigation helpers


def item_for_label(destination: NavigationDestination) -> str:
    """Return the sidebar label for a destination."""

    from associative_memory.presentation.navigation import item_for_destination

    return item_for_destination(destination).label


# ------------------------------------------------- architecture boundaries


def test_presentation_imports_no_business_logic():
    """Presentation code must not import algorithm/domain/infrastructure layers directly.

    Per AGENTS.md, the presentation layer calls application services
    (e.g. ``ImageService``) rather than implementing algorithm equations or
    touching domain/infrastructure directly. Application imports are
    therefore allowed and expected.
    """

    forbidden_prefixes = (
        "associative_memory.algorithms",
        "associative_memory.domain",
        "associative_memory.infrastructure",
    )
    presentation_root = PROJECT_ROOT / "src" / "associative_memory" / "presentation"
    sources = list(presentation_root.rglob("*.py"))
    sources.append(PROJECT_ROOT / "src" / "associative_memory" / "main.py")

    assert len(sources) > 1

    for source_file in sources:
        module = ast.parse(source_file.read_text(encoding="utf-8"))
        for node in ast.walk(module):
            if isinstance(node, ast.Import):
                imported_names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_names = [node.module]
            else:
                continue
            for name in imported_names:
                for prefix in forbidden_prefixes:
                    assert not name.startswith(prefix), (
                        f"{source_file.name} imports {name}, violating layering rules"
                    )


def test_presentation_has_no_inline_widget_stylesheet_strings():
    """Views get looks via object names/properties, not ad-hoc QSS."""

    presentation_root = PROJECT_ROOT / "src" / "associative_memory" / "presentation"

    violations: list[str] = []
    themed_files = 0
    for source_file in presentation_root.rglob("*.py"):
        tree = ast.parse(source_file.read_text(encoding="utf-8"))
        for call in ast.walk(tree):
            is_literal_styleset = (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "setStyleSheet"
                and call.args
                and isinstance(call.args[0], ast.Constant)
                and isinstance(call.args[0].value, str)
                and call.args[0].value.strip() != ""
            )
            if is_literal_styleset:
                violations.append(source_file.name)
        if "setStyleSheet" in ast.dump(tree):
            themed_files += 1

    assert violations == []
    assert themed_files >= 1


def test_all_presentation_modules_import_cleanly(qt_app):
    """Every presentation module is importable in one process."""

    import importlib

    for module_name in (
        "associative_memory.presentation.navigation",
        "associative_memory.presentation.theme.palette",
        "associative_memory.presentation.theme.typography",
        "associative_memory.presentation.theme.spacing",
        "associative_memory.presentation.theme.icons",
        "associative_memory.presentation.widgets.metric_card",
        "associative_memory.presentation.widgets.empty_state",
        "associative_memory.presentation.widgets.status",
        "associative_memory.presentation.widgets.image_canvas",
        "associative_memory.presentation.widgets.pixel_grid",
        "associative_memory.presentation.widgets.vector_view",
        "associative_memory.presentation.widgets.matrix_view",
        "associative_memory.presentation.widgets.weight_matrix_view",
        "associative_memory.presentation.widgets.formula_panel",
        "associative_memory.presentation.views",
        "associative_memory.presentation.views.dashboard_view",
        "associative_memory.presentation.views.image_dataset_view",
        "associative_memory.presentation.views.training_view",
        "associative_memory.presentation.views.recall_view",
        "associative_memory.presentation.views.experiment_view",
        "associative_memory.presentation.views.settings_view",
        "associative_memory.presentation.main_window",
        "associative_memory.presentation.app",
        "associative_memory.main",
    ):
        assert importlib.import_module(module_name) is not None
