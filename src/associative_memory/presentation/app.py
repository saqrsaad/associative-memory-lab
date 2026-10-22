"""Application factory for the Associative Memory Lab desktop application.

This module owns the intended PySide6 bootstrap sequence: it creates or
reuses the ``QApplication``, applies identity metadata, sets the default
font, and installs the centralized stylesheet produced by
:func:`associative_memory.presentation.theme.load_stylesheet`. Views must
not call ``setStyleSheet`` with their own strings.
"""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from associative_memory.presentation import APPLICATION_NAME
from associative_memory.presentation.main_window import MainWindow
from associative_memory.presentation.theme import load_stylesheet
from associative_memory.presentation.theme.typography import UI_FONT_FAMILY


def create_application(args: Sequence[str] | None = None) -> QApplication:
    """Create (or reuse) the ``QApplication`` with the centralized theme.

    Reapplication is idempotent: calling again with an existing instance
    refreshes metadata, font, and stylesheet without a second ``QApplication``
    (Qt allows only one).

    Args:
        args: Optional command-line arguments forwarded to Qt on first
            creation.

    Returns:
        The configured application instance.
    """

    app = QApplication.instance()
    if app is None:
        app = QApplication(list(args) if args is not None else [])
    app.setApplicationName(APPLICATION_NAME)
    app.setOrganizationName(APPLICATION_NAME)
    app.setFont(QFont(UI_FONT_FAMILY))
    app.setStyleSheet(load_stylesheet())
    return app


def run_application(args: Sequence[str] | None = None) -> int:
    """Create the minimal styled main window and run the event loop.

    The current window is the Task 05 design-system preview; it is
    extended with real navigation in the following task.
    """

    app = create_application(args)
    window = MainWindow()
    window.show()
    return app.exec()
