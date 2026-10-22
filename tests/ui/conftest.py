"""Shared fixtures for headless Qt GUI tests.

The tests run under Qt's offscreen platform by default so they need no
real display (Task 05 requirement). Set ``QT_QPA_PLATFORM`` before the
fixture import to override (e.g. to inspect widgets interactively).
``pytest-qt`` is deliberately not used: the project depends only on
PySide6 and plain pytest, so a single session-scoped ``QApplication``
fixture provides the required application context instead.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qt_app() -> QApplication:
    """Provide exactly one ``QApplication`` for the whole test session."""

    app = QApplication.instance() or QApplication([])
    app.setApplicationName("Associative Memory Lab")
    yield app
