"""Shared fixtures for headless Qt GUI unit tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture(scope="session")
def qt_app() -> QApplication:
    """Provide exactly one ``QApplication`` for the whole test session."""

    app = QApplication.instance() or QApplication([])
    app.setApplicationName("Associative Memory Lab")
    yield app


@pytest.fixture(scope="session")
def reference_cases() -> dict:
    """Load all algorithm reference cases from the fixture file."""

    with open(FIXTURES_DIR / "algorithm_reference_cases.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def reference_case(reference_cases: dict):
    """Return a reference case lookup function by case id."""

    def _get(case_id: str) -> dict:
        for case in reference_cases["cases"]:
            if case["id"] == case_id:
                return case
        raise KeyError(f"Reference case {case_id!r} not found")

    return _get
