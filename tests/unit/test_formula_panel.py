"""Unit tests for the FormulaPanel widget."""

from __future__ import annotations

from PySide6.QtWidgets import QApplication

from associative_memory.presentation.widgets.formula_panel import FormulaPanel


def test_formula_panel_instantiates(qt_app: QApplication) -> None:
    panel = FormulaPanel()
    assert panel.objectName() == "FormulaPanel"


def test_formula_panel_set_equation_and_caption(qt_app: QApplication) -> None:
    panel = FormulaPanel()
    panel.set_equation("W = Σ outer(s, t)")
    panel.set_caption("Heteroassociative · binary encoding")

    assert panel._equation == "W = Σ outer(s, t)"
    assert panel._caption == "Heteroassociative · binary encoding"


def test_formula_panel_set_learning_rule(qt_app: QApplication) -> None:
    panel = FormulaPanel()
    panel.set_learning_rule(
        memory_type="autoassociative",
        encoding="bipolar",
        source_dim=4,
        target_dim=4,
        pair_count=2,
    )

    assert "autoassociative" in panel._caption
    assert "bipolar" in panel._caption
    assert "4×4" in panel._caption
    assert "2 pair(s)" in panel._caption
    assert "outer(s, s)" in panel._caption


def test_formula_panel_set_learning_rule_hetero(qt_app: QApplication) -> None:
    panel = FormulaPanel()
    panel.set_learning_rule(
        memory_type="heteroassociative",
        encoding="binary",
        source_dim=2,
        target_dim=3,
        pair_count=5,
    )

    assert "heteroassociative" in panel._caption
    assert "outer(s, t)" in panel._caption


def test_formula_panel_clear(qt_app: QApplication) -> None:
    panel = FormulaPanel()
    panel.set_equation("W = Σ outer(s, t)")
    panel.set_caption("Heteroassociative · binary encoding")
    panel.clear()

    assert panel._equation == ""
    assert panel._caption == ""
    assert panel._equation_label.isHidden()
    assert panel._caption_label.isHidden()


def test_formula_panel_paint_does_not_crash(qt_app: QApplication) -> None:
    panel = FormulaPanel()
    panel.set_equation("W = Σ outer(s, t)")
    panel.set_caption("Autoassociative · bipolar encoding · 4x4 matrix")
    panel.show()

    pixmap = panel.grab()
    assert not pixmap.isNull()
