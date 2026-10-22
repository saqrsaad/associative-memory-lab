"""Behavior tests for the shared widgets (headless, offscreen)."""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

from associative_memory.presentation.widgets import (
    NOT_CALCULATED_TEXT,
    EmptyState,
    MetricCard,
    StatusBadge,
    StatusKind,
)

# ------------------------------------------------------------ MetricCard


def test_metric_card_instantiates_with_title_and_placeholder_value(qt_app):
    card = MetricCard("Pattern count")

    assert card.title == "Pattern count"
    assert card.value_label.text() == NOT_CALCULATED_TEXT == "Not calculated"
    assert card.description_label.isHidden()
    assert card.status_badge is None


def test_metric_card_renders_integer_and_preformatted_values(qt_app):
    card = MetricCard("Pair count")

    card.set_value(7)
    assert card.value_label.text() == "7"

    card.set_value("220:45")
    assert card.value_label.text() == "220:45"

    card.set_value(True)
    assert card.value_label.text() == "True"


def test_metric_card_formats_floats_with_documented_significance(qt_app):
    card = MetricCard("Match ratio")

    card.set_value(3.14159)
    assert card.value_label.text() == "3.142"  # four significant digits


def test_metric_card_missing_value_uses_deliberate_placeholder_not_zero(qt_app):
    card = MetricCard("Mismatches")

    card.set_value(12)  # start from a real value
    card.set_value(None)

    assert card.value_label.text() == NOT_CALCULATED_TEXT
    assert card.property("state") == "missing"
    assert card.value_label.property("state") == "missing"


def test_metric_card_restores_normal_state_after_value_return(qt_app):
    card = MetricCard("Mismatches")

    card.set_value(None)
    card.set_value(0)

    assert card.value_label.text() == "0"  # a genuine computed zero passes
    assert card.property("state") == ""


def test_metric_card_description_shows_and_hides(qt_app):
    card = MetricCard("Pair count")

    card.set_description("Across all training pairs")
    assert card.description_label.text() == "Across all training pairs"
    assert not card.description_label.isHidden()

    card.set_description(None)
    assert card.description_label.isHidden()


def test_metric_card_status_badge_lifecycle(qt_app):
    card = MetricCard("Recall match")

    card.set_status(StatusKind.SUCCESS, "All patterns matched")
    badge = card.status_badge
    assert badge is not None
    assert badge.status_kind is StatusKind.SUCCESS
    assert badge.text_label.text() == "All patterns matched"
    assert not badge.isHidden()

    card.set_status(StatusKind.WARNING)  # replace kind; custom text preserved
    assert card.status_badge is badge
    assert badge.status_kind is StatusKind.WARNING
    assert badge.text_label.text() == "All patterns matched"

    card.set_status(StatusKind.ERROR, "Known consequence")  # explicit text wins
    assert badge.status_kind is StatusKind.ERROR
    assert badge.text_label.text() == "Known consequence"

    card.clear_status()
    assert card.status_badge is None


def test_metric_card_exposes_accessible_names(qt_app):
    card = MetricCard("Pair count")

    card.set_value(7)
    assert "Pair count" in card.accessibleName()
    assert "Pair count: 7" in card.value_label.accessibleName()

    card.set_value(None)
    assert NOT_CALCULATED_TEXT in card.value_label.accessibleName()


# -------------------------------------------------------------- EmptyState


def test_empty_state_renders_without_application_data(qt_app):
    state = EmptyState("No dataset loaded", "Import images to get started.")

    assert state.title == "No dataset loaded"
    assert not state.description_label.isHidden()
    assert state.description_label.text() == "Import images to get started."
    assert state.action_button is None
    assert state.grab().isNull() is False


def test_empty_state_hides_description_when_absent(qt_app):
    state = EmptyState("Nothing here")

    assert state.description_label.isHidden()


def test_empty_state_action_appears_invokes_and_clears(qt_app):
    state = EmptyState("No dataset", "Nothing imported yet.")
    clicks: list[str] = []

    state.set_action("Import images", lambda: clicks.append("clicked"))
    button = state.action_button
    assert button is not None and not button.isHidden()
    assert button.text() == "Import images"

    button.click()
    assert clicks == ["clicked"]

    state.clear_action()
    assert state.action_button is not None  # retained but hidden
    assert state.action_button.isHidden()

    state.action_button.click()  # callback disconnected semantics: no-op
    assert clicks == ["clicked"]


def test_empty_state_action_is_keyboard_reachable(qt_app):
    state = EmptyState("No dataset", "Nothing imported yet.")

    state.set_action("Import images", lambda: None)
    assert state.action_button.focusPolicy() != Qt.FocusPolicy.NoFocus


def test_empty_state_supports_error_variant(qt_app):
    state = EmptyState("Dataset invalid", "One image could not be loaded.")

    state.set_variant("error")
    assert state.property("variant") == "error"

    with pytest.raises(ValueError, match="Unknown empty-state variant"):
        state.set_variant("banana")


def test_empty_state_optional_icon(qt_app):
    from PySide6.QtGui import QPixmap

    shown = EmptyState("Dataset preview")

    pixmap = QPixmap(16, 16)
    pixmap.fill()
    shown.set_icon(QIcon(pixmap))
    assert not shown.icon_label.isHidden()

    shown.set_icon(QIcon())  # a null icon simply does not display
    assert shown.icon_label.isHidden()

    hidden = EmptyState("T")
    assert hidden.icon_label.isHidden()  # no icon set: stays invisible


# ------------------------------------------------------------ StatusBadge


def test_status_badge_defaults_per_kind(qt_app):
    expected = {
        StatusKind.SUCCESS: "Success",
        StatusKind.INFO: "Information",
        StatusKind.WARNING: "Warning",
        StatusKind.ERROR: "Error",
        StatusKind.IN_PROGRESS: "In progress",
    }
    for kind, text in expected.items():
        badge = StatusBadge(kind)
        assert badge.status_kind is kind
        assert badge.text_label.text() == text
        assert badge.property("status") == kind.value
        assert not badge.icon_label.pixmap().isNull()
        assert kind.value in badge.accessibleName()


def test_status_badge_accepts_plain_string_kind(qt_app):
    badge = StatusBadge("error", "Load failed")

    assert badge.status_kind is StatusKind.ERROR
    assert badge.text_label.text() == "Load failed"

    with pytest.raises(ValueError):
        badge.set_status("magenta")


def test_status_badge_preserves_custom_text_on_kind_change(qt_app):
    badge = StatusBadge(StatusKind.INFO, "Loading training data")

    badge.set_status(StatusKind.IN_PROGRESS)
    assert badge.text_label.text() == "Loading training data"

    badge.set_status(StatusKind.IN_PROGRESS, "Encoding patterns…")
    assert badge.text_label.text() == "Encoding patterns…"


def test_status_badge_set_text_updates_accessibility(qt_app):
    badge = StatusBadge(StatusKind.WARNING, "Encoding changed")

    badge.set_text("Threshold mismatch")
    assert "Threshold mismatch" in badge.accessibleDescription()
    assert "warning" in badge.accessibleName()
