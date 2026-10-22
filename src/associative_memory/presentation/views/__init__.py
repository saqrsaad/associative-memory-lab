"""Shared helpers for presentation pages."""

from __future__ import annotations

from PySide6.QtWidgets import QVBoxLayout, QWidget

from associative_memory.presentation.widgets.empty_state import EmptyState


def placeholder_page(
    title: str,
    description: str,
    parent: QWidget | None = None,
) -> QWidget:
    """Build a clearly-labeled placeholder page for an unimplemented destination.

    The returned widget owns a layout containing an :class:`EmptyState`
    that explains the destination is not yet implemented. Callers add the
    widget to their own page layout. The page never pretends that the
    destination's functionality exists.
    """

    page = QWidget(parent)
    layout = QVBoxLayout(page)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    layout.addWidget(EmptyState(title, description, parent=page))
    return page
