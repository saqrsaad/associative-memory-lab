"""Reusable presentation widgets shared by all application views."""

from __future__ import annotations

from associative_memory.presentation.widgets.empty_state import EmptyState
from associative_memory.presentation.widgets.formula_panel import FormulaPanel
from associative_memory.presentation.widgets.image_canvas import ImageCanvas
from associative_memory.presentation.widgets.matrix_view import MatrixView
from associative_memory.presentation.widgets.metric_card import (
    NOT_CALCULATED_TEXT,
    MetricCard,
)
from associative_memory.presentation.widgets.pixel_grid import PixelGrid
from associative_memory.presentation.widgets.status import (
    STATUS_DEFAULT_TEXT,
    StatusBadge,
    StatusKind,
)
from associative_memory.presentation.widgets.vector_view import VectorView
from associative_memory.presentation.widgets.weight_matrix_view import WeightMatrixView

__all__ = [
    "NOT_CALCULATED_TEXT",
    "STATUS_DEFAULT_TEXT",
    "EmptyState",
    "FormulaPanel",
    "ImageCanvas",
    "MatrixView",
    "MetricCard",
    "PixelGrid",
    "StatusBadge",
    "StatusKind",
    "VectorView",
    "WeightMatrixView",
]

