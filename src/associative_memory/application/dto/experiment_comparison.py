"""Experiment-comparison DTOs for the application layer.

A comparison is always reported side by side with explicit notes: when
stored values are unavailable or not directly comparable (different
memory types, encodings, dimensions, or preprocessing settings), the note
explains why instead of presenting incompatible numbers as equivalents.
"""

from __future__ import annotations

from dataclasses import dataclass

from associative_memory.application.dto.saved_experiment import (
    SavedExperimentSummary,
)


@dataclass(frozen=True)
class ComparisonRow:
    """One labeled two-column comparison row with an optional note.

    ``left``/``right`` are display-ready strings; missing values render
    as ``"(not stored)"`` in the presentation layer. ``note`` explains
    availability or incomparability and is empty when the row is plain.
    """

    label: str
    left: str
    right: str
    note: str = ""


@dataclass(frozen=True)
class WeightComparison:
    """Element-by-element comparison of two equal-shape weight matrices."""

    element_count: int
    differing_count: int
    max_abs_difference: float
    identical: bool


@dataclass(frozen=True)
class ExperimentComparison:
    """Structured result of comparing two saved experiments."""

    left: SavedExperimentSummary
    right: SavedExperimentSummary
    rows: tuple[ComparisonRow, ...]
    weight_comparison: WeightComparison | None
    notes: tuple[str, ...]
