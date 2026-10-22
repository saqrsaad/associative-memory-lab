"""Navigation model for the application shell.

Qt-free: destinations, labels, tooltips, and ordering are plain data so
the navigation structure can be inspected and tested without a
``QApplication``. The ``main_window`` module owns the Qt wiring.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class NavigationDestination(StrEnum):
    """The application's top-level navigation destinations."""

    DASHBOARD = "dashboard"
    IMAGE_DATASET = "image_dataset"
    TRAINING = "training"
    RECALL_RESULTS = "recall_results"
    EXPERIMENT_HISTORY = "experiment_history"
    SETTINGS = "settings"


#: Destination order displayed in the sidebar navigation rail.
ALL_DESTINATIONS: tuple[NavigationDestination, ...] = (
    NavigationDestination.DASHBOARD,
    NavigationDestination.IMAGE_DATASET,
    NavigationDestination.TRAINING,
    NavigationDestination.RECALL_RESULTS,
    NavigationDestination.EXPERIMENT_HISTORY,
    NavigationDestination.SETTINGS,
)


@dataclass(frozen=True)
class NavigationItem:
    """One navigation destination and its user-facing label."""

    destination: NavigationDestination
    label: str
    tooltip: str


#: Human-readable labels and tooltips for every destination.
NAVIGATION_ITEMS: tuple[NavigationItem, ...] = (
    NavigationItem(
        NavigationDestination.DASHBOARD,
        "Dashboard",
        "Overview of datasets, models, and recent experiments.",
    ),
    NavigationItem(
        NavigationDestination.IMAGE_DATASET,
        "Image Dataset",
        "Import and organize images for training and recall.",
    ),
    NavigationItem(
        NavigationDestination.TRAINING,
        "Training",
        "Train associative memories from image datasets or vectors.",
    ),
    NavigationItem(
        NavigationDestination.RECALL_RESULTS,
        "Recall and Results",
        "Query trained memories and inspect recall results.",
    ),
    NavigationItem(
        NavigationDestination.EXPERIMENT_HISTORY,
        "Experiment History",
        "Inspect, load, and compare saved experiments.",
    ),
    NavigationItem(
        NavigationDestination.SETTINGS,
        "Settings",
        "Application preferences and about information.",
    ),
)


def item_for_destination(destination: NavigationDestination) -> NavigationItem:
    """Return the navigation item for ``destination``.

    Raises:
        KeyError: If ``destination`` is not a known navigation target.
    """

    for item in NAVIGATION_ITEMS:
        if item.destination == destination:
            return item
    raise KeyError(f"Unknown navigation destination: {destination!r}")
