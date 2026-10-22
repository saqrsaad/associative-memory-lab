"""Centralized color palette for the Associative Memory Lab interface.

All interface colors live here; widgets and views reference tokens from
this module (or via the stylesheet tokens in ``theme/__init__.py``) and
must not hardcode their own color strings.

Visual direction: a calm neutral background, white content surfaces, a
restrained indigo primary accent, and reserved status hues (green =
confirmed success, amber = warning, red = error, teal = in-progress,
blue = information). Dark mode is deliberately out of scope for the
initial design (see ``docs/tasks/05-design-system.md``).

Contrast policy
---------------
Text colors were chosen against their expected backgrounds and verified
with the WCAG relative-luminance formula:

* ``TEXT_PRIMARY`` on ``SURFACE``: 17.7:1 (WCAG AA AAA range).
* ``TEXT_SECONDARY`` on ``SURFACE`` / ``BG_CANVAS``: 7.6:1 / 6.9:1.
* ``PRIMARY`` as filled-button background with ``SURFACE`` text: 6.3:1.
* Status ``*_TEXT`` on the matching ``*_SOFT`` background: 4.9-6.8:1.

``TEXT_DISABLED`` is intentionally lower-contrast (2.5:1); WCAG exempts
disabled controls, and the reduced contrast communicates inactivity.
Status must never be communicated through color alone: status widgets
always pair color with an icon (see ``theme/icons.py``) and text.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Palette:
    """Design tokens for the light theme."""

    # Surfaces and canvas
    BG_CANVAS: str = "#F3F4F6"
    SURFACE: str = "#FFFFFF"
    SURFACE_ALT: str = "#F9FAFB"

    # Text
    TEXT_PRIMARY: str = "#111827"
    TEXT_SECONDARY: str = "#4B5563"
    TEXT_DISABLED: str = "#9CA3AF"

    # Borders
    BORDER: str = "#D1D5DB"
    BORDER_STRONG: str = "#9CA3AF"

    # Primary accent (indigo)
    PRIMARY: str = "#4F46E5"
    PRIMARY_HOVER: str = "#4338CA"
    PRIMARY_PRESSED: str = "#3730A3"
    PRIMARY_SOFT: str = "#EEF2FF"
    PRIMARY_TEXT: str = "#3730A3"

    # Status hues; *_TEXT sits on the matching *_SOFT background.
    SUCCESS_TEXT: str = "#166534"
    SUCCESS_SOFT: str = "#DCFCE7"

    INFO_TEXT: str = "#1D4ED8"
    INFO_SOFT: str = "#DBEAFE"

    WARNING_TEXT: str = "#92400E"
    WARNING_SOFT: str = "#FEF3C7"

    ERROR_TEXT: str = "#991B1B"
    ERROR_SOFT: str = "#FEE2E2"

    IN_PROGRESS_TEXT: str = "#0F766E"
    IN_PROGRESS_SOFT: str = "#CCFBF1"

    # Generic neutral soft background (e.g. "not calculated" placeholders).
    NEUTRAL_SOFT: str = "#E5E7EB"
    INPUT_BACKGROUND: str = "#FFFFFF"


#: The single shared palette instance for the light theme.
PALETTE = Palette()
