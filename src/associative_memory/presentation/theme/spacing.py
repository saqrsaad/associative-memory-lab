"""Shared spacing, radius, and icon-size constants.

Layout conventions for every view (all values in CSS px units):

* Page margins: ``SPACE_MD`` (24) around a workspace page.
* Between sibling cards/sections: ``SPACE_SM`` (16).
* Between a label and its control: ``SPACE_XXS`` (8).
* Inside cards and controls: ``SPACE_XXS``/``SPACE_XXXS`` (8/4).
* Corner radii: cards ``CARD_RADIUS`` (10), controls ``CONTROL_RADIUS`` (6).
* Border widths are 1 px for surfaces; controls reserve a 2 px border so
  the visible focus ring never shifts their layout.

Widgets must import these constants instead of scattering raw pixel
values; the stylesheet substitutes them through ``theme/__init__.py``.
"""

from __future__ import annotations

SPACE_XXXS = 4
SPACE_XXS = 8
SPACE_XS = 12
SPACE_SM = 16
SPACE_MD = 24
SPACE_LG = 32

CARD_RADIUS = 10
CONTROL_RADIUS = 6

BORDER_WIDTH = 1
FOCUS_BORDER_WIDTH = 2

#: Icon sizes: small status/badge glyphs and larger preview glyphs.
ICON_STATUS_PX = 16
ICON_EMPTY_PX = 32
CONTROL_MIN_HEIGHT = 32


def spacing_scale() -> dict[str, int]:
    """Return the spacing token mapping used by the stylesheet loader."""

    return {
        "SPACE_XXXS": SPACE_XXXS,
        "SPACE_XXS": SPACE_XXS,
        "SPACE_XS": SPACE_XS,
        "SPACE_SM": SPACE_SM,
        "SPACE_MD": SPACE_MD,
        "SPACE_LG": SPACE_LG,
        "CARD_RADIUS": CARD_RADIUS,
        "CONTROL_RADIUS": CONTROL_RADIUS,
        "BORDER_WIDTH": BORDER_WIDTH,
        "FOCUS_BORDER_WIDTH": FOCUS_BORDER_WIDTH,
        "CONTROL_MIN_HEIGHT": CONTROL_MIN_HEIGHT,
    }
