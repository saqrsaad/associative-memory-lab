"""Tests for the design tokens, stylesheet loader, and icon policy."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from PySide6.QtCore import QSize
from PySide6.QtGui import QIcon

from associative_memory.presentation.theme import (
    PALETTE,
    Palette,
    load_stylesheet,
    stylesheet_tokens,
)
from associative_memory.presentation.theme.icons import status_icon
from associative_memory.presentation.theme.spacing import spacing_scale
from associative_memory.presentation.theme.typography import type_scale

_HEX_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")


def _relative_luminance(hex_color: str) -> float:
    """WCAG relative luminance of an sRGB ``#RRGGBB`` color."""

    digits = hex_color.lstrip("#")
    channels = [int(digits[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]
    linearized = [
        channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]
    return (
        0.2126 * linearized[0] + 0.7152 * linearized[1] + 0.0722 * linearized[2]
    )


def _contrast_ratio(first: str, second: str) -> float:
    """WCAG contrast ratio between two sRGB colors."""

    lighter = max(_relative_luminance(first), _relative_luminance(second))
    darker = min(_relative_luminance(first), _relative_luminance(second))
    return (lighter + 0.05) / (darker + 0.05)


# ---------------------------------------------------------------- palette


def test_every_palette_token_is_a_valid_hex_color():
    for name in Palette.__dataclass_fields__:
        value = getattr(PALETTE, name)
        assert _HEX_COLOR.match(value), f"{name} is not a #RRGGBB color: {value!r}"


# Accessible-contrast policy, verified numerically instead of trusting
# the palette docstring. Threshold 4.5:1 = WCAG AA for normal text.
_CONTRAST_PAIRS = [
    ("TEXT_PRIMARY", "SURFACE"),
    ("TEXT_PRIMARY", "BG_CANVAS"),
    ("TEXT_SECONDARY", "SURFACE"),
    ("TEXT_SECONDARY", "BG_CANVAS"),
    ("SURFACE", "PRIMARY"),  # default-primary button text on fill
    ("PRIMARY_TEXT", "PRIMARY_SOFT"),
    ("SUCCESS_TEXT", "SUCCESS_SOFT"),
    ("INFO_TEXT", "INFO_SOFT"),
    ("WARNING_TEXT", "WARNING_SOFT"),
    ("ERROR_TEXT", "ERROR_SOFT"),
    ("IN_PROGRESS_TEXT", "IN_PROGRESS_SOFT"),
]


@pytest.mark.parametrize("text_token,surface_token", _CONTRAST_PAIRS)
def test_text_surface_pairs_meet_wcag_aa_contrast(text_token: str, surface_token: str):
    ratio = _contrast_ratio(getattr(PALETTE, text_token), getattr(PALETTE, surface_token))
    assert ratio >= 4.5, f"{text_token} on {surface_token} has contrast {ratio:.2f}"


def test_disabled_text_is_deliberately_muted():
    # Disabled controls are exempt from WCAG AA contrast; the token is
    # explicitly lower-contrast by design. Guard that it stays below the
    # normal-text threshold so it keeps signaling "disabled".
    ratio = _contrast_ratio(PALETTE.TEXT_DISABLED, PALETTE.SURFACE)
    assert 1.0 < ratio < 4.5


# ------------------------------------------------------------- typography


def test_font_stacks_are_non_empty_and_monospaced_fallback_exists():
    from associative_memory.presentation.theme.typography import (
        MONO_FONT_FAMILY,
        MONO_FONT_STACK,
        UI_FONT_FAMILY,
        UI_FONT_STACK,
    )

    assert UI_FONT_STACK.count(",") >= 1  # a fallback chain, not one family
    assert MONO_FONT_STACK.count(",") >= 1
    assert UI_FONT_FAMILY and MONO_FONT_FAMILY


def test_type_scale_sizes_are_positive_integers():
    for name, size in type_scale().items():
        assert isinstance(size, int) and size > 0, f"{name} = {size}"


def test_spacing_scale_values_are_positive_integers():
    for name, value in spacing_scale().items():
        assert isinstance(value, int) and value > 0, f"{name} = {value}"


# ------------------------------------------------------------- stylesheet


def test_stylesheet_tokens_render_sizes_with_px_units():
    tokens = stylesheet_tokens()

    assert tokens["FONT_BODY"].endswith("px")
    assert tokens["SPACE_MD"].endswith("px")
    assert tokens["CARD_RADIUS"].endswith("px")
    assert tokens["PRIMARY"] == PALETTE.PRIMARY  # colors stay raw hex


def test_stylesheet_fully_substitutes_all_tokens():
    rendered = load_stylesheet()

    assert rendered  # non-empty
    assert "@" not in rendered, "a token placeholder was not substituted"


def test_stylesheet_contains_the_component_selectors():
    rendered = load_stylesheet()

    for selector in (
        "QFrame#MetricCard",
        "QLabel#MetricCardValue",
        "QFrame#StatusBadge",
        "QFrame#EmptyState",
        "QPushButton:focus",
        'QPushButton[variant="primary"]',
        "QToolTip",
        "QScrollArea",
        "QFrame#MatrixView",
        "QFrame#WeightMatrixView",
        "QFrame#FormulaPanel",
        "QFrame#LegendSwatch",
    ):
        assert selector in rendered, f"missing selector: {selector}"


def test_stylesheet_references_the_centralized_font_stacks():
    from associative_memory.presentation.theme.typography import (
        MONO_FONT_STACK,
        UI_FONT_STACK,
    )

    rendered = load_stylesheet()

    assert UI_FONT_STACK in rendered
    assert MONO_FONT_STACK in rendered  # numeric data stays monospaced


def test_theme_package_imports_without_qt():
    """Importing the theme must not pull Qt bindings into a process."""

    project_root = Path(__file__).resolve().parents[2]
    script = (
        "import sys\n"
        "import associative_memory.presentation.theme\n"
        "import associative_memory.presentation.theme.palette\n"
        "import associative_memory.presentation.theme.typography\n"
        "import associative_memory.presentation.theme.spacing\n"
        "loaded = [n for n in sys.modules if n.startswith('PySide6')]\n"
        "print(loaded)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=project_root,
        check=True,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert completed.stdout.strip() == "[]"


# ------------------------------------------------------------------ icons


def test_status_icons_resolve_without_files_or_network(qt_app):
    for status_value in ("success", "info", "warning", "error", "in_progress"):
        icon = status_icon(status_value)
        assert isinstance(icon, QIcon)
        assert not icon.isNull()


def test_status_icon_size_follows_the_policy(qt_app):
    size = status_icon("info").pixmap(16, 16).size()
    assert isinstance(size, QSize)
    assert size.width() == 16 and size.height() == 16


def test_unknown_status_icon_raises():
    with pytest.raises(KeyError):
        status_icon("magenta")
