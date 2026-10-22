"""Theme package: centralized colors, typography, spacing, and stylesheet.

Architecture
------------
Design tokens are defined once in :mod:`palette`, :mod:`typography`, and
:mod:`spacing`. ``styles.qss`` is a template; :func:`load_stylesheet`
substitutes ``@TOKEN@`` placeholders from the token modules at load time,
so views never embed styles and no value is defined twice. The loader
raises when a placeholder is left unsubstituted, catching token typos
early.

Creating views
--------------
Widgets never draw their own colors or fonts. They use layout constants
from :mod:`spacing`, reference fonts via :mod:`typography`, and get
their visual identity from object names / dynamic ``property`` selectors
matched by ``styles.qss`` (e.g. ``QFrame#MetricCard``).
The Qt-dependent icon policy lives in :mod:`icons` and is imported by
widgets directly; importing this package stays Qt-free.
"""

from __future__ import annotations

import re
from importlib import resources

from associative_memory.presentation.theme.palette import PALETTE, Palette
from associative_memory.presentation.theme.spacing import spacing_scale
from associative_memory.presentation.theme.typography import (
    MONO_FONT_STACK,
    UI_FONT_STACK,
    type_scale,
)

__all__ = [
    "PALETTE",
    "Palette",
    "load_stylesheet",
    "stylesheet_tokens",
]


_TRAILING_TOKEN = re.compile(r"@([A-Z0-9_]+)@")


def stylesheet_tokens() -> dict[str, str]:
    """Return every stylesheet token as an already-rendered string.

    Size tokens (fonts, spacing, radii, borders) are strings ending in
    ``px``; color and font tokens are used verbatim.
    """

    tokens: dict[str, str] = {
        name: getattr(PALETTE, name) for name in Palette.__dataclass_fields__
    }
    for name, size in type_scale().items():
        tokens[name] = f"{size}px"

    for name, size in spacing_scale().items():
        tokens[name] = f"{size}px"

    tokens["UI_FONT_STACK"] = UI_FONT_STACK
    tokens["MONO_FONT_STACK"] = MONO_FONT_STACK
    return tokens


def load_stylesheet() -> str:
    """Load ``styles.qss`` and substitute every design token.

    Returns:
        The fully rendered Qt stylesheet.

    Raises:
        RuntimeError: If a ``@TOKEN@`` placeholder has no matching token
            (e.g. a typo in the template or a missing constant).
    """

    template = (
        resources.files("associative_memory.presentation.theme")
        .joinpath("styles.qss")
        .read_text(encoding="utf-8")
    )
    tokens = stylesheet_tokens()

    rendered = template
    for name, value in tokens.items():
        rendered = rendered.replace(f"@{name}@", value)

    unsubstituted = sorted(set(_TRAILING_TOKEN.findall(rendered)))
    if unsubstituted:
        raise RuntimeError(
            "Stylesheet contains unsubstituted tokens (missing or mispelled "
            f"design tokens): {', '.join(unsubstituted)}."
        )
    return rendered
