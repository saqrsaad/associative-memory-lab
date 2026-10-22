"""Centralized typography for the Associative Memory Lab interface.

Font stacks
-----------
``UI_FONT_STACK`` is a QSS-ready ``font-family`` value whose first choice
is Segoe UI: it ships with Windows, has full Arabic and Latin coverage,
and falls back to widely available fonts elsewhere. ``MONO_FONT_STACK``
always resolves to a monospaced face and is reserved for vectors,
matrices, dimensions, and numerical traces.

Arabic and English strategy (documented, per the design task):

* Interface copy may be translated to Arabic or English at any time;
  Qt layouts mirror automatically from the layout direction, with no
  per-widget hacks in this project.
* Technical content — equations, matrix notation, encoded vectors, and
  numeric values — is always rendered left-to-right in the monospaced
  stack and is never manually re-ordered to imitate right-to-left
  rendering.
* No widget hardcodes a font family; every font reference comes from
  this module.

The intentionally neutral fallback chain keeps results deterministic in
headless/offscreen test sessions, where system font enumeration may
return no families; tests therefore assert on the font stack strings,
not on render engine behavior.
"""

from __future__ import annotations

UI_FONT_FAMILY = "Segoe UI"

#: QSS-ready family list for interface text (Arabic + Latin coverage).
UI_FONT_STACK = '"Segoe UI", "Tahoma", "Noto Sans", "Helvetica Neue", Arial, sans-serif'

MONO_FONT_FAMILY = "Consolas"

#: QSS-ready family list for numerical data (vectors, matrices, traces).
MONO_FONT_STACK = '"Consolas", "Cascadia Mono", "Source Code Pro", Menlo, monospace'

#: Type scale in CSS px units, used to build the stylesheet.
FONT_CAPTION_PX = 12  # auxiliary labels, footnotes, badge text
FONT_SMALL_PX = 13  # descriptions and de-emphasized text
FONT_BODY_PX = 14  # default interface text
FONT_TITLE_PX = 18  # section titles and empty-state titles
FONT_HEADER_PX = 20  # window/header titles
FONT_METRIC_PX = 22  # metric-card values (monospaced)


def type_scale() -> dict[str, int]:
    """Return the token-size mapping used by the stylesheet loader."""

    return {
        "FONT_CAPTION": FONT_CAPTION_PX,
        "FONT_SMALL": FONT_SMALL_PX,
        "FONT_BODY": FONT_BODY_PX,
        "FONT_TITLE": FONT_TITLE_PX,
        "FONT_HEADER": FONT_HEADER_PX,
        "FONT_METRIC": FONT_METRIC_PX,
    }
