"""Icon policy for the Associative Memory Lab interface.

Policy
------
* Source: Qt's built-in :class:`~PySide6.QtWidgets.QStyle` standard
  icons. They ship inside Qt itself, are theme/rendering consistent on
  every supported platform, and require no local image files, bundled
  resources, or runtime network downloads (a Task 05 packaging
  requirement). Emoji are not used as interface icons.
* Size policy: icons are rendered at the sizes defined in
  ``theme/spacing.py`` (``ICON_STATUS_PX`` for badges, ``ICON_EMPTY_PX``
  for empty-state previews). Callers never pick their own sizes.
* Redundancy: a colored icon never carries status information alone —
  status widgets always pair the icon with text (color is only the
  third channel).

If a dedicated icon set is introduced later (e.g. an embedded Qt
resource file), only this module changes; call sites keep using
:func:`status_icon`.
"""

from __future__ import annotations

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QCommonStyle, QStyle

from associative_memory.presentation.theme.spacing import ICON_STATUS_PX

#: QStyle standard pixmap chosen per status. Requires no files on disk.
_STATUS_STYLE_ICONS: dict[str, QStyle.StandardPixmap] = {
    "success": QStyle.StandardPixmap.SP_DialogApplyButton,
    "info": QStyle.StandardPixmap.SP_MessageBoxInformation,
    "warning": QStyle.StandardPixmap.SP_MessageBoxWarning,
    "error": QStyle.StandardPixmap.SP_MessageBoxCritical,
    "in_progress": QStyle.StandardPixmap.SP_BrowserReload,
}


def status_icon(
    status: str,
    style: QStyle | None = None,
    *,
    size: int = ICON_STATUS_PX,
) -> QIcon:
    """Return the configured standard icon for ``status``.

    Args:
        status: One of the status keys documented in
            :mod:`associative_memory.presentation.widgets.status`
            (``success``, ``info``, ``warning``, ``error``,
            ``in_progress``).
        style: Style to resolve the icon with; defaults to a shared
            default style so testable icons exist without a widget.
        size: Requested icon size in pixels.

    Raises:
        KeyError: If ``status`` has no configured icon.
    """

    resolved_style = (
        style
        if style is not None
        else (
            QApplication.instance().style()
            if QApplication.instance() is not None
            else QCommonStyle()
        )
    )
    standard_enum = _STATUS_STYLE_ICONS[status]
    icon = resolved_style.standardIcon(standard_enum)
    if icon.isNull():
        raise RuntimeError(f"Standard icon for status {status!r} resolved to null.")
    return icon
