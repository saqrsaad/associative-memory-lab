"""Presentation layer: PySide6 widgets, views, viewmodels, and theming.

This package contains all GUI code. Widgets consume application services
and structured results; they never implement numerical logic. Importing
this package stays lightweight on purpose: it defines only shared
identity metadata, while Qt-dependent subpackages (:mod:`theme`,
:mod:`widgets`, and later :mod:`views`/:mod:`viewmodels`) are imported
explicitly by the code that needs them.
"""

APPLICATION_NAME = "Associative Memory Lab"

__all__ = ["APPLICATION_NAME"]
