"""Application entry point for the Associative Memory Lab.

Run with ``python -m associative_memory.main``. The import of the
presentation layer happens inside :func:`main` so that importing this
module in tests or tooling stays free of any Qt side effects.
"""

from __future__ import annotations


def main(args: list[str] | None = None) -> int:
    """Bootstrap the Qt application, show the main window, and run it."""

    from associative_memory.presentation.app import run_application

    return run_application(args)


if __name__ == "__main__":
    raise SystemExit(main())
