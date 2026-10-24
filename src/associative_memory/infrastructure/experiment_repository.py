"""Default storage location and repository construction for experiments.

This module owns *where* saved experiments live by default and builds the
concrete repository implementation for the application service. The JSON
and NumPy implementation itself lives in
:mod:`associative_memory.infrastructure.model_repository_json`; this
module only adds the default-location policy so application code never
hardcodes a private path.
"""

from __future__ import annotations

from pathlib import Path

from associative_memory.application.interfaces.model_repository import (
    ModelRepository,
)
from associative_memory.infrastructure.model_repository_json import (
    JsonModelRepository,
)

#: Project root derived from this source file, never from the current
#: working directory. ``parents[3]`` is the repository root for
#: ``<root>/src/associative_memory/infrastructure/experiment_repository.py``.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: Project-local folder holding saved experiments.
_STORAGE_EXPERIMENTS_DIR_NAME = "experiments"


def default_experiments_directory() -> Path:
    """Return the default saved-experiment storage folder.

    The default is ``<project-root>/experiments``. The project root is
    resolved from this module's location, so launching the application
    from another working directory does not change the storage location.
    The folder is created lazily by the repository when the first
    experiment is saved; nothing is written merely by calling this
    function.
    """

    return _PROJECT_ROOT / _STORAGE_EXPERIMENTS_DIR_NAME


def build_default_repository() -> ModelRepository:
    """Build the default experiment repository for the application.

    The default location is :func:`default_experiments_directory`; the
    application service can wrap a different repository for tests or
    alternative storage.
    """

    return JsonModelRepository(default_experiments_directory())
