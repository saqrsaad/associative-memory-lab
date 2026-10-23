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

#: Subdirectory of the user's home folder holding saved experiments.
_STORAGE_DIR_NAME = ".associative_memory_lab"
_STORAGE_EXPERIMENTS_DIR_NAME = "experiments"


def default_experiments_directory() -> Path:
    """Return the default saved-experiment storage folder.

    The default is ``<home>/.associative_memory_lab/experiments``. The
    folder is created lazily by the repository when the first experiment
    is saved; nothing is written merely by calling this function.
    """

    return Path.home() / _STORAGE_DIR_NAME / _STORAGE_EXPERIMENTS_DIR_NAME


def build_default_repository() -> ModelRepository:
    """Build the default experiment repository for the application.

    The default location is :func:`default_experiments_directory`; the
    application service can wrap a different repository for tests or
    alternative storage.
    """

    return JsonModelRepository(default_experiments_directory())
