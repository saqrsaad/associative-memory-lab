"""Repository interface for persisted associative-memory experiments.

The application layer defines *what* experiment persistence means
(:class:`ModelRepository`); the infrastructure layer provides *how* it is
stored (``infrastructure.model_repository_json``). Views and view models
must never touch storage files directly — they call the application-layer
:class:`ExperimentHistoryService`, which delegates to a repository
implementation behind these interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from associative_memory.application.dto.saved_experiment import (
    LoadedExperiment,
    SavedExperiment,
    SavedExperimentSummary,
)


class ExperimentRepositoryError(Exception):
    """Base error for the experiment persistence interface.

    The message describes the problem at the application level; the
    original lower-level exception remains available as ``__cause__`` for
    logging. Nothing in a failing repository call replaces stored data.
    """


class ExperimentValidationError(ExperimentRepositoryError):
    """A stored experiment is unreadable, incomplete, or inconsistent.

    Raised for corrupt JSON, missing or malformed NumPy array files,
    mismatched shapes or metadata, or values violating the stored
    encoding. Loading the experiment is refused instead of returning
    partially reconstructed numbers.
    """


class UnsupportedSchemaError(ExperimentValidationError):
    """A stored experiment uses an unknown manifest schema version.

    This happens when a file was written by a newer application version or
    by an abandoned legacy format. The application refuses the file rather
    than guessing about unknown fields.
    """


class ExperimentNotFoundError(ExperimentRepositoryError):
    """The requested experiment identifier has no stored entry."""


class ExperimentIdConflictError(ExperimentRepositoryError):
    """A save produced an identifier that already exists in storage."""


class ModelRepository(ABC):
    """Persistence operations for saved experiments.

    Implementations are responsible for validating stored content, using
    safe numeric deserialization (never pickle or arbitrary object
    loading), and performing writes so that a partially written experiment
    is never visible under a new identifier.
    """

    @abstractmethod
    def save(self, experiment: SavedExperiment) -> SavedExperimentSummary:
        """Store ``experiment`` under a fresh identifier.

        The caller-provided record must not carry an ``experiment_id``
        yet; the implementation assigns one, writes the complete entry
        atomically, and returns the resulting summary. Raises a
        repository error on failure without leaving a partial entry.
        """

        raise NotImplementedError

    @abstractmethod
    def list_experiments(self) -> tuple[SavedExperimentSummary, ...]:
        """Return every stored experiment's history-list summary.

        Invalid on-disk entries are included with ``readable=False`` and
        ``unreadable_reason`` set so they remain deletable; they never
        raise from the listing itself.
        """

        raise NotImplementedError

    @abstractmethod
    def load(self, experiment_id: str) -> LoadedExperiment:
        """Load one full experiment, or raise a repository error.

        Corrupt or unsupported entries raise
        :class:`ExperimentValidationError`; unknown identifiers raise
        :class:`ExperimentNotFoundError`.
        """

        raise NotImplementedError

    @abstractmethod
    def delete(self, experiment_id: str) -> None:
        """Remove one stored experiment, or raise a repository error."""

        raise NotImplementedError

    @property
    def storage_location(self) -> Path | None:
        """The on-disk storage location, when the implementation has one.

        Returns ``None`` for repositories without a visible folder; used
        only to inform the user, never for file access.
        """

        return None
