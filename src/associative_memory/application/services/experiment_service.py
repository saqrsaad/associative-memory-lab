"""Experiment orchestration for the application layer.

This service coordinates the application services (image processing,
training, recall) into one complete workflow and returns an
:class:`ExperimentReport` describing the whole run.

Error-handling policy
---------------------
``TrainingError`` and ``RecallError`` are application-level errors and
propagate unchanged. Infrastructure failures that are expected during image
experiments (``ImageLoadError``) are converted into
:class:`ExperimentError` while the original exception stays available as
``__cause__`` for logging. Unexpected exceptions are never converted into
fabricated results.

Reproducibility
---------------
Both experiment methods place enough metadata in the report (data source,
pair count, and — for image experiments — the complete preprocessing
configuration) to reproduce the run in a later task, as required by the
preprocessing persistence rules.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from associative_memory.algorithms.encoding import Encoding
from associative_memory.application.dto.experiment_report import (
    DATA_SOURCE_IMAGES,
    DATA_SOURCE_VECTORS,
    STATUS_COMPLETED,
    ExperimentReport,
)
from associative_memory.application.dto.training_result import (
    MemoryType,
    TrainingResult,
)
from associative_memory.application.services.image_service import ImageService
from associative_memory.application.services.recall_service import RecallService
from associative_memory.application.services.training_service import (
    TrainingDataset,
    TrainingService,
)
from associative_memory.infrastructure.image_loader import ImageLoadError
from associative_memory.infrastructure.image_preprocessor import (
    PreprocessingConfig,
)

_SupportedPath = str | Path
_MEMORY_TYPE_OPTIONS = ", ".join(member.value for member in MemoryType)


class ExperimentError(Exception):
    """Raised when an experiment use case cannot be prepared or executed.

    The message is user-facing; the original lower-level exception remains
    available as ``__cause__`` for diagnostic logging.
    """


class ExperimentService:
    """Coordinate complete training-and-recall experiments."""

    def __init__(
        self,
        image_service: ImageService | None = None,
        training_service: TrainingService | None = None,
        recall_service: RecallService | None = None,
    ) -> None:
        self._image_service = image_service if image_service is not None else ImageService()
        self._training_service = (
            training_service if training_service is not None else TrainingService()
        )
        self._recall_service = (
            recall_service if recall_service is not None else RecallService()
        )

    def run_vector_experiment(
        self,
        *,
        training_dataset: TrainingDataset,
        encoding: Encoding | str,
        queries: Sequence[object],
        expected_targets: Sequence[object] | None = None,
        name: str | None = None,
        trace: bool = False,
    ) -> ExperimentReport:
        """Train from ``training_dataset`` and recall every vector query.

        Args:
            training_dataset: Dataset built with
                :meth:`TrainingDataset.from_patterns` (autoassociative) or
                :meth:`TrainingDataset.from_pairs` (heteroassociative).
            encoding: Encoding used for training and recall validation.
            queries: One-dimensional query patterns.
            expected_targets: Optional expected pattern per query; its
                length must match ``queries``. Missing entries perform no
                comparison instead of manufacturing a score.
            name: Optional experiment name carried into the report.
            trace: When ``True``, individual outer-product contributions
                are kept on the training result.

        Returns:
            A completed :class:`ExperimentReport`.
        """

        expected_list = self._expectation_list(queries, expected_targets)

        trained = self._training_service.train(
            training_dataset,
            encoding=encoding,
            trace=trace,
        )

        recall_results = tuple(
            self._recall_service.recall(trained.memory, query, expected_pattern=target)
            for query, target in zip(queries, expected_list)
        )

        return ExperimentReport(
            name=name,
            status=STATUS_COMPLETED,
            memory_type=trained.memory_type,
            encoding=trained.encoding,
            data_source=DATA_SOURCE_VECTORS,
            training=trained,
            recall_results=recall_results,
            metadata=self._base_metadata(trained, training_dataset.name),
        )

    def run_image_experiment(
        self,
        *,
        preprocessing_config: PreprocessingConfig,
        training_sources: Sequence[_SupportedPath],
        training_targets: Sequence[_SupportedPath] | None = None,
        queries: Sequence[_SupportedPath] | None = None,
        query_targets: Sequence[_SupportedPath] | None = None,
        memory_type: MemoryType | str | None = None,
        name: str | None = None,
        trace: bool = False,
    ) -> ExperimentReport:
        """Train and recall from image files through the image service.

        All images are preprocessed with the same configuration, so
        encoding, dimensions, and flattening are consistent by construction
        (the training uses ``preprocessing_config.encoding``). Source image
        files are used as training patterns/targets; with ``queries=None``
        every training source itself is recalled afterward.

        Args:
            preprocessing_config: Explicit preprocessing configuration.
            training_sources: Image files used as sources (heteroassociative)
                or as the stored patterns themselves (autoassociative).
            training_targets: Image files paired with the sources; required
                for heteroassociative memories and forbidden otherwise.
            queries: Image files to recall; defaults to the sources.
            query_targets: Optional expected image per query.
            memory_type: Optional restatement of the requested memory mode;
                must match the supplied data.
            name: Optional experiment name carried into the report.
            trace: When ``True``, individual outer-product contributions
                are kept on the training result.

        Returns:
            A completed :class:`ExperimentReport` whose metadata contains
            the full preprocessing configuration.
        """

        requested_memory_type = self._coerce_memory_type(memory_type)
        inferred_memory_type = (
            MemoryType.HETEROASSOCIATIVE
            if training_targets is not None
            else MemoryType.AUTOASSOCIATIVE
        )
        if (
            requested_memory_type is not None
            and requested_memory_type is not inferred_memory_type
        ):
            raise ExperimentError(
                f"Requested memory type {requested_memory_type.value!r} does "
                f"not match the supplied data "
                f"({inferred_memory_type.value!r})."
            )

        source_vectors = self._image_vectors(training_sources, preprocessing_config)
        if training_targets is not None:
            target_vectors = self._image_vectors(training_targets, preprocessing_config)
            if len(target_vectors) != len(source_vectors):
                raise ExperimentError(
                    "Heteroassociative experiments require one target image "
                    f"for each source image, but {len(source_vectors)} sources "
                    f"were paired with {len(target_vectors)} targets."
                )
            dataset_name = name
            dataset = TrainingDataset.from_pairs(
                zip(source_vectors, target_vectors, strict=True), name=dataset_name
            )
        else:
            dataset_name = name
            dataset = TrainingDataset.from_patterns(source_vectors, name=dataset_name)

        trained = self._training_service.train(
            dataset,
            encoding=preprocessing_config.encoding,
            trace=trace,
        )

        query_paths = (
            training_sources if queries is None else tuple(queries)
        )
        query_vectors = self._image_vectors(query_paths, preprocessing_config)
        query_target_vectors: list[object] | None
        if query_targets is None:
            query_target_vectors = [None] * len(query_vectors)
        else:
            query_target_vectors = list(
                self._image_vectors(
                    tuple(query_targets), preprocessing_config
                )
            )
            if len(query_target_vectors) != len(query_vectors):
                raise ExperimentError(
                    f"query_targets must supply one expected image per query "
                    f"({len(query_vectors)} queries, "
                    f"{len(query_target_vectors)} targets)."
                )

        recall_results = tuple(
            self._recall_service.recall(
                trained.memory, query, expected_pattern=target
            )
            for query, target in zip(query_vectors, query_target_vectors)
        )

        metadata = self._base_metadata(
            trained, dataset_name, data_source=DATA_SOURCE_IMAGES
        )
        metadata.update(
            {
                "image_width": str(preprocessing_config.width),
                "image_height": str(preprocessing_config.height),
                "image_threshold": str(preprocessing_config.threshold),
                "image_polarity": preprocessing_config.polarity.value,
                "image_encoding": preprocessing_config.encoding.value,
                "image_resize_mode": preprocessing_config.resize_mode.value,
                "image_flattening_order": preprocessing_config.flattening_order.value,
                "image_grayscale_method": preprocessing_config.grayscale_method.value,
                "image_resampling": preprocessing_config.resampling.name,
            }
        )

        return ExperimentReport(
            name=name,
            status=STATUS_COMPLETED,
            memory_type=trained.memory_type,
            encoding=trained.encoding,
            data_source=DATA_SOURCE_IMAGES,
            training=trained,
            recall_results=recall_results,
            metadata=metadata,
        )

    def _image_vectors(
        self,
        paths: Sequence[_SupportedPath],
        preprocessing_config: PreprocessingConfig,
    ) -> tuple[object, ...]:
        """Preprocess every image file through the image service.

        ``ImageLoadError``, operating-system file errors (e.g. a missing or
        unreadable file), and configuration misuse are converted into
        :class:`ExperimentError`; the original exception is retained as
        ``__cause__`` for logging.
        """

        vectors = []
        for path in paths:
            try:
                preprocessed = self._image_service.preprocess_file(
                    path, preprocessing_config
                )
            except ImageLoadError as exc:
                raise ExperimentError(
                    f"Training or recall image could not be preprocessed: {exc}"
                ) from exc
            except (OSError, TypeError, ValueError) as exc:
                raise ExperimentError(
                    f"Image file or preprocessing configuration could not be "
                    f"used: {exc}"
                ) from exc
            vectors.append(preprocessed.encoded_vector)
        return tuple(vectors)

    def _expectation_list(
        self,
        queries: Sequence[object],
        expected_targets: Sequence[object] | None,
    ) -> list[object]:
        """Resolve the expected target per query, or ``None`` per query."""

        if expected_targets is None:
            return [None] * len(queries)
        targets = list(expected_targets)
        if len(targets) != len(queries):
            raise ExperimentError(
                f"expected_targets must supply one expected pattern per query "
                f"({len(queries)} queries, {len(targets)} targets)."
            )
        return targets

    def _base_metadata(
        self,
        trained: TrainingResult,
        dataset_name: str | None,
        *,
        data_source: str = DATA_SOURCE_VECTORS,
    ) -> dict[str, str]:
        """Build the reproducibility metadata shared by both methods."""

        metadata: dict[str, str] = {
            "data_source": data_source,
            "memory_type": trained.memory_type.value,
            "encoding": trained.encoding.value,
            "pair_count": str(trained.pair_count),
            "input_dimension": str(trained.input_dimension),
            "output_dimension": str(trained.output_dimension),
        }
        if dataset_name:
            metadata["dataset_name"] = dataset_name
        return metadata

    def _coerce_memory_type(
        self,
        value: MemoryType | str | None,
    ) -> MemoryType | None:
        """Return the :class:`MemoryType` member for ``value`` or ``None``."""

        if value is None:
            return None
        if isinstance(value, MemoryType):
            return value
        if isinstance(value, str):
            try:
                return MemoryType(value)
            except ValueError:
                pass
        raise ExperimentError(
            f"Unknown memory type {value!r}; expected one of: "
            f"{_MEMORY_TYPE_OPTIONS}."
        )
