"""JSON + NumPy ``.npz`` implementation of the experiment repository.

Storage layout
--------------
One directory per experiment, under a caller-supplied root directory:

    <root>/<experiment_id>/manifest.json   # readable, versioned metadata
    <root>/<experiment_id>/arrays.npz      # float64 numerical arrays

``experiment_id`` is the directory name: a 32-character lowercase hex
UUID. ``manifest.json`` lists every expected array key with its shape, so
``arrays.npz`` can be validated against the manifest before use.

Numerical fidelity
------------------
All arrays are stored as ``float64`` and are read back without rounding.

Security properties
-------------------
* Arrays are read with ``np.load(..., allow_pickle=False)``, so no
  arbitrary Python object serialization (pickle) is ever executed by a
  stored file.
* The manifest is plain JSON validated field-by-field; loading it cannot
  execute code, and unknown or future schema versions are rejected.
* Saved files contain no executable content: reloading an experiment
  reconstructs a memory object from numbers only.

Safe, atomic writes
-------------------
A save writes both files into a hidden temporary directory inside the
same root and then atomically renames that directory to its final
experiment id. If any write fails, the temporary directory is removed and
the final id is never visible, so no partially written experiment can be
observed or later loaded. An id conflict is detected before writing.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from associative_memory.algorithms.activation import activation_name
from associative_memory.algorithms.autoassociative import AutoassociativeMemory
from associative_memory.algorithms.encoding import Encoding
from associative_memory.algorithms.heteroassociative import HeteroassociativeMemory
from associative_memory.application.dto.saved_experiment import (
    APPLICATION_VERSION,
    DEFAULT_EXPERIMENT_NAME,
    MAX_EXPERIMENT_NAME_LENGTH,
    SCHEMA_VERSION,
    STORED_COMPARISON_BASES,
    LoadedExperiment,
    PreprocessingSettings,
    SavedExperiment,
    SavedExperimentSummary,
    StoredComparison,
    StoredRecallRecord,
    StoredTraining,
)
from associative_memory.application.dto.training_result import (
    MemoryType,
    TrainingResult,
)
from associative_memory.application.interfaces.model_repository import (
    ExperimentIdConflictError,
    ExperimentNotFoundError,
    ExperimentRepositoryError,
    ExperimentValidationError,
    ModelRepository,
    UnsupportedSchemaError,
)
from associative_memory.infrastructure.image_preprocessor import (
    FlatteningOrder,
    ForegroundPolarity,
    GrayscaleMethod,
    ResizeMode,
)

_FILE_NAME_MANIFEST = "manifest.json"
_FILE_NAME_ARRAYS = "arrays.npz"
_KEY_WEIGHT = "weight_matrix"
_KEY_RECALL_INPUT = "recall_input"
_KEY_RECALL_NET = "recall_net_input"
_KEY_RECALL_OUTPUT = "recall_output"
_KEY_RECALL_EXPECTED = "recall_expected"
_KEY_CONTRIBUTION_PREFIX = "contribution_"

_MEMORY_TYPE_VALUES = tuple(member.value for member in MemoryType)
_ENCODING_VALUES = tuple(member.value for member in Encoding)
_GRAYSCALE_VALUES = tuple(member.value for member in GrayscaleMethod)
_POLARITY_VALUES = tuple(member.value for member in ForegroundPolarity)
_RESIZE_VALUES = tuple(member.value for member in ResizeMode)
_FLATTENING_VALUES = tuple(member.value for member in FlatteningOrder)
_RESAMPLING_NAMES = frozenset(
    member.name for member in Image.Resampling
)

#: Comparison metrics whose redefinition would signal corruption.
_MATCH_RATIO_TOLERANCE = 1e-9

_EXPERIMENT_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_TEMP_DIR_PREFIX = ".save-in-progress-"


class JsonModelRepository(ModelRepository):
    """Experiment repository storing JSON manifests plus ``.npz`` arrays."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        """The directory containing one subdirectory per experiment."""

        return self._root

    @property
    def storage_location(self) -> Path:
        """The on-disk storage location shown to the user."""

        return self._root

    # ------------------------------------------------------------------ save

    def save(self, experiment: SavedExperiment) -> SavedExperimentSummary:
        """Store ``experiment`` under a new identifier (see class docs)."""

        _validate_saved_structure(experiment)
        if experiment.experiment_id is not None:
            raise ExperimentIdConflictError(
                "The experiment record already carries an identifier; "
                "saving it again would overwrite an existing entry."
            )

        self._ensure_root()
        experiment_id = uuid4().hex
        final_dir = self._root / experiment_id
        if final_dir.exists():
            raise ExperimentIdConflictError(
                f"An experiment with id {experiment_id!r} already exists."
            )

        created_at = datetime.now(timezone.utc).isoformat()
        data = _ManifestData(
            schema_version=SCHEMA_VERSION,
            experiment_id=experiment_id,
            name=experiment.name,
            created_at=created_at,
            application_version=APPLICATION_VERSION,
            training=experiment.training,
            preprocessing=experiment.preprocessing,
            target_preprocessing=experiment.target_preprocessing,
            recall=experiment.recall,
            arrays=_planned_array_shapes(experiment),
        )
        manifest = _compose_manifest(data)

        temp_dir = self._root / f"{_TEMP_DIR_PREFIX}{uuid4().hex}"
        try:
            temp_dir.mkdir(parents=True)
            payload = _compose_npz_payload(experiment)
            npz_path = temp_dir / _FILE_NAME_ARRAYS
            with npz_path.open("wb") as handle:
                np.savez_compressed(handle, **payload)
            manifest_path = temp_dir / _FILE_NAME_MANIFEST
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(temp_dir, final_dir)
        except ExperimentIdConflictError:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        except FileExistsError as exc:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise ExperimentIdConflictError(
                "An experiment with this id already exists (concurrent "
                "save); nothing was overwritten."
            ) from exc
        except Exception as exc:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise ExperimentRepositoryError(
                f"Saving the experiment failed: {exc}"
            ) from exc

        return _summary_from_data(data)

    # ------------------------------------------------------------------ listing

    def list_experiments(self) -> tuple[SavedExperimentSummary, ...]:
        """Return every stored experiment summary, newest first.

        Unreadable directories are described with ``readable=False`` and
        sorted last so they remain visible and deletable.
        """

        if not self._root.is_dir():
            return ()

        readable: list[SavedExperimentSummary] = []
        unreadable: list[SavedExperimentSummary] = []
        for entry in sorted(self._root.iterdir(), key=lambda path: path.name):
            if not entry.is_dir() or not _is_experiment_id(entry.name):
                continue
            try:
                data = self._read_entry(entry)
            except ExperimentRepositoryError as exc:
                unreadable.append(
                    _unreadable_summary(entry.name, str(exc))
                )
            else:
                readable.append(_summary_from_data(data))

        readable.sort(
            key=lambda summary: (summary.created_at or "", summary.experiment_id),
            reverse=True,
        )
        return tuple(readable + unreadable)

    # ------------------------------------------------------------------ load

    def load(self, experiment_id: str) -> LoadedExperiment:
        """Load one full experiment (validated; see class docs)."""

        resolved_id = _require_experiment_id(experiment_id)
        directory = self._root / resolved_id
        if not directory.is_dir():
            raise ExperimentNotFoundError(
                f"Unknown experiment {resolved_id!r}."
            )

        data = self._read_entry(directory)
        arrays = _load_arrays(directory, data.arrays)
        arrays = _validate_loaded_arrays(data, arrays)

        saved = SavedExperiment(
            training=data.training,
            name=data.name,
            preprocessing=data.preprocessing,
            target_preprocessing=data.target_preprocessing,
            recall=data.recall,
            weight_matrix=arrays[_KEY_WEIGHT],
            contributions=_extract_contributions(data, arrays),
            recall_input=arrays.get(_KEY_RECALL_INPUT),
            recall_net_input=arrays.get(_KEY_RECALL_NET),
            recall_output=arrays.get(_KEY_RECALL_OUTPUT),
            recall_expected=arrays.get(_KEY_RECALL_EXPECTED),
            experiment_id=resolved_id,
        )
        training_result = _rebuild_training_result(saved)
        return LoadedExperiment(
            summary=_summary_from_data(data),
            name=data.name,
            training=training_result,
            preprocessing=data.preprocessing,
            target_preprocessing=data.target_preprocessing,
            recall_record=data.recall,
        )

    # ------------------------------------------------------------------ delete

    def delete(self, experiment_id: str) -> None:
        """Remove the stored experiment ``experiment_id``."""

        resolved_id = _require_experiment_id(experiment_id)
        directory = self._root / resolved_id
        if not directory.is_dir():
            raise ExperimentNotFoundError(
                f"Unknown experiment {resolved_id!r}."
            )
        try:
            shutil.rmtree(directory)
        except OSError as exc:
            raise ExperimentRepositoryError(
                f"Deleting experiment {resolved_id!r} failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------ internals

    def _ensure_root(self) -> None:
        try:
            self._root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ExperimentRepositoryError(
                f"The experiment storage folder could not be created: {exc}"
            ) from exc

    def _read_entry(self, directory: Path) -> _ManifestData:
        manifest_path = directory / _FILE_NAME_MANIFEST
        if not manifest_path.is_file():
            raise ExperimentValidationError(
                "manifest.json is missing; the experiment entry is incomplete."
            )
        try:
            text = manifest_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ExperimentValidationError(
                f"manifest.json could not be read: {exc}"
            ) from exc
        try:
            document = json.loads(text)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ExperimentValidationError(
                f"manifest.json is not valid JSON: {exc}"
            ) from exc
        return _parse_manifest(document, expected_id=directory.name)


@dataclass(frozen=True)
class _ManifestData:
    """Typed view of one validated manifest document."""

    schema_version: int
    experiment_id: str
    name: str
    created_at: str
    application_version: str
    training: StoredTraining
    preprocessing: PreprocessingSettings | None
    target_preprocessing: PreprocessingSettings | None
    recall: StoredRecallRecord | None
    arrays: dict[str, tuple[int, ...]]


# --------------------------------------------------------------------------
# Manifest composing and validation


def _compose_manifest(data: _ManifestData) -> dict[str, Any]:
    """Build the JSON manifest for ``data`` (schema version 1)."""

    training = data.training
    manifest: dict[str, Any] = {
        "schema_version": data.schema_version,
        "experiment_id": data.experiment_id,
        "name": data.name,
        "created_at": data.created_at,
        "application_version": data.application_version,
        "training": {
            "memory_type": training.memory_type.value,
            "encoding": training.encoding.value,
            "activation_name": training.activation_name,
            "threshold": float(training.threshold),
            "input_dimension": training.input_dimension,
            "output_dimension": training.output_dimension,
            "pair_count": training.pair_count,
            "dataset_name": training.dataset_name,
            "contribution_count": training.contribution_count,
        },
        "preprocessing": (
            _settings_to_manifest(data.preprocessing)
            if data.preprocessing is not None
            else None
        ),
        "target_preprocessing": (
            _settings_to_manifest(data.target_preprocessing)
            if data.target_preprocessing is not None
            else None
        ),
        "recall": (
            {
                "comparison_basis": data.recall.comparison_basis,
                "warnings": list(data.recall.warnings),
                "input_source_file": data.recall.input_source_file,
                "target_source_file": data.recall.target_source_file,
                "comparison": (
                    {
                        "element_count": data.recall.comparison.element_count,
                        "mismatch_count": data.recall.comparison.mismatch_count,
                        "match_ratio": float(
                            data.recall.comparison.match_ratio
                        ),
                        "exact_match": data.recall.comparison.exact_match,
                    }
                    if data.recall.comparison is not None
                    else None
                ),
            }
            if data.recall is not None
            else None
        ),
        "arrays": {
            key: {"shape": list(shape), "dtype": "float64"}
            for key, shape in data.arrays.items()
        },
    }
    return manifest


def _settings_to_manifest(settings: PreprocessingSettings) -> dict[str, Any]:
    """Serialize preprocessing settings for the manifest."""

    return {
        "width": settings.width,
        "height": settings.height,
        "grayscale_method": settings.grayscale_method,
        "resampling": settings.resampling,
        "threshold": settings.threshold,
        "polarity": settings.polarity,
        "encoding": settings.encoding.value,
        "resize_mode": settings.resize_mode,
        "flattening_order": settings.flattening_order,
    }


def _settings_from_manifest(value: object, where: str) -> PreprocessingSettings:
    """Rebuild :class:`PreprocessingSettings` from a manifest mapping."""

    mapping = _require_mapping(value, where)
    resampling_name = _require_str(mapping.get("resampling"), f"{where}.resampling")
    if resampling_name not in _RESAMPLING_NAMES:
        raise ExperimentValidationError(
            f"{where}.resampling {resampling_name!r} is not a supported "
            "Pillow resampling filter."
        )
    encoding_value = _require_str(mapping.get("encoding"), f"{where}.encoding")
    if encoding_value not in _ENCODING_VALUES:
        raise ExperimentValidationError(
            f"{where}.encoding {encoding_value!r} is unsupported; expected "
            f"one of {', '.join(_ENCODING_VALUES)}."
        )
    grayscale_value = _require_str(
        mapping.get("grayscale_method"), f"{where}.grayscale_method"
    )
    if grayscale_value not in _GRAYSCALE_VALUES:
        raise ExperimentValidationError(
            f"{where}.grayscale_method {grayscale_value!r} is unsupported."
        )
    polarity_value = _require_str(mapping.get("polarity"), f"{where}.polarity")
    if polarity_value not in _POLARITY_VALUES:
        raise ExperimentValidationError(
            f"{where}.polarity {polarity_value!r} is unsupported."
        )
    resize_value = _require_str(mapping.get("resize_mode"), f"{where}.resize_mode")
    if resize_value not in _RESIZE_VALUES:
        raise ExperimentValidationError(
            f"{where}.resize_mode {resize_value!r} is unsupported."
        )
    flattening_value = _require_str(
        mapping.get("flattening_order"), f"{where}.flattening_order"
    )
    if flattening_value not in _FLATTENING_VALUES:
        raise ExperimentValidationError(
            f"{where}.flattening_order {flattening_value!r} is unsupported."
        )
    width = _require_positive_int(mapping.get("width"), f"{where}.width")
    height = _require_positive_int(mapping.get("height"), f"{where}.height")
    threshold = _require_int(mapping.get("threshold"), f"{where}.threshold")
    try:
        settings = PreprocessingSettings(
            width=width,
            height=height,
            grayscale_method=grayscale_value,
            resampling=resampling_name,
            threshold=threshold,
            polarity=polarity_value,
            encoding=Encoding(encoding_value),
            resize_mode=resize_value,
            flattening_order=flattening_value,
        )
    except ValueError as exc:
        raise ExperimentValidationError(
            f"{where} is not a valid preprocessing configuration: {exc}"
        ) from exc
    try:
        # Verify against the authoritative infrastructure validator so a
        # loaded configuration is always exactly rebuildable (this also
        # enforces documented ranges such as threshold in [0, 255]).
        settings.to_config()
    except (TypeError, ValueError) as exc:
        raise ExperimentValidationError(
            f"{where} does not describe a rebuildable preprocessing "
            f"configuration: {exc}"
        ) from exc
    return settings


def _parse_manifest(document: object, *, expected_id: str) -> _ManifestData:
    """Validate a manifest document field-by-field and return typed data."""

    mapping = _require_mapping(document, "manifest")
    schema_version = _require_int(
        mapping.get("schema_version"), "manifest.schema_version"
    )
    if schema_version > SCHEMA_VERSION:
        raise UnsupportedSchemaError(
            f"The experiment uses schema version {schema_version}, which is "
            f"newer than the version this application supports "
            f"({SCHEMA_VERSION}); it was probably created with a newer "
            "application release. The stored file was left unchanged."
        )
    if schema_version != SCHEMA_VERSION:
        raise UnsupportedSchemaError(
            f"The experiment uses unsupported schema version "
            f"{schema_version}; this application supports version "
            f"{SCHEMA_VERSION}. The stored file was left unchanged."
        )

    experiment_id = _require_str(
        mapping.get("experiment_id"), "manifest.experiment_id"
    )
    if experiment_id != expected_id:
        raise ExperimentValidationError(
            f"manifest.experiment_id {experiment_id!r} does not match its "
            f"folder {expected_id!r}."
        )
    name = _require_experiment_name(mapping.get("name"), "manifest.name")
    created_at = _require_iso_timestamp(
        mapping.get("created_at"), "manifest.created_at"
    )
    application_version = _require_str(
        mapping.get("application_version"), "manifest.application_version"
    )

    training = _parse_training(mapping.get("training"), "manifest.training")
    preprocessing = mapping.get("preprocessing")
    if preprocessing is not None:
        preprocessing = _settings_from_manifest(
            preprocessing, "manifest.preprocessing"
        )
    target_preprocessing = mapping.get("target_preprocessing")
    if target_preprocessing is not None:
        target_preprocessing = _settings_from_manifest(
            target_preprocessing, "manifest.target_preprocessing"
        )
    recall_raw = mapping.get("recall")
    recall = (
        None if recall_raw is None else _parse_recall(recall_raw, "manifest.recall")
    )

    arrays_block = _require_mapping(mapping.get("arrays"), "manifest.arrays")
    plan = _planned_array_shapes_for(training, recall)
    arrays = _parse_array_descriptions(arrays_block, plan)

    return _ManifestData(
        schema_version=schema_version,
        experiment_id=experiment_id,
        name=name,
        created_at=created_at,
        application_version=application_version,
        training=training,
        preprocessing=preprocessing,
        target_preprocessing=target_preprocessing,
        recall=recall,
        arrays=arrays,
    )


def _parse_training(
    value: object, where: str
) -> StoredTraining:
    """Validate the manifest's training block into :class:`StoredTraining`."""

    mapping = _require_mapping(value, where)
    memory_type_value = _require_str(
        mapping.get("memory_type"), f"{where}.memory_type"
    )
    try:
        memory_type = MemoryType(memory_type_value)
    except ValueError as exc:
        raise ExperimentValidationError(
            f"{where}.memory_type {memory_type_value!r} is unsupported; "
            f"expected one of {', '.join(_MEMORY_TYPE_VALUES)}."
        ) from exc

    encoding_value = _require_str(mapping.get("encoding"), f"{where}.encoding")
    if encoding_value not in _ENCODING_VALUES:
        raise ExperimentValidationError(
            f"{where}.encoding {encoding_value!r} is unsupported; expected "
            f"one of {', '.join(_ENCODING_VALUES)}."
        )

    activation = _require_str(
        mapping.get("activation_name"), f"{where}.activation_name"
    )
    expected_activation = activation_name(Encoding(encoding_value))
    if activation != expected_activation:
        raise ExperimentValidationError(
            f"{where}.activation_name {activation!r} does not match the "
            f"stored encoding {encoding_value!r} (expected "
            f"{expected_activation!r})."
        )

    threshold = _require_finite_float(mapping.get("threshold"), f"{where}.threshold")
    input_dim = _require_positive_int(mapping.get("input_dimension"),
                                       f"{where}.input_dimension")
    output_dim = _require_positive_int(mapping.get("output_dimension"),
                                        f"{where}.output_dimension")
    pair_count = _require_positive_int(mapping.get("pair_count"),
                                        f"{where}.pair_count")

    dataset_name = mapping.get("dataset_name")
    if dataset_name is not None:
        dataset_name = _require_str(dataset_name, f"{where}.dataset_name")

    contribution_count = mapping.get("contribution_count")
    if contribution_count is not None:
        contribution_count = _require_non_negative_int(
            contribution_count, f"{where}.contribution_count"
        )

    return StoredTraining(
        memory_type=memory_type,
        encoding=Encoding(encoding_value),
        activation_name=activation,
        threshold=threshold,
        input_dimension=input_dim,
        output_dimension=output_dim,
        pair_count=pair_count,
        dataset_name=dataset_name,
        contribution_count=contribution_count,
    )


def _parse_recall(value: object, where: str) -> StoredRecallRecord:
    """Validate the manifest's recall block (if present)."""

    mapping = _require_mapping(value, where)
    basis = mapping.get("comparison_basis")
    if basis is not None:
        basis = _require_str(basis, f"{where}.comparison_basis")
        if basis not in STORED_COMPARISON_BASES:
            raise ExperimentValidationError(
                f"{where}.comparison_basis {basis!r} is unsupported."
            )

    warnings_raw = mapping.get("warnings")
    if not isinstance(warnings_raw, list):
        raise ExperimentValidationError(
            f"{where}.warnings must be a list of strings; received "
            f"{type(warnings_raw).__name__}."
        )
    warnings = tuple(
        _require_str(item, f"{where}.warnings[{index}]")
        for index, item in enumerate(warnings_raw)
    )

    input_source_file = mapping.get("input_source_file")
    if input_source_file is not None:
        input_source_file = _require_str(
            input_source_file, f"{where}.input_source_file"
        )
    target_source_file = mapping.get("target_source_file")
    if target_source_file is not None:
        target_source_file = _require_str(
            target_source_file, f"{where}.target_source_file"
        )

    comparison_raw = mapping.get("comparison")
    comparison = None
    if comparison_raw is not None:
        comparison = _parse_stored_comparison(comparison_raw, f"{where}.comparison")
    if comparison is None and basis is not None:
        raise ExperimentValidationError(
            f"{where}: a comparison basis is declared without comparison "
            "metrics."
        )
    if comparison is not None and basis is None:
        raise ExperimentValidationError(
            f"{where}: comparison metrics are declared without a comparison "
            "basis."
        )

    return StoredRecallRecord(
        comparison_basis=basis,
        warnings=warnings,
        input_source_file=input_source_file,
        target_source_file=target_source_file,
        comparison=comparison,
    )


def _parse_stored_comparison(value: object, where: str) -> StoredComparison:
    """Validate one manifest comparison-metrics mapping."""

    mapping = _require_mapping(value, where)
    element_count = _require_positive_int(
        mapping.get("element_count"), f"{where}.element_count"
    )
    mismatch_count = _require_non_negative_int(
        mapping.get("mismatch_count"), f"{where}.mismatch_count"
    )
    if mismatch_count > element_count:
        raise ExperimentValidationError(
            f"{where}.mismatch_count must not exceed element_count "
            f"({mismatch_count} > {element_count})."
        )
    match_ratio = _require_finite_float(
        mapping.get("match_ratio"), f"{where}.match_ratio"
    )
    if not 0.0 <= match_ratio <= 1.0:
        raise ExperimentValidationError(
            f"{where}.match_ratio must lie in [0, 1]; received {match_ratio!r}."
        )
    exact_match = _require_bool(mapping.get("exact_match"), f"{where}.exact_match")
    if exact_match is not (mismatch_count == 0):
        raise ExperimentValidationError(
            f"{where}.exact_match must be True exactly when mismatch_count "
            "is zero."
        )
    return StoredComparison(
        element_count=element_count,
        mismatch_count=mismatch_count,
        match_ratio=match_ratio,
        exact_match=exact_match,
    )


def _planned_array_shapes_for(
    training: StoredTraining, recall: StoredRecallRecord | None
) -> dict[str, tuple[int, ...]]:
    """Build the manifest's expected array-key -> shape mapping."""

    plan: dict[str, tuple[int, ...]] = {
        _KEY_WEIGHT: (
            training.input_dimension,
            training.output_dimension,
        )
    }
    count = training.contribution_count or 0
    for index in range(count):
        plan[f"{_KEY_CONTRIBUTION_PREFIX}{index:06d}"] = (
            training.input_dimension,
            training.output_dimension,
        )
    if recall is not None:
        plan[_KEY_RECALL_INPUT] = (training.input_dimension,)
        plan[_KEY_RECALL_NET] = (training.output_dimension,)
        plan[_KEY_RECALL_OUTPUT] = (training.output_dimension,)
        if recall.comparison is not None:
            plan[_KEY_RECALL_EXPECTED] = (training.output_dimension,)
    return plan


def _planned_array_shapes(experiment: SavedExperiment) -> dict[str, tuple[int, ...]]:
    """Plan array keys from a real :class:`SavedExperiment` record."""

    return _planned_array_shapes_for(experiment.training, experiment.recall)


def _parse_array_descriptions(
    arrays_block: Mapping[str, object],
    plan: Mapping[str, tuple[int, ...]],
) -> dict[str, tuple[int, ...]]:
    """Validate the manifest arrays block against the expected key plan."""

    unexpected = set(arrays_block) - set(plan)
    if unexpected:
        raise ExperimentValidationError(
            "manifest.arrays contains unexpected entries: "
            f"{', '.join(sorted(unexpected))}."
        )
    missing = set(plan) - set(arrays_block)
    if missing:
        raise ExperimentValidationError(
            "manifest.arrays is missing declared entries: "
            f"{', '.join(sorted(missing))}."
        )

    shapes: dict[str, tuple[int, ...]] = {}
    for key in sorted(plan):
        description = _require_mapping(arrays_block[key], f"manifest.arrays.{key}")
        dtype = _require_str(description.get("dtype"), f"manifest.arrays.{key}.dtype")
        if dtype != "float64":
            raise ExperimentValidationError(
                f"manifest.arrays.{key}.dtype {dtype!r} is unsupported; "
                "expected float64."
            )
        shape_raw = description.get("shape")
        if not isinstance(shape_raw, list) or not shape_raw:
            raise ExperimentValidationError(
                f"manifest.arrays.{key}.shape must be a non-empty list of "
                "dimension lengths."
            )
        shape = tuple(
            _require_positive_int(value, f"manifest.arrays.{key}.shape")
            for value in shape_raw
        )
        expected_shape = plan[key]
        if shape != expected_shape:
            raise ExperimentValidationError(
                f"manifest.arrays.{key} declares shape {shape}, but the "
                f"experiment metadata requires {expected_shape}."
            )
        shapes[key] = shape
    return shapes


# --------------------------------------------------------------------------
# Array storage and validation


def _compose_npz_payload(experiment: SavedExperiment) -> dict[str, NDArray[np.float64]]:
    """Collect the arrays of ``experiment`` into one ``.npz`` payload."""

    payload: dict[str, NDArray[np.float64]] = {
        _KEY_WEIGHT: np.asarray(experiment.weight_matrix, dtype=np.float64)
    }
    contributions = experiment.contributions or ()
    for index, contribution in enumerate(contributions):
        payload[f"{_KEY_CONTRIBUTION_PREFIX}{index:06d}"] = np.asarray(
            contribution, dtype=np.float64
        )
    if experiment.recall is not None:
        payload[_KEY_RECALL_INPUT] = np.asarray(
            experiment.recall_input, dtype=np.float64
        )
        payload[_KEY_RECALL_NET] = np.asarray(
            experiment.recall_net_input, dtype=np.float64
        )
        payload[_KEY_RECALL_OUTPUT] = np.asarray(
            experiment.recall_output, dtype=np.float64
        )
        if experiment.recall_expected is not None:
            payload[_KEY_RECALL_EXPECTED] = np.asarray(
                experiment.recall_expected, dtype=np.float64
            )
    return payload


def _load_arrays(
    directory: Path, plan: Mapping[str, tuple[int, ...]]
) -> dict[str, NDArray[Any]]:
    """Read every planned array from the entry's ``arrays.npz`` file.

    Reading uses ``allow_pickle=False``; the returned arrays are copies so
    the archive can be closed immediately.
    """

    path = directory / _FILE_NAME_ARRAYS
    if not path.is_file():
        raise ExperimentValidationError(
            "arrays.npz is missing; the experiment's numerical data is "
            "incomplete and cannot be loaded safely."
        )
    try:
        with np.load(path, allow_pickle=False) as archive:
            raw = {key: np.array(archive[key], copy=True) for key in archive.files}
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, EOFError) as exc:
        raise ExperimentValidationError(
            f"arrays.npz could not be read as a safe NumPy archive: {exc}"
        ) from exc

    unexpected = set(raw) - set(plan)
    if unexpected:
        raise ExperimentValidationError(
            "arrays.npz contains unexpected arrays: "
            f"{', '.join(sorted(unexpected))}."
        )
    missing = set(plan) - set(raw)
    if missing:
        raise ExperimentValidationError(
            "arrays.npz is missing the declared arrays: "
            f"{', '.join(sorted(missing))}."
        )
    for key, expected_shape in plan.items():
        array = raw[key]
        if array.dtype != np.dtype(np.float64):
            raise ExperimentValidationError(
                f"arrays.npz entry {key!r} has dtype {array.dtype!s}, but "
                "float64 is required."
            )
        if tuple(array.shape) != tuple(expected_shape):
            raise ExperimentValidationError(
                f"arrays.npz entry {key!r} has shape {tuple(array.shape)}, "
                f"but the manifest declares {tuple(expected_shape)}."
            )
    return raw


def _validate_loaded_arrays(
    data: _ManifestData, arrays: Mapping[str, NDArray[Any]]
) -> dict[str, NDArray[np.float64]]:
    """Check value domains of loaded arrays; recompute stored metrics."""

    output = {key: array for key, array in arrays.items()}

    def check_finite(key: str) -> None:
        if not np.all(np.isfinite(output[key])):
            raise ExperimentValidationError(
                f"The stored array {key!r} contains non-finite values."
            )

    def check_encoded(key: str) -> None:
        allowed = _allowed_values(data.training.encoding)
        if not np.isin(output[key], allowed).all():
            raise ExperimentValidationError(
                f"The stored array {key!r} contains values that do not "
                f"belong to the {data.training.encoding.value} encoding "
                f"({', '.join(str(v) for v in allowed)})."
            )

    check_finite(_KEY_WEIGHT)
    count = data.training.contribution_count or 0
    for index in range(count):
        check_finite(f"{_KEY_CONTRIBUTION_PREFIX}{index:06d}")

    recall = data.recall
    if recall is not None:
        check_encoded(_KEY_RECALL_INPUT)
        check_finite(_KEY_RECALL_NET)
        check_encoded(_KEY_RECALL_OUTPUT)
        if recall.comparison is not None:
            check_encoded(_KEY_RECALL_EXPECTED)
            expected = output[_KEY_RECALL_EXPECTED]
            recalled = output[_KEY_RECALL_OUTPUT]
            element_count = int(recalled.shape[0])
            if recall.comparison.element_count != element_count:
                raise ExperimentValidationError(
                    "The stored comparison element_count "
                    f"{recall.comparison.element_count} does not match the "
                    f"stored output pattern length {element_count}."
                )
            mismatch = int(np.count_nonzero(recalled != expected))
            if mismatch != recall.comparison.mismatch_count:
                raise ExperimentValidationError(
                    "The stored comparison metrics do not match the stored "
                    f"arrays (declared {recall.comparison.mismatch_count} "
                    f"mismatches, stored arrays contain {mismatch})."
                )
            recomputed_ratio = (element_count - mismatch) / element_count
            if abs(recomputed_ratio - recall.comparison.match_ratio) > _MATCH_RATIO_TOLERANCE:
                raise ExperimentValidationError(
                    "The stored match ratio does not match the stored "
                    f"arrays (declared {recall.comparison.match_ratio!r}, "
                    f"stored arrays imply {recomputed_ratio!r})."
                )
    return {key: np.asarray(value, dtype=np.float64) for key, value in output.items()}


def _extract_contributions(
    data: _ManifestData, arrays: Mapping[str, NDArray[Any]]
) -> tuple[NDArray[np.float64], ...] | None:
    """Return the stored contribution arrays, or ``None`` when absent."""

    count = data.training.contribution_count
    if count is None or count == 0:
        return None
    return tuple(
        arrays[f"{_KEY_CONTRIBUTION_PREFIX}{index:06d}"] for index in range(count)
    )


def _allowed_values(encoding: Encoding) -> tuple[float, float]:
    """The exact encoded values permitted for ``encoding``."""

    if encoding is Encoding.BINARY:
        return (0.0, 1.0)
    return (-1.0, 1.0)


def _rebuild_training_result(saved: SavedExperiment) -> TrainingResult:
    """Reconstruct the algorithm memory object and training result."""

    training = saved.training
    weight = saved.weight_matrix
    if training.memory_type is MemoryType.AUTOASSOCIATIVE:
        memory = AutoassociativeMemory(
            weight_matrix=weight,
            pattern_dimension=training.input_dimension,
            encoding=training.encoding,
            pattern_count=training.pair_count,
            contributions=saved.contributions,
        )
    else:
        memory = HeteroassociativeMemory(
            weight_matrix=weight,
            source_dimension=training.input_dimension,
            target_dimension=training.output_dimension,
            encoding=training.encoding,
            pair_count=training.pair_count,
            contributions=saved.contributions,
        )
    return TrainingResult(
        memory_type=training.memory_type,
        encoding=training.encoding,
        activation_name=training.activation_name,
        threshold=training.threshold,
        input_dimension=training.input_dimension,
        output_dimension=training.output_dimension,
        pair_count=training.pair_count,
        memory=memory,
        weight_matrix=weight,
        contributions=saved.contributions,
        dataset_name=training.dataset_name,
    )


def _validate_saved_structure(experiment: SavedExperiment) -> None:
    """Validate cross-consistency of a caller-supplied record before storage.

    All checked failures raise :class:`ExperimentValidationError`; the
    repository never stores an inconsistent record.
    """

    training = experiment.training
    weight = np.asarray(experiment.weight_matrix)
    if weight.ndim != 2:
        raise ExperimentValidationError(
            f"The weight matrix must be two-dimensional; received {weight.ndim}."
        )
    expected_weight_shape = (
        training.input_dimension,
        training.output_dimension,
    )
    if weight.shape != expected_weight_shape:
        raise ExperimentValidationError(
            f"The weight matrix shape {weight.shape} does not match the "
            f"declared dimensions {expected_weight_shape}."
        )
    if not np.all(np.isfinite(weight)):
        raise ExperimentValidationError(
            "The weight matrix contains non-finite values."
        )

    contributions = experiment.contributions or ()
    declared_count = training.contribution_count
    if declared_count is not None and declared_count != len(contributions):
        raise ExperimentValidationError(
            f"contribution_count {declared_count} does not match the "
            f"contributions tuple length {len(contributions)}."
        )
    for index, contribution in enumerate(contributions):
        array = np.asarray(contribution)
        if array.shape != expected_weight_shape:
            raise ExperimentValidationError(
                f"Contribution {index} has shape {array.shape}, expected "
                f"{expected_weight_shape}."
            )
        if not np.all(np.isfinite(array)):
            raise ExperimentValidationError(
                f"Contribution {index} contains non-finite values."
            )

    if experiment.recall is None:
        if any(
            getattr(experiment, field) is not None
            for field in (
                "recall_input",
                "recall_net_input",
                "recall_output",
                "recall_expected",
            )
        ):
            raise ExperimentValidationError(
                "Recall arrays are present without a stored recall record."
            )
    else:
        _validate_recall_arrays(experiment)

    if experiment.preprocessing is not None:
        _validate_settings_dims(
            experiment.preprocessing,
            "preprocessing",
            training.input_dimension,
        )
    if experiment.target_preprocessing is not None:
        _validate_settings_dims(
            experiment.target_preprocessing,
            "target_preprocessing",
            training.output_dimension,
        )


def _validate_recall_arrays(experiment: SavedExperiment) -> None:
    """Validate stored recall arrays against the record and its metrics."""

    training = experiment.training
    record = experiment.recall
    if record is None:  # defensive: the caller guarantees a record here
        raise ExperimentValidationError(
            "A recall record is required to validate stored recall arrays."
        )

    required = {
        _KEY_RECALL_INPUT: (
            training.input_dimension,
            experiment.recall_input,
        ),
        _KEY_RECALL_NET: (
            training.output_dimension,
            experiment.recall_net_input,
        ),
        _KEY_RECALL_OUTPUT: (
            training.output_dimension,
            experiment.recall_output,
        ),
    }
    allowed = _allowed_values(training.encoding)
    for key, (length, array) in required.items():
        if array is None:
            raise ExperimentValidationError(
                f"The stored recall record requires the array {key!r}, "
                "which is missing."
            )
        resolved = np.asarray(array)
        if resolved.shape != (length,):
            raise ExperimentValidationError(
                f"The stored recall array {key!r} has shape {resolved.shape}, "
                f"expected ({length},)."
            )
        if not np.all(np.isfinite(resolved)):
            raise ExperimentValidationError(
                f"The stored recall array {key!r} contains non-finite values."
            )
        if key in (_KEY_RECALL_INPUT, _KEY_RECALL_OUTPUT):
            if not np.isin(resolved, allowed).all():
                raise ExperimentValidationError(
                    f"The stored recall array {key!r} contains values that "
                    f"do not belong to the {training.encoding.value} encoding."
                )

    if record.comparison is None and experiment.recall_expected is not None:
        raise ExperimentValidationError(
            "The stored expected pattern is present without comparison "
            "metrics; the declared arrays and the recall record must "
            "agree."
        )

    if record.comparison is not None:
        expected = experiment.recall_expected
        if expected is None:
            raise ExperimentValidationError(
                "The stored recall record has comparison metrics but no "
                "stored expected pattern."
            )
        resolved = np.asarray(expected)
        if resolved.shape != (training.output_dimension,):
            raise ExperimentValidationError(
                "The stored expected pattern has shape "
                f"{resolved.shape}, expected "
                f"({training.output_dimension},)."
            )
        if not np.all(np.isfinite(resolved)):
            raise ExperimentValidationError(
                "The stored expected pattern contains non-finite values."
            )
        if not np.isin(resolved, allowed).all():
            raise ExperimentValidationError(
                "The stored expected pattern contains values that do not "
                f"belong to the {training.encoding.value} encoding."
            )
        recalled = np.asarray(experiment.recall_output)
        element_count = int(recalled.shape[0])
        if record.comparison.element_count != element_count:
            raise ExperimentValidationError(
                "The stored comparison element_count "
                f"{record.comparison.element_count} does not match the "
                f"stored output pattern length {element_count}."
            )
        mismatch = int(np.count_nonzero(recalled != resolved))
        if mismatch != record.comparison.mismatch_count:
            raise ExperimentValidationError(
                "The stored comparison metrics do not match the stored "
                f"arrays (declared {record.comparison.mismatch_count} "
                f"mismatches, stored arrays contain {mismatch})."
            )
        recomputed_ratio = (element_count - mismatch) / element_count
        if (
            abs(recomputed_ratio - record.comparison.match_ratio)
            > _MATCH_RATIO_TOLERANCE
        ):
            raise ExperimentValidationError(
                "The stored match ratio does not match the stored arrays "
                f"(declared {record.comparison.match_ratio!r}, stored arrays "
                f"imply {recomputed_ratio!r})."
            )


def _validate_settings_dims(
    settings: PreprocessingSettings, where: str, dimension: int
) -> None:
    """Check that a stored preprocessing grid matches a vector dimension."""

    if settings.width * settings.height != dimension:
        raise ExperimentValidationError(
            f"{where} describes a {settings.width}x{settings.height} grid "
            f"({settings.width * settings.height} pixels), which does not "
            f"match the stored pattern dimension {dimension}."
        )


def _summary_from_data(data: _ManifestData) -> SavedExperimentSummary:
    """Build the history-list summary of a validated manifest."""

    recall = data.recall
    recall_present = recall is not None
    comparison = recall.comparison if recall is not None else None
    weight_shape = data.arrays[_KEY_WEIGHT]
    return SavedExperimentSummary(
        experiment_id=data.experiment_id,
        name=data.name,
        readable=True,
        schema_version=data.schema_version,
        created_at=data.created_at,
        application_version=data.application_version,
        memory_type=data.training.memory_type.value,
        encoding=data.training.encoding.value,
        activation_name=data.training.activation_name,
        threshold=data.training.threshold,
        input_dimension=data.training.input_dimension,
        output_dimension=data.training.output_dimension,
        pair_count=data.training.pair_count,
        dataset_name=data.training.dataset_name,
        weight_shape=weight_shape,
        contribution_count=data.training.contribution_count,
        preprocessing=data.preprocessing,
        recall_present=recall_present,
        recall_basis=recall.comparison_basis if recall_present else None,
        recall_exact_match=comparison.exact_match if comparison else None,
        recall_mismatch_count=comparison.mismatch_count if comparison else None,
        recall_element_count=comparison.element_count if comparison else None,
    )


def _unreadable_summary(
    experiment_id: str, reason: str
) -> SavedExperimentSummary:
    """Build a placeholder summary for an invalid on-disk entry."""

    return SavedExperimentSummary(
        experiment_id=experiment_id,
        name=DEFAULT_EXPERIMENT_NAME,
        readable=False,
        unreadable_reason=reason,
    )


# --------------------------------------------------------------------------
# Primitive validation helpers


def _require_mapping(value: object, where: str) -> dict[str, Any]:
    """Require a JSON object at ``where``."""

    if not isinstance(value, dict):
        raise ExperimentValidationError(
            f"{where} must be a JSON object; received "
            f"{type(value).__name__}."
        )
    return value


def _require_str(value: object, where: str) -> str:
    """Require a JSON string at ``where``."""

    if not isinstance(value, str):
        raise ExperimentValidationError(
            f"{where} must be a string; received {type(value).__name__}."
        )
    return value


def _require_bool(value: object, where: str) -> bool:
    """Require a JSON boolean at ``where``."""

    if not isinstance(value, bool):
        raise ExperimentValidationError(
            f"{where} must be a boolean; received {type(value).__name__}."
        )
    return value


def _require_int(value: object, where: str) -> int:
    """Require a JSON integer (not bool) at ``where``."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ExperimentValidationError(
            f"{where} must be an integer; received {type(value).__name__}."
        )
    return value


def _require_positive_int(value: object, where: str) -> int:
    """Require an integer >= 1 at ``where``."""

    number = _require_int(value, where)
    if number < 1:
        raise ExperimentValidationError(
            f"{where} must be at least 1; received {number}."
        )
    return number


def _require_non_negative_int(value: object, where: str) -> int:
    """Require an integer >= 0 at ``where``."""

    number = _require_int(value, where)
    if number < 0:
        raise ExperimentValidationError(
            f"{where} must not be negative; received {number}."
        )
    return number


def _require_finite_float(value: object, where: str) -> float:
    """Require a finite JSON number (int or float, not bool) at ``where``."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExperimentValidationError(
            f"{where} must be a number; received {type(value).__name__}."
        )
    number = float(value)
    if not np.isfinite(number):
        raise ExperimentValidationError(
            f"{where} must be finite; received {number!r}."
        )
    return number


def _require_experiment_name(value: object, where: str) -> str:
    """Require a non-empty experiment name within the documented length."""

    name = _require_str(value, where).strip()
    if not name:
        raise ExperimentValidationError(f"{where} must not be empty.")
    if len(name) > MAX_EXPERIMENT_NAME_LENGTH:
        raise ExperimentValidationError(
            f"{where} must not exceed {MAX_EXPERIMENT_NAME_LENGTH} "
            "characters."
        )
    return name


def _require_iso_timestamp(value: object, where: str) -> str:
    """Require a parseable ISO-8601 timestamp string at ``where``."""

    text = _require_str(value, where)
    try:
        datetime.fromisoformat(text)
    except ValueError as exc:
        raise ExperimentValidationError(
            f"{where} is not a valid ISO-8601 timestamp: {text!r}."
        ) from exc
    return text


def _require_experiment_id(experiment_id: object) -> str:
    """Validate an experiment identifier, rejecting malformed values.

    Malformed identifiers (never valid experiment ids) are reported as
    unknown-experiment errors, mapping every hostile or accidental value
    to one clear outcome.
    """

    if not isinstance(experiment_id, str) or not _is_experiment_id(experiment_id):
        raise ExperimentNotFoundError(
            f"Unknown experiment {experiment_id!r}."
        )
    return experiment_id


def _is_experiment_id(experiment_id: str) -> bool:
    """Whether ``experiment_id`` is a well-formed experiment identifier."""

    return bool(_EXPERIMENT_ID_PATTERN.fullmatch(experiment_id))
