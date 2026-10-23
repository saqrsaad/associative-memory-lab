"""Persistence regression tests for the experiment repository.

Covers the required tests of Task 10 (experiment history) and the
persistence demands of Task 11: exact round trips, preprocessing
metadata preservation, corruption and unsupported-schema rejection,
missing files, identifier conflicts, deletion, comparable and
incomparable comparisons, and the guarantee that loading a stored
experiment never executes serialized code.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import associative_memory.infrastructure.model_repository_json as repo_module
from associative_memory.algorithms.encoding import Encoding
from associative_memory.application.dto.recall_result import (
    COMPARISON_BASIS_EXPECTED,
)
from associative_memory.application.dto.saved_experiment import (
    APPLICATION_VERSION,
    SCHEMA_VERSION,
    SavedExperiment,
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
from associative_memory.application.services.experiment_history_service import (
    ExperimentHistoryService,
)
from associative_memory.application.services.training_service import (
    TrainingDataset,
    TrainingService,
)
from associative_memory.infrastructure.experiment_repository import (
    build_default_repository,
    default_experiments_directory,
)
from associative_memory.infrastructure.image_preprocessor import (
    FlatteningOrder,
    ForegroundPolarity,
    GrayscaleMethod,
    PreprocessingConfig,
    ResizeMode,
)
from associative_memory.infrastructure.model_repository_json import (
    JsonModelRepository,
)

_TRAINING_SERVICE = TrainingService()

_AUTO_PATTERNS_A = (
    [1.0, -1.0, 1.0, -1.0],
    [-1.0, 1.0, 1.0, -1.0],
)
_AUTO_PATTERNS_B = (
    [1.0, -1.0, 1.0, 1.0],
    [-1.0, 1.0, -1.0, 1.0],
)

# Marker list recording whether a pickled object was ever instantiated:
# loading a stored experiment must never execute serialized code.
_PICKLE_RECORD: list[str] = []


def _record_pickle_execution() -> None:
    """Marker callable used as a pickled object's reducer."""

    _PICKLE_RECORD.append("executed")


class _HostilePickledObject:
    """Object that would run code when pickled data is unpickled."""

    def __reduce__(self):
        return (_record_pickle_execution, ())


# ---------------------------------------------------------------------- fixtures


def _auto_result(
    patterns: tuple = _AUTO_PATTERNS_A, trace: bool = True
) -> TrainingResult:
    """Train a fresh autoassociative memory from known bipolar patterns."""

    dataset = TrainingDataset.from_patterns(list(patterns), name="dataset-test")
    return _TRAINING_SERVICE.train(dataset, encoding=Encoding.BIPOLAR, trace=trace)


def _config(encoding: Encoding = Encoding.BIPOLAR) -> PreprocessingConfig:
    """A 2x2 preprocessing configuration valid for the 4-element patterns."""

    return PreprocessingConfig(
        width=2,
        height=2,
        grayscale_method=GrayscaleMethod.LUMINANCE_601,
        resampling=Image.Resampling.NEAREST,
        threshold=128,
        polarity=ForegroundPolarity.DARK_IS_ON,
        encoding=encoding,
        resize_mode=ResizeMode.PRESERVE_ASPECT,
        flattening_order=FlatteningOrder.ROW_MAJOR,
    )


def _make_saved(result: TrainingResult | None = None) -> SavedExperiment:
    """Build an internally consistent, savable record for repository tests."""

    result = result or _auto_result()
    training = StoredTraining(
        memory_type=result.memory_type,
        encoding=result.encoding,
        activation_name=result.activation_name,
        threshold=result.threshold,
        input_dimension=result.input_dimension,
        output_dimension=result.output_dimension,
        pair_count=result.pair_count,
        dataset_name=result.dataset_name,
        contribution_count=(
            None if result.contributions is None else len(result.contributions)
        ),
    )
    return SavedExperiment(
        training=training,
        name="Repository probe",
        weight_matrix=result.weight_matrix,
        contributions=result.contributions or None,
    )


def _single_entry_id(repo: JsonModelRepository) -> str:
    """Save one experiment into an empty repository and return its id."""

    repo.save(_make_saved())
    entries = repo.list_experiments()
    assert len(entries) == 1
    return entries[0].experiment_id


def _save_two_experiments(repo: JsonModelRepository) -> tuple[str, str]:
    """Save two comparable autoassociative experiments; return their ids."""

    service = ExperimentHistoryService(repository=repo)
    left = service.save_experiment(
        name="Left pattern set",
        training_result=_auto_result(_AUTO_PATTERNS_A),
        preprocessing_config=_config(),
    )
    right = service.save_experiment(
        name="Right pattern set",
        training_result=_auto_result(_AUTO_PATTERNS_B),
        preprocessing_config=_config(),
    )
    return left.experiment_id, right.experiment_id


def _wide_config() -> PreprocessingConfig:
    """A 3x2 preprocessing configuration for six-element patterns."""

    return PreprocessingConfig(
        width=3,
        height=2,
        grayscale_method=GrayscaleMethod.LUMINANCE_601,
        resampling=Image.Resampling.NEAREST,
        threshold=128,
        polarity=ForegroundPolarity.DARK_IS_ON,
        encoding=Encoding.BIPOLAR,
        resize_mode=ResizeMode.PRESERVE_ASPECT,
        flattening_order=FlatteningOrder.ROW_MAJOR,
    )


# ---------------------------------------------------------------------- interfaces


def test_repository_satisfies_the_application_interface(tmp_path: Path) -> None:
    repo = JsonModelRepository(tmp_path)
    assert isinstance(repo, ModelRepository)


def test_default_repository_uses_the_documented_location() -> None:
    repo = build_default_repository()
    assert isinstance(repo, JsonModelRepository)
    assert repo.storage_location == default_experiments_directory()


# ---------------------------------------------------------------------- round trips


def test_save_and_reload_preserves_every_array_exactly(tmp_path: Path) -> None:
    repo = JsonModelRepository(tmp_path)
    experiment_id = _single_entry_id(repo)

    loaded = repo.load(experiment_id)

    expected = np.outer(_AUTO_PATTERNS_A[0], _AUTO_PATTERNS_A[0]) + np.outer(
        _AUTO_PATTERNS_A[1], _AUTO_PATTERNS_A[1]
    )
    assert loaded.name == "Repository probe"
    assert loaded.summary.experiment_id == experiment_id
    np.testing.assert_array_equal(loaded.training.weight_matrix, expected)
    assert loaded.training.weight_matrix.flags.writeable is False
    assert loaded.training.encoding is Encoding.BIPOLAR
    assert loaded.training.memory_type is MemoryType.AUTOASSOCIATIVE
    assert loaded.training.pair_count == 2
    assert loaded.training.input_dimension == 4
    assert loaded.training.output_dimension == 4
    assert loaded.training.threshold == 0.0
    assert loaded.training.dataset_name == "dataset-test"

    contributions = loaded.training.contributions
    assert contributions is not None and len(contributions) == 2
    np.testing.assert_array_equal(
        contributions[0], np.outer(_AUTO_PATTERNS_A[0], _AUTO_PATTERNS_A[0])
    )
    np.testing.assert_array_equal(
        contributions[1], np.outer(_AUTO_PATTERNS_A[1], _AUTO_PATTERNS_A[1])
    )
    assert all(
        contribution.flags.writeable is False for contribution in contributions
    )


def test_summary_metadata_matches_the_stored_experiment(tmp_path: Path) -> None:
    repo = JsonModelRepository(tmp_path)
    summary = repo.save(_make_saved())

    assert summary.schema_version == SCHEMA_VERSION
    assert summary.application_version == APPLICATION_VERSION
    assert summary.name == "Repository probe"
    assert summary.memory_type == "autoassociative"
    assert summary.encoding == "bipolar"
    assert summary.activation_name == "bipolar_activation"
    assert summary.threshold == 0.0
    assert summary.input_dimension == 4
    assert summary.output_dimension == 4
    assert summary.pair_count == 2
    assert summary.dataset_name == "dataset-test"
    assert summary.weight_shape == (4, 4)
    assert summary.contribution_count == 2
    assert summary.readable is True
    assert summary.recall_present is False
    uuid.UUID(summary.experiment_id)  # well-formed UUID hex
    created_at = summary.created_at
    assert created_at is not None and "T" in created_at


def test_preprocessing_metadata_round_trip(tmp_path: Path) -> None:
    repo = JsonModelRepository(tmp_path)
    service = ExperimentHistoryService(repository=repo)
    summary = service.save_experiment(
        name="With preprocessing",
        training_result=_auto_result(),
        preprocessing_config=_config(Encoding.BIPOLAR),
    )

    loaded = service.load_experiment(summary.experiment_id)
    settings = loaded.preprocessing
    assert settings is not None
    assert settings.width == 2
    assert settings.height == 2
    assert settings.grayscale_method == GrayscaleMethod.LUMINANCE_601.value
    assert settings.resampling == "NEAREST"
    assert settings.threshold == 128
    assert settings.polarity == ForegroundPolarity.DARK_IS_ON.value
    assert settings.encoding is Encoding.BIPOLAR
    assert settings.resize_mode == ResizeMode.PRESERVE_ASPECT.value
    assert settings.flattening_order == FlatteningOrder.ROW_MAJOR.value

    rebuilt = settings.to_config()
    assert rebuilt == _config(Encoding.BIPOLAR)


def test_loaded_model_reproduces_inference_results(tmp_path: Path) -> None:
    repo = JsonModelRepository(tmp_path)
    result = _auto_result()
    experiment_id = _single_entry_id(repo)

    loaded = repo.load(experiment_id)
    query = np.array(_AUTO_PATTERNS_A[0])
    recalled_loaded = loaded.training.memory.infer(query)
    recalled_original = result.memory.infer(query)

    np.testing.assert_array_equal(
        recalled_loaded.net_input, recalled_original.net_input
    )
    np.testing.assert_array_equal(
        recalled_loaded.activated_output, recalled_original.activated_output
    )


# ---------------------------------------------------------------------- recall records


def test_recall_record_round_trips_with_its_metrics(tmp_path: Path) -> None:
    repo = JsonModelRepository(tmp_path)
    result = _auto_result()
    query = np.array(_AUTO_PATTERNS_A[0])
    inference = result.memory.infer(query)
    expected = np.array(_AUTO_PATTERNS_A[0])
    mismatch = int(np.count_nonzero(inference.activated_output != expected))

    training = StoredTraining(
        memory_type=result.memory_type,
        encoding=result.encoding,
        activation_name=result.activation_name,
        threshold=result.threshold,
        input_dimension=result.input_dimension,
        output_dimension=result.output_dimension,
        pair_count=result.pair_count,
        dataset_name=result.dataset_name,
        contribution_count=len(result.contributions),
    )
    record = StoredRecallRecord(
        comparison_basis=COMPARISON_BASIS_EXPECTED,
        warnings=("Interference possible with two stored patterns.",),
        comparison=StoredComparison(
            element_count=4,
            mismatch_count=mismatch,
            match_ratio=(4 - mismatch) / 4,
            exact_match=mismatch == 0,
        ),
    )
    saved = SavedExperiment(
        training=training,
        name="With recall",
        weight_matrix=result.weight_matrix,
        contributions=result.contributions,
        recall=record,
        recall_input=inference.input_pattern,
        recall_net_input=inference.net_input,
        recall_output=inference.activated_output,
        recall_expected=expected,
    )
    summary = repo.save(saved)
    loaded = repo.load(summary.experiment_id)

    stored_record = loaded.recall_record
    assert stored_record is not None and stored_record.comparison is not None
    assert stored_record.comparison.element_count == 4
    assert stored_record.comparison.mismatch_count == mismatch
    assert stored_record.comparison.exact_match is (mismatch == 0)
    assert stored_record.comparison_basis == COMPARISON_BASIS_EXPECTED
    assert stored_record.warnings == (
        "Interference possible with two stored patterns.",
    )
    # The arrays survive: loading validates them against the stored
    # metrics, and the weight matrix that produced them is exact.
    np.testing.assert_array_equal(
        loaded.training.weight_matrix, result.weight_matrix
    )


def test_save_rejects_recall_metrics_that_contradict_the_arrays(
    tmp_path: Path,
) -> None:
    repo = JsonModelRepository(tmp_path)
    result = _auto_result()
    query = np.array(_AUTO_PATTERNS_A[0])
    inference = result.memory.infer(query)

    training = StoredTraining(
        memory_type=result.memory_type,
        encoding=result.encoding,
        activation_name=result.activation_name,
        threshold=result.threshold,
        input_dimension=result.input_dimension,
        output_dimension=result.output_dimension,
        pair_count=result.pair_count,
        contribution_count=len(result.contributions),
    )
    saved = SavedExperiment(
        training=training,
        name="Inconsistent metrics",
        weight_matrix=result.weight_matrix,
        contributions=result.contributions,
        recall=StoredRecallRecord(
            comparison_basis=COMPARISON_BASIS_EXPECTED,
            comparison=StoredComparison(
                element_count=4,
                mismatch_count=1,
                match_ratio=0.75,
                exact_match=False,
            ),
        ),
        recall_input=inference.input_pattern,
        recall_net_input=inference.net_input,
        recall_output=inference.activated_output,
        recall_expected=np.array(_AUTO_PATTERNS_A[0]),
    )

    with pytest.raises(ExperimentValidationError, match="mismatch"):
        repo.save(saved)
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------- listing


def test_list_orders_savable_entries_newest_first(tmp_path: Path) -> None:
    repo = JsonModelRepository(tmp_path)
    first_id, second_id = _save_two_experiments(repo)

    entries = repo.list_experiments()
    assert [entry.experiment_id for entry in entries] == [second_id, first_id]
    assert all(entry.readable for entry in entries)


def test_list_skips_foreign_and_temporary_directories(tmp_path: Path) -> None:
    repo = JsonModelRepository(tmp_path)
    experiment_id = _single_entry_id(repo)

    (tmp_path / "not-an-experiment").mkdir()
    (tmp_path / ".save-in-progress-aborted").mkdir()
    broken_id = "a" * 32
    (tmp_path / broken_id).mkdir()
    (tmp_path / "loose_file.txt").write_text("not an entry", encoding="utf-8")

    entries = repo.list_experiments()
    assert [entry.experiment_id for entry in entries] == [experiment_id, broken_id]
    assert entries[0].readable is True
    assert entries[1].readable is False
    assert entries[1].unreadable_reason is not None
    # Unreadable entries remain deletable.
    repo.delete(broken_id)
    assert all(
        entry.experiment_id != broken_id for entry in repo.list_experiments()
    )


# ---------------------------------------------------------------------- rejection


def test_corrupt_manifest_is_rejected_and_reported_as_unreadable(
    tmp_path: Path,
) -> None:
    repo = JsonModelRepository(tmp_path)
    experiment_id = _single_entry_id(repo)
    (repo.root / experiment_id / "manifest.json").write_text(
        "{ definitely not json", encoding="utf-8"
    )

    with pytest.raises(ExperimentValidationError, match="valid JSON"):
        repo.load(experiment_id)
    entry = repo.list_experiments()[0]
    assert entry.readable is False
    assert entry.unreadable_reason is not None


def test_missing_manifest_and_arrays_are_rejected(tmp_path: Path) -> None:
    repo = JsonModelRepository(tmp_path)
    experiment_id = _single_entry_id(repo)
    (repo.root / experiment_id / "arrays.npz").unlink()

    with pytest.raises(ExperimentValidationError, match="arrays.npz"):
        repo.load(experiment_id)

    (repo.root / experiment_id / "manifest.json").unlink()
    with pytest.raises(ExperimentValidationError, match="manifest.json"):
        repo.load(experiment_id)


def test_corrupt_npz_is_rejected(tmp_path: Path) -> None:
    repo = JsonModelRepository(tmp_path)
    experiment_id = _single_entry_id(repo)
    (repo.root / experiment_id / "arrays.npz").write_bytes(b"not a zip archive")

    with pytest.raises(ExperimentValidationError, match="arrays.npz"):
        repo.load(experiment_id)


def test_unsupported_schema_versions_are_rejected(tmp_path: Path) -> None:
    repo = JsonModelRepository(tmp_path)
    experiment_id = _single_entry_id(repo)
    manifest_path = repo.root / experiment_id / "manifest.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))

    document["schema_version"] = SCHEMA_VERSION + 1
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(
        UnsupportedSchemaError, match="newer than the version this application"
    ):
        repo.load(experiment_id)

    document["schema_version"] = 0
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(UnsupportedSchemaError, match="unsupported schema version"):
        repo.load(experiment_id)


def test_missing_and_malformed_identifiers_report_unknown_experiment(
    tmp_path: Path,
) -> None:
    repo = JsonModelRepository(tmp_path)
    _single_entry_id(repo)

    with pytest.raises(ExperimentNotFoundError):
        repo.load("f" * 32)
    with pytest.raises(ExperimentNotFoundError):
        repo.load("")
    with pytest.raises(ExperimentNotFoundError):
        repo.load("../escape")
    with pytest.raises(ExperimentNotFoundError):
        repo.delete("f" * 32)


def test_save_rejects_records_that_already_carry_an_id(tmp_path: Path) -> None:
    repo = JsonModelRepository(tmp_path)
    saved = _make_saved()
    object.__setattr__(saved, "experiment_id", "a" * 32)

    with pytest.raises(ExperimentIdConflictError):
        repo.save(saved)


def test_save_detects_a_concurrent_id_conflict_without_overwriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = JsonModelRepository(tmp_path)
    conflict_id = uuid.uuid4().hex
    (tmp_path / conflict_id).mkdir()
    marker_file = tmp_path / conflict_id / "sentinel.txt"
    marker_file.write_text("pre-existing", encoding="utf-8")

    # Force the generated id to collide with the pre-created directory.
    monkeypatch.setattr(repo_module, "uuid4", lambda: uuid.UUID(conflict_id))

    with pytest.raises(ExperimentIdConflictError):
        repo.save(_make_saved())

    assert marker_file.read_text(encoding="utf-8") == "pre-existing"
    names = sorted(path.name for path in tmp_path.iterdir())
    assert names == [conflict_id]


def test_failed_save_cleans_up_and_never_publishes_an_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = JsonModelRepository(tmp_path)

    def broken_savez_compressed(*args: object, **kwargs: object) -> None:
        raise OSError("simulated disk failure")

    monkeypatch.setattr(np, "savez_compressed", broken_savez_compressed)
    with pytest.raises(ExperimentRepositoryError, match="Saving the experiment"):
        repo.save(_make_saved())

    assert list(tmp_path.iterdir()) == []


def test_delete_removes_the_entry_and_unknown_ids_fail(tmp_path: Path) -> None:
    repo = JsonModelRepository(tmp_path)
    experiment_id = _single_entry_id(repo)

    repo.delete(experiment_id)
    assert repo.list_experiments() == ()
    with pytest.raises(ExperimentNotFoundError):
        repo.load(experiment_id)


# ---------------------------------------------------------------------- no code execution


def test_loading_never_executes_serialized_code(tmp_path: Path) -> None:
    repo = JsonModelRepository(tmp_path)
    experiment_id = _single_entry_id(repo)
    arrays_path = repo.root / experiment_id / "arrays.npz"

    # Keep the manifest valid, but let the weight-matrix key carry a
    # pickled object array instead of float64 numbers.
    np.savez_compressed(
        arrays_path,
        weight_matrix=np.array([[_HostilePickledObject()]], dtype=object),
    )

    with pytest.raises(ExperimentValidationError, match="safe NumPy archive"):
        repo.load(experiment_id)
    assert _PICKLE_RECORD == []


# ---------------------------------------------------------------------- comparison


def test_compatible_comparisons_report_weight_differences(
    tmp_path: Path,
) -> None:
    repo = JsonModelRepository(tmp_path)
    left_id, right_id = _save_two_experiments(repo)
    service = ExperimentHistoryService(repository=repo)

    comparison = service.compare_experiments(left_id, right_id)
    left_weight = repo.load(left_id).training.weight_matrix
    right_weight = repo.load(right_id).training.weight_matrix
    expected_mismatches = int(np.count_nonzero(left_weight != right_weight))

    assert comparison.weight_comparison is not None
    assert comparison.weight_comparison.element_count == 16
    assert comparison.weight_comparison.differing_count == expected_mismatches
    assert comparison.weight_comparison.differing_count > 0
    assert comparison.weight_comparison.identical is False
    max_diff = float(np.max(np.abs(left_weight - right_weight)))
    assert comparison.weight_comparison.max_abs_difference == pytest.approx(max_diff)


def test_identical_experiments_compare_as_identical(tmp_path: Path) -> None:
    repo = JsonModelRepository(tmp_path)
    left_id, _ = _save_two_experiments(repo)
    service = ExperimentHistoryService(repository=repo)

    # The twin is trained from the left experiment's exact patterns, so
    # both stored weight matrices must be element-for-element identical.
    twin = service.save_experiment(
        name="Twin",
        training_result=_auto_result(_AUTO_PATTERNS_A),
        preprocessing_config=_config(),
    )
    comparison = service.compare_experiments(left_id, twin.experiment_id)

    assert comparison.weight_comparison is not None
    assert comparison.weight_comparison.differing_count == 0
    assert comparison.weight_comparison.identical is True
    assert comparison.weight_comparison.max_abs_difference == pytest.approx(0.0)


def test_incompatible_dimensions_are_not_compared_elementwise(
    tmp_path: Path,
) -> None:
    repo = JsonModelRepository(tmp_path)
    left_id, _ = _save_two_experiments(repo)
    service = ExperimentHistoryService(repository=repo)

    other = service.save_experiment(
        name="Six-element patterns",
        training_result=_auto_result(
            ([1.0, -1.0, 1.0, -1.0, 1.0, -1.0],),
        ),
        preprocessing_config=_wide_config(),
    )

    comparison = service.compare_experiments(left_id, other.experiment_id)
    assert comparison.weight_comparison is None
    shape_row = next(
        row for row in comparison.rows if row.label == "Weight-matrix shape"
    )
    assert shape_row.note != ""
    dims_row = next(
        row for row in comparison.rows if row.label == "Input dimension (n)"
    )
    assert dims_row.note != ""
