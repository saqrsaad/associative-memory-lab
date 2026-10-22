"""Integration tests for the experiment orchestration workflow (GUI-free)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from associative_memory.algorithms.encoding import Encoding
from associative_memory.application.dto.experiment_report import (
    DATA_SOURCE_IMAGES,
    DATA_SOURCE_VECTORS,
    STATUS_COMPLETED,
)
from associative_memory.application.dto.training_result import MemoryType
from associative_memory.application.services.experiment_service import (
    ExperimentError,
    ExperimentService,
)
from associative_memory.application.services.recall_service import RecallService
from associative_memory.application.services.training_service import (
    TrainingDataset,
    TrainingError,
    TrainingService,
)
from associative_memory.infrastructure.image_preprocessor import PreprocessingConfig

SERVICE = ExperimentService()


@pytest.fixture(scope="module")
def hetero_pair_vectors():
    """The single hand-calculated binary source/target pair."""

    return np.array([1, 0, 1, 0]), np.array([1, 1, 0, 0])


def _shape_images(tmp_path: Path) -> tuple[Path, Path]:
    """Two 4x4 images whose dark blocks are disjoint (orthogonal patterns)."""

    left = Image.new("L", (4, 4), 255)
    for y in range(2):
        for x in range(2):
            left.putpixel((x, y), 0)  # top-left dark block
    right = Image.new("L", (4, 4), 255)
    for y in range(2):
        for x in range(2, 4):
            right.putpixel((x, y), 0)  # top-right dark block
    left_path = tmp_path / "image_a.png"
    right_path = tmp_path / "image_b.png"
    left.save(left_path)
    right.save(right_path)
    return left_path, right_path


def test_vector_experiment_reports_the_full_workflow(hetero_pair_vectors):
    source, target = hetero_pair_vectors
    dataset = TrainingDataset.from_pairs(
        [(source, target)], name="hetero-vector-exp"
    )

    report = SERVICE.run_vector_experiment(
        training_dataset=dataset,
        encoding=Encoding.BINARY,
        queries=[source, np.array([0, 1, 0, 1])],
        expected_targets=[target, None],
        name="VectorDemo",
    )

    assert report.status == STATUS_COMPLETED
    assert report.name == "VectorDemo"
    assert report.memory_type is MemoryType.HETEROASSOCIATIVE
    assert report.encoding is Encoding.BINARY
    assert report.data_source == DATA_SOURCE_VECTORS
    assert report.training.pair_count == 1
    assert report.total_recall_count == 2
    assert report.compared_recall_count == 1
    assert report.exact_recall_count == 1
    assert report.mismatched_recall_count == 0
    assert report.metadata["data_source"] == "vectors"
    assert report.metadata["pair_count"] == "1"
    assert report.metadata["dataset_name"] == "hetero-vector-exp"
    assert report.metadata["encoding"] == "binary"
    assert report.recall_results[0].comparison.exact_match is True
    assert report.recall_results[1].comparison is None


def test_vector_experiment_without_expected_targets_reports_no_comparisons():
    report = SERVICE.run_vector_experiment(
        training_dataset=TrainingDataset.from_patterns([[1, 0, 1, 0]], name="auto-run"),
        encoding=Encoding.BINARY,
        queries=[np.array([1, 0, 1, 0])],
        name=None,
    )

    # Autoassociative recall compares against the input pattern by default.
    assert report.memory_type is MemoryType.AUTOASSOCIATIVE
    assert report.compared_recall_count == 1
    assert report.exact_recall_count == 1
    assert report.metadata["dataset_name"] == "auto-run"


def test_vector_experiment_reports_mismatched_recalls(hetero_pair_vectors):
    source, target = hetero_pair_vectors

    report = SERVICE.run_vector_experiment(
        training_dataset=TrainingDataset.from_pairs([(source, target)]),
        encoding=Encoding.BINARY,
        queries=[np.array([0, 1, 0, 1])],
        expected_targets=[target],
    )

    assert report.exact_recall_count == 0
    assert report.mismatched_recall_count == 1
    assert report.recall_results[0].warnings


def test_vector_experiment_rejects_a_target_count_mismatch(hetero_pair_vectors):
    source, target = hetero_pair_vectors

    with pytest.raises(
        ExperimentError,
        match="expected_targets must supply one expected pattern per query",
    ):
        SERVICE.run_vector_experiment(
            training_dataset=TrainingDataset.from_pairs([(source, target)]),
            encoding=Encoding.BINARY,
            queries=[source],
            expected_targets=[target, target],
        )


def test_image_experiment_trains_and_recalls_disjoint_patterns(tmp_path: Path):
    left_path, right_path = _shape_images(tmp_path)

    report = SERVICE.run_image_experiment(
        preprocessing_config=PreprocessingConfig(4, 4),
        training_sources=[left_path, right_path],
        name="shapes-auto",
    )

    assert report.status == STATUS_COMPLETED
    assert report.memory_type is MemoryType.AUTOASSOCIATIVE
    assert report.data_source == DATA_SOURCE_IMAGES
    assert report.training.pair_count == 2
    assert report.total_recall_count == 2
    assert report.exact_recall_count == 2
    assert report.metadata["data_source"] == "images"
    assert report.metadata["image_width"] == "4"
    assert report.metadata["image_height"] == "4"
    assert report.metadata["image_threshold"] == "128"
    assert report.metadata["image_polarity"] == "dark_is_on"
    assert report.metadata["image_encoding"] == "binary"
    assert report.metadata["image_resize_mode"] == "preserve_aspect"
    assert report.metadata["image_flattening_order"] == "row_major"
    assert report.metadata["image_grayscale_method"] == "luminance_601"
    assert report.metadata["image_resampling"] == "NEAREST"


def test_hetero_image_experiment_with_expected_queries(tmp_path: Path):
    left_path, right_path = _shape_images(tmp_path)

    report = SERVICE.run_image_experiment(
        preprocessing_config=PreprocessingConfig(4, 4),
        training_sources=[left_path],
        training_targets=[right_path],
        queries=[left_path],
        query_targets=[right_path],
        name="shapes-hetero",
    )

    assert report.memory_type is MemoryType.HETEROASSOCIATIVE
    assert report.total_recall_count == 1
    assert report.exact_recall_count == 1
    assert report.recall_results[0].comparison.mismatch_count == 0


def test_hetero_image_experiment_without_expects_reports_no_comparison(tmp_path):
    left_path, right_path = _shape_images(tmp_path)

    report = SERVICE.run_image_experiment(
        preprocessing_config=PreprocessingConfig(4, 4),
        training_sources=[left_path],
        training_targets=[right_path],
    )

    assert report.recall_results[0].comparison is None


def test_requested_memory_type_must_match_supplied_image_data(tmp_path: Path):
    left_path, right_path = _shape_images(tmp_path)

    with pytest.raises(
        ExperimentError,
        match="does not match the supplied data",
    ):
        SERVICE.run_image_experiment(
            preprocessing_config=PreprocessingConfig(4, 4),
            training_sources=[left_path],
            training_targets=[right_path],
            memory_type="autoassociative",
        )
    with pytest.raises(
        ExperimentError,
        match="does not match the supplied data",
    ):
        SERVICE.run_image_experiment(
            preprocessing_config=PreprocessingConfig(4, 4),
            training_sources=[left_path, right_path],
            memory_type="heteroassociative",
        )


def test_unmatched_target_image_count_is_rejected(tmp_path: Path):
    left_path, right_path = _shape_images(tmp_path)

    with pytest.raises(
        ExperimentError,
        match="require one target image for each source image",
    ):
        SERVICE.run_image_experiment(
            preprocessing_config=PreprocessingConfig(4, 4),
            training_sources=[left_path, right_path],
            training_targets=[right_path],
        )


def test_missing_image_file_is_converted_into_an_experiment_error(tmp_path: Path):
    missing = tmp_path / "missing.png"

    with pytest.raises(ExperimentError, match="missing.png") as excinfo:
        SERVICE.run_image_experiment(
            preprocessing_config=PreprocessingConfig(4, 4),
            training_sources=[missing],
        )

    assert excinfo.value.__cause__ is not None


def test_empty_image_dataset_propagates_the_training_error(tmp_path: Path):
    with pytest.raises(TrainingError, match="at least one pattern"):
        SERVICE.run_image_experiment(
            preprocessing_config=PreprocessingConfig(4, 4),
            training_sources=[],
        )


def test_experiment_report_arrays_are_read_only(hetero_pair_vectors):
    source, target = hetero_pair_vectors

    report = SERVICE.run_vector_experiment(
        training_dataset=TrainingDataset.from_pairs([(source, target)]),
        encoding=Encoding.BINARY,
        queries=[source],
        expected_targets=[target],
    )

    assert report.training.weight_matrix.flags.writeable is False
    recall_result = report.recall_results[0]
    assert recall_result.output_pattern.flags.writeable is False
    assert recall_result.net_input.flags.writeable is False
    with pytest.raises(TypeError):
        report.metadata["new_key"] = "value"


def test_experiment_uses_injected_services():
    """The service accepts injected dependencies (no hidden global state)."""

    experiment = ExperimentService(
        image_service=None,
        training_service=TrainingService(),
        recall_service=RecallService(),
    )

    report = experiment.run_vector_experiment(
        training_dataset=TrainingDataset.from_patterns([[1, 0, 1, 0]]),
        encoding=Encoding.BINARY,
        queries=[np.array([1, 0, 1, 0])],
        name=None,
    )

    assert report.exact_recall_count == 1
