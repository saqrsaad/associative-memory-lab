"""Integration tests for the application recall workflow (GUI-free)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from associative_memory.algorithms.encoding import Encoding
from associative_memory.application.dto.recall_result import (
    COMPARISON_BASIS_EXPECTED,
    COMPARISON_BASIS_INPUT,
)
from associative_memory.application.dto.training_result import MemoryType
from associative_memory.application.services.recall_service import (
    RecallError,
    RecallService,
)
from associative_memory.application.services.training_service import (
    TrainingDataset,
    TrainingService,
)

TRAINING_SERVICE = TrainingService()
RECALL_SERVICE = RecallService()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = (
    PROJECT_ROOT / "tests" / "fixtures" / "algorithm_reference_cases.json"
)


@pytest.fixture(scope="module")
def hetero_memory():
    """A binary heteroassociative memory trained on one hand-calculated pair."""

    dataset = TrainingDataset.from_pairs([
        ([1, 0, 1, 0], [1, 1, 0, 0]),
    ])
    return TRAINING_SERVICE.train(dataset, encoding=Encoding.BINARY).memory


@pytest.fixture(scope="module")
def auto_memory():
    """A binary autoassociative memory with two disjoint stored patterns."""

    dataset = TrainingDataset.from_patterns([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ])
    return TRAINING_SERVICE.train(dataset, encoding=Encoding.BINARY).memory


def test_successful_hetero_recall_with_expected_target(hetero_memory):
    result = RECALL_SERVICE.recall(
        hetero_memory,
        np.array([1, 0, 1, 0]),
        expected_pattern=np.array([1, 1, 0, 0]),
    )

    assert result.memory_type is MemoryType.HETEROASSOCIATIVE
    assert result.input_dimension == 4
    assert result.output_dimension == 4
    # y_in = x @ W = (x . s) t = 2 * t.
    assert np.array_equal(result.net_input, np.array([2.0, 2.0, 0.0, 0.0]))
    assert np.array_equal(result.output_pattern, np.array([1.0, 1.0, 0.0, 0.0]))
    assert result.comparison is not None
    assert result.comparison.mismatch_count == 0
    assert result.comparison.match_ratio == 1.0
    assert result.comparison.exact_match is True
    assert result.comparison_basis == COMPARISON_BASIS_EXPECTED
    assert result.expected_pattern is not None
    assert result.expected_pattern.tolist() == [1.0, 1.0, 0.0, 0.0]
    assert result.warnings == ()


def test_interference_failure_reports_mismatches_and_warning(hetero_memory):
    # The query is orthogonal to the stored source, so y_in = 0 and the
    # binary output is all OFF; a genuine recall failure, not a defect.
    result = RECALL_SERVICE.recall(
        hetero_memory,
        np.array([0, 1, 0, 1]),
        expected_pattern=np.array([1, 1, 0, 0]),
    )

    assert np.array_equal(result.net_input, np.array([0.0, 0.0, 0.0, 0.0]))
    assert np.array_equal(result.output_pattern, np.array([0.0, 0.0, 0.0, 0.0]))
    assert result.comparison.mismatch_count == 2
    assert result.comparison.match_ratio == 0.5
    assert result.comparison.exact_match is False
    assert len(result.warnings) == 1
    assert "interference" in result.warnings[0]


def test_hetero_recall_without_expected_target_makes_no_comparison(hetero_memory):
    result = RECALL_SERVICE.recall(hetero_memory, np.array([1, 0, 1, 0]))

    assert result.comparison is None
    assert result.comparison_basis is None
    assert result.expected_pattern is None
    assert result.output_pattern.tolist() == [1.0, 1.0, 0.0, 0.0]
    assert len(result.warnings) == 1
    assert "comparison is None" in result.warnings[0]


def test_genuine_zero_match_ratio_is_distinguishable_from_no_comparison():
    dataset = TrainingDataset.from_pairs([([1, 1, 1, 1], [1, 1, 1, 1])])
    memory = TRAINING_SERVICE.train(dataset, encoding=Encoding.BINARY).memory

    result = RECALL_SERVICE.recall(
        memory,
        np.array([0, 0, 0, 0]),
        expected_pattern=np.array([1, 1, 1, 1]),
    )

    assert result.comparison is not None
    assert result.comparison.match_ratio == 0.0
    assert result.comparison.mismatch_count == 4
    assert result.comparison.exact_match is False


def test_autoassociative_recall_compares_against_the_input_pattern(auto_memory):
    result = RECALL_SERVICE.recall(auto_memory, np.array([1, 0, 1, 0]))

    assert result.memory_type is MemoryType.AUTOASSOCIATIVE
    assert result.comparison_basis == COMPARISON_BASIS_INPUT
    assert result.expected_pattern.tolist() == [1.0, 0.0, 1.0, 0.0]
    assert result.comparison.exact_match is True
    assert result.comparison.mismatch_count == 0


def test_corrupted_autoassociative_query_shows_interference(auto_memory):
    result = RECALL_SERVICE.recall(auto_memory, np.array([0, 1, 1, 0]))

    # y_in = (C . A) A + (C . B) B = [1, 1, 1, 1] -> all ON.
    assert np.array_equal(result.output_pattern, np.array([1.0, 1.0, 1.0, 1.0]))
    assert result.comparison.exact_match is False
    assert result.comparison.mismatch_count == 2
    assert result.comparison.match_ratio == 0.5


def test_recall_matches_the_reference_fixture_case():
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    case = next(c for c in fixture["cases"] if c["kind"] == "recall_comparison")
    patterns = case["inputs"]["patterns"]
    query = case["inputs"]["query"]

    memory = TRAINING_SERVICE.train(
        TrainingDataset.from_patterns(patterns), encoding=case["encoding"]
    ).memory
    result = RECALL_SERVICE.recall(
        memory, query, expected_pattern=case["expected"]["target"]
    )

    assert np.array_equal(result.net_input, case["expected"]["net_input"])
    assert np.array_equal(result.output_pattern, case["expected"]["output"])
    assert result.comparison.mismatch_count == case["expected"]["mismatch_count"]
    assert result.comparison.exact_match == case["expected"]["exact_recall"]
    assert result.comparison.match_ratio == pytest.approx(0.75)


def test_incompatible_query_dimensions_are_rejected(hetero_memory):
    with pytest.raises(
        RecallError, match="Query length 3 does not match the trained source"
    ):
        RECALL_SERVICE.recall(hetero_memory, np.array([1, 0, 1]))


def test_incompatible_expected_length_is_rejected(hetero_memory):
    with pytest.raises(
        RecallError, match="expected_pattern length 2 does not match"
    ):
        RECALL_SERVICE.recall(
            hetero_memory,
            np.array([1, 0, 1, 0]),
            expected_pattern=np.array([1, 1]),
        )


def test_query_values_that_violate_the_memory_encoding_are_rejected(hetero_memory):
    with pytest.raises(RecallError, match="not valid for binary encoding"):
        RECALL_SERVICE.recall(hetero_memory, np.array([2, 0, 1, 0]))


def test_expected_values_that_violate_the_memory_encoding_are_rejected(hetero_memory):
    with pytest.raises(RecallError, match="not valid for binary encoding"):
        RECALL_SERVICE.recall(
            hetero_memory,
            np.array([1, 0, 1, 0]),
            expected_pattern=np.array([1, 1, 0, 2]),
        )


def test_unsupported_memory_object_is_rejected():
    with pytest.raises(RecallError, match="must be an AutoassociativeMemory"):
        RECALL_SERVICE.recall("not a memory", np.array([1, 0]))
    with pytest.raises(RecallError, match="must be an AutoassociativeMemory"):
        RECALL_SERVICE.recall(None, np.array([1, 0]))


def test_recall_result_arrays_are_read_only(hetero_memory):
    result = RECALL_SERVICE.recall(
        hetero_memory,
        np.array([1, 0, 1, 0]),
        expected_pattern=np.array([1, 1, 0, 0]),
    )

    for array in (
        result.input_pattern,
        result.net_input,
        result.output_pattern,
        result.expected_pattern,
    ):
        assert array.flags.writeable is False
        with pytest.raises(ValueError):
            array[0] = 7.0


def test_recalled_memory_object_still_supports_direct_inference(hetero_memory):
    """The trained model remains usable with the algorithm package API."""

    direct = hetero_memory.infer(np.array([1, 0, 1, 0]))

    assert np.array_equal(direct.activated_output, np.array([1.0, 1.0, 0.0, 0.0]))


def test_application_layer_imports_without_pyside6_or_gui_launch():
    """The complete workflow runs without launching the GUI.

    A fresh interpreter imports the full application layer; the check asserts
    that neither Qt nor PySide6 is loaded by those imports.
    """

    script = (
        "import sys\n"
        "import associative_memory.application.services.experiment_service\n"
        "import associative_memory.application.services.training_service\n"
        "import associative_memory.application.services.recall_service\n"
        "import associative_memory.application.services.image_service\n"
        "loaded = [name for name in sys.modules if 'PySide6' in name or name == 'Qt']\n"
        "print(loaded)\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        env={
            **os.environ,
            "PYTHONPATH": str(PROJECT_ROOT / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        check=True,
    )

    assert completed.stdout.strip() == "[]"
