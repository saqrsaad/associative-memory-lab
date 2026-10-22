"""Unit tests for binary and bipolar activation."""

from __future__ import annotations

import numpy as np
import pytest

from associative_memory.algorithms.activation import (
    ACTIVATION_THRESHOLD,
    activate,
    binary_activation,
    bipolar_activation,
)
from associative_memory.algorithms.encoding import Encoding


def test_binary_activation_matches_reference_case(reference_case):
    case = reference_case("binary_activation_boundaries")

    result = binary_activation(case["inputs"]["net_input"])

    assert np.array_equal(
        result,
        np.asarray(case["expected"]["output"], dtype=float),
    )


def test_bipolar_activation_matches_reference_case(reference_case):
    case = reference_case("bipolar_activation_boundaries")

    result = bipolar_activation(case["inputs"]["net_input"])

    assert np.array_equal(
        result,
        np.asarray(case["expected"]["output"], dtype=float),
    )


def test_activation_threshold_is_zero():
    assert ACTIVATION_THRESHOLD == 0.0


def test_binary_activation_scalar_boundaries():
    assert float(binary_activation(2.0)) == 1.0
    assert float(binary_activation(0.0)) == 0.0
    assert float(binary_activation(-0.0)) == 0.0
    assert float(binary_activation(-3.0)) == 0.0


def test_bipolar_activation_scalar_boundaries():
    assert float(bipolar_activation(2.0)) == 1.0
    assert float(bipolar_activation(0.0)) == -1.0
    assert float(bipolar_activation(-0.0)) == -1.0
    assert float(bipolar_activation(-3.0)) == -1.0


def test_activate_dispatches_on_encoding():
    net_input = [2.0, 0.0, -3.0]

    assert np.array_equal(
        activate(net_input, Encoding.BINARY),
        np.array([1.0, 0.0, 0.0]),
    )
    assert np.array_equal(
        activate(net_input, "bipolar"),
        np.array([1.0, -1.0, -1.0]),
    )


def test_activate_rejects_unknown_encoding():
    with pytest.raises(ValueError):
        activate([1.0], "ternary")


@pytest.mark.parametrize(
    "net_input",
    [
        np.zeros((2, 2)),
        [1.0, np.nan],
        [1.0, np.inf],
        ["a", "b"],
    ],
)
def test_activation_rejects_invalid_net_inputs(net_input):
    with pytest.raises((ValueError, TypeError)):
        binary_activation(net_input)
    with pytest.raises((ValueError, TypeError)):
        bipolar_activation(net_input)


def test_activation_preserves_vector_shape():
    result = binary_activation([1.0, -1.0, 0.0])
    assert result.shape == (3,)

    scalar_result = binary_activation(1.0)
    assert scalar_result.shape == ()
