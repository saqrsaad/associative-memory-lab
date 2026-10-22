"""Recall use cases for the application layer.

The service validates the query and expected pattern at the use-case
boundary, executes inference through the existing memory objects, and
performs the exact comparison criterion through ``algorithms.metrics``.
No numerical formula is implemented here.
"""

from __future__ import annotations

from numpy.typing import ArrayLike

from associative_memory.algorithms.autoassociative import AutoassociativeMemory
from associative_memory.algorithms.encoding import Encoding, validate_vector
from associative_memory.algorithms.heteroassociative import HeteroassociativeMemory
from associative_memory.algorithms.metrics import compare_vectors
from associative_memory.algorithms.results import InferenceResult
from associative_memory.application.dto.recall_result import (
    COMPARISON_BASIS_EXPECTED,
    COMPARISON_BASIS_INPUT,
    RecallComparison,
    RecallResult,
)
from associative_memory.application.dto.training_result import MemoryType

_MUTUAL_WARNING = (
    "Recall did not exactly reproduce the expected pattern "
    "({mismatch} of {count} elements differ). Simple Hebbian associative "
    "memories superpose stored patterns and can produce corrupted or "
    "spurious outputs through interference, especially when stored patterns "
    "are correlated or the query is corrupted."
)

_NO_COMPARISON_WARNING = (
    "No expected target was supplied, so recall success could not be "
    "evaluated (comparison is None, not a zero score). For heteroassociative "
    "memory, compare the output with the paired target pattern, not with "
    "the source cue."
)


class RecallError(Exception):
    """Raised when a recall use case cannot be performed.

    The message describes the problem at the use-case level; the original
    lower-level exception remains available as ``__cause__`` for logging.
    """


class RecallService:
    """Coordinate query validation, inference, and comparison."""

    def recall(
        self,
        memory: AutoassociativeMemory | HeteroassociativeMemory,
        query: ArrayLike,
        *,
        expected_pattern: ArrayLike | None = None,
    ) -> RecallResult:
        """Recall an output pattern from ``query`` using ``memory``.

        The query (and expected pattern, when supplied) are validated at the
        boundary with the memory's encoding before inference runs.
        Comparison policy follows the mathematical conventions: the output
        is compared against the supplied expected pattern, or — for
        autoassociative memory without one — against the supplied input
        pattern (the intended original). Heteroassociative recall without
        an expected target performs no comparison; it does not manufacture
        a score.

        Raises:
            RecallError: For unsupported memory objects, invalid query or
                expected patterns, or incompatible dimensions/encodings.
        """

        if not isinstance(memory, (AutoassociativeMemory, HeteroassociativeMemory)):
            raise RecallError(
                "memory must be an AutoassociativeMemory or a "
                "HeteroassociativeMemory trained by the algorithm package."
            )

        memory_type = (
            MemoryType.AUTOASSOCIATIVE
            if isinstance(memory, AutoassociativeMemory)
            else MemoryType.HETEROASSOCIATIVE
        )

        try:
            validate_vector(query, name="query", encoding=memory.encoding)
            inference: InferenceResult = memory.infer(query)
        except (TypeError, ValueError) as exc:
            raise RecallError(str(exc)) from exc

        if expected_pattern is not None:
            try:
                expected_vector = validate_vector(
                    expected_pattern,
                    name="expected_pattern",
                    encoding=memory.encoding,
                )
            except (TypeError, ValueError) as exc:
                raise RecallError(str(exc)) from exc
            if expected_vector.shape[0] != inference.target_dimension:
                raise RecallError(
                    f"expected_pattern length {expected_vector.shape[0]} does "
                    f"not match the memory output dimension "
                    f"{inference.target_dimension}."
                )
            comparison = self._compare(
                inference.activated_output, expected_vector, memory.encoding
            )
            comparison_basis: str | None = COMPARISON_BASIS_EXPECTED
            expected_output = expected_vector
        elif memory_type is MemoryType.AUTOASSOCIATIVE:
            expected_output = inference.input_pattern
            comparison = self._compare(
                inference.activated_output, expected_output, memory.encoding
            )
            comparison_basis = COMPARISON_BASIS_INPUT
        else:
            expected_output = None
            comparison: RecallComparison | None = None
            comparison_basis = None

        warnings: list[str] = []
        if comparison is None:
            warnings.append(_NO_COMPARISON_WARNING)
        elif comparison.mismatch_count > 0:
            warnings.append(
                _MUTUAL_WARNING.format(
                    mismatch=comparison.mismatch_count,
                    count=comparison.element_count,
                )
            )

        return RecallResult(
            memory_type=memory_type,
            encoding=memory.encoding,
            activation_name=inference.activation_name,
            threshold=inference.threshold,
            input_dimension=inference.source_dimension,
            output_dimension=inference.target_dimension,
            input_pattern=inference.input_pattern,
            net_input=inference.net_input,
            output_pattern=inference.activated_output,
            expected_pattern=expected_output,
            comparison=comparison,
            comparison_basis=comparison_basis,
            warnings=tuple(warnings),
        )

    def _compare(
        self,
        output: ArrayLike,
        expected: ArrayLike,
        encoding: Encoding,
    ) -> RecallComparison:
        """Run the algorithm-package comparison, wrapped for the boundary."""

        try:
            comparison = compare_vectors(output, expected, encoding)
        except (TypeError, ValueError) as exc:
            raise RecallError(str(exc)) from exc
        return RecallComparison(
            element_count=comparison.element_count,
            mismatch_count=comparison.mismatch_count,
            match_ratio=comparison.match_ratio,
            exact_match=comparison.exact_match,
        )
