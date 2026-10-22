"""Core associative-memory algorithms.

This package contains the numerical Hebbian associative-memory operations. It
does not import PySide6, image-loading, persistence, or presentation modules.
"""

from .activation import (
    ACTIVATION_THRESHOLD,
    activate,
    binary_activation,
    bipolar_activation,
)
from .autoassociative import (
    AutoassociativeMemory,
    infer_autoassociative,
    train_autoassociative,
)
from .encoding import Encoding, validate_encoding_values, validate_vector
from .hebbian import HebbianTrainingResult, accumulate_hebbian, hebbian_weight_matrix
from .heteroassociative import (
    HeteroassociativeMemory,
    infer_heteroassociative,
    train_heteroassociative,
)
from .metrics import PatternComparison, compare_vectors
from .outer_product import outer_product
from .results import InferenceResult

__all__ = [
    "ACTIVATION_THRESHOLD",
    "AutoassociativeMemory",
    "Encoding",
    "HebbianTrainingResult",
    "HeteroassociativeMemory",
    "InferenceResult",
    "PatternComparison",
    "accumulate_hebbian",
    "activate",
    "binary_activation",
    "bipolar_activation",
    "compare_vectors",
    "hebbian_weight_matrix",
    "infer_autoassociative",
    "infer_heteroassociative",
    "outer_product",
    "train_autoassociative",
    "train_heteroassociative",
    "validate_encoding_values",
    "validate_vector",
]
