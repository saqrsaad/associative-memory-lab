# Architecture

## 1. Purpose and scope

Associative Memory Lab is an educational Python application for
building, training, inspecting, and testing associative-memory neural
networks using image patterns. The application should make the
relationship between image preprocessing, pattern encoding, Hebbian
learning, weight matrices, and recall observable to the learner.

This document defines the intended architecture. When the implementation
differs, update this document and the relevant Architecture Decision
Record (ADR) rather than silently allowing the documentation and code to
diverge.

## 2. Architectural principles

-   **Separation of concerns:** GUI code presents state and collects
    user input; it does not implement neural-network mathematics.
-   **Explicit data stages:** original image, preprocessed image,
    encoded pattern, trained model, and recall result are distinct
    objects.
-   **Numerical transparency:** matrix shapes, encoding conventions,
    thresholds, and recall rules are inspectable.
-   **Reproducibility:** an experiment records the configuration and
    inputs needed to understand how a result was produced.
-   **Testability:** domain logic and numerical operations should be
    testable without launching the GUI.
-   **Safe persistence:** load files as untrusted input, validate schema
    and array dimensions, and avoid arbitrary object deserialization.
-   **Incremental development:** keep interfaces small and add
    capabilities only when supported by a defined task.

## 3. Proposed layers

### Presentation layer

Contains the PySide6 application shell, windows, pages, reusable
widgets, dialogs, view models/controllers, and UI state handling.

Responsibilities: - Collect user choices and validate basic form
input. - Display datasets, image stages, pattern vectors, model
information, and recall comparisons. - Show progress, validation errors,
warnings, and operation status. - Call application services and render
their results.

The presentation layer must not directly calculate outer products,
update weight matrices, or implement recall equations.

### Application layer

Coordinates user workflows through use-case services, for example: -
Import image / create image pair. - Preprocess and encode pattern. -
Train associative-memory model. - Recall output from a query pattern. -
Save, load, compare, and export an experiment.

Services validate workflow-level requirements, call domain components,
and return structured results suitable for presentation. They should
avoid depending on concrete GUI widgets.

### Domain layer

Defines the core concepts and rules: - Image and preprocessing
configuration. - Encoded pattern and pattern pair. - Memory type and
encoding type. - Training configuration and trained memory model. -
Recall configuration and recall result. - Experiment metadata and
validation rules.

Domain objects should describe meaning and invariants, not GUI layout.

### Numerical / algorithm layer

Implements pure or near-pure numerical operations: - Image-to-pattern
encoding helpers where appropriate. - Hebbian outer-product learning. -
Autoassociative and heteroassociative weight construction. - Optional
diagonal/self-connection policy for autoassociative memory. - Recall
activation and thresholding. - Similarity, mismatch, and match-ratio
metrics.

This layer should make matrix dimensions explicit and raise clear errors
for incompatible shapes. It should be testable with small
hand-computable examples.

### Infrastructure layer

Handles external concerns: - Image file loading and export. - Experiment
serialization and deserialization. - File-system paths and application
settings. - Logging and diagnostic information.

Infrastructure should translate external formats into validated internal
objects. It must not silently accept malformed or dimensionally
inconsistent experiment data.

## 4. Dependency direction

The intended dependency direction is:

`Presentation -> Application -> Domain / Numerical`

`Application -> Infrastructure interfaces <- Infrastructure implementations`

Domain and numerical code should not import PySide6 or depend on
widgets. Infrastructure may depend on domain data definitions, but
domain logic should not depend on infrastructure implementations.

## 5. Suggested package layout

Adapt this layout to the repository's existing structure rather than
duplicating modules that already exist.

``` text
src/
  associative_memory_lab/
    main.py
    presentation/
      main_window.py
      pages/
      widgets/
      dialogs/
      resources/
    application/
      services/
      dto/
    domain/
      models.py
      enums.py
      validation.py
    algorithms/
      hebbian.py
      recall.py
      metrics.py
    infrastructure/
      image_io.py
      experiment_repository.py
      settings.py
tests/
  unit/
  integration/
docs/
  architecture.md
  algorithm-theory.md
  mathematical-conventions.md
  ui-design-system.md
  user-guide.md
  development-guide.md
  decisions/
```

This is a recommendation, not a claim that these exact modules already
exist.

## 6. Core data flow

1.  The user imports an image or defines a source-target pair.
2.  Infrastructure loads the image and preserves its source identity.
3.  Preprocessing produces a separate, inspectable image representation.
4.  Encoding converts the processed pixels into a one-dimensional Binary
    or Bipolar pattern.
5.  The application validates the selected memory type and compatible
    source-target dimensions.
6.  The algorithm layer builds the weight matrix from the encoded
    training patterns.
7.  The application stores the model and its configuration in the
    experiment state.
8.  A query pattern is passed to the recall operation.
9.  The result includes the raw net input, activated output, decoded
    pattern/image, and applicable comparison metrics.
10. The presentation layer displays the intermediate and final results.

The original image must never be silently overwritten by preprocessing.
A recall result must not be presented as guaranteed to reproduce the
target: crosstalk, interference, threshold choices, and model
limitations can affect the result.

## 7. Main domain concepts

-   **ImageAsset:** source path or import identity plus source metadata.
-   **PreprocessingConfig:** resize dimensions, grayscale policy,
    threshold, and foreground polarity.
-   **EncodedPattern:** values, encoding type, logical dimensions, and
    source reference.
-   **PatternPair:** source pattern and target pattern for
    heteroassociative learning.
-   **MemoryConfig:** memory type, encoding, diagonal policy, and recall
    rule.
-   **MemoryModel:** validated weight matrix plus configuration and
    training metadata.
-   **RecallResult:** query, net input, activated output, decoded
    pattern, and metrics.
-   **Experiment:** dataset references, configuration, model state,
    results, and schema version.

Concrete field names may vary, but the meanings and invariants should
remain explicit.

## 8. Error handling

Use actionable, user-readable errors for: - Unsupported or unreadable
image files. - Missing source-target pairs. - Empty datasets. -
Incompatible vector lengths or matrix dimensions. - Invalid thresholds
or unsupported configuration combinations. - Corrupt, incomplete, or
incompatible experiment files.

Detailed diagnostic information may be logged for developers, but raw
tracebacks should not be the normal user-facing error message.

## 9. Persistence and compatibility

Persist human-readable metadata separately from numeric arrays where
practical. Record a schema version, algorithm settings, preprocessing
settings, encoding, memory type, pattern dimensions, and software
version when available. Validate all data on load. Do not use
unrestricted pickle loading for files supplied by users.

A saved experiment should either load correctly, be migrated through an
explicit migration, or fail with a clear explanation. It must not
silently reinterpret an old weight matrix under new conventions.

## 10. Testing strategy

-   Unit-test encoding, weight construction, activation, metrics, and
    validation without a GUI.
-   Use tiny arrays with known expected results to verify equations and
    orientation.
-   Test image preprocessing on fixtures with known dimensions and pixel
    values.
-   Integration-test application services using temporary files and
    deterministic data.
-   Test persistence round-trips and malformed-file rejection.
-   Keep GUI tests focused on navigation, input validation, and
    rendering of service results.

## 11. Architecture change policy

A change that crosses layer boundaries, changes matrix conventions,
changes serialized formats, or introduces a new major GUI dependency
should be documented in the relevant ADR. Update this document when the
actual architecture changes.
