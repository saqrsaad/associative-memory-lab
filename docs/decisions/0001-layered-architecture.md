# ADR 0001: Use a Layered Architecture

-   **Status:** Proposed
-   **Date:** 2026-10-19

## Context

Associative Memory Lab combines a desktop GUI, image processing,
numerical learning algorithms, recall, and experiment persistence. If
these responsibilities are mixed in widgets or a single application
module, mathematical behavior becomes difficult to test, UI changes can
break algorithms, and persistence details can leak into domain logic.

## Decision

Organize the application around distinct presentation, application,
domain/numerical, and infrastructure responsibilities.

-   Presentation handles user interaction and rendering.
-   Application services coordinate workflows.
-   Domain and numerical modules define core concepts, validation,
    learning, recall, and metrics.
-   Infrastructure handles file I/O, persistence, settings, and external
    formats.

Dependencies should point inward toward stable domain and numerical
concepts. Domain and algorithm modules must not depend on PySide6
widgets.

## Alternatives considered

1.  **Single-module or widget-centric implementation:** initially quick,
    but difficult to test and extend.
2.  **Highly distributed microservice architecture:** adds operational
    complexity without a clear benefit for a local educational desktop
    application.
3.  **Layered modular monolith:** selected because it supports local
    execution, testable numerical code, and a manageable codebase.

## Consequences

### Positive

-   Numerical algorithms can be tested without starting the GUI.
-   UI redesigns are less likely to change mathematical behavior.
-   Persistence can validate data at a clear boundary.
-   Responsibilities and ownership are easier to understand.

### Trade-offs

-   Requires small interfaces between layers.
-   May introduce additional files and data-transfer objects.
-   Developers must avoid unnecessary abstraction for trivial
    operations.

## Implementation notes

This ADR describes the intended design. Mark it Accepted only after the
repository structure and dependencies conform to the decision. If the
existing project uses different but equivalent boundaries, update the
ADR to describe the actual choice rather than restructuring code
mechanically.
