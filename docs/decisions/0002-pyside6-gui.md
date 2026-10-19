# ADR 0002: Use PySide6 for the Desktop GUI

-   **Status:** Proposed
-   **Date:** 2026-10-19

## Context

The application needs a desktop interface for image previews, dataset
management, configuration forms, numeric matrix/vector inspection,
recall comparisons, and experiment history. The interface must support
reusable widgets, layouts, event handling, and a maintainable separation
from the algorithm layer.

## Decision

Use PySide6 (the official Python bindings for Qt 6) as the intended
desktop GUI framework, subject to confirmation against the repository's
actual dependencies and task requirements.

Keep UI code in the presentation layer. Communicate with application
services through clear methods, signals/slots, or controller-style
interfaces. Use Qt layouts and reusable widgets rather than absolute
positioning. Long-running operations should not block the GUI thread.

## Alternatives considered

1.  **Tkinter:** included with many Python installations and suitable
    for small tools, but may require more custom work for a polished,
    data-rich desktop laboratory UI.
2.  **Web-based UI:** offers web styling and distribution options but
    introduces a different application and packaging model.
3.  **Other Qt Python bindings:** could provide similar capabilities,
    but using multiple bindings would complicate dependencies and
    maintenance.

## Consequences

### Positive

-   Mature widget, layout, event, and desktop application facilities.
-   Suitable for reusable image, matrix, vector, and workflow
    components.
-   Supports a clear separation between GUI and numerical services.

### Trade-offs

-   PySide6 must be included in the environment and packaging
    configuration.
-   Developers need familiarity with Qt's event model, signals/slots,
    and thread-affinity rules.
-   Packaging the desktop application requires deliberate handling of Qt
    resources and deployment.

## Implementation notes

This ADR is proposed until the project configuration and source code
confirm PySide6 is in use. If another framework has already been
selected and implemented, revise this ADR rather than introducing
PySide6 solely to match this document.
