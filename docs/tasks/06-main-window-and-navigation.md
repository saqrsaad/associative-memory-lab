Task 06 — Main Window, Navigation, and Application Shell

1. Objective

Build the desktop application's main window and navigation structure using the design system established in Task 05.

The result must provide a stable application shell into which the feature pages can be integrated.

2. Prerequisites

- Task 05 is complete.
- The application can launch a minimal PySide6 window.

3. Scope

Implement:

- Application entry point.
- Main window.
- Sidebar navigation.
- Top bar.
- Central content area.
- Page switching.
- Window sizing and layout behavior.
- Consistent page titles.
- Basic application-level error presentation.

Do not implement the complete business logic of the dataset, training, or recall pages.

4. Expected Modules

- src/associative_memory/presentation/app.py
- src/associative_memory/presentation/main_window.py
- src/associative_memory/presentation/navigation.py
- src/associative_memory/main.py
- Basic placeholder views for the planned navigation destinations.
- Relevant UI tests.

Reuse the existing project entry point if it already serves the same purpose.

5. Navigation Destinations

Provide navigation for:

1. Dashboard.
2. Image Dataset.
3. Training.
4. Recall and Results.
5. Experiment History.
6. Settings.

Use a single main application window for the primary workflow.

Do not create a separate top-level window for every navigation destination.

6. Layout Requirements

The application shell must include:

- A persistent navigation area.
- A compact header with the current page title.
- A central content area.
- Consistent page margins and spacing.
- Appropriate window minimum dimensions.
- Reasonable behavior when resized.

Use a clear visual distinction between the active navigation item and inactive items.

Avoid fixed pixel positions for the entire page. Prefer layouts and size policies that respond to window resizing.

7. Navigation Behavior

- Selecting a navigation item changes the central page.
- The active page is visually identifiable.
- Switching pages must not create duplicate page instances unnecessarily.
- Unsaved user changes must not be discarded silently when a future workflow requires confirmation.
- Navigation must remain responsive.

Placeholder pages must be clearly identified as not yet implemented; they must not pretend that functionality exists.

8. Error Handling

If a page cannot be initialized, display an understandable error or log the diagnostic details without crashing the entire application where feasible.

Do not suppress exceptions silently.

9. Required Tests

- Main window initializes.
- All planned navigation destinations are reachable.
- The active page changes correctly.
- Window resizing does not break the main layout.
- The application can be launched through the documented entry point.
- The test suite can run in a supported headless environment.

10. Acceptance Criteria

- The application launches into a consistent main window.
- Sidebar navigation works.
- The central content area can host all planned pages.
- The visual system from Task 05 is reused.
- No algorithm or preprocessing logic is embedded in the main window.
- The application remains usable when resized.

11. Completion Report

Report the navigation destinations, application entry point, files changed, tests run, actual results, and any placeholder functionality that remains.

Do not begin Task 07 until the application shell works reliably.