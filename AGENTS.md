AGENTS.md

Associative Memory Lab — Agent Development Rules

1. Project Mission

Build a maintainable educational desktop application in Python that demonstrates the associative memory algorithms covered in the course material.

The application must help users understand both:

- The numerical behavior of the algorithms.
- The relationship between input images, numerical patterns, weight matrices, activation, and recall results.

The project must remain readable, testable, documented, and suitable for future contributors.

2. Mandatory Working Rules

Before making changes:

1. Read this file.
2. Read the relevant task specification in docs/tasks/ when available.
3. Inspect the existing code and tests.
4. Identify the files expected to change.
5. Implement only the current task.

Do not implement all project phases in one attempt.

Do not proceed to a later phase unless the current phase satisfies its acceptance criteria.

Do not replace existing work without inspecting it first.

3. Architecture Boundaries

Follow the documented layered architecture.

- domain contains core data structures and domain rules.
- algorithms contains mathematical implementations.
- application coordinates use cases and produces result objects.
- infrastructure handles file access, persistence, logging, and external resources.
- presentation contains PySide6 widgets, views, viewmodels, and visual styling.
- tests validate behavior at unit, integration, and UI levels.

Dependency rules:

- algorithms must not import PySide6.
- domain must not import PySide6, image-loading libraries, or presentation modules.
- application must not depend on concrete GUI widgets.
- presentation must call application services rather than implement algorithm equations.
- infrastructure must not own presentation logic.
- avoid circular imports.

4. Mathematical Conventions

Follow docs/mathematical-conventions.md.

For the row-vector convention:

- input patterns have shape (n,).
- target patterns have shape (m,).
- heteroassociative weight matrices have shape (n, m).
- inference uses y_in = x @ W.
- heteroassociative training uses W = sum(outer(s, t)).
- autoassociative training uses W = sum(outer(s, s)).

Use the exact activation threshold documented for the experiment.

Never silently switch between row-vector and column-vector conventions.

Never silently change binary/bipolar encoding, weight updates, diagonal handling, or the learning rule.

Document mathematical changes and add tests.

5. Image Processing Rules

- Support PNG, JPEG, and BMP input where supported by the installed image library.
- Validate files before processing.
- Make grayscale conversion, resizing, thresholding, and encoding explicit.
- Use consistent preprocessing for training and recall.
- Keep the original image separate from the processed representation.
- Never assume that every image uses a dark foreground and light background.
- Display the selected threshold, output dimensions, and encoding convention.
- Report unsupported formats and invalid dimensions clearly.
- Do not silently discard color or aspect-ratio information without documenting the operation.

6. Algorithm Requirements

- Implement the course's Hebbian and outer-product methods directly.
- Keep autoassociative and heteroassociative behavior distinct.
- Validate vector and matrix dimensions.
- Return structured results rather than UI-specific objects.
- Make intermediate numerical results available for educational visualization.
- Do not claim successful recall unless the result meets an explicitly defined criterion.
- Explain that simple Hebbian memory can exhibit interference and spurious outputs.

7. User Interface Requirements

- Use PySide6.
- Build a consistent, modern, responsive desktop interface.
- Separate views, widgets, viewmodels, and application services.
- Keep visual styling centralized in the theme system.
- Use consistent spacing, typography, colors, borders, and interaction states.
- Provide clear empty, loading, success, warning, and error states.
- Use accessible labels, tooltips, keyboard navigation, and readable contrast.
- Do not block the GUI thread during long-running operations.
- Avoid unnecessary animations and excessive decoration.
- Never display fake metrics or fabricated training results.

8. Result Visualization Requirements

When applicable, show:

- Original input image.
- Preprocessed image.
- Binary or bipolar pattern.
- Flattened vector and its dimensions.
- Individual outer-product matrices.
- Final weight matrix.
- Net input before activation.
- Output after activation.
- Expected target or original pattern.
- Recall comparison and mismatch count.
- Relevant warnings about interference or unsupported assumptions.

Use clear labels, dimensions, legends, and numeric formatting.

For large matrices, provide pagination, cropping, or summarized views while preserving access to the full data.

Do not round or hide values in the underlying calculation merely to simplify the display.

9. Code Quality

- Use clear English identifiers.
- Use type hints for public interfaces.
- Add concise docstrings where they clarify behavior.
- Prefer small, focused functions.
- Avoid duplicated logic and unnecessary abstraction.
- Handle expected errors explicitly.
- Never suppress exceptions silently.
- Avoid global mutable state.
- Do not introduce dependencies without a clear reason.

10. Testing Requirements

Use pytest.

- Test algorithm calculations with known numerical examples.
- Test matrix dimensions and activation boundaries.
- Test binary and bipolar conversion.
- Test invalid files and incompatible dimensions.
- Test application workflows independently of the GUI.
- Add UI tests for critical user flows.
- Run relevant tests after every meaningful change.
- Report the exact checks run and their results.
- Never claim a test passed unless it actually ran successfully.

11. Documentation Requirements

- Update README.md when setup, execution, or user workflows change.
- Document architecture and dependency boundaries.
- Document mathematical conventions and preprocessing assumptions.
- Keep task specifications and acceptance criteria up to date.
- Record significant architectural decisions in docs/decisions/.
- Do not document unimplemented features as complete.

12. Git and Security

- Make small, reviewable changes.
- Never commit secrets, API keys, personal data, local environments, or generated caches.
- Inspect git status and the diff before proposing a commit.
- Do not discard unrelated user changes.
- Do not force-push, reset destructively, or push changes without explicit authorization.
- Use descriptive commit messages.

13. Required Task Completion Report

At the end of each task, report:

1. What was implemented.
2. Which files were created or changed.
3. Which tests or checks were executed.
4. Their actual results.
5. Any known limitations.
6. Whether all acceptance criteria were met.
7. The exact next task that may begin.

If acceptance criteria are not met, explain the blocker and do not mark the task complete.

14. Scope Discipline

- Implement only the active task.
- Do not add unrelated features.
- Do not replace the course algorithm with CNNs, modern Hopfield variants, or other models.
- Prefer simple, maintainable code over premature optimization.
- Ask for clarification when ambiguity materially affects correctness.