# Development Guide

## 1. Purpose

This guide explains how to work on Associative Memory Lab while
preserving reproducibility, numerical correctness, and a maintainable
GUI architecture. Adjust commands to the repository's actual package
manager and entry point; do not assume a command exists until it is
defined by project configuration.

## 2. Prerequisites

The project is intended to use Python and PySide6. Check the repository
configuration for the supported Python version and declared dependencies
before installing anything. Use a separate virtual environment for this
project to avoid mixing dependencies with other Python projects.

Typical environment setup on a Unix-like shell:

``` bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell, activation is commonly:

``` powershell
.venv\Scripts\Activate.ps1
```

Install the project's declared dependencies using its documented
dependency file or package manager. Do not add dependencies solely
because they are convenient without checking whether the project already
has an approved solution.

## 3. Repository orientation

Use the actual repository tree as the source of truth. A recommended
separation is:

-   `presentation/`: PySide6 windows, pages, widgets, dialogs, themes,
    and UI resources.
-   `application/`: workflow services and data-transfer objects.
-   `domain/`: core concepts, enums, invariants, and validation.
-   `algorithms/`: Hebbian learning, recall, activation, and metrics.
-   `infrastructure/`: image I/O, experiment persistence, settings, and
    logging.
-   `tests/`: unit and integration tests.
-   `docs/`: architecture, theory, conventions, guides, and ADRs.

Do not create duplicate packages if equivalent modules already exist.
Prefer small, focused changes consistent with the existing structure.

## 4. Development workflow

1.  Read `AGENTS.md` and the relevant task specification before editing.
2.  Inspect the existing code and tests for the affected behavior.
3.  Identify the smallest set of files needed.
4.  Implement the requested change within scope.
5.  Add or update tests when the task permits and the behavior warrants
    them.
6.  Run only the checks authorized by the task or project instructions.
7.  Review the diff for unrelated changes, accidental generated files,
    and convention mismatches.
8.  Update documentation when behavior, interfaces, conventions, or
    persistence formats change.
9.  Report changed files, behavior, checks performed, and known
    limitations.

Do not silently expand a task into unrelated refactoring, dependency
installation, formatting of unrelated files, or release/publishing
actions.

## 5. Numerical implementation rules

-   Follow `mathematical-conventions.md` as the canonical source for
    vector orientation and dimensions.
-   Validate vector lengths and matrix shapes before numerical
    operations.
-   Keep preprocessing, encoding, learning, activation, and decoding as
    explicit stages.
-   Do not let widgets implement learning or recall equations.
-   Use deterministic, hand-computable fixtures for core algorithm
    tests.
-   Test zero net input and threshold-boundary behavior explicitly.
-   Validate values for NaN and infinity where numerical arrays enter or
    leave the system.
-   Make dtype conversions explicit when they can affect results.
-   Avoid accidental NumPy broadcasting that masks incompatible shapes.

A one-dimensional NumPy array does not itself encode row-versus-column
semantics. The operation and its shape tests must enforce the documented
convention.

## 6. GUI implementation rules

-   Use PySide6 widgets and layouts consistent with the existing
    architecture.
-   Keep styling centralized and reusable.
-   Use signals/slots or explicit service interfaces to communicate
    actions.
-   Pass structured data to views instead of embedding algorithm logic
    in widgets.
-   Avoid blocking the GUI thread during expensive work.
-   Return results to the GUI thread using Qt-safe mechanisms.
-   Provide empty, loading, error, and success states where relevant.
-   Ensure errors are understandable to users and actionable.
-   Do not report completion before the underlying operation succeeds.

## 7. Testing strategy

The exact commands depend on the repository's configured tooling. Check
project configuration and documentation before choosing commands.

Recommended coverage: - Unit tests for preprocessing and encoding. -
Unit tests for outer-product matrices and recall orientation. - Tests
for Binary and Bipolar activation, including ties and threshold
boundaries. - Tests for mismatch and match-ratio metrics. - Validation
tests for incompatible dimensions and invalid values. - Persistence
round-trip tests and malformed-file rejection. - Integration tests for
application services. - Focused GUI tests for navigation and
user-visible states.

Use small arrays with manually calculated expected matrices. A test that
merely verifies "no exception occurred" is not sufficient for a
mathematical operation.

## 8. Persistence and security

-   Store a schema version in experiment files.
-   Validate all fields and numeric array shapes during loading.
-   Avoid unrestricted pickle deserialization for user-provided files.
-   Keep file paths and user-provided names separate from executable
    behavior.
-   Handle missing, truncated, malformed, and incompatible data
    gracefully.
-   Do not overwrite an existing experiment without the intended user
    action and a clear outcome.

## 9. Documentation and ADRs

Update documentation when a change affects: - Layer boundaries or
package responsibilities. - Vector/matrix orientation or flattening
order. - Encoding or activation behavior. - GUI framework or major UI
architecture. - Experiment file schema or compatibility. - User-visible
workflow.

Use an ADR for decisions with meaningful long-term consequences. ADRs
should state the context, decision, alternatives, consequences, and
status. Avoid treating a proposed decision as implemented until the code
and project configuration confirm it.

## 10. Code review checklist

-   Is the change within the requested scope?
-   Are numerical shapes and conventions explicit?
-   Are invalid inputs rejected with useful messages?
-   Does the UI display service results rather than reimplementing them?
-   Are tests meaningful and deterministic?
-   Are persisted data and compatibility handled safely?
-   Is documentation consistent with the actual behavior?
-   Are there unrelated files or generated artifacts in the diff?

## 11. Contribution expectations

Keep pull requests focused and describe the user-visible or algorithmic
behavior changed. Include the rationale, relevant tests/checks,
screenshots for significant UI changes when appropriate, and known
limitations. Do not claim checks passed if they were not run.
