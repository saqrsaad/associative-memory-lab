# ADR 0003: Use Row-Vector Numerical Conventions

- **Status:** Accepted for the Task 01 specification; implementation pending Task 02
- **Date:** 2026-10-19

## Context

Associative-memory equations are sensitive to vector orientation and
multiplication order. A mismatch between the learning equation and the recall
equation can produce transposed or dimensionally invalid outputs. Image encoding
and flattening order also affect whether a recalled vector reconstructs the
intended image.

Task 01 and `AGENTS.md` require row vectors throughout the application. The
repository contained no algorithm implementation when this decision was recorded,
so there was no existing code convention to preserve or migrate. The
authoritative numerical details and reference examples are in
`docs/mathematical-conventions.md` and `docs/algorithm-theory.md`.

## Decision

Adopt the following conventions for documentation, tests, and algorithm code:

1. Use row vectors throughout the application. A NumPy pattern is a
   one-dimensional array of shape `(n,)` and is treated as a row vector in the
   equations. Weight matrices use shape `(n, m)`.
2. For heteroassociative memory, define source `s` with shape `(n,)`, target `t`
   with shape `(m,)`, and `W = sum(outer(s, t))` with shape `(n, m)`.
3. Recall with `y_in = x @ W`, producing a net-input vector of shape `(m,)`, then
   apply the configured activation function.
4. For autoassociative memory, use `W = sum(outer(x, x))` with shape `(n, n)` and
   the same recall equation `y_in = x @ W`. Keep the full diagonal by default.
5. Define binary encoding as off/background `0` and on/foreground `1`. Define
   bipolar encoding as off/background `-1` and on/foreground `+1`. Treat encoding
   as explicit configuration and require an explicit conversion before changing
   representations.
6. Define binary activation as `1` when the net input is strictly greater than
   zero and `0` otherwise. Define bipolar activation as `+1` when the net input
   is strictly greater than zero and `-1` otherwise. At exactly zero, binary
   output is `0` and bipolar output is `-1`; never substitute `>= 0`.
7. Use row-major order for flattening a processed image into a pattern and for
   reconstructing a display grid.
8. Count a recall as exact only when the activated output matches the expected
   pattern elementwise with zero mismatches. Report mismatch count and match
   ratio otherwise.
9. Store the hand-calculated reference cases in
   `tests/fixtures/algorithm_reference_cases.json`; a later test must be able to
   parse and compare them.

## Alternatives considered

1. **Column vectors as the canonical notation:** valid in the literature and
   equivalent when all equations are adapted, but not selected for this project.
   A column-vector presentation would write learning with `s t^T` and recall with
   `W^T x`; mixing those with the row-vector equations above is forbidden.
2. **Unspecified conventions:** rejected because separate modules could implement
   incompatible equations and produce plausible but wrong shapes.
3. **Implicit library defaults for orientation, thresholds, and flattening:**
   rejected because defaults can be ambiguous or vary between operations.
4. **Preserve-the-cue or `>= 0` tie handling:** rejected for the basic algorithm
   because Task 01 defines the zero boundary exactly as `0` for binary and `-1`
   for bipolar output.
5. **Zero-diagonal or iterative Hopfield variants as defaults:** rejected because
   they are separate modeling decisions outside the basic course algorithm.

## Consequences

### Positive

- Matrix dimensions can be checked at every algorithm boundary.
- Unit tests can verify orientation using small, hand-calculated examples.
- Encoded patterns can be reconstructed consistently into images.
- Experiment files can record the conventions needed for future compatibility.
- Binary and bipolar results have an unambiguous interpretation at zero.

### Trade-offs

- Column-vector equations found in textbooks or lecture notes must be translated
  consistently before use.
- Changes to these rules can affect saved experiments and require explicit
  versioning or migration.
- Full-diagonal Hebbian memory can exhibit crosstalk and is not guaranteed to
  reconstruct every stored pattern.

## Implementation notes

Task 02 must implement the row-vector equations and the strict zero boundary in
the algorithm package and cover them with unit tests. This ADR does not claim
that implementation exists. If a different convention is later adopted, update
`docs/mathematical-conventions.md`, this ADR, affected tests, and persisted schema
or migration rules together. Do not change orientation or serialized matrix
interpretation silently.
