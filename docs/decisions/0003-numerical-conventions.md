# ADR 0003: Standardize Numerical Conventions

-   **Status:** Proposed
-   **Date:** 2026-10-19

## Context

Associative-memory equations are sensitive to vector orientation and
multiplication order. A mismatch between the learning equation and
recall equation can produce transposed or dimensionally invalid outputs.
Image encoding and flattening order also affect whether a recalled
vector reconstructs the intended image.

## Decision

Adopt the following canonical conventions for documentation, tests, and
new algorithm code:

1.  Use column vectors in the mathematical specification.
2.  For heteroassociative memory, define source (s
    `\in `{=tex}`\mathbb{R}`{=tex}\^{n `\times 1`{=tex}}), target (t
    `\in `{=tex}`\mathbb{R}`{=tex}\^{m `\times 1`{=tex}}), and
    (W=`\sum`{=tex}\_p s^{(p)}(t^{(p)})\^T), so (W) has shape (n
    `\times `{=tex}m).
3.  Recall a target using (h_t=W\^T s_q), producing a vector of length
    (m).
4.  For autoassociative memory, use (W=`\sum`{=tex}\_p
    x^{(p)}(x^{(p)})\^T), with shape (n `\times `{=tex}n), and a
    documented recall rule such as (h=Wx_q).
5.  Define Binary encoding as off/background (0), on/foreground (1).
6.  Define Bipolar encoding as off/background (-1), on/foreground (+1).
7.  Choose and test one image flattening order; row-major order is the
    recommended default.
8.  Specify activation threshold and exact tie behavior, including zero
    net input.
9.  Treat diagonal removal for autoassociative memory as an explicit
    configuration rather than an implicit universal rule.

NumPy implementations may use one-dimensional arrays, but operations and
tests must preserve these intended shapes.

## Alternatives considered

1.  **Unspecified conventions:** rejected because separate modules could
    implement incompatible equations.
2.  **Row vectors as the canonical mathematical notation:** valid, but
    not selected for the documentation. A row-vector implementation
    remains acceptable if all equations and tests are adapted
    consistently.
3.  **Implicit library defaults for ties and flattening:** rejected
    because defaults may be ambiguous or change behavior
    unintentionally.

## Consequences

### Positive

-   Matrix dimensions can be checked at every algorithm boundary.
-   Unit tests can verify orientation using small, hand-calculated
    examples.
-   Encoded patterns can be reconstructed consistently into images.
-   Experiment files can record conventions for future compatibility.

### Trade-offs

-   Existing code may need adapters or documentation updates if it uses
    another consistent convention.
-   Changes to these rules can affect saved experiments and therefore
    require explicit versioning or migration.

## Implementation notes

Before accepting this ADR, inspect the existing implementation and
tests. If a different convention is already used consistently, either
migrate it deliberately or amend this ADR to document the actual
convention. Do not change matrix orientation or serialized model
interpretation without updating tests, documentation, and compatibility
handling.
