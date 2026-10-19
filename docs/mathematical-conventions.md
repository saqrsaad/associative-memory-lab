# Mathematical Conventions

**Status:** Authoritative for Task 01 and all later tasks. Last updated: 2026-10-19.

This document defines the numerical conventions used by Associative Memory Lab.
Algorithm code, tests, user-interface labels, and persisted experiment metadata
must follow these conventions. `docs/algorithm-theory.md` explains the ideas and
contains worked examples; it does not override this document. The architectural
decision is recorded in `docs/decisions/0003-numerical-conventions.md`.

## 1. Symbols

| Symbol | Meaning | Shape |
| --- | --- | --- |
| `P` | Number of training patterns or training pairs | scalar |
| `n` | Source/input pattern length | scalar |
| `m` | Target/output pattern length | scalar |
| `s^(p)` | Source row vector for training pair `p` | `(n,)` |
| `t^(p)` | Target row vector for training pair `p` | `(m,)` |
| `x^(p)` | Autoassociative training pattern `p` | `(n,)` |
| `W` | Weight matrix | `(n, m)` for heteroassociative memory, `(n, n)` for autoassociative memory |
| `x` | Query row vector | `(n,)` |
| `y_in` | Net-input vector before activation | `(m,)` |
| `y` | Activated recalled output | `(m,)` |
| `N` | Number of compared pattern elements | scalar |
| `M` | Mismatch count | scalar |

In this project a pattern is represented as a one-dimensional numeric array.
Mathematically it is treated as a row vector; if an explicit two-dimensional row
is required it has shape `(1, n)`. NumPy one-dimensional arrays do not encode
row-versus-column semantics, so operations, shape checks, and tests must enforce
the convention below.

## 2. Pattern representation and encoding

### 2.1 Binary encoding

- ON/foreground pixel: `1`
- OFF/background pixel: `0`
- Valid encoded values: `{0, 1}`

### 2.2 Bipolar encoding

- ON/foreground pixel: `+1`
- OFF/background pixel: `-1`
- Valid encoded values: `{-1, +1}`

### 2.3 Conversion

For a binary value `x_binary` and its bipolar equivalent `x_bipolar`:

$$
x_{\text{bipolar}} = 2x_{\text{binary}} - 1
$$

$$
x_{\text{binary}} = \frac{x_{\text{bipolar}} + 1}{2}
$$

For valid integer inputs both conversions must produce exact integers. A binary
`0` maps to bipolar `-1`, and a binary `1` maps to bipolar `+1`; the reverse
conversion returns the original binary values.

### 2.4 Explicit encoding configuration

Encoding is explicit configuration. It must never be inferred silently from the
numeric range of a vector, and binary and bipolar values must never be mixed
inside one calculation. A vector containing only `0` and `1` is not treated as
bipolar unless it is explicitly converted first.

Every encoded pattern and every comparison must retain its encoding metadata.
Training and recall for one experiment must use compatible preprocessing,
dimensions, traversal order, and encoding. Any intentional conversion must be
recorded as an explicit step.

## 3. Training pairs and memory modes

A training pair `p` is written:

$$
s^{(p)} \rightarrow t^{(p)}
$$

where the source has shape `(n,)` and the target has shape `(m,)`.

- **Heteroassociative memory:** the input and target patterns may differ, so
  `n` and `m` may differ. Each pair associates a source pattern with a distinct
  target pattern.
- **Autoassociative memory:** the target is the same pattern as the input. Using
  the autoassociative notation, `x^(p)` is both source and target and the
  dimension is `n = m`.

Training pairs are accumulated in their original order only for tracing and
reporting. The mathematical summation is order-independent apart from ordinary
floating-point considerations.

## 4. Canonical vector and matrix convention

**Row vectors are used throughout the application.**

For input dimension `n` and target dimension `m`:

| Quantity | Shape |
| --- | --- |
| Source `s` | `(n,)` |
| Target `t` | `(m,)` |
| Weight matrix `W` | `(n, m)` |
| Query `x` | `(n,)` |
| Net input `y_in = x @ W` | `(m,)` |
| Activated output `y` | `(m,)` |

Inference therefore uses:

$$
y_{\text{in}} = xW
$$

where `@` denotes ordinary matrix multiplication. For a query of shape `(n,)`
and a weight matrix of shape `(n, m)`, the result has shape `(m,)`.

Many texts use column vectors and write the equivalent operations with
transposes. Those column-vector conventions exist in the literature but are not
used in this project. Do not combine a column-vector learning equation with a
row-vector recall equation. If a source uses column vectors, translate the whole
calculation consistently before applying it here.

## 5. Hebbian learning and outer products

For each training pair, the heteroassociative contribution is the outer product:

$$
W^{(p)} = \operatorname{outer}\left(s^{(p)}, t^{(p)}\right)
$$

with entries:

$$
W_{ij}^{(p)} = s_i^{(p)} t_j^{(p)}
$$

The contribution has shape `(n, m)`. The accumulated heteroassociative weight
matrix is:

$$
W = \sum_{p=1}^{P} W^{(p)}
  = \sum_{p=1}^{P} \operatorname{outer}\left(s^{(p)}, t^{(p)}\right)
$$

and also has shape `(n, m)`.

For autoassociative memory:

$$
W = \sum_{p=1}^{P} \operatorname{outer}\left(x^{(p)}, x^{(p)}\right)
$$

which has shape `(n, n)`.

The outer product is the matrix formulation of the pairwise Hebbian
contribution; it is not a separate competing learning rule. For bipolar data,
negative products and negative accumulated weights are expected and must be
preserved.

The basic course algorithm keeps the full diagonal of `W`. Setting the diagonal
to zero is not a default, is not an implicit normalization step, and is outside
the basic course algorithm. Any zero-diagonal variant requires a separate,
documented configuration decision. Hopfield iterative dynamics are likewise
outside this task and must not be introduced as hidden behavior.

## 6. Inference

Inference uses the row-vector multiplication:

$$
y_{\text{in}} = xW
$$

For a query `x` of shape `(n,)` and a weight matrix `W` of shape `(n, m)`, the
net-input vector `y_in` has shape `(m,)`. For autoassociative memory `W` is
square and the same equation applies.

The net input is the value before activation. It is not the recalled pattern and
must not be reported as the final output. Activation is applied elementwise
afterward.

## 7. Activation functions

Activation converts each net-input component to an encoded output value.

### 7.1 Binary activation

$$
f_{\text{binary}}(a) =
\begin{cases}
1, & a > 0 \\
0, & a \le 0
\end{cases}
$$

### 7.2 Bipolar activation

$$
f_{\text{bipolar}}(a) =
\begin{cases}
+1, & a > 0 \\
-1, & a \le 0
\end{cases}
$$

| Net input `a` | Binary output | Bipolar output |
| --- | --- | --- |
| `a < 0` | `0` | `-1` |
| `a = 0` | `0` | `-1` |
| `a > 0` | `1` | `+1` |

The comparison is strictly `a > 0`. Zero maps to `0` for binary activation and
to `-1` for bipolar activation. Do not substitute `a >= 0`, do not preserve the
query value at zero, and do not apply a different tie rule. The zero case is part
of the algorithm and must be covered by algorithm tests in Task 02.

The output encoding must match the configured encoding. Do not apply binary
activation to a bipolar experiment or vice versa without an explicit,
documented conversion.

## 8. Recall evaluation, interference, and success criteria

A numerical computation can succeed while pattern recall fails. The net-input
vector and the activated output must therefore be distinguished from successful
reconstruction.

For a recalled output `y` and expected target `t` of the same length `N > 0` and
the same encoding:

$$
M = \sum_{i=1}^{N} \mathbf{1}\left[y_i \ne t_i\right]
$$

$$
\text{match ratio} = \frac{N - M}{N}
$$

**Exact recall succeeds only when `M = 0`.** If `M > 0`, report the mismatch
count and match ratio rather than describing the result as a successful recall.
Negative results and unexpected outputs are valid outcomes of the numerical
computation when the storage assumptions are not met.

Simple Hebbian memory superposes all stored contributions. Correlated or
overlapping patterns can cause crosstalk, producing corrupted patterns, mixtures,
or outputs that match no stored target (spurious outputs). An autoassociative
network can fail to reconstruct a stored pattern for the same reason, especially
when the query is corrupted or when the stored patterns are strongly correlated.

Do not claim a universal capacity limit unless the model, assumptions, and
derivation are explicitly specified. Do not introduce iterative Hopfield cleanup
or a zero-diagonal weight rule as a default in order to improve recall.

For autoassociative recall, compare the output with the intended original
pattern. For heteroassociative recall, compare the output with the paired target,
not with the source cue. Comparisons are valid only when dimensions and encoding
semantics are compatible. If they are not compatible, report the comparison as
invalid rather than manufacturing a score.

## 9. Image flattening and dimensions

If a processed image has height `H` and width `K`, its flattened pattern length
is:

$$
N = HK
$$

Use row-major order (left-to-right across a row, then top-to-bottom across rows)
for flattening and for reconstructing a display grid. The same order must be used
for training and recall. A mismatch between flattening and reshaping can scramble
a recalled image even when the numeric vector is correct.

## 10. Numeric representation and validation

- Validate that pattern vectors are one-dimensional and non-empty at algorithm
  boundaries, or normalize them in one explicit helper.
- Validate that all values are finite. Reject NaN and infinity unless a
  deliberately supported feature defines otherwise.
- Validate valid encoding values before learning: binary values must be `0` or
  `1`; bipolar values must be `-1` or `+1`.
- Validate expected lengths and matrix shapes before every multiplication.
- Avoid implicit broadcasting that could produce a plausible but incorrect
  result.
- Use a numeric dtype capable of representing the selected encoding and the
  accumulated values. Do not hide or round underlying calculation values merely
  for display.
- Keep accumulation deterministic for a fixed input order and configuration.
  Preserve the original pair order for diagnostic traces.

The reference examples use small integers so their expected values can be checked
exactly by hand. An implementation may use floating-point net inputs, but the
small reference cases must still reproduce the documented integer values.

## 11. Reference examples and machine-readable fixture

Hand-calculated reference examples are documented in
`docs/algorithm-theory.md`, Section 7. They cover a single outer product,
accumulation of multiple Hebbian contributions, heteroassociative inference,
autoassociative inference, binary and bipolar activation at positive, negative,
and zero inputs, encoding conversion, and a recall failure caused by
interference.

The same hand-calculated cases are stored in
`tests/fixtures/algorithm_reference_cases.json` for use by tests in later tasks.
The fixture is data, not an executable algorithm; its values were derived by hand
and must not be replaced by values generated from an implementation.

### 11.1 Fixture schema

The fixture is a JSON object with these top-level fields:

- `schema_version`: integer, currently `1`.
- `vector_orientation`: string, must be `"row"`.
- `matrix_layout`: human-readable description of matrix row ordering.
- `description`: provenance statement.
- `cases`: array of reference-case objects.

Each case object has:

- `id`: unique string identifier.
- `kind`: one of `encoding_conversion`, `outer_product`, `hebbian_sum`,
  `heteroassociative_inference`, `autoassociative_inference`, `activation`, or
  `recall_comparison`.
- `encoding`: `"binary"` or `"bipolar"`.
- `inputs`: object mapping input names to vectors or matrices. A collection of
  `P` vectors is represented as a `P` by `n` array.
- `expected`: object containing expected values. It may include a `dimensions`
  object mapping each input or result name to its array shape, for example
  `[3, 2]` for a matrix with three rows and two columns.
- Optional fields such as `mismatch_count` and `exact_recall` may appear in
  `expected` for recall comparisons.

All numeric values in the fixture are integers. Matrix entries are written as
arrays of rows. A later test may load the file with the standard `json` module
and compare shapes and values exactly.

## 12. Convention changes

Changing vector orientation, weight layout, flattening order, activation tie
handling, encoding semantics, or serialized matrix interpretation is a
compatibility-sensitive change. Update this document, ADR 0003, affected tests,
and persisted schema or migration rules together. Do not allow different modules
to follow different conventions.
