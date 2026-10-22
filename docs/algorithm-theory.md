# Algorithm Theory

## 1. Purpose and scope

Associative memory stores relationships between patterns and retrieves an
associated pattern when given a cue. In this project, patterns are vectors
derived from images after preprocessing and encoding.

Two memory modes are in scope:

- **Autoassociative memory:** stores a pattern in association with itself. A
  query is intended to recover or clean up that same pattern.
- **Heteroassociative memory:** stores an association between a source pattern
  and a different target pattern.

This document explains the concepts, the learning rule, recall, interference,
and the worked reference examples. The normative numerical rules — vector
orientation, shapes, equations, encoding conversions, and activation boundaries
— are defined in `docs/mathematical-conventions.md`. If the two documents appear
to disagree, the conventions document is authoritative.

The project is educational. Results depend on the training patterns,
representation, learning rule, and recall configuration. Exact recall is not
guaranteed for every dataset.

## 2. From image to pattern

An image file is not directly a neural pattern. It passes through distinct
stages:

1. **Load:** decode the source image and preserve its identity as the original.
2. **Grayscale (optional or configured):** convert color information to
   intensity.
3. **Resize:** normalize the relevant patterns to the required dimensions.
4. **Threshold:** map intensity values to foreground and background.
5. **Foreground polarity:** define whether the object is treated as ON or OFF.
6. **Encode:** map pixels to Binary or Bipolar numeric values.
7. **Flatten:** convert the two-dimensional grid to a one-dimensional vector
   using one documented traversal order.

The original image must remain separate from the processed representation.
Training and recall for one model must use compatible preprocessing, dimensions,
traversal order, and encoding.

### 2.1 Encoding is explicit

Binary and Bipolar encodings have the same ON/OFF meaning but different numeric
representations. The selected encoding is configuration; it is not inferred from
whether a vector happens to contain zeros or negative values. The exact values
and conversion formula are defined in the conventions document.

A binary pattern and a bipolar pattern must never be mixed inside one weight
calculation or one comparison. If a conversion is needed, perform it explicitly
and record it. In particular, do not pass a vector of zeros and ones into a
bipolar-only decoder, and do not interpret a bipolar vector as binary by
inspection.

## 3. Training pairs and memory modes

A training pair is written:

$$
s^{(p)} \rightarrow t^{(p)}
$$

The source pattern is the cue and the target pattern is what the memory should
associate with that cue. The source and target lengths need not match for
heteroassociative memory.

- In **heteroassociative** memory, the pair may connect different patterns or
  different dimensions. The model learns to map the source to the target.
- In **autoassociative** memory, the target is the same pattern as the input.
  There is no separate label: the stored pattern is associated with itself.

Both modes use the same outer-product learning formulation. The distinction is
the target pattern and the resulting meaning of recall, not a different
multiplication rule. Autoassociative and heteroassociative models must remain
separate in configuration and reporting.

Training pairs are retained in their original order for tracing and reporting.
The mathematical sum does not depend on that order, apart from ordinary
floating-point effects.

## 4. Hebbian learning as outer products

Hebbian learning strengthens connections between units that are active together.
For a stored source-target pair, the outer product creates a matrix whose entry
at row `i` and column `j` is the product of source component `i` and target
component `j`. This matrix is the pairwise Hebbian contribution for that pair.

The complete weight matrix is the sum of the contributions from all training
pairs. The exact row-vector equations and shapes are defined in
`docs/mathematical-conventions.md`, Section 5. In summary:

- The heteroassociative weight matrix has shape `(n, m)`.
- The autoassociative weight matrix has shape `(n, n)`.
- The outer product is the matrix formulation of the pairwise Hebbian update; it
  is not a separate, competing learning rule.

For bipolar patterns, negative contributions are expected. They are part of the
memory, not an error or a value to clamp.

### 4.1 Diagonal policy

The basic course algorithm keeps the full diagonal of the autoassociative weight
matrix. The diagonal contains self-connections produced by the outer product.
Setting the diagonal to zero is not a default and is not silently applied. A
zero-diagonal variant is a separate modeling decision and is outside the basic
algorithm covered by this task. The same restriction applies to Hopfield-style
iterative dynamics: they are not part of the basic single-step model.

## 5. Recall and activation

Recall applies a query pattern to the learned weight matrix and then applies an
activation function to the resulting net input. The normative row-vector
multiplication and activation equations are in
`docs/mathematical-conventions.md`, Sections 6 and 7.

The two stages must remain visible and distinct:

- The **net input** is the raw result of the matrix multiplication. It may
  contain arbitrary real values, including negative values and zero.
- The **activated output** is the encoded pattern produced by applying the
  selected activation function elementwise.

For autoassociative memory, the query has the same length as the stored patterns
and the weight matrix is square. For heteroassociative memory, the query has the
source length and the output has the target length.

### 5.1 The zero boundary

The basic activation rules use a strict positive test: a net input is ON only
when it is strictly greater than zero. At exactly zero, binary activation
produces `0` and bipolar activation produces `-1`. This is a deliberate
convention. Replacing the strict comparison with `>= 0`, preserving the cue at
zero, or using a different tie rule changes the algorithm and is not permitted
without a documented decision and corresponding tests.

The output encoding must be consistent with the experiment encoding. The
activated output is not automatically a successful recall; it must still be
compared with the expected pattern.

### 5.2 Single-step recall

The basic course algorithm performs one matrix multiplication followed by one
activation. It does not iterate recall to a fixed point. Iterative dynamics and
other cleanup procedures are outside this task and must not be introduced as
implicit behavior.

## 6. Interference and recall evaluation

A simple outer-product memory superposes the contributions of all stored
patterns. When patterns overlap or are correlated, their contributions can
interfere. This is called crosstalk. A query may therefore produce:

- a corrupted version of a stored pattern;
- a mixture of stored patterns; or
- an output that does not match any stored target (a spurious output).

These are possible outcomes of the model, not automatically implementation
defects.

A numerical computation and a successful pattern recall are different claims.
The computation succeeds when the matrix multiplication and activation return
finite values of the expected shape. Exact recall succeeds only when the
activated output matches the expected pattern elementwise under a defined
comparison criterion. The conventions document defines the mismatch count and
match ratio; exact recall requires zero mismatches.

An autoassociative Hebbian network can fail to reconstruct a stored pattern when
the cue is corrupted, when stored patterns are strongly correlated, or when the
activation boundary changes a marginal net input. Heteroassociative recall can
produce an output outside the stored target set for the same reasons. Do not
describe partial or incorrect output as a successful recall merely because the
calculation completed.

Do not claim a universal capacity limit unless the model, assumptions, and
derivation are explicitly supplied. Capacity depends on pattern count, pattern
dimensionality, correlation, encoding, diagonal policy, threshold handling, and
whether the cue is exact. Report observed recall quality for the actual
experiment instead of promising perfect retrieval.

For autoassociative recall, compare the output with the intended original
pattern. For heteroassociative recall, compare it with the paired target, not the
source cue. A comparison is valid only when the vectors have the same dimensions
and encoding semantics.

## 7. Reference examples

All examples below use the row-vector convention and bipolar encoding unless
stated otherwise. Dimensions are shown for every input, intermediate result, and
output; `@` denotes the NumPy matrix-multiplication operator. Every value was
calculated by hand; no algorithm implementation was used to generate these
expected results. The same cases are available in
`tests/fixtures/algorithm_reference_cases.json`.

### 7.1 Single outer product

Let the first heteroassociative training pair be:

$$
s^{(1)} = \begin{bmatrix}1 & -1 & 1\end{bmatrix}, \qquad
t^{(1)} = \begin{bmatrix}-1 & 1\end{bmatrix}
$$

The source has shape `(3,)` and the target has shape `(2,)`. The outer product
is:

$$
W^{(1)} = \operatorname{outer}\left(s^{(1)}, t^{(1)}\right)
= \begin{bmatrix}
-1 &  1 \\
 1 & -1 \\
-1 &  1
\end{bmatrix}
$$

Shape: `(3, 2)`. Each entry is `s_i t_j`; for example, the top-left entry is
`1 × (-1) = -1` and the bottom-right entry is `1 × 1 = 1`.

### 7.2 Sum of multiple Hebbian contributions

Add a second pair with the same dimensions:

$$
s^{(2)} = \begin{bmatrix}-1 & 1 & 1\end{bmatrix}, \qquad
t^{(2)} = \begin{bmatrix}1 & 1\end{bmatrix}
$$

The second contribution is:

$$
W^{(2)} = \operatorname{outer}\left(s^{(2)}, t^{(2)}\right)
= \begin{bmatrix}
-1 & -1 \\
 1 &  1 \\
 1 &  1
\end{bmatrix}
$$

Shape: `(3, 2)`. The accumulated weight matrix is:

$$
W = W^{(1)} + W^{(2)}
= \begin{bmatrix}
-2 & 0 \\
 2 & 0 \\
 0 & 2
\end{bmatrix}
$$

Shape: `(3, 2)`. The summation would produce the same mathematical result in any
pair order; the original order is retained only for tracing.

### 7.3 Heteroassociative inference

Use `W^(1)` from Section 7.1 and query it with the original source pattern:

$$
x = s^{(1)} = \begin{bmatrix}1 & -1 & 1\end{bmatrix}
$$

The net input is:

$$
y_{\text{in}} = xW^{(1)}
= \begin{bmatrix}1 & -1 & 1\end{bmatrix}
\begin{bmatrix}
-1 &  1 \\
 1 & -1 \\
-1 &  1
\end{bmatrix}
= \begin{bmatrix}-3 & 3\end{bmatrix}
$$

Dimensions: `(3,) @ (3, 2) -> (2,)`. The first component is
`1(-1) + (-1)(1) + 1(-1) = -3`; the second is
`1(1) + (-1)(-1) + 1(1) = 3`.

Applying bipolar activation to `[-3, 3]` gives:

$$
y = \begin{bmatrix}-1 & 1\end{bmatrix}
$$

The expected target is `t^(1) = [-1, 1]`, so this example has zero mismatches and
exact recall succeeds. The result is exact for this example; it is not a
guarantee for other patterns.

### 7.4 Autoassociative inference

Let a stored bipolar pattern be:

$$
x^{(1)} = \begin{bmatrix}1 & -1 & 1\end{bmatrix}
$$

Its autoassociative weight matrix is:

$$
W = \operatorname{outer}\left(x^{(1)}, x^{(1)}\right)
= \begin{bmatrix}
 1 & -1 &  1 \\
-1 &  1 & -1 \\
 1 & -1 &  1
\end{bmatrix}
$$

Shape: `(3, 3)`. Query with the same pattern, `q = [1, -1, 1]`:

$$
y_{\text{in}} = qW
= \begin{bmatrix}1 & -1 & 1\end{bmatrix}
\begin{bmatrix}
 1 & -1 &  1 \\
-1 &  1 & -1 \\
 1 & -1 &  1
\end{bmatrix}
= \begin{bmatrix}3 & -3 & 3\end{bmatrix}
$$

Dimensions: `(3,) @ (3, 3) -> (3,)`. Bipolar activation maps `[3, -3, 3]` to
`[1, -1, 1]`, which equals the stored pattern. The mismatch count is zero.

### 7.5 Activation at positive, negative, and zero inputs

Take the net-input vector:

$$
a = \begin{bmatrix}2 & -3 & 0\end{bmatrix}
$$

with shape `(3,)`. Applying the two activation rules elementwise gives:

| Component | Input | Binary output | Bipolar output |
| --- | --- | --- | --- |
| 1 | `2` | `1` | `+1` |
| 2 | `-3` | `0` | `-1` |
| 3 | `0` | `0` | `-1` |

Thus the binary output is `[1, 0, 0]` and the bipolar output is `[1, -1, -1]`,
both of shape `(3,)`. The zero component demonstrates the strict boundary: it is
not treated as a positive input.

### 7.6 Binary-to-bipolar conversion

Let:

$$
x_{\text{binary}} = \begin{bmatrix}0 & 1 & 1 & 0 & 1\end{bmatrix}
$$

with shape `(5,)`. The explicit conversion is:

$$
x_{\text{bipolar}} = 2x_{\text{binary}} - 1
= \begin{bmatrix}-1 & 1 & 1 & -1 & 1\end{bmatrix}
$$

The result has shape `(5,)`. Reversing the conversion returns
`[0, 1, 1, 0, 1]`. No mixing of the two encodings occurs in this calculation.

### 7.7 Interference and a failed autoassociative recall

This example shows that a failed recall can be a correct outcome of the basic
algorithm. Store two bipolar patterns:

$$
s^{(1)} = \begin{bmatrix}1 & 1 & -1 & -1\end{bmatrix}, \qquad
s^{(2)} = \begin{bmatrix}1 & -1 & 1 & -1\end{bmatrix}
$$

Each has shape `(4,)`. The autoassociative weight matrix is:

$$
W = \operatorname{outer}\left(s^{(1)}, s^{(1)}\right)
  + \operatorname{outer}\left(s^{(2)}, s^{(2)}\right)
= \begin{bmatrix}
 2 &  0 &  0 & -2 \\
 0 &  2 & -2 &  0 \\
 0 & -2 &  2 &  0 \\
-2 &  0 &  0 &  2
\end{bmatrix}
$$

Shape: `(4, 4)`. Query with a corrupted version of the first pattern, where only
the last component is flipped:

$$
q = \begin{bmatrix}1 & 1 & -1 & 1\end{bmatrix}
$$

The net input is:

$$
y_{\text{in}} = qW
= \begin{bmatrix}0 & 4 & -4 & 0\end{bmatrix}
$$

Dimensions: `(4,) @ (4, 4) -> (4,)`. Bipolar activation gives:

$$
y = \begin{bmatrix}-1 & 1 & -1 & -1\end{bmatrix}
$$

Compared with the intended original `s^(1) = [1, 1, -1, -1]`, the first component
differs. The mismatch count is `1`, the match ratio is `3/4`, and exact recall
fails. The zero net-input components became `-1` under the documented bipolar
activation rule. This is a legitimate interference result, not evidence that the
matrix multiplication was implemented incorrectly.

## 8. Metrics and interpretation

The net input and the activated output explain why a recall result occurred. For
image-based patterns, the interface should show the original image, the
preprocessed image, the encoded pattern and its encoding, the pattern dimensions,
the individual outer-product contributions where supported, the accumulated
weight matrix, the net input, the activated output, the expected pattern, and the
comparison metrics.

The mismatch count and match ratio are defined in the conventions document.
Metrics are meaningful only when the compared vectors have the same dimensions,
traversal order, and encoding. If the comparison is invalid, report that fact
rather than displaying a fabricated score.

Do not display fake metrics or claim successful recall without an explicit
comparison criterion. A completed calculation is not the same claim as exact
pattern reconstruction.

## 9. Source material, notation, and clarifications

No lecture notes, slides, scanned examples, or other course source material were
present in the repository during Task 01. Therefore no lecture input/output pairs
could be preserved, and no apparent typographical error in course notation could
be verified. The examples in Section 7 are independently constructed and
hand-calculated.

The following clarifications record conventions that can differ between
presentations of the same basic model. They are clarifications, not silent
corrections of a lecture formula:

- **Vector orientation.** Many presentations use column vectors, with learning
  written using a transpose of the target and recall written with a transposed
  weight matrix. This project uses row vectors, so the equations and
  multiplication order are adapted consistently. The learning rule is the same;
  the orientation is not. Do not combine formulas from the two conventions.
- **Zero activation boundary.** Some presentations use `>= 0` or preserve the cue
  at zero. This project uses the strict boundary documented in Task 01: zero maps
  to `0` for binary activation and `-1` for bipolar activation.
- **Diagonal and iterative variants.** Some presentations zero the diagonal of an
  autoassociative matrix or iterate recall as a Hopfield-style dynamic. Those are
  separate modeling choices. The basic course algorithm in this project keeps the
  full diagonal and performs a single recall step.
- **Capacity claims.** No universal capacity formula is assumed. Any capacity
  statement requires explicit model assumptions and a derivation.

If course material is added later, preserve its original inputs and expected
results and record any discrepancy here instead of silently editing the source
formula.

## 10. Educational transparency requirements

The application should let users inspect:

- the original image and the preprocessed image separately;
- the encoded pattern and its encoding;
- each training pair and its dimensions;
- each individual outer-product contribution where supported;
- the accumulated weight matrix;
- the net input and the activation configuration;
- the activated output, expected pattern, and comparison metrics; and
- warnings about interference, invalid comparisons, or unsupported assumptions.

These views are explanatory tools. Displayed calculations must be generated from
the same validated algorithm services used by the actual training and recall
operations, not from a separate demonstration implementation.
