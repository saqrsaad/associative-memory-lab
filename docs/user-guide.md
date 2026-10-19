# User Guide

## 1. About Associative Memory Lab

Associative Memory Lab is an educational desktop application for
exploring associative-memory neural networks using image patterns. It is
designed to help users observe how image preprocessing and numeric
encoding affect training, weight matrices, and recall results.

Exact menu names may vary with the implemented version. This guide
describes the intended workflow; only use actions that are present in
your installed build.

## 2. Key concepts

-   **Autoassociative memory:** learns to associate each pattern with
    itself.
-   **Heteroassociative memory:** learns source-to-target pattern pairs.
-   **Binary encoding:** represents off/on pixels as 0/1.
-   **Bipolar encoding:** represents off/on pixels as -1/+1.
-   **Preprocessing:** transforms an imported image into a consistent
    image representation.
-   **Weight matrix:** numeric matrix formed by the selected learning
    rule.
-   **Recall:** applies a query pattern to a trained memory to produce
    an output.
-   **Net input:** values before the activation function converts them
    to output states.

## 3. Prepare your images

1.  Open the Dataset workspace.
2.  Import supported image files using the application's import control.
3.  Inspect the original preview and check that the intended
    foreground/background is visible.
4.  For autoassociative experiments, select the patterns you want the
    model to store.
5.  For heteroassociative experiments, create source-target pairs and
    confirm that each pair is correctly assigned.
6.  Remove or correct invalid entries before training.

Use simple, high-contrast images when first learning the workflow. Keep
in mind that every image used by a single model must be processed into
compatible dimensions and conventions.

## 4. Preprocess and encode

1.  Select an image or pattern pair.
2.  Choose the supported resize dimensions.
3.  Select grayscale and threshold settings if exposed by the
    application.
4.  Set foreground polarity so the intended object is mapped to the
    on-state.
5.  Choose Binary or Bipolar encoding.
6.  Inspect the processed image, pixel grid, and encoded vector.
7.  Confirm the displayed vector length matches `height × width`.

Do not assume that a dark pixel always means foreground. The configured
threshold and polarity determine its interpretation. If the object
appears inverted, correct the foreground polarity or preprocessing
settings and regenerate the encoded pattern.

The original image should remain available separately from its derived
representations.

## 5. Train an autoassociative memory

1.  Open the Training workspace.
2.  Select Autoassociative memory.
3.  Choose the encoding and any available diagonal/self-connection
    policy.
4.  Confirm all selected patterns have compatible dimensions and
    encoding.
5.  Start training.
6.  Inspect the resulting weight matrix and its dimensions.
7.  Review any warnings about duplicate, correlated, or invalid
    patterns.

A common Hebbian rule accumulates an outer product of each pattern with
itself. The exact matrix convention and diagonal policy are documented
in `mathematical-conventions.md`.

## 6. Train a heteroassociative memory

1.  Prepare source-target pairs in the Dataset workspace.
2.  Confirm that each source is paired with the intended target.
3.  Open Training and select Heteroassociative memory.
4.  Confirm source dimensions agree across all pairs and target
    dimensions agree across all pairs.
5.  Choose the supported encoding and activation settings.
6.  Start training.
7.  Inspect the weight matrix and verify that its dimensions correspond
    to source length by target length under the documented convention.

Heteroassociative memory learns a mapping from source patterns to target
patterns. When evaluating recall, compare the output with the paired
target, not with the source cue.

## 7. Run recall

1.  Open Recall & Results.
2.  Select a trained model and a compatible query pattern.
3.  If supported, choose whether to use a clean query or a deliberately
    corrupted cue.
4.  Run recall.
5.  Inspect the query, net input, activation settings, output vector,
    and reconstructed image.
6.  If an expected target is available, inspect the mismatch count and
    match ratio.

A recalled output may differ from the intended pattern. Outer-product
memories can experience interference between stored patterns, and
outcomes depend on pattern similarity, encoding, thresholds, and model
configuration.

## 8. Understand the results

-   **Weight matrix:** contains accumulated connection weights; its
    dimensions depend on the memory type and vector convention.
-   **Net input:** values before activation.
-   **Recalled output:** pattern after activation.
-   **Mismatch count:** number of positions that differ from the
    expected pattern.
-   **Match ratio:** fraction of positions that match, from 0 to 1 (or
    0% to 100%).

Only compare patterns that share compatible dimensions, flattening
order, and encoding semantics. If the app reports that a comparison is
invalid, fix the compatibility issue rather than interpreting the
metric.

## 9. Save and reopen experiments

If experiment persistence is available in your build:

1.  Give the experiment a meaningful name.
2.  Save it through the application's save control.
3.  Confirm the save operation completed successfully.
4.  Reopen it through Experiment History or the relevant load control.
5.  Verify that the memory type, encoding, preprocessing configuration,
    patterns, and results are consistent.

Treat experiment files as data, not executable code. Do not load files
from unknown sources. If a file is rejected as incompatible or
malformed, do not attempt to bypass validation.

## 10. Troubleshooting

**An image cannot be imported** - Confirm that the file exists and uses
a supported image format. - Try opening it in another image viewer to
check whether it is corrupt. - Check the application's error message.

**The foreground is inverted** - Revisit threshold and
foreground-polarity settings. - Regenerate the processed image and
encoded vector.

**Training is disabled** - Confirm that valid patterns or source-target
pairs exist. - Check that dimensions and encodings are compatible. -
Resolve any validation messages.

**Recall output looks scrambled** - Check image dimensions and vector
flattening order. - Confirm that encoding and decoding use the same
convention. - Verify that the model and query have compatible
dimensions.

**Recall is inaccurate** - Inspect pattern similarity, encoding,
diagonal policy, and activation settings. - Try a smaller or
less-correlated training set. - Compare the net input with the activated
output. - Remember that exact recall is not guaranteed.

**An experiment will not load** - Confirm that it was produced by a
compatible version. - Check whether the file is incomplete or has been
edited manually. - Keep a backup before attempting any supported
migration.

## 11. Responsible interpretation

The application is a learning and visualization tool, not a guarantee of
a particular neural-network result. Use displayed equations,
configuration, and numerical outputs to understand each experiment.
Report the actual observed outcome rather than assuming the network
always reconstructs the desired image.
