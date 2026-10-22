Task 09 — Recall Workspace and Results Visualization
1. Objective
Implement the main recall workspace and educational results dashboard.
This is a core feature of the project. It must let users execute recall and understand how the network produced its output.
2. Prerequisites
- Tasks 02, 03, 04, 05, 06, 07, and 08 are complete.
3. Scope
Implement:
- Input-pattern selection.- Recall execution.- Input and output visualization.- Expected-target comparison.- Numerical net-input inspection.- Calculation trace.- Recall metrics.- Weight-matrix inspection.- Warnings about interference and mismatches.- Export of a basic experiment report if feasible within the existing architecture.
Do not implement unsupported neural-network variants or claim that the model always reconstructs the original image.
4. Expected Modules
- src/associative_memory/presentation/views/recall_view.py- Relevant viewmodel modules.- src/associative_memory/presentation/widgets/vector_view.py- src/associative_memory/presentation/widgets/weight_matrix_view.py- src/associative_memory/presentation/widgets/formula_panel.py- src/associative_memory/presentation/widgets/step_navigator.py- Relevant result and integration tests.
Reuse the existing image and matrix viewers.
5. Recall Workflow
The user must be able to:
1. Select a previously trained model.2. Select an input image or compatible pattern.3. Review the preprocessing configuration.4. Run recall.5. Inspect the input representation.6. Inspect the net-input vector before activation.7. Inspect the output vector after activation.8. View the output as an image when the output dimension maps to the configured image dimensions.9. Compare the result with the expected target or original pattern where applicable.10. Review metrics and warnings.
Disable recall when no compatible trained model exists.
Do not allow incompatible input dimensions to reach matrix multiplication without validation.
6. Result Summary
Show the following where applicable:
- Memory type.- Encoding.- Input dimensions.- Weight-matrix dimensions.- Output dimensions.- Expected target.- Exact-match status.- Mismatch count.- Match ratio.- Experiment warnings.
Do not display metrics as if they were available when no valid comparison target exists.
For a single recall example, distinguish pattern match ratio from dataset-level classification accuracy.
7. Image Comparison
For image-compatible results, display:
- Input image.- Processed input pattern.- Expected target or reference pattern, if available.- Actual recalled output.
Provide a difference view that highlights mismatched pixels.
Use a consistent visual convention:
- Matching pixels.- Missing pixels.- Extra pixels.- Invalid or unavailable comparison.
Include a legend and textual mismatch counts so the interpretation does not depend only on color.
If the output is not compatible with the image dimensions, display the vector instead of fabricating an image.
8. Calculation Trace
Provide a step navigator or equivalent interaction to inspect:
1. Preprocessed input.2. Flattened pattern.3. Weight-matrix dimensions.4. Net-input computation.5. Net-input values.6. Activation rule.7. Final output.8. Comparison with the expected target.
Display the equation:
y_in = x @ W
Show actual values and dimensions.
For larger calculations, summarize the operation by default while retaining the ability to inspect individual values. Avoid flooding the page with hundreds or thousands of unstructured numbers.
9. Metrics and Interpretation
Define metrics precisely.
For equal-length vectors:
- Exact match: every output component equals the reference component.- Mismatch count: number of unequal components.- Match ratio: number of matching components divided by the total number of components.
When there are no components, do not report a valid match ratio.
For image patterns, report pixel agreement only when the output and reference have compatible dimensions and encoding.
For heteroassociative memory, compare against the expected target.
For autoassociative memory, compare against the original input or the designated reference pattern.
Do not label a result as a successful classification merely because its vector is nonzero or has the correct dimensions.
10. Interference Warnings
If the recalled output does not match the expected stored target, show a clear warning.
Where possible, explain:
- The output differs from the target.- The network may exhibit interference between stored patterns.- The current training set or encoding may be unsuitable for the input.- The result should be inspected rather than assumed correct.
Do not claim to have identified the exact cause unless the application has sufficient evidence.
11. Required Tests
- Successful recall of a reference example.- Recall that produces a nonmatching output.- Correct net-input vector.- Correct activation output.- Correct exact-match calculation.- Correct mismatch count and match ratio.- Missing comparison target.- Incompatible input dimensions.- Correct image reconstruction dimensions.- Difference-view consistency with the underlying vectors.- Correct handling of an output that cannot be reshaped into an image.- Confirmation that displayed values match the application service results.
12. Acceptance Criteria
- The user can run recall through the GUI.- Input and output patterns are clearly visible.- Numerical calculations are inspectable.- Comparison metrics are mathematically correct.- Interference is reported without overstating its cause.- The interface never fabricates images or metrics.- The result workspace remains usable for both small and moderately large patterns.
13. Completion Report
Report the recall flow, visualizations, metrics, tests run, actual results, and known limitations.
Do not begin Task 10 until the results workspace is reliable.