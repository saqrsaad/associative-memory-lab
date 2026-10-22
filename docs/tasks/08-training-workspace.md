Task 08 — Training Workspace and Learning Trace
1. Objective
Implement the training workspace that allows users to configure a memory, train it using the dataset workspace, and inspect the learning calculations.
The interface must show the relationship between the training examples and the resulting weight matrix.
2. Prerequisites
- Tasks 02, 04, 05, 06, and 07 are complete.
3. Scope
Implement:
- Memory type selection.- Encoding selection.- Activation selection.- Dataset validation.- Training execution.- Training progress and status.- Per-pair outer-product visualization.- Accumulated weight-matrix visualization.- Training summary.- Clear handling of invalid configurations.
Do not implement the full recall workspace in this task.
4. Expected Modules
- src/associative_memory/presentation/views/training_view.py- src/associative_memory/presentation/viewmodels/training_viewmodel.py- src/associative_memory/presentation/widgets/matrix_view.py- src/associative_memory/presentation/widgets/weight_matrix_view.py- src/associative_memory/presentation/widgets/formula_panel.py- Relevant integration and UI tests.
Reuse shared widgets rather than creating a new matrix component for every page.
5. Configuration Controls
Provide controls for:
- Heteroassociative or autoassociative memory.- Binary or bipolar encoding.- Activation function appropriate to the selected encoding.- Selected dataset or training set.- Optional detailed calculation trace.
Prevent incompatible combinations or explain them before training.
Do not silently reinterpret patterns when the encoding changes.
If switching encoding invalidates existing processed patterns, require reprocessing or provide a clear, explicit conversion path.
6. Training Workflow
The user must be able to:
1. Select the memory type.2. Select a compatible training dataset.3. Review the training examples and target assignments.4. Confirm encoding and activation configuration.5. Start training.6. See the current operation status.7. View the resulting weight matrix and its dimensions.8. Inspect individual outer-product contributions.9. Inspect the accumulated weight matrix.10. Receive a clear success or failure message.
Training must not proceed when the dataset is empty or incompatible.
7. Mathematical Trace
When detailed trace is enabled, show:
- The input pattern for each training pair.- The target pattern for heteroassociative memory.- The outer-product matrix for that pair.- The accumulated weight matrix after each pair.- The final weight matrix.- The dimensions of all matrices.- The relevant equation and encoding.
For autoassociative memory, show that each contribution is outer(s, s).
For heteroassociative memory, show that each contribution is outer(s, t).
The displayed matrices must come from the actual algorithm result, not from a separately reimplemented calculation in the GUI.
8. Weight-Matrix Viewer
Support:
- Numeric inspection.- Row and column labels or indices.- Matrix dimensions.- Scrolling.- A heatmap where appropriate.- A readable legend for signed values.- Exact underlying numerical values.
For large matrices, avoid creating one GUI widget per matrix cell. Use an efficient table model or custom drawing.
Do not render every matrix cell as a separate card.
Do not silently truncate the underlying matrix. If a summarized preview is shown, make it clear and provide a way to inspect the full matrix.
9. Responsiveness
Long-running work must not block the GUI thread.
Use an appropriate Qt worker mechanism, such as QThread or a worker object moved to a QThread, when needed.
Keep thread boundaries explicit.
Do not access GUI widgets directly from a background worker.
Handle success, failure, cancellation where supported, and cleanup of worker resources.
Avoid introducing concurrency complexity for calculations that complete immediately, but do not block the GUI for larger image or matrix operations.
10. Training State
Represent training state explicitly, for example:
- Not ready.- Ready to train.- Training.- Completed.- Failed.
Disable actions that cannot safely run during training.
If the user changes the dataset or preprocessing settings after training, indicate whether the existing model is still compatible or must be retrained.
11. Required Tests
- Train a heteroassociative memory.- Train an autoassociative memory.- Reject an empty dataset.- Reject incompatible dimensions.- Confirm the final weight matrix matches the algorithm result.- Confirm that individual outer products are shown in the correct order.- Confirm the GUI remains responsive during a simulated long-running task.- Confirm error handling restores the UI to a usable state.- Confirm incompatible dataset changes invalidate the model state.
12. Acceptance Criteria
- A user can train a model through the GUI.- The final weights match the core algorithm.- The mathematical trace is accurate.- Matrix dimensions and encoding are visible.- Training errors are understandable.- The GUI remains responsive during lengthy operations.- No duplicate algorithm implementation exists in the presentation layer.
13. Completion Report
Report the training workflow, mathematical trace, matrix-view behavior, tests executed, actual results, and limitations.
Do not begin Task 09 until training can be executed and inspected reliably.