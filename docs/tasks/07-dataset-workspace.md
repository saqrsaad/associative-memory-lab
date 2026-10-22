Task 07 — Image Dataset Workspace
1. Objective
Implement the dataset workspace where users import images, inspect their processed representations, organize training examples, and define input-target associations.
The workspace must make the image-to-vector conversion visible and understandable.
2. Prerequisites
- Tasks 03 and 04 are complete.- Task 05 is complete.- Task 06 is complete.
3. Scope
Implement:
- Image import through a file dialog.- Image previews.- Dataset list and selection.- Image preprocessing controls.- Training-pair organization.- Target assignment for heteroassociative memory.- Pattern inspection.- Input validation.- Empty, loading, warning, and error states.
Do not implement the full training or recall execution workflow in this task.
4. Expected Modules
- src/associative_memory/presentation/views/dataset_view.py- src/associative_memory/presentation/viewmodels/dataset_viewmodel.py- src/associative_memory/presentation/widgets/image_canvas.py- src/associative_memory/presentation/widgets/pixel_grid.py- src/associative_memory/presentation/widgets/vector_view.py- Additional small dataset widgets when necessary.- Relevant UI and integration tests.
5. User Workflow
The user must be able to:
1. Open the dataset workspace.2. Import an image from the file system.3. See the original image and its dimensions.4. Select preprocessing settings.5. Inspect the grayscale and thresholded representations.6. Select binary or bipolar encoding.7. Inspect the encoded pixel grid and flattened vector.8. Add the processed pattern to the dataset.9. Assign a target or class label when using heteroassociative memory.10. Review and remove dataset entries.11. Correct invalid configuration or file errors.
6. Image Import
Use a native file dialog or a consistent application-styled file-selection workflow.
Initially support PNG, JPEG, and BMP.
Do not assume the file extension guarantees valid image data.
Display useful errors for:
- Unsupported formats.- Corrupt files.- Unreadable files.- Invalid image dimensions.- Failed preprocessing.
Never crash the application because a user selected an invalid file.
7. Preprocessing Controls
Expose the settings supported by Task 03, including:
- Target width and height.- Grayscale conversion.- Threshold.- Foreground polarity.- Binary or bipolar encoding.- Resizing strategy.
Use validated input controls with clear labels and tooltips.
When a setting changes, update the preview only after successful preprocessing. Do not replace the last valid preview with invalid data.
Clearly indicate that changing preprocessing settings may change the encoded pattern and invalidate compatibility with previously trained weights.
8. Image and Pattern Visualization
Display:
- Original image.- Processed image.- Encoded pixel grid.- Vector length.- Image dimensions.- Encoding.- Current preprocessing configuration.
The pixel grid must preserve the row-major mapping used by the algorithm.
Allow inspection of pixel values where practical.
For large images, do not create one heavyweight GUI widget per pixel. Use an efficient drawing or table-based visualization.
9. Training-Pair Management
Support a clear representation of training examples.
For heteroassociative memory, a training example consists of an input pattern and its target pattern or target vector.
For autoassociative memory, the target is the same pattern as the input. Do not ask the user to provide a redundant target image unless the workflow explicitly supports an advanced mode.
For the initial implementation, use consistent image dimensions within a single model unless a documented transformation maps different source dimensions to a common representation.
Prevent incompatible patterns from being added to the active training set.
10. State and Architecture
- The view must not implement image-processing formulas.- The viewmodel must coordinate user selections and application services.- Application services must remain independent of Qt.- Do not store the entire dataset in global mutable state.- Use a clear ownership policy for images and NumPy arrays.- Do not claim that imported patterns are persisted permanently unless a persistence implementation is available.
11. Required Tests
- Import a valid image.- Reject an invalid image.- Change preprocessing settings.- Verify that the displayed dimensions match the processed data.- Verify that the encoded vector length is correct.- Add and remove dataset entries.- Reject incompatible patterns.- Assign and validate heteroassociative targets.- Confirm that changing preprocessing settings invalidates or refreshes the relevant preview and pattern representation.- Confirm that the view can operate using mocked application services.
12. Acceptance Criteria
- Users can import and inspect supported images.- Preprocessing settings are visible and reproducible.- Image, pixel-grid, and vector views agree.- Training examples can be organized without entering raw arrays manually.- Invalid inputs produce clear messages.- No algorithm formulas are embedded in the GUI.- All relevant tests pass.
13. Completion Report
Report implemented user flows, files changed, tests executed, actual results, and known limitations.
Do not begin Task 08 until dataset preparation and inspection work correctly.