Task 03 — Image Loading and Preprocessing
1. Objective
Implement a reproducible image-processing pipeline that converts ordinary image files into vectors suitable for the associative memory algorithms.
Users must be able to select a PNG, JPEG, or BMP image rather than manually entering pixel arrays.
The pipeline must also expose the intermediate representations for visualization.
2. Prerequisites
- Tasks 00 and 01 are complete.- Task 02 is complete, or its encoding interfaces are stable.
3. Scope
Implement:
- Image file loading.- File validation.- Grayscale conversion.- Resizing to a configurable target size.- Threshold-based binarization.- Binary and bipolar encoding.- Flattening to a one-dimensional vector.- Structured preprocessing metadata.- Clear error reporting.
Do not implement GUI file dialogs or GUI widgets in this task.
4. Expected Modules
- src/associative_memory/infrastructure/image_loader.py- src/associative_memory/infrastructure/image_preprocessor.py- src/associative_memory/application/services/image_service.py- Relevant unit tests under tests/unit/
Choose and document a suitable image library, such as Pillow, based on compatibility with the project.
5. Supported Inputs
Initially support:
- PNG.- JPEG/JPG.- BMP.
Validate file existence, readability, and actual image content. Do not rely only on the file extension.
Handle corrupt files, unsupported formats, and invalid dimensions with user-readable error messages.
Do not silently accept unsupported animated or multi-frame content. Either explicitly support it under a documented policy or reject it clearly.
6. Preprocessing Configuration
Provide explicit configuration for:
- Target width and height.- Grayscale conversion.- Resampling method.- Threshold value.- Foreground polarity.- Output encoding: binary or bipolar.- Flattening order.
The default configuration must be documented and reproducible.
Do not silently assume that black pixels always represent ON or that white pixels always represent OFF.
For grayscale thresholding, define precisely whether pixels below or above the threshold become ON.
Keep this convention consistent in training, dataset inspection, and recall.
Aspect Ratio
Make resizing behavior explicit.
Prefer an option that preserves the aspect ratio with padding, alongside a documented direct-resize option if useful.
If the application distorts the image, clearly indicate this to the user.
Flattening
Use a documented, consistent flattening order, such as row-major order.
The same pixel must always map to the same vector index for a given image size and preprocessing configuration.
7. Intermediate Data
The preprocessing result should make the following available:
- Original image dimensions.- Original image or a safe image representation.- Grayscale representation.- Resized representation.- Binary pixel matrix.- Encoded binary or bipolar matrix.- Flattened vector.- Target dimensions.- Threshold and polarity settings.- Flattening order.- Encoding.- Relevant processing metadata.
Do not store large duplicated image arrays unnecessarily when a safe immutable representation or explicit conversion is sufficient.
Ensure arrays returned to callers cannot be accidentally mutated in a way that invalidates the stored experiment without notice.
8. Reproducibility
Training and recall must use the same preprocessing configuration when their input dimensions are intended to match.
Persist the preprocessing configuration with experiment metadata in a later task.
Changing image size, threshold, polarity, or encoding can change the resulting pattern and must be visible to the user.
9. Required Tests
- Load valid PNG, JPEG, and BMP files.- Reject missing and unreadable files.- Reject corrupt image data.- Validate resizing behavior.- Validate grayscale conversion.- Validate exact threshold boundary behavior.- Validate foreground/background polarity.- Validate binary and bipolar conversion.- Validate flattening order.- Verify vector length equals width multiplied by height.- Verify repeated processing with identical settings gives identical patterns.- Verify invalid configuration values produce clear errors.
Use small synthetic test images so the tests do not depend on external files or network access.
10. Acceptance Criteria
- Supported image formats load correctly.- Invalid files fail with meaningful errors.- Preprocessing settings are explicit and reproducible.- The pipeline exposes intermediate representations.- The resulting vector has the expected dimension.- The algorithm package remains independent of image-loading details.- No GUI dependencies are introduced into the preprocessing implementation.
11. Completion Report
Report supported formats, default preprocessing choices, files changed, tests run, actual outcomes, and known limitations.
Do not begin Task 04 until the preprocessing pipeline is tested.