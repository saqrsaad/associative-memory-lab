Task 04 — Application Services and Result DTOs
1. Objective
Create an application layer that coordinates image processing, training, recall, and experiment creation without depending on GUI components.
The application layer must provide a stable interface between the algorithm package and the presentation layer.
2. Prerequisites
- Tasks 00, 01, and 02 are complete.- Task 03 has stable preprocessing interfaces.
3. Scope
Implement:
- Training service.- Recall service.- Image-processing service integration.- Experiment orchestration.- Typed result objects.- Input validation at use-case boundaries.- Clear application-level errors.
Do not build GUI screens or implement numerical formulas again in the service layer.
4. Expected Modules
- src/associative_memory/application/services/image_service.py- src/associative_memory/application/services/training_service.py- src/associative_memory/application/services/recall_service.py- src/associative_memory/application/services/experiment_service.py- src/associative_memory/application/dto/training_result.py- src/associative_memory/application/dto/recall_result.py- src/associative_memory/application/dto/experiment_report.py- Relevant integration tests.
Reuse existing modules where possible instead of creating duplicate implementations.
5. Training Service
The training service must:
- Accept a validated training dataset.- Accept the memory type.- Accept the encoding and activation configuration.- Invoke the appropriate algorithm implementation.- Return the trained model information and training trace.- Report input and target dimensions.- Report the number of training pairs.- Preserve enough metadata to reproduce the experiment.
Reject incompatible patterns before training begins.
Do not silently skip invalid examples.
6. Recall Service
The recall service must:
- Accept a trained model and an input pattern.- Validate compatible dimensions and encoding.- Execute inference.- Return net input and activated output.- Compare the output against an expected target when one is provided.- Compare autoassociative output against the original pattern when appropriate.- Report exact match, mismatch count, and match ratio.- Distinguish unavailable comparison data from a genuine zero score.
Do not label an output as a successful recall without an explicit comparison criterion.
7. Result DTO Requirements
Use typed, documented result objects.
A training result should include:
- Model or weight-matrix reference.- Memory type.- Encoding.- Activation configuration.- Training sample count.- Input and output dimensions.- Optional individual outer-product contributions.- Experiment metadata.
A recall result should include:
- Input pattern.- Net input.- Output pattern.- Model dimensions.- Optional expected pattern.- Comparison metrics when applicable.- Warnings and relevant configuration.
An experiment report should provide a consistent summary of a complete workflow.
Do not expose mutable internal arrays without a documented ownership policy.
8. Error Handling
Define application-level exceptions where useful.
Convert expected lower-level failures into clear application errors while retaining diagnostic information for logging.
Do not catch all exceptions and return a fabricated result.
9. Dependency Rules
- Application services may depend on domain, algorithms, and infrastructure interfaces.- Application services must not import presentation modules or PySide6 widgets.- Algorithm formulas must remain in the algorithms package.- File-system operations must be delegated to infrastructure components.
10. Required Tests
- Successful training workflow.- Successful recall workflow.- Incompatible dimensions.- Missing training examples.- Invalid encoding configuration.- Correct result metadata.- Correct comparison metrics.- Correct handling of missing expected targets.- Verification that application services work without launching the GUI.
11. Acceptance Criteria
- A complete training and recall workflow can run without a GUI.- All result objects expose the values required by later presentation tasks.- No algorithm formula is duplicated in application services.- Invalid operations return clear errors.- Integration tests pass.
12. Completion Report
Report the public service interfaces, DTOs, tests executed, actual results, and remaining limitations.
Do not begin Task 05 until the application interfaces are stable.