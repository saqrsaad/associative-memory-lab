Task 10 — Experiment History and Persistence
1. Objective
Allow users to save, reopen, inspect, and compare experiments without losing the information required to interpret their results.
Persistence must preserve the numerical model and the preprocessing settings associated with it.
2. Prerequisites
- Tasks 02 through 09 are complete.- The experiment and result data structures are stable.
3. Scope
Implement:
- Experiment metadata.- Persistent model storage.- Saved experiment history.- Experiment loading.- Experiment deletion with confirmation.- Basic experiment comparison.- Schema versioning.- Validation of saved files.- Safe error handling for corrupt or incompatible data.
4. Expected Modules
- src/associative_memory/application/interfaces/model_repository.py- src/associative_memory/infrastructure/model_repository_json.py- src/associative_memory/infrastructure/experiment_repository.py- src/associative_memory/presentation/views/experiment_view.py- Relevant viewmodel modules.- Relevant persistence tests.
Reuse or extend existing interfaces rather than introducing duplicate repositories.
5. Persistence Format
Choose an explicit format for metadata and numerical arrays.
A practical initial design is:
- JSON for readable metadata.- NumPy's ".npy" or ".npz" format for numerical arrays where appropriate.- A versioned experiment manifest linking metadata and numerical data.
Do not rely on arbitrary Python object serialization for untrusted experiment files.
If a single-file package format is selected, document its structure and security properties.
6. Required Experiment Metadata
Store, where applicable:
- Experiment ID.- Schema version.- Creation timestamp.- Memory type.- Encoding.- Activation rule and threshold.- Training dataset references or pattern data.- Input and target dimensions.- Image preprocessing configuration.- Final weight matrix.- Relevant training trace.- Recall input.- Recall output.- Expected target.- Metrics.- Warnings.- Application version.
Do not include unnecessary private file paths or unrelated personal information.
Document whether original images are copied into the experiment or referenced externally. Prefer a clear and reliable strategy over silent dependencies on files that may later move.
7. Saving
Saving must:
- Validate the experiment structure.- Write files safely.- Avoid leaving a partially written experiment when feasible.- Detect filename or ID conflicts.- Provide a clear success or failure message.- Preserve exact numerical values.
Do not report successful saving before the write operation has completed.
8. Loading
Loading must:
- Validate the schema version.- Validate array dimensions.- Validate supported encoding and memory type.- Handle missing files.- Handle corrupted data.- Handle unsupported future schema versions.- Reject invalid values safely.
Do not execute code contained in a saved experiment.
If migration is supported, implement explicit version-to-version migrations and test them.
9. History Interface
Display useful metadata such as:
- Experiment name or ID.- Creation date.- Memory type.- Image or pattern dimensions.- Encoding.- Training sample count.- Recall result status where available.
Support opening an experiment and inspecting its results.
Confirm destructive deletion.
Do not expose implementation details or large binary arrays in the history list.
10. Experiment Comparison
Provide a basic comparison between two compatible experiments.
Show relevant differences, such as:
- Memory type.- Encoding.- Preprocessing configuration.- Dataset size.- Weight-matrix dimensions.- Recall outputs and metrics.
Do not compare incompatible metrics as if they were directly equivalent.
If the weight matrices have different shapes, explain that a direct element-by-element comparison is unavailable.
11. Repository Interface
Keep persistence behind repository interfaces.
The presentation layer must not read JSON or NumPy files directly.
The algorithm package must not perform persistence.
12. Required Tests
- Save and reload a valid experiment.- Preserve numerical arrays exactly.- Preserve preprocessing metadata.- Reject corrupted files.- Reject unsupported schema versions.- Handle missing files.- Handle ID conflicts.- Test deletion behavior.- Test compatible experiment comparison.- Test incompatible matrix dimensions.- Verify that loading an experiment does not execute serialized code.
13. Acceptance Criteria
- Experiments can be saved and reopened.- The saved model can reproduce the relevant inference results.- Preprocessing and encoding settings are preserved.- Invalid saved files fail safely.- Repository interfaces isolate persistence from the GUI and algorithms.- The history view shows accurate metadata.- Relevant tests pass.
14. Completion Report
Report the storage format, schema version, repository implementation, tests executed, actual outcomes, and known migration or portability limitations.
Do not begin Task 11 until experiment persistence is validated.