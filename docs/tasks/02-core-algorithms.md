Task 02 — Core Associative Memory Algorithms
1. Objective
Implement the mathematical core of the course's Hebbian associative memory algorithms using NumPy.
The algorithm package must be usable independently of the GUI, file system, and image-loading infrastructure.
2. Prerequisites
- Task 00 is complete.- Task 01 is complete.- docs/mathematical-conventions.md is authoritative.
3. Scope
Implement:
- Outer-product calculation.- Hebbian weight accumulation.- Binary and bipolar activation.- Heteroassociative memory training and inference.- Autoassociative memory training and inference.- Structured numerical results.- Validation of vector and matrix dimensions.
Do not implement image loading, GUI widgets, persistence, or Hopfield iterative dynamics.
4. Expected Modules
- src/associative_memory/algorithms/outer_product.py- src/associative_memory/algorithms/hebbian.py- src/associative_memory/algorithms/activation.py- src/associative_memory/algorithms/heteroassociative.py- src/associative_memory/algorithms/autoassociative.py- src/associative_memory/algorithms/metrics.py, if needed- Relevant unit tests under tests/unit/
Keep public exports in init.py minimal and intentional.
5. Design Requirements
5.1 Outer Product
Provide a clearly named function that accepts one-dimensional input and target vectors and returns their outer product.
Validate:
- Both arguments are one-dimensional.- Neither argument is empty.- Values are finite.- Values comply with the selected encoding where encoding validation is applicable.
The returned matrix must have shape (n, m).
5.2 Hebbian Accumulation
Provide a function or small class that accumulates outer-product contributions for a sequence of training pairs.
Requirements:
- Reject an empty training dataset.- Validate that input dimensions are consistent across pairs.- Validate that target dimensions are consistent across pairs.- Preserve per-pair contributions when detailed tracing is requested.- Return the accumulated weight matrix.- Avoid hidden in-place mutation of caller-owned arrays.
For bipolar data, negative contributions are expected and must be preserved.
5.3 Activation
Implement the activation rules documented in Task 01.
The threshold behavior must be exact:
- Binary: zero maps to 0.- Bipolar: zero maps to -1.
Support scalar and vector inputs if this can be implemented consistently and tested clearly.
5.4 Heteroassociative Memory
Provide an implementation that:
- Trains from input-target pairs.- Stores or exposes the trained weight matrix.- Computes net input using x @ W.- Applies the configured activation function.- Returns both net input and activated output.- Validates inference dimensions.
The expected output dimension must match the target dimension.
5.5 Autoassociative Memory
Provide an implementation that:
- Trains from input patterns.- Uses W = sum(outer(s, s)).- Computes x @ W.- Applies the configured activation function.- Returns net input and output.- Validates that inference inputs have the same dimension as stored patterns.
Do not silently set the diagonal of W to zero.
Do not claim that recall always reproduces a stored pattern.
5.6 Structured Results
Design typed result objects, or equivalent well-documented structures, containing the values needed by later application services.
At minimum, inference results must expose:
- Input pattern.- Weight matrix or an explicitly documented reference to it.- Net input.- Activated output.- Encoding.- Activation rule or threshold metadata.- Relevant dimensions.
Training traces should make individual outer products available when requested.
Avoid embedding presentation-specific labels, Qt objects, or GUI state in algorithm results.
6. Error Handling
Use meaningful exceptions or clear validation errors for:
- Incorrect dimensionality.- Incompatible input and target dimensions.- Empty training data.- Unsupported encoding values.- NaN or infinite numerical values.- Inference with incompatible dimensions.
Do not convert invalid input to zeros or ignore invalid training pairs silently.
7. Required Tests
Create tests for:
- Hand-calculated outer products.- Hebbian accumulation from two or more pairs.- Positive, negative, and zero activation inputs.- Binary and bipolar activation.- Heteroassociative training and inference.- Autoassociative training and inference.- Output dimensions.- Invalid dimensions.- Empty training data.- Invalid numerical values.- Non-mutation of caller-owned arrays.- At least one reference example from Task 01.
Include tests that demonstrate a legitimate failure of recall due to interference. Do not treat all nonmatching outputs as implementation defects.
8. Quality Requirements
- Use NumPy for numerical operations.- Add type hints to public functions.- Document non-obvious mathematical choices.- Keep functions small and focused.- Avoid importing PySide6 or application/presentation modules.- Avoid global mutable state.- Do not introduce a machine-learning framework for these basic equations.
9. Acceptance Criteria
- All algorithm unit tests pass.- Numerical outputs match the reference examples.- Dimensions are validated consistently.- The algorithm package runs without starting the GUI.- No algorithm module imports PySide6.- Intermediate numerical values are accessible for educational use.- No unsupported guarantee of successful recall is made.
10. Completion Report
List the implemented APIs, files changed, test commands and actual results, known limitations, and acceptance status.
Do not begin Task 03 until the core algorithm tests pass.