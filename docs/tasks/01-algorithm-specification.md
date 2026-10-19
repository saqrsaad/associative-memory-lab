Task 01 — Algorithm Specification and Mathematical Conventions
1. Objective
Establish an unambiguous mathematical specification for the associative memory algorithms taught in the course.
This task defines the numerical conventions that all subsequent algorithm, application, and presentation tasks must follow.
This is a specification and verification task. Do not implement the complete algorithm package or build the GUI in this task.
2. Prerequisites
- Task 00 — Project Foundation is complete.- AGENTS.md has been reviewed.- The repository and existing documentation have been inspected.
3. Required Deliverables
Create or update:
- docs/algorithm-theory.md- docs/mathematical-conventions.md- docs/architecture.md, if architectural clarifications are required- docs/decisions/0003-numerical-conventions.md- tests/fixtures/algorithm_reference_cases.json, if a machine-readable fixture is useful
Do not duplicate the same mathematical rules across multiple documents. Make docs/mathematical-conventions.md the authoritative source for numerical conventions.
4. Mathematical Scope
Document the following topics.
4.1 Pattern Representation
Define:
- Binary encoding: {0, 1}.- Bipolar encoding: {-1, +1}.- The ON and OFF pixel conventions.- The binary-to-bipolar conversion: x_bipolar = 2*x_binary - 1.- The bipolar-to-binary conversion: x_binary = (x_bipolar + 1)/2.
State that encoding is explicit configuration, not something inferred silently from input values.
4.2 Training Pairs
Define each training pair as:
s^(p) -> t^(p)
Explain the distinction between:
- Heteroassociative memory: input and target patterns may differ.- Autoassociative memory: the target is the same pattern as the input.
4.3 Vector and Matrix Convention
Use row vectors throughout the application.
For input dimension n and target dimension m:
- s has shape (n,).- t has shape (m,).- W has shape (n, m).- Inference computes y_in = x @ W.- The output net-input vector has shape (m,).
Explicitly state that alternative column-vector conventions exist in the literature but are not used in this project.
4.4 Hebbian Learning and Outer Product
Document:
W^(p) = outer(s^(p), t^(p))
W = sum_p outer(s^(p), t^(p))
For autoassociative memory:
W = sum_p outer(s^(p), s^(p))
Explain that the outer product expresses the pairwise Hebbian contribution as a matrix. It is the matrix formulation used by this project, not a separate competing learning rule.
Clarify that training pairs are accumulated in their original order only for tracing and reporting; summation itself is order-independent, apart from any floating-point considerations.
4.5 Activation Functions
Define binary activation as:
f(a) = 1 if a > 0, otherwise 0.
Define bipolar activation as:
f(a) = +1 if a > 0, otherwise -1.
The zero threshold must be handled exactly as documented above. Do not substitute >= 0.
Explain why the zero case matters and include tests for it in Task 02.
4.6 Interference and Recall Evaluation
Explain:
- Crosstalk between stored patterns.- The possibility of outputs that do not match any stored target.- The difference between a successful numerical computation and successful pattern recall.- Why an autoassociative Hebbian network can fail to reconstruct a stored pattern.
Do not claim a universal capacity limit unless the assumptions and model are explicitly specified.
Do not introduce Hopfield iterative dynamics or zero-diagonal weight rules as defaults. These require separate documented decisions and are outside the basic course algorithm.
5. Reference Examples
Document hand-calculated examples for:
- A single outer product.- The sum of multiple Hebbian contributions.- Heteroassociative inference.- Autoassociative inference.- Binary and bipolar activation at positive, negative, and zero inputs.
Use small vectors and show the dimensions of all intermediate results.
Where lecture examples are available, preserve their original inputs and expected results. Record any ambiguity or apparent typo instead of silently modifying the lecture formula.
6. Implementation Guidance
This task must not introduce unnecessary runtime dependencies.
Use Python and NumPy notation in examples where appropriate.
If a reference calculation is encoded as a JSON fixture, define its schema and ensure the fixture can be parsed by a test in a later task.
Do not write an independent algorithm implementation merely to produce expected results. Reference values must be hand-calculated or independently verified.
7. Required Validation
- Verify the dimensions of every documented matrix multiplication.- Verify each numerical example independently.- Check that binary and bipolar encodings are never mixed within a calculation.- Confirm that activation threshold rules agree across all documents.- Review all equations for consistency with the row-vector convention.
8. Acceptance Criteria
- The numerical convention is explicit and unambiguous.- All reference examples include expected dimensions and results.- The zero-threshold activation rule is documented.- Autoassociative and heteroassociative learning are clearly distinguished.- Any discrepancy in lecture notation is recorded as a clarification, not silently resolved.- No GUI or full algorithm implementation is added.- Relevant documentation is internally consistent.
9. Completion Report
Report:
1. Documents created or changed.2. Reference examples verified.3. Any ambiguity remaining in the course material.4. Validation checks performed.5. Acceptance criteria status.
Do not begin Task 02 if the core mathematical conventions remain ambiguous.