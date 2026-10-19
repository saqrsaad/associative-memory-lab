# UI Design System — Associative Memory Lab

---

## 1. Purpose & Product Goal

* **Goal:** Create a modern desktop scientific laboratory that makes associative memory algorithms visually understandable and tangible, rather than merely executable.
* **General Impression:** The interface should feel like a polished scientific application: clean, precise, calm, and information-rich without visual clutter, highlighting intermediate representations and numerical results without overwhelming the user.
* **Nature of the Document:** This document defines interaction, visual, and architectural principles rather than prescribing a rigid mockup.

---

## 2. Experience Principles

1. **Workflow Visibility:**
`Dataset → Preprocessing → Encoding → Training → Recall → Analysis → Experiment History`.
2. **Context Awareness:** Users should be able to identify the active dataset, memory type, encoding convention, dimensions, and model status at any moment.
3. **Separation of Input and Output:** Original, processed, encoded, and recalled data must not be visually conflated.
4. **Explainability:** Label all units, dimensions, encoding conventions, and activation rules clearly.
5. **Progressive Disclosure:** Show essential information first, with expandable technical detail for matrices and vectors.
6. **Predictable Actions:** Primary actions should be visually distinct and consistently placed.
7. **Accessibility by Default:** Support keyboard navigation, clear focus indicators, readable text, and non-color-only status cues.

---

## 3. Main Application Structure & Layout

### Layout

* **Persistent Left Navigation Sidebar / Rail:** For navigation across main sections.
* **Compact Top Bar / Header Area:** Displays workspace title, experiment name, model status, and global actions.
* **Central Workspace:** Clear hierarchy using task-specific cards and panels, avoiding excessive nested cards and unnecessary borders.
* **Right Contextual Panel:** Used only when it adds true contextual value.
* **Status Area:** Shows operation progress, concise validation messages, and non-blocking notifications.
* **Resizing Flexibility:** Support window resizing across different display scales without overlapping or clipping content.

### Required Navigation Sections

1. Dashboard
2. Image Dataset
3. Training
4. Recall and Results
5. Experiment History
6. Settings / About

---

## 4. Workflow Workspaces

### A. Dataset Workspace

* **Function:** Import images, view thumbnails, organize autoassociative patterns or heteroassociative source-target pairs, and select items for processing.
* **Display Elements:**
* Original image preview with file name and basic metadata.
* Pair role (source or target) where relevant.
* Validation state and any incompatibility warning.



### B. Preprocessing & Encoding Workspace

* **Independent Stages Display:**
* Original image.
* Grayscale preview / resized / thresholded result.
* Final binary or bipolar preview on a pixel grid.
* Flattened encoded vector with explicit Binary/Bipolar values.
* Width, height, vector length, threshold, and foreground/background polarity.
* Preserve aspect ratio in previews unless the user explicitly requests distortion.


* **Interaction Note:** The UI must make it clear that preprocessing creates a derived representation and does not overwrite the original file.

### C. Training Workspace

* **Display Elements:**
* Memory type and encoding.
* Training pairs, target labels, and number of valid patterns.
* Source and target dimensions, learning rule, and diagonal policy where applicable.
* Activation threshold.
* Training progress and validation errors.
* Weight matrix dimensions and a dedicated inspectable matrix view.
* Access to individual outer-product contributions and the accumulated weight matrix.



### D. Recall & Results Workspace

* **Display Elements:**
* Selected cue / query.
* Net-input values before activation.
* Activation function, threshold, and tie rule.
* Recalled output grid/image.
* Expected target or original pattern when known.
* Mismatch count and match ratio when comparison is valid.
* Matrix and vector inspection alongside a step-by-step calculation trace.
* Warnings about interference and invalid assumptions.


* **Display Rules:**
* Clearly label whether a comparison is against the source pattern or the target pattern.
* Do not imply perfect recall when the output differs.
* All displayed numerical values must come from actual computation results and must never be hardcoded.



### E. Experiment History Workspace

* **Display Elements:** Saved experiments with name, timestamp, memory type, encoding, dimensions, and summary metrics.
* **Available Actions:** Inspect, load, duplicate, compare, and delete actions only when implemented and appropriately confirmed.

---

## 5. Visual Hierarchy and Tokens

* **Default Theme:** Light theme.
* **Colors and Surfaces:**
* Use a restrained indigo or blue primary accent for key actions and selected navigation.
* Use neutral backgrounds and surfaces for most panels.
* Reserve green for confirmed success, amber for warnings, and red for errors.
* **Strict Rule:** Do not communicate system state or data values through color alone.


* **Design Elements:**
* Use consistent spacing, corner radii, typography scales, and icon sizes.
* Avoid excessive shadows, gradients, and decorative animations; prioritize the legibility of images and numerical data.
* Centralize theme colors, typography sizes, and component styles in a theme module rather than scattering hardcoded values across widgets.



---

## 6. Typography

* **Fonts:** Use a readable font with Arabic and Latin coverage.
* **Numbers and Matrices:** Use a monospaced font for all matrices, vectors, dimensions, and numerical traces.
* **Mathematical Notation:** Keep mathematical notation visually distinct from explanatory text.
* **Alignment and Labeling:** Align numerical values consistently, and use readable labels rather than abbreviations beginners may not understand.

---

## 7. Data & Matrix Visualization

* **Matrix Viewer:** Use a dedicated matrix viewer component rather than dumping large arrays into plain text.
* **Controls & Scrolling:** Support scrolling, zooming, numeric cell/row inspection, and dimension labels.
* **Explicit Truncation:** Make truncation explicit for large matrices.
* **Colors & Heatmaps:**
* Use a symmetric color scale around zero for signed weight matrices.
* Provide a legend for all heatmaps.
* Heatmaps should supplement, not replace, exact numerical values.
* Do not rely on red/green alone to communicate positive and negative values.


* **Data Integrity:** Never fabricate metrics or present illustrative values as live experiment results.

---

## 8. Components and States

* **Reusable Components:** Navigation item, workspace header, image preview panel, pattern grid, vector viewer, matrix viewer, metric tile, configuration form row, empty state, loading/progress state, inline validation message, and confirmation dialog.
* **Data States:** Every data-dependent component must support at least:
`Empty | Loading | Ready | Error/Invalid | Disabled`.
* **Uncomputed Values:** Do not show a fake zero value when a result has not yet been computed; use a clear "Not calculated" or empty state.

---

## 9. Interaction Rules

* **Action Control:** Disable actions that cannot currently succeed and explain the unmet prerequisite.
* **Input Validation:** Validate user input near the relevant control.
* **Confirmation:** Require explicit confirmation for destructive actions (e.g., deleting a saved experiment).
* **Non-blocking UI:** Avoid blocking the UI during long-running operations; use proper worker tasks and progress reporting.
* **Cancellation:** Keep operations cancellable only when cancellation is genuinely supported.
* **State Preservation:** Preserve the user's work when switching workspaces.
* **Interaction Honesty:** Do not report a training or save operation as successful before the operation completes.

---

## 10. Arabic & English / Accessibility

### Linguistic Support (Arabic & RTL):

* Support right-to-left (RTL) layout for Arabic content and interface.
* **Explicit Exceptions:**
* Keep equations and matrix notation left-to-right (LTR).
* Do not reverse the ordering of numerical vectors.
* Keep code identifiers and file paths in their original form.
* Use consistent technical terminology across the entire application.



### Accessibility and Usability:

* Ensure every interactive control has an accessible name.
* Maintain sufficient contrast for text, focus indicators, and status labels.
* Support keyboard focus order and documented shortcuts.
* Use tooltips for supplementary explanations, not as the only place for essential instructions.
* Avoid tiny click targets and dense numerical layouts.
* Respect operating-system scaling and high-DPI displays.

---

## 11. PySide6 Implementation Guidance

* **Components and Layouts:** Use reusable Qt widgets and dynamic layouts rather than fixed absolute positioning.
* **Centralized Styling:** Centralize styling in a theme module or shared stylesheet (QSS).
* **Separation of Concerns:** Use signals/slots or controller/service interfaces to communicate user actions. GUI widgets must consume structured service results and should not own the learning algorithm.
* **Threading:** Run long-running operations on background worker threads to avoid freezing the GUI thread, marshalling results back safely.

---

## 12. UI Acceptance Checklist

The interface is accepted only when:

* [ ] Every major page and workspace follows the same visual system and consistent styling.
* [ ] The active workflow, experiment context, and original vs. derived image stages are visibly distinct.
* [ ] Memory type and encoding are always clear when interpreting any pattern.
* [ ] All core user flows can be completed without confusion.
* [ ] Training and recall expose enough detail (visual and numerical evidence, step-by-step trace) to explain results.
* [ ] Large matrices remain usable and scrollable.
* [ ] Errors and empty states are intentionally designed rather than left blank, providing helpful guidance.
* [ ] Layouts remain usable at different window sizes and display scales.
* [ ] **No displayed value is hardcoded as if it were a real experiment result.**