UI Design System — Associative Memory Lab
Product Design Goal
Create a modern desktop scientific laboratory that makes associative memory algorithms visually understandable, not merely executable.
The interface should feel like a polished scientific application: clean, precise, calm, and information-rich without visual clutter.
Layout
- Use a persistent left navigation sidebar.- Use a compact top bar for page title, experiment status, and global actions.- Use a central workspace with a clear hierarchy.- Use a right contextual panel only when it adds value.- Keep the primary action visually obvious on each page.- Avoid excessive nested cards and unnecessary borders.- Support resizing without overlapping or clipping content.
Navigation
Required sections:
1. Dashboard2. Image Dataset3. Training4. Recall and Results5. Experiment History6. Settings
Visual Style
- Default to a light theme.- Use a restrained indigo or blue primary accent.- Use neutral backgrounds and surfaces.- Reserve green for confirmed success, amber for warnings, and red for errors.- Do not communicate status through color alone.- Use consistent spacing, corner radii, typography, and icon sizes.- Avoid excessive shadows, gradients, and decorative animations.
Typography
- Use a readable font with Arabic and Latin coverage.- Use a monospaced font for matrices, vectors, dimensions, and numerical traces.- Keep mathematical notation visually distinct from explanatory text.- Align numerical values consistently.- Use readable labels rather than abbreviations that beginners may not understand.
Image Workspace
Display:
- Original image.- Grayscale preview.- Binary or bipolar preview.- Image dimensions.- Preprocessing settings.- Flattened vector length.- Pixel-grid inspection.
Preserve aspect ratio in previews unless the user explicitly requests distortion.
Training Workspace
Display:
- Memory type.- Training pairs and target labels.- Encoding convention.- Activation threshold.- Training progress and validation errors.- Weight matrix dimensions.- Access to individual outer products and the accumulated weight matrix.
Results Workspace
Display:
- Input, expected target, and actual output where applicable.- Image comparisons.- Net input before activation.- Output after activation.- Exact-match and mismatch metrics.- Matrix and vector inspection.- Step-by-step calculation trace.- Warnings about interference and invalid assumptions.
All displayed numerical values must come from the actual computation result.
Interaction
- Provide clear empty states with an obvious next action.- Disable actions that cannot yet be performed.- Show progress during lengthy operations.- Display actionable error messages.- Use tooltips for technical terms.- Support keyboard navigation and accessible contrast.- Keep image processing settings visible and reproducible.
Data Visualization
- Use a dedicated matrix viewer rather than dumping large arrays into plain text.- Support scrolling, zooming, numeric inspection, and dimension labels.- Use a symmetric color scale around zero for signed weight matrices.- Provide a legend for all heatmaps.- Preserve access to exact numerical values.- Never fabricate metrics or present illustrative values as live experiment results.
Arabic and English
- Support right-to-left layout for Arabic content.- Keep equations and matrix notation left-to-right.- Do not reverse the ordering of numerical vectors.- Keep code identifiers and file paths in their original form.- Use consistent technical terminology across the application.
Acceptance Criteria
The interface is accepted only when:
- Every major page follows the same visual system.- All core user flows can be completed without confusion.- The results workspace exposes both visual and numerical evidence.- Large matrices remain usable.- Errors and empty states are designed rather than left blank.- No displayed value is hardcoded as if it were a real experiment result.




# UI Design System

## 1. Purpose

The UI should feel like a modern desktop laboratory: clear, focused,
visually consistent, and oriented toward experimentation. It must make
intermediate representations and numerical results understandable
without overwhelming the user.

This document defines interaction and visual principles rather than
prescribing a particular pixel-perfect mock-up.

## 2. Experience principles

-   **Show the workflow:** Dataset → Preprocessing → Encoding → Training
    → Recall → Analysis → Experiment history.
-   **Keep context visible:** users should be able to identify the
    active dataset, memory type, encoding, dimensions, and model status.
-   **Separate input from output:** original, processed, encoded, and
    recalled data must not be visually conflated.
-   **Explain numerical results:** label units, dimensions, encodings,
    and activation rules.
-   **Progressive disclosure:** show essential information first, with
    expandable technical detail for matrices and vectors.
-   **Predictable actions:** primary actions should be visually distinct
    and consistently placed.
-   **Accessible by default:** support keyboard navigation, clear focus
    indicators, readable text, and non-color-only status cues.

## 3. Main application structure

A suggested main window includes:

-   **Navigation rail or sidebar:** Dataset, Training, Recall & Results,
    Experiment History, Settings/About.
-   **Header/context area:** current workspace, experiment name, model
    status, and relevant actions.
-   **Main work area:** task-specific panels or cards.
-   **Status area:** operation progress, concise validation messages,
    and non-blocking notifications where appropriate.

Avoid forcing every function into a single crowded screen. Use separate
workspaces while keeping the current experiment context consistent.

## 4. Workflow workspaces

### Dataset workspace

Should support importing images, viewing thumbnails, organizing
autoassociative patterns or heteroassociative source-target pairs, and
selecting items for processing.

Show: - Original image preview. - File name and basic metadata. - Pair
role (source or target) where relevant. - Validation state and any
incompatibility warning.

### Preprocessing and encoding workspace

Show each stage independently: - Original image. -
Grayscale/resized/thresholded result as applicable. - Final binary image
or pixel grid. - Encoded vector with explicit Binary/Bipolar values. -
Width, height, vector length, threshold, and foreground polarity.

The UI must make it clear that preprocessing creates a derived
representation and does not overwrite the original file.

### Training workspace

Show: - Memory type and encoding. - Number of valid patterns or pairs. -
Source and target dimensions. - Learning rule and diagonal policy, where
applicable. - Training status and validation errors. - Weight matrix
dimensions and an inspectable matrix view.

If individual outer-product contributions are displayed, distinguish a
single contribution from the accumulated weight matrix.

### Recall and results workspace

Show: - Selected cue/query. - Net-input values before activation. -
Activation function, threshold, and tie rule. - Recalled output
grid/image. - Expected target or original pattern when known. - Mismatch
count and match ratio when the comparison is valid.

Clearly label whether a comparison is against the source pattern or the
target pattern. Do not imply perfect recall when the output differs.

### Experiment history workspace

Show saved experiments with name, timestamp, memory type, encoding,
dimensions, and summary metrics where available. Offer inspect, load,
duplicate, compare, and delete actions only when those operations are
implemented and appropriately confirmed.

## 5. Visual hierarchy and tokens

Use a restrained, consistent visual language: - One primary accent for
key actions and selected navigation. - Neutral surfaces for most
panels. - Consistent spacing, corner radius, border, and typography
scales. - Distinct semantic states for success, warning, error, and
information. - Limited decorative effects; prioritize legibility of
images and numerical data.

Centralize theme colors, typography sizes, spacing values, and reusable
component styles rather than scattering hard-coded values across
widgets. Exact token values may be selected during implementation and
should then be documented in the theme module.

## 6. Components and states

Reusable components may include: - Navigation item. - Workspace
header. - Image preview panel. - Pattern grid. - Vector viewer. - Matrix
viewer. - Metric tile. - Configuration form row. - Empty state. -
Loading/progress state. - Inline validation message. - Confirmation
dialog.

Each data-dependent component should support at least the relevant
states: empty, loading, ready, invalid/error, and disabled. Do not show
a fake zero value when a result has not yet been computed; use a clear
"Not calculated" or empty state.

## 7. Matrix and vector presentation

Numerical displays are a core educational feature: - Show row and column
dimensions. - Make truncation explicit for large matrices. - Support
scrolling and selecting a cell or row where feasible. - Format values
consistently without changing the underlying stored values. - Provide a
way to inspect raw values or export them if supported. - Use a heatmap
only as a supplement to, not a replacement for, numerical values. - Do
not rely on red/green alone to communicate positive and negative values.

## 8. Interaction rules

-   Disable actions that cannot currently succeed, and explain the unmet
    prerequisite.
-   Validate user input near the relevant control.
-   Confirm destructive actions such as deleting a saved experiment.
-   Avoid blocking the UI during long-running operations; use
    appropriate worker/task mechanisms and progress reporting.
-   Keep operations cancellable only when cancellation is genuinely
    supported.
-   Preserve the user's work when switching workspaces.
-   Do not report a training or save operation as successful before the
    operation completes.

## 9. Accessibility and usability

-   Every interactive control should have an accessible name.
-   Maintain sufficient contrast for text, focus indicators, and status
    labels.
-   Support keyboard focus order and shortcuts only when their behavior
    is documented.
-   Use tooltips for supplementary explanations, not as the only place
    for essential instructions.
-   Avoid tiny click targets and dense numerical layouts.
-   Respect operating-system scaling and high-DPI displays.
-   Provide readable text alternatives for status conveyed through
    icons.

## 10. PySide6 implementation guidance

Use reusable Qt widgets and layouts rather than fixed absolute
positioning. Keep styling centralized, for example in a theme module or
shared stylesheet. Use signals/slots or controller/service interfaces to
communicate user actions. GUI widgets should consume structured service
results and should not own the learning algorithm.

Long operations should not run synchronously on the GUI thread if they
can noticeably freeze the interface. Keep thread/task results marshalled
safely back to the GUI thread.

## 11. UI acceptance checklist

-   The active workflow and experiment are obvious.
-   Original and derived image stages are visibly distinct.
-   Memory type and encoding are always clear when interpreting a
    pattern.
-   Training and recall expose enough detail to explain the result.
-   Errors explain what the user can do next.
-   Empty, loading, and error states are intentionally designed.
-   Layouts remain usable at different window sizes and display scales.
-   Visual styling is consistent across workspaces.
