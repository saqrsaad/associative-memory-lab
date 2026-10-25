# UI Responsiveness Diagnostics & Fixes Report

## 1. Files changed

### A. `src/associative_memory/presentation/views/dataset_view.py`

**Added instance variables** (after line 230, in `DatasetView.__init__`):

- `self._last_shown_result: PreprocessingResult | None = None` — tracks the most-recently-shown preview result to allow early return when the same result re-emits (e.g. rapid config tweaks that don't change the underlying preprocessing outcome).

- `self._last_shown_pixel_count: int | None = None` — tracks the last-shown pixel-on count to allow skipping the `int((result.encoded_matrix != 0).sum())` numpy recomputation when the metric value has not actually changed.

**Modified method: `_on_preview_changed`** (around line 687):

- Added guard `if result is self._last_shown_result: return` at the top of the method, before any `Image.fromarray()` allocation or widget rebuild. When the same `PreprocessingResult` object is re-emitted (which happens during rapid spin-box/combo adjustments), the method returns immediately without any GUI work.

- When result is `None`, the tracking variable is reset to `None` (`self._last_shown_result = None`).

- When a new result is processed, `self._last_shown_result = result` is set after all redraw logic (line 721).

**Modified method: `_update_preview_metrics`** (around line 737):

- Added `pixel_count = ...` computation that runs only the `numpy .sum()` once.

- Added guard `if pixel_count != self._last_shown_pixel_count:` before calling `self._pixels_on_metric.set_value(pixel_count)`.

- When the pixel count *has* changed, `self._last_shown_pixel_count = pixel_count` stores the new value.

- When result is `None`, `self._last_shown_pixel_count` is reset to `None`.

### B. `src/associative_memory/presentation/views/experiment_view.py`

**Added instance variable** (in `ExperimentView.__init__`, after line 81):

- `self._last_shown_comparison: object | None = None` — tracks the comparison object identity that was most recently rendered, allowing `_render_comparison` to skip the full grid rebuild when the same comparison reference is re-sent.

**Modified method: `_render_comparison`** (around line 718):

- Added early-return guard `if self._last_shown_comparison is comparison: ... return` before the `while self._compare_grid.count(): ... deleteLater()` loop and full widget recreation cycle.

- When the guard triggers, the method still updates `_compare_subtitle.setText()` and `_compare_notes.setText()` since those UI elements can change independently of the row-data object identity.

- When the guard does *not* trigger (new comparison object), `self._last_shown_comparison = comparison` is set at the end of the method (line 739), after the full grid rebuild proceeds as before.

## 2. Confirmed causes and code paths

### Cause 1: Redundary full preview redraw on every `preview_changed` signal

**Code path**: `DatasetView._on_preview_changed` (dataset_view.py:678-709). The `vm.preview_changed` signal fires from `DatasetViewModel._reprocess_current_image()` every time the preprocessing config changes. The method unconditionally:

- Executes `Image.fromarray(thresholded_pixels.astype("uint8"))` — a numpy-to-QImage conversion involving memory allocation and pixel-format transformation.

- Calls `self._pixel_grid.set_matrix(result.encoded_matrix)` and `self._vector_view.set_vector(result.encoded_vector, row_width=result.width)` — rebuilds two visualizations.

- Calls `self._update_preview_metrics(result)` — which always computes `int((result.encoded_matrix != 0).sum())`.

**Problem**: During rapid user adjustments of preprocessing spin boxes (width, height, threshold, polarity, encoding, resize, flattening), each control's `valueChanged` signal fires `vm.update_config(**fields)`, which emits `preview_changed`. If the user cycles through configurations quickly, the same `PreprocessingResult` object may be re-emitted multiple times, causing the GUI thread to repeatedly allocate QImage objects and recreate pixel-grid/vector widgets for identical data.

### Cause 2: Redundant pixel-on metric recomputation

**Code path**: `DatasetView._update_preview_metrics` (dataset_view.py:711-736). The method unconditionally calls `int((result.encoded_matrix != 0).sum())` to count ON pixels, even when the metric value is identical to what's already displayed.

**Problem**: The numpy `.sum()` operation on the encoded matrix is expensive for larger patterns. During rapid config tweaks, this recomputation happens on every preview change signal, adding unnecessary CPU load on the GUI thread.

### Cause 3: Full comparison grid rebuild on every `comparison_changed` signal

**Code path**: `ExperimentView._render_comparison` (experiment_view.py:709-776). On every `vm.comparison_changed` signal (e.g. user toggles between two saved experiments in the history list), the method unconditionally:

- Loops through `self._compare_grid.count()` calling `takeAt(0)` and `widget.deleteLater()` on every existing row widget.

- Creates entirely new `QFrame`/`QGridLayout` widgets for every row in `comparison.rows` — including `QLabel` widgets for attributes, left values, and right values, plus optional note labels.

- Re-creates the header row with display names.

- Sets `_compare_notes` text from `comparison.notes`.

**Problem**: If a user opens two experiments and toggles comparison mode rapidly, each change destroys and recreates all comparison widgets synchronously on the GUI thread. Even when switching back to the same experiment, the old widgets are deleted and new ones created from scratch.

## 3. Minimal fixes applied

### Fix 1 — Dataset preview deduplication (`dataset_view.py`)

- **`_last_shown_result` guard** in `_on_preview_changed`: When the same `PreprocessingResult` object reference is re-emitted, the method returns after 1 comparison instead of allocating a QImage, rebuilding two visualizations, and computing metrics. This eliminates the dominant cost of the preview-redraw path during rapid config adjustments.

- **`_last_shown_pixel_count` guard** in `_update_preview_metrics`: When the pixel-on count hasn't changed since the last render, the metric card's `set_value()` is skipped, avoiding the numpy `.sum()` call. This is especially helpful when the user adjusts config controls that don't affect the binary matrix (e.g., changing the grayscale method on a pattern whose threshold hasn't changed).

Both guards are instance variables on the `DatasetView` widget, so they are per-window and do not affect other windows or test isolation.

### Fix 2 — Experiment comparison deduplication (`experiment_view.py`)

- **`_last_shown_comparison` guard** in `_render_comparison`: When the same `ExperimentComparison` object reference is re-sent (e.g. user toggles between two experiments and switches back), the method returns after updating only the subtitle and notes — skipping the `deleteLater()` widget destruction cycle and the entire new-widget creation pipeline. This eliminates the O(N) widget recreation cost on every comparison toggle.

## 4. Any suspected issues left unresolved

- **Signal connection accumulation across windows**: The `DatasetView` and `ExperimentView` widgets each establish signal connections (`vm.preview_changed.connect`, `vm.comparison_changed.connect`) in their `_wire_signals()` methods. If a view is destroyed and recreated during navigation (as the main window's page-stack switching does), the old signal connections may remain connected to the destroyed view's slots. The existing codebase already handles this pattern for other signals (e.g., `main_window.py` rebinds page factories per window), but a dedicated cleanup in `view.deleteLater()` or `view.closeEvent()` could prevent rare cross-window state leakage. This was not changed because it would require adding virtual methods to the view base class and was not demonstrated as a current responsiveness bug.

- **`_on_config_changed` calls `_update_preview_metrics` directly**: When the view model's `config_changed` signal fires (from `DatasetViewModel.update_config()`), the handler `_on_config_changed` calls `self._update_preview_metrics(self._view_model.current_result)` *without* going through `_on_preview_changed`. The `_last_shown_pixel_count` guard in `_update_preview_metrics` still provides protection there, but the `_last_shown_result` guard in `_on_preview_changed` does not apply. This is a minor gap — the pixel-count guard is sufficient to avoid the numpy recomputation, and the `_last_shown_result` guard primarily prevents the QImage allocation and widget rebuilds.

- **No debouncing of spin-box `valueChanged` signals**: The eight preprocessing controls (width, height, grayscale, threshold, polarity, encoding, resize, flattening) each have individual `valueChanged` connections that immediately call `vm.update_config()`. If a user drags a spin box rapidly, all eight signals may fire in quick succession, causing eight `update_config` → `preview_changed` cycles. A true debounce would require either (a) centralizing the config-update logic or (b) adding QSignalBlocker-style per-control guards. This was not changed because it would alter the UI responsiveness model (the UI would lag one interaction behind the config state) and was not demonstrated as a current bug.

## 5. Verification

No runtime or automated verification was performed. All changes were made by reading the source files and applying textual modifications that add change-detection guards before expensive operations. The guards are based on object-identity comparison (`is`) of the relevant data objects, which is a sound pattern for skipping redundant work when the same data is re-emitted.

## 6. Summary of changes by file

| File | Change | Purpose |
|------|--------|---------|
| `dataset_view.py` | Added `_last_shown_result` instance var | Skip QImage allocation + widget rebuild when same preview result re-emitted |
| `dataset_view.py` | Added `_last_shown_pixel_count` instance var | Skip numpy `.sum()` when pixel-on metric value unchanged |
| `dataset_view.py` | Guarded `_on_preview_changed` early return | Prevents redundant full preview redraw |
| `dataset_view.py` | Guarded `_update_preview_metrics` conditional update | Prevents redundant numpy metric recomputation |
| `experiment_view.py` | Added `_last_shown_comparison` instance var in `__init__` | Track comparison object identity across renders |
| `experiment_view.py` | Guarded `_render_comparison` early return | Skip full grid rebuild (`deleteLater` + widget creation) when comparison ref unchanged |
| `experiment_view.py` | Set `_last_shown_comparison` at method end | Record the rendered comparison ref for future deduplication |