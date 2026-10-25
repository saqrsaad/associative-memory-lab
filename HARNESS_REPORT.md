# Dashboard & Settings Task — Completion Report

## 1. Exact files changed

1. `src/associative_memory/presentation/views/dataset_view.py`
   - Added `self._last_shown_result: PreprocessingResult | None = None` instance variable in `DatasetView.__init__`
   - Added `self._last_shown_pixel_count: int | None = None` instance variable in `DatasetView.__init__`
   - Guarded `_on_preview_changed` with `if result is self._last_shown_result: return` early return
   - Guarded `_update_preview_metrics` with `if pixel_count != self._last_shown_pixel_count:` conditional metric update
   - Both tracking variables reset when result is `None`

2. `src/associative_memory/presentation/views/experiment_view.py`
   - Added `self._last_shown_comparison: object | None = None` instance variable in `ExperimentView.__init__`
   - Guarded `_render_comparison` with `if self._last_shown_comparison is comparison: ... return` early return before widget-destruction loop
   - Set `self._last_shown_comparison = comparison` at the end of `_render_comparison` when full rebuild proceeds

No other files were changed. All pre-existing user changes in the working tree were preserved.

## 2. Dashboard data sources, update signals, and empty-state behavior

- Dashboard (`DashboardViewModel`) reads exclusively from four shared window view models: `DatasetViewModel`, `TrainingViewModel`, `RecallViewModel`, `ExperimentHistoryViewModel`. No state is duplicated or fabricated.
- Update path: 9 observed signals (`entries_changed`, `memory_type_changed`, `config_changed`, `preview_changed`, `state_changed`, `result_ready`, `models_changed`, `selection_changed`, `history_changed`) → `_emit_changed()` → `changed` → `DashboardView._refresh()`.
- Empty states: each section is a `QStackedWidget` switching between an `EmptyState` (with action signals routed by `MainWindow._build_dashboard_page` to `_open_page(...)`) and live content. `has_model_content` correctly shows content during `TRAINING`/`FAILED` states.

## 3. Settings preferences, storage mechanism, and unsupported settings

- Exactly one editable preference: `window/remember_geometry` (bool, default `False`), persisted through `QSettings` under the application identity.
- `window/geometry` (QByteArray) saved in `closeEvent` when the preference is enabled, restored in `_restore_window_geometry`.
- Toggling off removes saved geometry; re-enabling does not resurrect a stale shape.
- Read-only facts: storage location from shared history service (`or "Not available"`), `APPLICATION_VERSION = "0.1.0"`, `DEFAULT_PREPROCESSING_CONFIG` (16×16 etc.), theme "Light (the only theme implemented)".
- Explicitly declared unsupported in the page: theme selection, configurable storage, startup-page preference. No fake controls exist.

## 4. Factory compatibility and remaining concerns

- `page_for_destination` calls factories positionally with `parent` only; every registered callable is compatible.
- `_is_rebindable_page_factory` rebinds Dashboard/Settings to shared view models only when the registry holds the module default or an earlier `MainWindow` bound method. The `parent`-only `failing_view` monkeypatched in `tests/ui/test_main_window.py` is preserved untouched.
- Cross-window: the latest constructed window owns the Dashboard/Settings binding; no global registry mutation.

## 5. Issues deliberately left unresolved

- **Signal connection accumulation across windows**: If a view is destroyed and recreated during navigation, old signal connections may remain connected. Not changed — not demonstrated as a current bug.
- **`_on_config_changed` calls `_update_preview_metrics` directly**: The `_last_shown_pixel_count` guard in `_update_preview_metrics` still provides protection, but the `_last_shown_result` guard in `_on_preview_changed` does not apply. Minor gap; pixel-count guard is sufficient for the numpy recomputation.
- **No debouncing of spin-box `valueChanged` signals**: Eight preprocessing controls each have individual `valueChanged` connections. A true debounce would alter the UI responsiveness model and was not changed.

## 6. Explicit verification statement

No application launch, test run, lint, or type check was performed. All conclusions and changes are based solely on source-code inspection.

### Next step

Wait for user review. No further action until explicitly authorized.