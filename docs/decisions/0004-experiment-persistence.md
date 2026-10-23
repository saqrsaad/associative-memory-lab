# ADR 0004: Experiment Persistence Format and Portability Policy

- **Status:** Accepted for the Task 10 implementation
- **Date:** 2026-10-23

## Context

Task 10 requires saving, reopening, and comparing experiments. A saved
experiment must preserve the trained weight matrix, the preprocessing
configuration, and (when available) the numerical trace and the most recent
recall operation, so that reopening an experiment reproduces the same numbers.
The persistence format was not prescribed; the design must keep persistence out
of the presentation and algorithm layers and must never execute code from
untrusted experiment files.

## Decision

1. **Storage layout:** one directory per experiment under
   `<home>/.associative_memory_lab/experiments/<experiment_id>/` containing
   `manifest.json` (readable, UTF-8 JSON metadata) and `arrays.npz` (all
   numerical arrays as `float64`). `experiment_id` is the directory name: the
   32-character lowercase hex of a UUID4.
2. **Manifest schema version:** `SCHEMA_VERSION = 1` recorded in
   `application.dto.saved_experiment`. Schema version 1 declares every expected
   array key with its shape and dtype inside `"arrays"`, so `arrays.npz` is
   validated against the manifest before any number is used.
3. **Security properties:** arrays are read with
   `np.load(..., allow_pickle=False)`; the manifest is plain JSON validated
   field-by-field (strict types, no `NaN`/`Infinity`, no unknown keys).
   Loading a saved experiment reconstructs a memory object from numbers only;
   no pickle and no arbitrary object deserialization are ever used.
4. **Atomic writes:** a save writes both files into a hidden temporary
   directory (`<root>/.save-in-progress-<uuid>`) inside the same root and then
   atomically renames it to the final experiment id. Any failure removes the
   temporary directory and no partial entry becomes visible. Identifier
   conflicts (existing final directory, or a concurrent rename race) raise
   `ExperimentIdConflictError` and overwrite nothing.
5. **Image portability:** original images are **not** copied into saved
   experiments. The exact encoded vectors, the complete preprocessing
   configuration (grid, grayscale method, resampling filter, threshold,
   polarity, encoding, resize mode, flattening order), and the source file
   *names* are stored instead. A later move or deletion of an image file never
   invalidates stored numerical results; only the original-image preview may
   become unavailable.
6. **Repository boundary:** persistence lives only behind the
   `ModelRepository` interface (`application.interfaces.model_repository`) and
   its JSON/NPZ implementation. Views and view models reach storage solely
   through `application.services.experiment_history_service`; they never read
   JSON/NumPy files, and the algorithm package performs no persistence.
7. **Compatibility handling:** manifest schema versions other than 1 are
   rejected with `UnsupportedSchemaError` (newer files are described as
   probably created by a newer release; older files as legacy). Corrupt,
   incomplete, dimensionally inconsistent, or encoding-violating entries raise
   `ExperimentValidationError`; unreadable entries remain listed in the
   history with `readable=False` so they stay visible and deletable. Stored
   comparison metrics are recomputed from the stored arrays at load time and
   must agree exactly (mismatch count) to detect tampering or corruption.

## Alternatives considered

1. **Single-file package (e.g. one `.npz` with a JSON entry):** slightly
   simpler deletion, but manifests would be harder for users to inspect and
   the directory rename gives the same atomicity, so the two-file directory
   layout was preferred.
2. **Copying original images into the experiment:** makes previews always
   available but duplicates potentially large media, complicates deletion, and
   risks storing private file content beyond the numerical evidence needed;
   rejected in favor of exact numerics plus source *name* references and full
   configuration.
3. **Pickle or `allow_pickle=True` for partial-trace structures:** rejected
   unconditionally; loading untrusted files must never execute code.
4. **Migrating legacy/lower versions silently:** rejected; without a defined
   reader for an older version any migration would invent data. Files are
   rejected with a clear message instead.

## Consequences

### Positive

- Saving produces a complete, self-describing, human-inspectable entry whose
  numerical content round-trips exactly.
- Corrupt or foreign data fails safely before any number reaches statistics,
  view models, or the recall workspace.
- History comparison can explain every incompatible pair instead of equating
  incommensurable numbers.

### Trade-offs

- Future schema changes bump `SCHEMA_VERSION` and require an explicit
  version-to-version migration before older files load.
- Deleting an original image only affects preview availability, which the
  history and detail views communicate textually.
- The recall record stored per experiment is the one existing at save time;
  recall runs performed after saving are not retroactively stored.

## Implementation notes

Implemented in `infrastructure.model_repository_json.JsonModelRepository`
(default root from `infrastructure.experiment_repository`),
`application.dto.saved_experiment` (`SCHEMA_VERSION = 1`,
`APPLICATION_VERSION` kept in sync with `pyproject.toml`), and
`application.services.experiment_history_service`. Tests for the acceptance
criteria (save/reload fidelity, corruption rejection, schema rejection,
missing files, id conflicts, deletion, comparison) are specified for this task
and remain deferred at the time of this decision record.
