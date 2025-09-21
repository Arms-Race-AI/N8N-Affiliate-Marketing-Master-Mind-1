This repository contains a small scientific-code toolchain (UPE L4) and supporting
artifacts. The guidance below is tailored to make AI coding agents productive
quickly when changing or extending the code in `upel4/src/upeel/` and the
associated tests in `tests/`.

High-level architecture (quick):
- Core library: `upel4/src/upeel/` — implements a symbolic operator registry
  (`symbolic_operators.py`), backend descriptors and generation helpers
  (`codegen/backends.py`, `codegen/helpers.py`), and lightweight telemetry
  (`telemetry/span_helpers.py`). Treat `upeel` as a single library exposing
  operator metadata and backend code-emission helpers.
- Tests: `tests/` — unit tests validate metadata and rendering behaviors
  (e.g., `test_backends_registry.py`). Use these tests as the authoritative
  behavioral examples when changing registry entries or comment rendering.
- Docs: `docs/` contain higher-level design notes but are not required for
  local development.

Key patterns and conventions (concrete):
- Backend descriptors are instances of `BackendDescriptor` (see
  `codegen/backends.py`). They encode: name, language, file extension,
  invocation_kind, compile/run command tuples and comment semantics. When
  adding or editing a backend, update `COMMENT_SYNTAX` in
  `codegen/helpers.py` and add matching unit tests in `tests/test_backends_registry.py`.
- Comment rendering: use `comment(backend, text)` from
  `codegen/helpers.py` to format clause-aware comments. `BackendDescriptor.render_invocation`
  uses this to attach the operator clause to emitted code.
- Registry pattern: operators are `Op` dataclasses and are registered into the
  `REGISTRY` in `symbolic_operators.py`. New operators must have unique names
  and include an `eval_py` callable and a short `clause` string used for traceability.
- Telemetry is optional and defensive: import of OpenTelemetry/Prometheus is
  guarded in `telemetry/span_helpers.py`. Code must work when these
  dependencies are missing — the module falls back to local in-memory records.

Developer workflows (how to build / test / debug):
- Run unit tests with pytest from repository root. The test harness manipulates
  PYTHONPATH to import `upel4/src`. Example (PowerShell):

```powershell
python -m pytest -q
```

- When editing backends or comment syntax, run `tests/test_backends_registry.py` to
  ensure the registry names, comment tokens and rendering remain consistent.

Project-specific gotchas and examples:
- Many backends include `{source}`, `{output}`, or `{session}` placeholders in
  their `compile_cmd`/`run_cmd`. These are intentionally not validated by the
  descriptor — rely on tests (or add new tests) to assert the absence of
  unexpanded braces when enumerating required binaries (see
  `BackendDescriptor.required_binaries` and `test_backends_registry.py`).
- Comment token mapping uses block/open tokens for some languages (e.g., `/*` or
  `(*`). `comment()` will convert them into balanced comments. Look at
  `codegen/helpers.py:COMMENT_SYNTAX` when adding languages.
- Numeric and sequence coercions in `symbolic_operators.py` accept multiple
  input shapes (scalars, mappings, sequences). Follow existing helper
  functions (e.g. `_vector_from_mapping`, `_is_matrix`) when adding new
  operator implementations to keep consistent error handling.

What to change carefully (and how to validate):
- Changing `COMMENT_SYNTAX` or backend names requires updating tests in
  `tests/test_backends_registry.py` which assert the mapping and rendering.
- Adding operators: ensure unique `Op.name`, correct `arity` and that
  `eval_py` accepts the expected inputs. Add unit tests that exercise both
  pure-python evaluation and any codegen rendering that relies on the clause.
- Telemetry changes must preserve the fallback behavior — tests or manual
  imports should not fail when OpenTelemetry/Prometheus are not installed.

If you need more context, inspect these representative files first:
- `upel4/src/upeel/symbolic_operators.py` (operator registry and math helpers)
- `upel4/src/upeel/codegen/backends.py` (backend descriptors and registry)
- `upel4/src/upeel/codegen/helpers.py` (comment rendering tokens)
- `upel4/src/upeel/telemetry/span_helpers.py` (telemetry fallbacks)
- `tests/test_backends_registry.py` (how the project asserts behavior)

After editing this file, tell me which area you'd like more depth on
(operator semantics, backend targets, or telemetry) and I'll expand examples
or add targeted tests/snippets.
