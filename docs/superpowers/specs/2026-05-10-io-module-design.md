# `io` Module Design

## Goal

Define the JSON and filesystem boundary for `TensorComputation`.

The module should make it easy for callers to load computations from JSON,
serialize computations back to JSON, and keep error reporting precise enough for
library consumers and a future CLI.

It should not own rewrite choice logic, rewrite application, validation policy,
argument parsing, progress reporting, or optimization.

## Scope

This module includes:

- parsing JSON strings into `TensorComputation`
- serializing `TensorComputation` to JSON strings
- reading JSON files from paths
- writing JSON files to paths
- a small error boundary that distinguishes filesystem errors from JSON errors

This module excludes:

- `TensorComputation::validate()` calls
- rewrite candidate generation
- rewrite decision construction
- rewrite application
- CLI argument parsing
- candidate ranking, greedy selection, MCTS, or cost logic
- stdin/stdout handling

A future CLI should call this module, but this module should not call CLI or
rewrite APIs.

## Public API

```rust
#[derive(Debug)]
pub enum IoJsonError {
    Io(std::io::Error),
    Json(serde_json::Error),
}

impl From<std::io::Error> for IoJsonError;
impl From<serde_json::Error> for IoJsonError;
impl std::fmt::Display for IoJsonError;
impl std::error::Error for IoJsonError;

pub fn read_json(
    path: impl AsRef<std::path::Path>,
) -> Result<TensorComputation, IoJsonError>;

pub fn write_json(
    path: impl AsRef<std::path::Path>,
    comp: &TensorComputation,
) -> Result<(), IoJsonError>;

pub fn from_json(input: &str) -> Result<TensorComputation, serde_json::Error>;

pub fn to_json(comp: &TensorComputation) -> Result<String, serde_json::Error>;
```

`IoJsonError` should not derive `PartialEq` or `Eq`, because the wrapped
standard library and serde errors do not support equality in a useful way.
Tests should match on variants instead.

## Behavior

`from_json` is a thin wrapper around:

```rust
serde_json::from_str::<TensorComputation>(input)
```

It returns serde's error directly. If a JSON payload is syntactically valid but
structurally invalid for the symbolic model, `from_json` still succeeds. The
caller decides whether and when to invoke `TensorComputation::validate()`.

`to_json` is a thin wrapper around:

```rust
serde_json::to_string_pretty(comp)
```

Pretty output matches the existing `rustymill` conversion behavior and produces
readable output for file-based workflows. The function should not append an
extra trailing newline.

`read_json` reads the entire file as UTF-8 text with `std::fs::read_to_string`
and then calls `from_json`.

`write_json` calls `to_json` and then writes the resulting string with
`std::fs::write`.

## Error Boundary

`IoJsonError` separates file access failures from JSON failures:

```rust
pub enum IoJsonError {
    Io(std::io::Error),
    Json(serde_json::Error),
}
```

Conversion rules:

- filesystem read/write failures become `IoJsonError::Io`
- serde parse/serialize failures become `IoJsonError::Json`
- string helpers return `serde_json::Error` directly because they do not perform
  filesystem work

`Display` should be concise and include the underlying error message. It does
not need to include path context; callers such as a CLI can add path context at
the call site.

`std::error::Error::source` should return the wrapped error for both variants.

## Relationship To Other Modules

`repr` owns the data model and serde-compatible schema. `io` only uses that
schema through serde.

`rewrite` owns candidate generation, decision validation, rewrite construction,
and rewrite application. `io` does not import `rewrite`.

The future file-based workflow should compose modules externally:

```text
io::read_json
caller-owned rewrite policy
io::write_json or caller-owned stdout handling
```

Keeping that composition outside `io` allows candidate selection and optimization
policy to evolve without changing JSON compatibility helpers.

## Module Export

The crate root should expose the module:

```rust
pub mod io;
```

No crate-root re-exports are required for the first implementation. Callers can
use `gristmill_symbolics::io::{read_json, write_json, from_json, to_json}`.

## Testing

Add focused integration tests for the module.

String helpers:

- `from_json` parses `tests/fixtures/repr/basic.json`
- `to_json` emits pretty JSON that reparses to the same `TensorComputation`
- unsupported legacy symmetry actions still fail through `from_json`

File helpers:

- `write_json` followed by `read_json` round-trips a small computation through a
  temporary file
- malformed JSON read from a file returns `IoJsonError::Json`
- a missing file returns `IoJsonError::Io`

The tests should avoid introducing a new test-only dependency. A unique path
under `std::env::temp_dir()` is sufficient, with best-effort cleanup at the end
of the test.

## Acceptance Criteria

The `io` module is complete when:

- `src/io.rs` exists
- `src/lib.rs` exposes `pub mod io`
- the public API matches this design
- JSON output uses `serde_json::to_string_pretty`
- `read_json` and `write_json` preserve compatible fixture round-tripping
- `IoJsonError` distinguishes filesystem errors from JSON errors
- no validation, rewrite, CLI, ranking, or optimization policy is added to `io`
