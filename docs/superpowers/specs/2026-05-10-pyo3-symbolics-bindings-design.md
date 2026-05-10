# PyO3 Symbolics Bindings Design

## Goal

Add a thin PyO3 layer that exposes the Rust symbolic rewrite kernel to Python.

The module is a converter and handle boundary. Rust remains authoritative for
construction, validation, cost, legal action-space generation, decision
validation, and rewrite application. Python receives faithful public snapshots
for feature extraction and keeps opaque handles for Rust values whose private
sidecars must be preserved.

This is the PyO3 submodule of the sampled AlphaZero RL prototype. It does not
implement RL search, sampling, feature padding, neural models, replay, or
training.

## Context

The root crate already exposes the deterministic rewrite kernel and cost
objective:

- `io::read_json`
- `io::from_json`
- `TensorComputation::validate`
- `cost::log_total_flops`
- `rewrite::next_action_space`
- `rewrite::validate_decision`
- `rewrite::build_rewrite`
- `rewrite::apply_rewrite`

The PyO3 module should wrap these APIs rather than reimplementing their logic in
Python.

The most important boundary is `rewrite::ActionSpace`. Its public fields include
the definition index and candidate factorization templates, but the struct also
contains private graph and biclique sidecars needed by `build_rewrite`. Python
must be able to hold and reuse the exact Rust `ActionSpace` that produced a
decision.

## Module Shape

The Python extension module is named:

```python
gristmill_symbolics
```

It exposes two Rust-backed Python classes:

```python
TensorComputation
ActionSpace
```

`TensorComputation` owns one Rust `repr::TensorComputation`. It is not a Python
data class. Python cannot construct it from dicts or lists.

`ActionSpace` owns one Rust `rewrite::ActionSpace`. It is an opaque live handle.
Python can inspect a public snapshot, but it cannot access or mutate private
graph or biclique sidecars.

Higher-level RL environment state, action-space caching, sampled PUCT, and
training logic belong in the later pure Python `gristmill_rl` package, not in
this module.

## TensorComputation API

Construction goes through Rust JSON loaders and validation:

```python
comp = TensorComputation.load_json(path)
comp = TensorComputation.from_json_string(text)
```

Both constructors call the existing Rust loader and then
`TensorComputation::validate()`. Invalid JSON or invalid symbolic representation
raises a Python exception and returns no handle.

Core methods:

```python
child = comp.clone()
state = comp.snapshot()
log_cost = comp.log_total_flops()
space = comp.next_action_space(start_from)
comp.apply_decision_with_space(space, decision)
```

`clone()` returns a new Rust-backed `TensorComputation` with a cloned Rust
computation.

`snapshot()` returns faithful public `repr.rs` data as plain Python `dict` and
`list` primitives.

`log_total_flops()` calls Rust `cost::log_total_flops()` and returns a Python
`float`.

`next_action_space(start_from)` calls Rust `rewrite::next_action_space()` and
returns either an `ActionSpace` handle or `None`.

`apply_decision_with_space(space, decision)` converts a Python decision dict
into `rewrite::Decision`, validates/builds/applies the rewrite through Rust, and
mutates the receiving `TensorComputation` in place. It returns `None`.

The next rewrite cursor is not returned because it is already available as
`space.def_index`. Python search code should use:

```python
space = comp.next_action_space(start_from)
decision = ...
comp.apply_decision_with_space(space, decision)
start_from = space.def_index
```

## ActionSpace API

`ActionSpace` exposes read-only convenience properties:

```python
space.def_index
space.candidate_count
space.snapshot()
```

`def_index` is the definition index selected by Rust
`rewrite::next_action_space`.

`candidate_count` is the length of `candidate_templates`.

`snapshot()` returns the public rewrite data as plain Python `dict` and `list`
primitives:

```python
{
    "def_index": 0,
    "candidate_templates": [
        {
            "left_definition": {...},
            "right_definition": {...},
            "rewritten_definition": {...},
        }
    ],
}
```

The nested definitions use the same `TensorDef` schema as
`TensorComputation.snapshot()`.

An `ActionSpace` is not consumed by `apply_decision_with_space`. It remains
reusable so Python MCTS nodes can store one action-space handle and apply
different sampled decisions to multiple clones of the same pre-rewrite
computation.

An `ActionSpace` is valid for the computation state that produced it and for
clones of that same pre-rewrite state. Applying it to an unrelated or already
rewritten computation is outside the API contract. The v1 PyO3 layer documents
this lifecycle rule but does not add state fingerprints or hidden ownership
checks.

The PyO3 layer does not cache action spaces. One-time action-space computation
per search node is a Python RL-layer invariant.

## Snapshot Schema

Snapshots mirror the public Rust representation from `repr.rs` and the public
parts of `rewrite.rs`.

IDs are converted from Rust newtypes to Python integers:

- `RangeId(0)` becomes `0`
- `TensorId(0)` becomes `0`
- `IndexId(0)` becomes `0`

`Rational` values are converted to explicit exact dictionaries:

```python
{"numer": 1, "denom": 2}
```

Enum variants become strings matching Rust variant names:

```python
"Identity"
"Negate"
```

A computation snapshot has this shape:

```python
{
    "ranges": [
        {"id": 0, "size": 10},
    ],
    "tensors": [
        {
            "id": 0,
            "symmetry": [
                {"perm": [0, 1], "action": "Identity"},
            ],
        },
    ],
    "definitions": [
        {
            "base": 2,
            "ext_indices": [
                {"id": 0, "range": 0},
            ],
            "terms": [
                {
                    "coeff": {"numer": 1, "denom": 1},
                    "sum_indices": [],
                    "factors": [
                        {"tensor": 0, "indices": [0]},
                    ],
                },
            ],
        },
    ],
}
```

The conversion is intentionally data-only. Python feature extraction may reshape
or pad these snapshots later, but PyO3 does not expose JAX arrays or learned
feature tensors.

## Decision Input

`Decision` is a Python input dict rather than a PyO3 class:

```python
{
    "candidate_index": 0,
    "left_mask": [True, True],
    "right_mask": [True],
}
```

The PyO3 layer converts this dict into Rust `rewrite::Decision`. Missing fields,
wrong field types, or non-boolean masks raise Python type/value errors before
calling rewrite logic. Semantically invalid decisions, such as out-of-range
candidate indices or empty masks, raise errors from Rust rewrite validation.

## Error Handling

The module exposes one custom base exception:

```python
GristmillSymbolicsError
```

Rust errors from JSON loading, representation validation, cost evaluation, and
rewrite operations are converted into this exception with clear messages.

Python-side malformed inputs may raise standard `TypeError` or `ValueError`.
Examples include a non-dict decision input, missing decision fields, and masks
that are not lists of booleans.

Invalid rewrite decisions do not return status dictionaries. They raise and
leave the receiving `TensorComputation` unchanged.

## Packaging Layout

The repository gains a `python/` directory for the PyO3 crate and package:

```text
python/
  Cargo.toml
  pyproject.toml
  src/lib.rs
  tests/
```

The PyO3 crate depends on the root crate by path. It builds the Python extension
module named `gristmill_symbolics`.

The later pure Python RL package can live under `python/gristmill_rl/`, but it is
outside this PyO3 submodule design.

## Testing

PyO3 tests should cover:

- `TensorComputation.load_json(path)` validates and loads a fixture
- `TensorComputation.from_json_string(text)` validates and loads a fixture
- invalid JSON or invalid representation raises an exception
- `TensorComputation.snapshot()` exposes the faithful public `repr.rs` shape
- `TensorComputation.log_total_flops()` returns a finite Python float for valid
  nonzero-cost inputs
- `TensorComputation.next_action_space(start_from)` returns `None` or an
  `ActionSpace` handle
- `ActionSpace.snapshot()` exposes `def_index` and candidate templates
- `apply_decision_with_space(space, decision)` mutates a cloned computation and
  returns `None`
- invalid decisions raise and do not mutate the computation
- one `ActionSpace` handle can be reused on multiple clones of the same original
  computation

Root Rust tests remain responsible for rewrite correctness, cost correctness,
and representation validation. PyO3 tests verify conversion, lifecycle, and
exception behavior at the Python boundary.

## Acceptance Criteria

The PyO3 submodule is complete when:

- Python can import `gristmill_symbolics`
- Python can load and validate a Rust `TensorComputation` from a JSON path or
  JSON string
- Python can clone, snapshot, and compute log FLOP cost for a computation
- Python can request a Rust `ActionSpace` handle from a computation
- Python can inspect the public part of an action space as dict/list data
- Python can apply a dict decision through the exact stored `ActionSpace` handle
- `apply_decision_with_space` mutates the receiving computation and returns
  `None`
- `ActionSpace` handles are reusable on clones of the same pre-rewrite state
- malformed inputs and Rust errors surface as Python exceptions

The PyO3 module remains a converter and handle layer. It does not implement RL
policy logic, search logic, action-space caching, feature padding, model
training, or experiment runners.
