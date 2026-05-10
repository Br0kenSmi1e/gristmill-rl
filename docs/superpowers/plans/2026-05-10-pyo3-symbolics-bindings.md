# PyO3 Symbolics Bindings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `gristmill_symbolics` PyO3 extension that exposes Rust-backed `TensorComputation` and `ActionSpace` handles plus faithful dict/list snapshots.

**Architecture:** Keep Rust authoritative for JSON construction, validation, cost, action-space generation, decision validation, and rewrite application. The Python extension owns only PyO3 classes, conversion helpers, decision input parsing, and Python exceptions. Python receives data-only snapshots and stores opaque `ActionSpace` handles for reuse on cloned pre-rewrite states.

**Tech Stack:** Rust 2024, PyO3 0.28, pythonize 0.28, maturin, uv, pytest.

---

## File Structure

- Create `python/Cargo.toml`: Rust crate manifest for the PyO3 extension.
- Create `python/pyproject.toml`: uv/maturin Python project metadata and dev dependency declaration.
- Create `python/src/lib.rs`: PyO3 classes, error conversion, snapshot conversion, and decision parsing.
- Create `python/tests/test_bindings.py`: Python tests covering import, constructors, snapshots, action spaces, apply behavior, and lifecycle.
- Generate `python/uv.lock`: uv dependency lockfile.
- Generate `python/Cargo.lock`: Cargo lockfile for the extension crate.
- Modify `.gitignore`: ignore Python build artifacts and virtualenv directories.

The root Rust crate files should not need changes. If implementation reveals a missing public accessor in the Rust core, stop and add the smallest root-crate API with Rust tests before continuing.

## References

- Spec: `docs/superpowers/specs/2026-05-10-pyo3-symbolics-bindings-design.md`
- Existing rewrite tests: `tests/rewrite.rs`
- Existing JSON fixtures: `tests/fixtures/repr/basic.json`
- Existing cost API: `src/cost.rs`
- Existing IO API: `src/io.rs`
- Existing representation API: `src/repr.rs`
- Existing rewrite API: `src/rewrite.rs`

## Task 1: Scaffold The Python Extension Package

**Files:**
- Create: `python/Cargo.toml`
- Create: `python/pyproject.toml`
- Create: `python/src/lib.rs`
- Create: `python/tests/test_bindings.py`
- Generate: `python/uv.lock`
- Generate: `python/Cargo.lock`
- Modify: `.gitignore`

- [ ] **Step 1: Write the failing import smoke test**

Run:

```bash
mkdir -p python/tests
```

Create `python/tests/test_bindings.py`:

```python
def test_module_exports_core_types():
    import gristmill_symbolics

    assert hasattr(gristmill_symbolics, "TensorComputation")
    assert hasattr(gristmill_symbolics, "ActionSpace")
    assert hasattr(gristmill_symbolics, "GristmillSymbolicsError")
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
cd python
uv run pytest tests/test_bindings.py::test_module_exports_core_types -q
```

Expected: FAIL because `gristmill_symbolics` is not importable yet. If `uv` is not initialized, the command may fail before pytest starts; continue with Step 3.

- [ ] **Step 3: Add Python project metadata**

Create `python/pyproject.toml`:

```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "gristmill-symbolics"
version = "0.1.0"
requires-python = ">=3.11"
description = "Python bindings for the gristmill-symbolics rewrite kernel"
dependencies = []

[dependency-groups]
dev = [
    "maturin>=1.0,<2.0",
    "pytest>=8.0",
]

[tool.maturin]
bindings = "pyo3"
module-name = "gristmill_symbolics"
manifest-path = "Cargo.toml"
```

- [ ] **Step 4: Add the PyO3 crate manifest**

Create `python/Cargo.toml`:

```toml
[package]
name = "gristmill-symbolics-python"
version = "0.1.0"
edition = "2024"
publish = false

[lib]
name = "gristmill_symbolics"
crate-type = ["cdylib"]

[dependencies]
gristmill-symbolics = { path = ".." }
pyo3 = { version = "0.28.3", features = ["extension-module"] }
pythonize = "0.28"
serde_json = "1"
```

- [ ] **Step 5: Add the minimal extension module**

Run:

```bash
mkdir -p python/src
```

Create `python/src/lib.rs`:

```rust
use pyo3::prelude::*;

pyo3::create_exception!(
    gristmill_symbolics,
    GristmillSymbolicsError,
    pyo3::exceptions::PyException
);

#[pyclass(name = "TensorComputation")]
struct PyTensorComputation;

#[pymethods]
impl PyTensorComputation {}

#[pyclass(name = "ActionSpace")]
struct PyActionSpace;

#[pymethods]
impl PyActionSpace {}

#[pymodule]
fn gristmill_symbolics(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyTensorComputation>()?;
    module.add_class::<PyActionSpace>()?;
    module.add(
        "GristmillSymbolicsError",
        py.get_type::<GristmillSymbolicsError>(),
    )?;
    Ok(())
}
```

- [ ] **Step 6: Ignore local Python build outputs**

Modify `.gitignore` so it contains:

```gitignore
/target
.worktrees/
/tmp
/python/.venv
/python/target
/python/.pytest_cache
```

- [ ] **Step 7: Build and install the extension into the uv environment**

Run:

```bash
cd python
uv sync
uv run maturin develop
```

Expected: `uv sync` creates `uv.lock`, and `maturin develop` builds and installs `gristmill_symbolics`.

- [ ] **Step 8: Run the import smoke test**

Run:

```bash
cd python
uv run pytest tests/test_bindings.py::test_module_exports_core_types -q
```

Expected: PASS.

- [ ] **Step 9: Commit the scaffold**

Run:

```bash
git add .gitignore python/Cargo.toml python/Cargo.lock python/pyproject.toml python/uv.lock python/src/lib.rs python/tests/test_bindings.py
git commit -m "feat: scaffold pyo3 symbolics package"
```

## Task 2: Add TensorComputation Loading, Validation, Cost, Clone, And Snapshots

**Files:**
- Modify: `python/src/lib.rs`
- Modify: `python/tests/test_bindings.py`

- [ ] **Step 1: Add tests for constructors, validation, clone, cost, and snapshots**

Append to `python/tests/test_bindings.py`:

```python
from pathlib import Path

import pytest

from gristmill_symbolics import GristmillSymbolicsError, TensorComputation


ROOT = Path(__file__).resolve().parents[2]
BASIC_FIXTURE = ROOT / "tests" / "fixtures" / "repr" / "basic.json"


def test_load_json_validates_and_snapshots_basic_fixture():
    comp = TensorComputation.load_json(BASIC_FIXTURE)

    snapshot = comp.snapshot()

    assert snapshot == {
        "ranges": [{"id": 0, "size": 3}],
        "tensors": [
            {
                "id": 0,
                "symmetry": [{"perm": [0], "action": "Identity"}],
            }
        ],
        "definitions": [
            {
                "base": 0,
                "ext_indices": [{"id": 0, "range": 0}],
                "terms": [
                    {
                        "coeff": {"numer": 1, "denom": 1},
                        "sum_indices": [],
                        "factors": [{"tensor": 0, "indices": [0]}],
                    }
                ],
            }
        ],
    }


def test_from_json_string_validates_and_clones():
    text = BASIC_FIXTURE.read_text()

    comp = TensorComputation.from_json_string(text)
    clone = comp.clone()

    assert clone.snapshot() == comp.snapshot()
    assert clone is not comp


def test_invalid_json_string_raises_gristmill_error():
    with pytest.raises(GristmillSymbolicsError):
        TensorComputation.from_json_string("{")


def test_invalid_representation_raises_gristmill_error():
    text = """
    {
      "ranges": [{ "id": 7, "size": 3 }],
      "tensors": [],
      "definitions": []
    }
    """

    with pytest.raises(GristmillSymbolicsError):
        TensorComputation.from_json_string(text)


def test_log_total_flops_returns_python_float():
    comp = TensorComputation.load_json(BASIC_FIXTURE)

    value = comp.log_total_flops()

    assert isinstance(value, float)
    assert value > 0.0
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run:

```bash
cd python
uv run pytest tests/test_bindings.py -q
```

Expected: FAIL because `TensorComputation` does not have `load_json`, `from_json_string`, `clone`, `snapshot`, or `log_total_flops`.

- [ ] **Step 3: Replace `python/src/lib.rs` with computation bindings and snapshot conversion**

Replace `python/src/lib.rs` with:

```rust
use gristmill_symbolics::cost;
use gristmill_symbolics::io;
use gristmill_symbolics::repr::{
    Factor, Index, Rational, SymAction, SymGenerator, TensorComputation as RustTensorComputation,
    TensorDef, TensorInfo, Term,
};
use gristmill_symbolics::rewrite::ActionSpace as RustActionSpace;
use pyo3::prelude::*;
use pythonize::pythonize;
use serde_json::{Value, json};
use std::fmt;
use std::path::PathBuf;

pyo3::create_exception!(
    gristmill_symbolics,
    GristmillSymbolicsError,
    pyo3::exceptions::PyException
);

fn py_gristmill_error(error: impl fmt::Debug) -> PyErr {
    GristmillSymbolicsError::new_err(format!("{error:?}"))
}

fn py_gristmill_display_error(error: impl fmt::Display) -> PyErr {
    GristmillSymbolicsError::new_err(error.to_string())
}

fn validate_loaded(comp: RustTensorComputation) -> PyResult<PyTensorComputation> {
    comp.validate().map_err(py_gristmill_error)?;
    Ok(PyTensorComputation { inner: comp })
}

fn range_value(range: &gristmill_symbolics::repr::Range) -> Value {
    json!({
        "id": range.id.0,
        "size": range.size,
    })
}

fn tensor_value(tensor: &TensorInfo) -> Value {
    json!({
        "id": tensor.id.0,
        "symmetry": tensor.symmetry.iter().map(sym_generator_value).collect::<Vec<_>>(),
    })
}

fn sym_generator_value(generator: &SymGenerator) -> Value {
    json!({
        "perm": generator.perm,
        "action": sym_action_name(generator.action),
    })
}

fn sym_action_name(action: SymAction) -> &'static str {
    match action {
        SymAction::Identity => "Identity",
        SymAction::Negate => "Negate",
    }
}

fn tensor_def_value(definition: &TensorDef) -> Value {
    json!({
        "base": definition.base.0,
        "ext_indices": definition.ext_indices.iter().map(index_value).collect::<Vec<_>>(),
        "terms": definition.terms.iter().map(term_value).collect::<Vec<_>>(),
    })
}

fn term_value(term: &Term) -> Value {
    json!({
        "coeff": rational_value(&term.coeff),
        "sum_indices": term.sum_indices.iter().map(index_value).collect::<Vec<_>>(),
        "factors": term.factors.iter().map(factor_value).collect::<Vec<_>>(),
    })
}

fn rational_value(rational: &Rational) -> Value {
    json!({
        "numer": *rational.numer(),
        "denom": *rational.denom(),
    })
}

fn index_value(index: &Index) -> Value {
    json!({
        "id": index.id.0,
        "range": index.range.0,
    })
}

fn factor_value(factor: &Factor) -> Value {
    json!({
        "tensor": factor.tensor.0,
        "indices": factor.indices.iter().map(|index| index.0).collect::<Vec<_>>(),
    })
}

fn computation_value(comp: &RustTensorComputation) -> Value {
    json!({
        "ranges": comp.ranges().iter().map(range_value).collect::<Vec<_>>(),
        "tensors": comp.tensors().iter().map(tensor_value).collect::<Vec<_>>(),
        "definitions": comp.definitions().iter().map(tensor_def_value).collect::<Vec<_>>(),
    })
}

#[pyclass(name = "TensorComputation")]
struct PyTensorComputation {
    inner: RustTensorComputation,
}

#[pymethods]
impl PyTensorComputation {
    #[staticmethod]
    fn load_json(path: PathBuf) -> PyResult<Self> {
        let comp = io::read_json(path).map_err(py_gristmill_display_error)?;
        validate_loaded(comp)
    }

    #[staticmethod]
    fn from_json_string(text: &str) -> PyResult<Self> {
        let comp = io::from_json(text).map_err(py_gristmill_display_error)?;
        validate_loaded(comp)
    }

    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }

    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        pythonize(py, &computation_value(&self.inner)).map_err(py_gristmill_display_error)
    }

    fn log_total_flops(&self) -> PyResult<f64> {
        cost::log_total_flops(&self.inner).map_err(py_gristmill_error)
    }
}

#[pyclass(name = "ActionSpace")]
struct PyActionSpace {
    inner: RustActionSpace,
}

#[pymethods]
impl PyActionSpace {}

#[pymodule]
fn gristmill_symbolics(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyTensorComputation>()?;
    module.add_class::<PyActionSpace>()?;
    module.add(
        "GristmillSymbolicsError",
        py.get_type::<GristmillSymbolicsError>(),
    )?;
    Ok(())
}
```

- [ ] **Step 4: Run formatting and the Python tests**

Run:

```bash
cd python
cargo fmt
uv run maturin develop
uv run pytest tests/test_bindings.py -q
```

Expected: all current Python tests PASS.

- [ ] **Step 5: Commit computation bindings**

Run:

```bash
git add python/Cargo.toml python/Cargo.lock python/src/lib.rs python/tests/test_bindings.py
git commit -m "feat: expose tensor computation bindings"
```

## Task 3: Add ActionSpace Handles And Public Snapshots

**Files:**
- Modify: `python/src/lib.rs`
- Modify: `python/tests/test_bindings.py`

- [ ] **Step 1: Add tests for `next_action_space`, `ActionSpace` properties, and action-space snapshots**

Append to `python/tests/test_bindings.py`:

```python
import json

from gristmill_symbolics import ActionSpace


def actionable_json() -> str:
    return json.dumps(
        {
            "ranges": [{"id": 0, "size": 8}],
            "tensors": [
                {"id": 0, "symmetry": []},
                {"id": 1, "symmetry": []},
                {"id": 2, "symmetry": []},
                {"id": 3, "symmetry": []},
            ],
            "definitions": [
                {
                    "base": 3,
                    "ext_indices": [
                        {"id": 0, "range": 0},
                        {"id": 1, "range": 0},
                    ],
                    "terms": [
                        {
                            "coeff": [1, 1],
                            "sum_indices": [{"id": 2, "range": 0}],
                            "factors": [
                                {"tensor": 0, "indices": [0, 2]},
                                {"tensor": 1, "indices": [2, 1]},
                            ],
                        },
                        {
                            "coeff": [1, 1],
                            "sum_indices": [{"id": 3, "range": 0}],
                            "factors": [
                                {"tensor": 0, "indices": [0, 3]},
                                {"tensor": 2, "indices": [3, 1]},
                            ],
                        },
                    ],
                }
            ],
        }
    )


def test_next_action_space_returns_none_for_basic_fixture():
    comp = TensorComputation.load_json(BASIC_FIXTURE)

    assert comp.next_action_space(0) is None


def test_next_action_space_returns_handle_and_public_snapshot():
    comp = TensorComputation.from_json_string(actionable_json())

    space = comp.next_action_space(0)
    snapshot = space.snapshot()

    assert isinstance(space, ActionSpace)
    assert space.def_index == 0
    assert space.candidate_count == len(snapshot["candidate_templates"])
    assert space.candidate_count > 0
    assert snapshot["def_index"] == 0
    first = snapshot["candidate_templates"][0]
    assert set(first) == {
        "left_definition",
        "right_definition",
        "rewritten_definition",
    }
    assert first["left_definition"]["terms"]
    assert first["right_definition"]["terms"]
    assert first["rewritten_definition"]["terms"]
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run:

```bash
cd python
uv run pytest tests/test_bindings.py::test_next_action_space_returns_none_for_basic_fixture tests/test_bindings.py::test_next_action_space_returns_handle_and_public_snapshot -q
```

Expected: FAIL because `next_action_space`, `ActionSpace.def_index`, `ActionSpace.candidate_count`, and `ActionSpace.snapshot` are not implemented.

- [ ] **Step 3: Add factorization conversion and action-space methods**

Patch `python/src/lib.rs` to import `Factorization` and `rewrite`:

```rust
use gristmill_symbolics::rewrite::{
    self, ActionSpace as RustActionSpace, Factorization,
};
```

Replace the earlier single-line `RustActionSpace` import with the block above.

Add these conversion helpers after `computation_value`:

```rust
fn factorization_value(factorization: &Factorization) -> Value {
    json!({
        "left_definition": tensor_def_value(&factorization.left_definition),
        "right_definition": tensor_def_value(&factorization.right_definition),
        "rewritten_definition": tensor_def_value(&factorization.rewritten_definition),
    })
}

fn action_space_value(space: &RustActionSpace) -> Value {
    json!({
        "def_index": space.def_index,
        "candidate_templates": space
            .candidate_templates
            .iter()
            .map(factorization_value)
            .collect::<Vec<_>>(),
    })
}
```

Add this method to the `PyTensorComputation` `#[pymethods]` block:

```rust
    fn next_action_space(&self, start_from: usize) -> PyResult<Option<PyActionSpace>> {
        rewrite::next_action_space(&self.inner, start_from)
            .map(|space| space.map(|inner| PyActionSpace { inner }))
            .map_err(py_gristmill_error)
    }
```

Replace the empty `PyActionSpace` `#[pymethods]` block with:

```rust
#[pymethods]
impl PyActionSpace {
    #[getter]
    fn def_index(&self) -> usize {
        self.inner.def_index
    }

    #[getter]
    fn candidate_count(&self) -> usize {
        self.inner.candidate_templates.len()
    }

    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        pythonize(py, &action_space_value(&self.inner)).map_err(py_gristmill_display_error)
    }
}
```

- [ ] **Step 4: Run formatting and action-space tests**

Run:

```bash
cd python
cargo fmt
uv run maturin develop
uv run pytest tests/test_bindings.py -q
```

Expected: all current Python tests PASS.

- [ ] **Step 5: Commit action-space bindings**

Run:

```bash
git add python/src/lib.rs python/tests/test_bindings.py
git commit -m "feat: expose action space handles"
```

## Task 4: Add Decision Parsing And Rewrite Application

**Files:**
- Modify: `python/src/lib.rs`
- Modify: `python/tests/test_bindings.py`

- [ ] **Step 1: Add tests for applying decisions, invalid decisions, malformed decisions, and handle reuse**

Append to `python/tests/test_bindings.py`:

```python
def first_full_decision(space):
    template = space.snapshot()["candidate_templates"][0]
    return {
        "candidate_index": 0,
        "left_mask": [True] * len(template["left_definition"]["terms"]),
        "right_mask": [True] * len(template["right_definition"]["terms"]),
    }


def test_apply_decision_with_space_mutates_clone_and_returns_none():
    comp = TensorComputation.from_json_string(actionable_json())
    space = comp.next_action_space(0)
    child = comp.clone()
    before = child.snapshot()
    decision = first_full_decision(space)

    result = child.apply_decision_with_space(space, decision)
    after = child.snapshot()

    assert result is None
    assert len(after["tensors"]) == len(before["tensors"]) + 2
    assert len(after["definitions"]) == len(before["definitions"]) + 2
    assert after != before


def test_invalid_decision_raises_and_does_not_mutate():
    comp = TensorComputation.from_json_string(actionable_json())
    space = comp.next_action_space(0)
    child = comp.clone()
    before = child.snapshot()
    bad_decision = {
        "candidate_index": 0,
        "left_mask": [],
        "right_mask": [True],
    }

    with pytest.raises(GristmillSymbolicsError):
        child.apply_decision_with_space(space, bad_decision)

    assert child.snapshot() == before


def test_malformed_decision_shape_raises_type_or_value_error():
    comp = TensorComputation.from_json_string(actionable_json())
    space = comp.next_action_space(0)

    with pytest.raises(TypeError):
        comp.clone().apply_decision_with_space(space, "not a dict")

    with pytest.raises(ValueError):
        comp.clone().apply_decision_with_space(
            space,
            {"candidate_index": 0, "left_mask": [True]},
        )


def test_action_space_handle_is_reusable_on_multiple_clones():
    comp = TensorComputation.from_json_string(actionable_json())
    space = comp.next_action_space(0)
    decision = first_full_decision(space)
    left = comp.clone()
    right = comp.clone()

    left.apply_decision_with_space(space, decision)
    right.apply_decision_with_space(space, decision)

    assert left.snapshot() == right.snapshot()
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run:

```bash
cd python
uv run pytest tests/test_bindings.py::test_apply_decision_with_space_mutates_clone_and_returns_none tests/test_bindings.py::test_invalid_decision_raises_and_does_not_mutate tests/test_bindings.py::test_malformed_decision_shape_raises_type_or_value_error tests/test_bindings.py::test_action_space_handle_is_reusable_on_multiple_clones -q
```

Expected: FAIL because `apply_decision_with_space` and decision parsing are not implemented.

- [ ] **Step 3: Add decision parsing imports**

Patch the imports in `python/src/lib.rs`:

```rust
use gristmill_symbolics::rewrite::{
    self, ActionSpace as RustActionSpace, Decision, Factorization,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::types::{PyBool, PyDict, PyList};
```

This replaces the previous `gristmill_symbolics::rewrite` import block and extends the PyO3 imports.

- [ ] **Step 4: Add decision parsing helpers**

Add these helpers after `action_space_value`:

```rust
fn required_dict_item<'py>(
    dict: &Bound<'py, PyDict>,
    field: &str,
) -> PyResult<Bound<'py, PyAny>> {
    dict.get_item(field)?
        .ok_or_else(|| PyValueError::new_err(format!("missing decision field '{field}'")))
}

fn parse_bool_mask(value: &Bound<'_, PyAny>, field: &str) -> PyResult<Vec<bool>> {
    let list = value
        .downcast::<PyList>()
        .map_err(|_| PyTypeError::new_err(format!("decision field '{field}' must be a list")))?;

    let mut mask = Vec::with_capacity(list.len());
    for item in list.iter() {
        if !item.is_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err(format!(
                "decision field '{field}' must contain only bool values"
            )));
        }
        mask.push(item.extract::<bool>()?);
    }
    Ok(mask)
}

fn parse_decision(value: &Bound<'_, PyAny>) -> PyResult<Decision> {
    let dict = value
        .downcast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("decision must be a dict"))?;

    let candidate_index = required_dict_item(dict, "candidate_index")?.extract::<usize>()?;
    let left_mask = parse_bool_mask(&required_dict_item(dict, "left_mask")?, "left_mask")?;
    let right_mask = parse_bool_mask(&required_dict_item(dict, "right_mask")?, "right_mask")?;

    Ok(Decision {
        candidate_index,
        left_mask,
        right_mask,
    })
}
```

- [ ] **Step 5: Add `apply_decision_with_space`**

Add this method to the `PyTensorComputation` `#[pymethods]` block:

```rust
    fn apply_decision_with_space(
        &mut self,
        space: &PyActionSpace,
        decision: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let decision = parse_decision(decision)?;
        let rewrite = rewrite::build_rewrite(&self.inner, &space.inner, &decision)
            .map_err(py_gristmill_error)?;
        rewrite::apply_rewrite(&mut self.inner, rewrite).map_err(py_gristmill_error)
    }
```

- [ ] **Step 6: Run formatting and full Python tests**

Run:

```bash
cd python
cargo fmt
uv run maturin develop
uv run pytest tests/test_bindings.py -q
```

Expected: all Python tests PASS.

- [ ] **Step 7: Commit rewrite application bindings**

Run:

```bash
git add python/src/lib.rs python/tests/test_bindings.py
git commit -m "feat: apply rewrite decisions from python"
```

## Task 5: Final Verification And Cleanup

**Files:**
- Modify only if verification exposes a concrete failure.

- [ ] **Step 1: Run Rust tests for the root crate**

Run:

```bash
cargo test
```

Expected: all root Rust tests PASS.

- [ ] **Step 2: Run Rust tests for the PyO3 crate**

Run:

```bash
cd python
cargo test
```

Expected: the PyO3 crate compiles and Rust test harness reports PASS.

- [ ] **Step 3: Run Python tests through uv**

Run:

```bash
cd python
uv run pytest -q
```

Expected: all Python tests PASS.

- [ ] **Step 4: Verify import manually through uv**

Run:

```bash
cd python
uv run python -c "from gristmill_symbolics import TensorComputation; print(TensorComputation.load_json('../tests/fixtures/repr/basic.json').snapshot()['ranges'][0]['size'])"
```

Expected output:

```text
3
```

- [ ] **Step 5: Inspect git status**

Run:

```bash
git status --short
```

Expected: only intentional files are modified or untracked. At this point the expected tracked changes are under `.gitignore` and `python/`.

- [ ] **Step 6: Commit final verification fixes if any were required**

If Step 5 shows no uncommitted changes, skip this step. If verification required small fixes, run:

```bash
git add .gitignore python/Cargo.toml python/Cargo.lock python/pyproject.toml python/uv.lock python/src/lib.rs python/tests/test_bindings.py
git commit -m "test: verify pyo3 symbolics bindings"
```

Expected: a commit is created only when verification caused additional file changes after Task 4.

## Self-Review

- Spec coverage: the plan covers module name, `TensorComputation`, `ActionSpace`, JSON loaders plus validation, clone, snapshot, cost, action-space generation, public action-space snapshots, dict decisions, in-place apply returning `None`, reusable action-space handles, uv/maturin packaging, and Python exception behavior.
- Scope check: the plan does not add RL sampling, PUCT, replay, JAX, feature padding, action-space caching, or experiment runners.
- Type consistency: public Python names are `TensorComputation`, `ActionSpace`, and `GristmillSymbolicsError`; methods match the approved spec.
- Verification: root Rust tests, extension crate tests, Python pytest, and a manual uv import check are included.
