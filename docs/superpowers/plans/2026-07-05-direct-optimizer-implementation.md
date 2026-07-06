# Direct Optimizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained direct optimizer package that trains and samples verified target-definition proposals for `TensorComputation` values without using the existing action-selector, REINFORCE trainer, or CLI checkpoint path.

**Architecture:** Add `gristmill_symbolics.direct_optimizer` as an independent Python package with converter, dataset, model, trainer, checkpoint, and sampler modules. Data flows from raw candidates to split processed JSONL, then to weighted supervised NNX training, then to verifier-gated sampling that reconstructs candidates from generated target definitions and returns the lowest-log-flops valid result.

**Tech Stack:** Python 3.11, PyO3 `TensorComputation` bindings, JAX, Flax NNX, Optax, Orbax checkpointing, NumPy, pytest, uv, Rust cargo tests.

---

## Baseline And Constraints

Work from the repository root:

```bash
cd /Users/longli/rcode/gristmill-symbolics
git branch --show-current
git log --oneline -7
git status --short
```

Expected branch and spec commits:

```text
main
999afdf docs: add direct optimizer sampler spec
13519dd docs: add direct optimizer raw split spec
1ec973d docs: add direct optimizer trainer spec
3d14275 docs: add direct optimizer model spec
65919a3 docs: add direct optimizer dataset spec
4b7729b docs: add direct optimizer converter spec
fa9d53d docs: add direct optimizer overview spec
```

Current unrelated untracked files may exist under `.superpowers/` and older `docs/superpowers/plans/*.md`. Do not stage or edit them while executing this plan.

All Python `uv run ...` commands in this plan must be run with working directory
`python/`. Test paths in command blocks are relative to `python/`. Run Rust
commands from the repository root.

Read these accepted specs before implementation:

```bash
sed -n '1,260p' docs/superpowers/specs/direct-optimizer/overview-design.md
sed -n '1,320p' docs/superpowers/specs/direct-optimizer/converter-design.md
sed -n '1,360p' docs/superpowers/specs/direct-optimizer/dataset-design.md
sed -n '1,360p' docs/superpowers/specs/direct-optimizer/model-design.md
sed -n '1,320p' docs/superpowers/specs/direct-optimizer/trainer-design.md
sed -n '1,320p' docs/superpowers/specs/direct-optimizer/sampler-design.md
```

Critical accepted decisions:

- Package path is `python/gristmill_symbolics/direct_optimizer/`.
- This is a self-contained path. Do not integrate with `gristmill_symbolics.cli.checkpoint`, the action-selector model, or the REINFORCE trainer.
- Use a direct optimizer DSL, not `gristmill_symbolics.model.tokenizer`.
- Model generates target definitions only; reconstruction copies ranges and input tensors from `x`.
- Ordered outputs are preserved and used for keys and verification.
- Structured tokens distinguish keyword, scalar type, and scalar value.
- Model uses Flax NNX, static padded shapes, and batched JAX execution.
- Trainer consumes already processed train/valid/test JSONL and never splits internally.
- Dataset flow is raw generation, raw split, then independent processing per split.
- Sampler is verifier-gated and returns the lowest-log-flops valid candidate.

## File Map

Create:

- `python/gristmill_symbolics/direct_optimizer/__init__.py` - package marker and final public exports.
- `python/gristmill_symbolics/direct_optimizer/tokens.py` - structured token constants, logical-token conversion, padding, scalar bounds checks, and token batch helpers.
- `python/gristmill_symbolics/direct_optimizer/converter.py` - direct optimizer DSL printer, parser, target reconstruction, and `TensorComputation` validation boundary.
- `python/gristmill_symbolics/direct_optimizer/dataset.py` - raw generation, raw splitting, processed-row building, JSONL IO, and dataset CLI.
- `python/gristmill_symbolics/direct_optimizer/model.py` - Flax NNX encoder-decoder Transformer, teacher-forcing helpers, log-probability helpers, and batched autoregressive sampling.
- `python/gristmill_symbolics/direct_optimizer/trainer.py` - processed JSONL collation, weighted objective, NNX train/eval steps, epoch loop, and trainer metrics.
- `python/gristmill_symbolics/direct_optimizer/checkpoint.py` - direct optimizer checkpoint schema, Orbax-backed model/optimizer state save/load, and static model-kwargs validation.
- `python/gristmill_symbolics/direct_optimizer/train.py` - command-line training entry point.
- `python/gristmill_symbolics/direct_optimizer/sample.py` - programmatic sampler API and command-line inference entry point.
- `python/tests/direct_optimizer/__init__.py` - test package marker.
- `python/tests/direct_optimizer/fixtures.py` - compact symbolic computation fixtures shared by direct optimizer tests.
- `python/tests/direct_optimizer/test_layout.py` - package import and forbidden dependency tests.
- `python/tests/direct_optimizer/test_converter.py` - DSL, reconstruction, and structured-token tests.
- `python/tests/direct_optimizer/test_dataset.py` - raw generation, split, build, JSONL, and dataset CLI tests.
- `python/tests/direct_optimizer/test_model.py` - NNX model shape, scoring, masking, and sampling tests.
- `python/tests/direct_optimizer/test_trainer.py` - collation, objective, train/eval, checkpoint integration, and train CLI tests.
- `python/tests/direct_optimizer/test_checkpoint.py` - direct checkpoint schema and restore tests.
- `python/tests/direct_optimizer/test_sampler.py` - sampler filtering, metrics, best-candidate, checkpoint loading, and sample CLI tests.

Modify:

- `python/pyproject.toml` - add explicit `orbax-checkpoint>=0.11` dependency because checkpointing is part of the public direct optimizer path.
- `python/uv.lock` - refresh after dependency change.

Do not modify:

- `python/gristmill_symbolics/cli/checkpoint.py`
- `python/gristmill_symbolics/cli/train.py`
- `python/gristmill_symbolics/cli/train_state.py`
- `python/gristmill_symbolics/model/tokenizer/`
- `python/gristmill_symbolics/model/transformer_action_selector/`
- `python/gristmill_symbolics/trainer/reinforce/`

## Dependency Rules To Enforce

Allowed:

```text
direct_optimizer.converter -> json, gristmill_symbolics.TensorComputation
direct_optimizer.tokens -> dataclasses, typing, numpy/jax arrays
direct_optimizer.dataset -> converter, TensorComputation, equivalent_computations, rewrite functions, json, math, pathlib, numpy
direct_optimizer.model -> tokens, jax, flax.nnx
direct_optimizer.trainer -> dataset JSONL readers, tokens, model, checkpoint, jax, flax.nnx, optax, numpy
direct_optimizer.checkpoint -> model, flax.nnx, optax, orbax.checkpoint, json, pathlib
direct_optimizer.sample -> converter, tokens, model, checkpoint, TensorComputation, equivalent_computations, json, jax
direct_optimizer.train -> trainer CLI only
```

Forbidden:

```text
direct_optimizer.* -> gristmill_symbolics.model.tokenizer
direct_optimizer.* -> gristmill_symbolics.model.transformer_action_selector
direct_optimizer.* -> gristmill_symbolics.trainer.reinforce
direct_optimizer.* -> gristmill_symbolics.cli.checkpoint
direct_optimizer.trainer -> gristmill_symbolics.TensorComputation
direct_optimizer.trainer -> gristmill_symbolics.equivalent_computations
direct_optimizer.sample -> direct_optimizer.trainer
```

## Task 1: Package Skeleton, Dependency, And Boundary Guard

**Files:**
- Create: `python/gristmill_symbolics/direct_optimizer/__init__.py`
- Create: `python/tests/direct_optimizer/__init__.py`
- Create: `python/tests/direct_optimizer/test_layout.py`
- Modify: `python/pyproject.toml`
- Modify: `python/uv.lock`

- [ ] **Step 1: Write the failing package and dependency tests**

Create `python/tests/direct_optimizer/test_layout.py`:

```python
import ast
import importlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "gristmill_symbolics" / "direct_optimizer"

FORBIDDEN_IMPORTS = {
    "gristmill_symbolics.model.tokenizer",
    "gristmill_symbolics.model.transformer_action_selector",
    "gristmill_symbolics.trainer.reinforce",
    "gristmill_symbolics.cli.checkpoint",
}


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_direct_optimizer_package_imports():
    package = importlib.import_module("gristmill_symbolics.direct_optimizer")

    assert package.__all__ == ()


def test_orbax_checkpoint_is_declared_as_direct_dependency():
    pyproject = (ROOT / "pyproject.toml").read_text()

    assert '"orbax-checkpoint>=0.11"' in pyproject


def test_direct_optimizer_modules_do_not_import_forbidden_training_paths():
    assert PACKAGE.exists()
    for path in PACKAGE.glob("*.py"):
        imported = _imported_modules(path)
        forbidden = {
            module
            for module in imported
            if any(
                module == blocked or module.startswith(blocked + ".")
                for blocked in FORBIDDEN_IMPORTS
            )
        }
        assert forbidden == set(), f"{path} imports {sorted(forbidden)}"
```

- [ ] **Step 2: Run the focused test and confirm it fails**

Run:

```bash
uv run pytest tests/direct_optimizer/test_layout.py -q
```

Expected: FAIL because `gristmill_symbolics.direct_optimizer` and the explicit Orbax dependency do not exist yet.

- [ ] **Step 3: Add the minimal package skeleton and explicit dependency**

Create `python/gristmill_symbolics/direct_optimizer/__init__.py`:

```python
"""Self-contained direct optimizer package."""

__all__ = ()
```

Create `python/tests/direct_optimizer/__init__.py`:

```python
"""Tests for the direct optimizer package."""
```

Modify `python/pyproject.toml` dependencies to include:

```toml
    "orbax-checkpoint>=0.11",
```

Refresh the lockfile:

```bash
cd python
uv lock
```

- [ ] **Step 4: Run the focused test and confirm it passes**

Run:

```bash
uv run pytest tests/direct_optimizer/test_layout.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/pyproject.toml python/uv.lock python/gristmill_symbolics/direct_optimizer/__init__.py python/tests/direct_optimizer/__init__.py python/tests/direct_optimizer/test_layout.py
git commit -m "feat: add direct optimizer package boundary"
```

## Task 2: Converter Source And Target DSL Round Trip

**Files:**
- Create: `python/gristmill_symbolics/direct_optimizer/converter.py`
- Create: `python/tests/direct_optimizer/fixtures.py`
- Create: `python/tests/direct_optimizer/test_converter.py`

- [ ] **Step 1: Write failing DSL round-trip tests**

Create `python/tests/direct_optimizer/fixtures.py`:

```python
import json

from gristmill_symbolics import TensorComputation


def source_comp_json() -> str:
    return json.dumps(
        {
            "ranges": [{"id": 0, "size": 8}],
            "tensors": [
                {"id": 0, "symmetry": [{"perm": [0], "action": "Identity"}]},
                {"id": 1, "symmetry": []},
            ],
            "definitions": [
                {
                    "base": 1,
                    "ext_indices": [{"id": 0, "range": 0}],
                    "terms": [
                        {
                            "coeff": [1, 1],
                            "sum_indices": [],
                            "factors": [{"tensor": 0, "indices": [0]}],
                        }
                    ],
                }
            ],
        }
    )


def source_comp() -> TensorComputation:
    return TensorComputation.from_json_string(source_comp_json())


def two_candidate_jsons() -> tuple[str, str]:
    low = json.loads(source_comp_json())
    high = json.loads(source_comp_json())
    high["definitions"][0]["terms"][0]["coeff"] = [2, 1]
    return json.dumps(low), json.dumps(high)
```

Create the first tests in `python/tests/direct_optimizer/test_converter.py`:

```python
import pytest

from gristmill_symbolics.direct_optimizer.converter import (
    computation_to_source_text,
    computation_to_target_text,
    source_text_to_snapshot,
    target_text_to_definitions,
)
from tests.direct_optimizer.fixtures import source_comp


def test_source_text_round_trips_full_snapshot():
    comp = source_comp()

    text = computation_to_source_text(comp)

    assert source_text_to_snapshot(text) == comp.snapshot()
    assert text.splitlines() == [
        "range id range_id:0 size dim_size:8",
        "tensor id tensor_id:0",
        "symmetry action sym_action:Identity",
        "perm axis:0",
        "endsymmetry",
        "endtensor",
        "tensor id tensor_id:1",
        "endtensor",
        "def base tensor_id:1",
        "ext id index_id:0 range range_id:0",
        "term",
        "coeff numer coeff_num:1 denom coeff_den:1",
        "factor tensor tensor_id:0",
        "index index_id:0",
        "endfactor",
        "endterm",
        "enddef",
    ]


def test_target_text_round_trips_definitions_only():
    comp = source_comp()

    text = computation_to_target_text(comp)

    assert target_text_to_definitions(text) == comp.snapshot()["definitions"]
    assert text.splitlines()[0] == "def base tensor_id:1"
    assert all(not line.startswith("range ") for line in text.splitlines())
    assert all(not line.startswith("tensor ") for line in text.splitlines())


@pytest.mark.parametrize(
    "bad_text, message",
    [
        ("foo id range_id:0", "unknown keyword"),
        ("def base range_id:1\nenddef", "expected tensor_id"),
        ("def base tensor_id:1\nterm\nenddef", "unclosed term"),
        ("endterm", "unexpected endterm"),
    ],
)
def test_parser_rejects_malformed_dsl(bad_text, message):
    with pytest.raises(ValueError, match=message):
        target_text_to_definitions(bad_text)
```

- [ ] **Step 2: Run the converter test and confirm it fails**

Run:

```bash
uv run pytest tests/direct_optimizer/test_converter.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'gristmill_symbolics.direct_optimizer.converter'`.

- [ ] **Step 3: Implement deterministic DSL printing and strict parsing**

Create `python/gristmill_symbolics/direct_optimizer/converter.py` with these public functions and internal parser names:

```python
from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from gristmill_symbolics import TensorComputation

CONVERTER_SCHEMA_VERSION = 1

SOURCE_RECORD_STARTS = {"range", "tensor", "symmetry", "perm", "endsymmetry", "endtensor", "def", "ext", "term", "coeff", "sum", "factor", "index", "endfactor", "endterm", "enddef"}
TARGET_RECORD_STARTS = {"def", "ext", "term", "coeff", "sum", "factor", "index", "endfactor", "endterm", "enddef"}


def computation_to_source_text(comp: TensorComputation) -> str:
    snapshot = comp.snapshot()
    return _source_snapshot_to_text(snapshot)


def computation_to_target_text(comp: TensorComputation) -> str:
    return _definitions_to_text(comp.snapshot()["definitions"])


def source_text_to_snapshot(text: str) -> dict[str, Any]:
    parser = _Parser(text, allow_source_records=True)
    return parser.parse_source()


def target_text_to_definitions(text: str) -> list[dict[str, Any]]:
    parser = _Parser(text, allow_source_records=False)
    return parser.parse_definitions_until_end()
```

Implement `_source_snapshot_to_text`, `_definitions_to_text`, `_range_to_lines`, `_tensor_to_lines`, `_definition_to_lines`, `_term_to_lines`, `_factor_to_lines`, and `_Parser` so that:

- `coeff` accepts Rust snapshot forms `{"numer": 1, "denom": 1}` and `[1, 1]`, but prints and parses to the snapshot form returned by `TensorComputation.snapshot()`.
- Every record has fixed field order exactly as asserted in the tests.
- Unknown keywords, malformed typed scalars, wrong scalar types, missing fields, extra fields, and invalid block nesting raise `ValueError` with the tested context.
- Parser output uses dictionaries with keys `ranges`, `tensors`, and `definitions` for source text, and a list of definition dictionaries for target text.

- [ ] **Step 4: Run the focused converter test and confirm it passes**

Run:

```bash
uv run pytest tests/direct_optimizer/test_converter.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/gristmill_symbolics/direct_optimizer/converter.py python/tests/direct_optimizer/fixtures.py python/tests/direct_optimizer/test_converter.py
git commit -m "feat: add direct optimizer DSL converter"
```

## Task 3: Converter Reconstruction And Structured Tokens

**Files:**
- Create: `python/gristmill_symbolics/direct_optimizer/tokens.py`
- Modify: `python/gristmill_symbolics/direct_optimizer/converter.py`
- Modify: `python/tests/direct_optimizer/test_converter.py`

- [ ] **Step 1: Add failing reconstruction and token tests**

Append to `python/tests/direct_optimizer/test_converter.py`:

```python
import numpy as np

from gristmill_symbolics.direct_optimizer.converter import (
    target_text_to_computation,
)
from gristmill_symbolics.direct_optimizer.tokens import (
    KIND,
    SCALAR_TYPE,
    decode_token_row_to_text,
    encode_text,
    pad_tokens,
)


def test_target_reconstruction_copies_input_envelope_and_registers_new_bases():
    x = source_comp()
    target_text = "\n".join(
        [
            "def base tensor_id:9",
            "ext id index_id:0 range range_id:0",
            "term",
            "coeff numer coeff_num:1 denom coeff_den:1",
            "factor tensor tensor_id:0",
            "index index_id:0",
            "endfactor",
            "endterm",
            "enddef",
        ]
    )

    candidate = target_text_to_computation(x, target_text)
    snapshot = candidate.snapshot()

    assert snapshot["ranges"] == x.snapshot()["ranges"]
    assert {"id": 0, "symmetry": [{"perm": [0], "action": "Identity"}]} in snapshot["tensors"]
    assert {"id": 9, "symmetry": []} in snapshot["tensors"]
    assert snapshot["definitions"][0]["base"] == 9


def test_target_reconstruction_rejects_unknown_factor_tensor():
    x = source_comp()
    target_text = "\n".join(
        [
            "def base tensor_id:9",
            "ext id index_id:0 range range_id:0",
            "term",
            "coeff numer coeff_num:1 denom coeff_den:1",
            "factor tensor tensor_id:99",
            "index index_id:0",
            "endfactor",
            "endterm",
            "enddef",
        ]
    )

    with pytest.raises(ValueError, match="unknown tensor_id:99"):
        target_text_to_computation(x, target_text)


def test_structured_token_round_trip_preserves_valid_dsl_text():
    text = computation_to_target_text(source_comp())

    tokens = encode_text(text)
    padded = pad_tokens(tokens, length=len(tokens["kind"]) + 3)
    decoded = decode_token_row_to_text({key: value[: len(tokens["kind"])] for key, value in padded.items()})

    assert decoded == text
    assert KIND["KEYWORD"] in set(np.asarray(tokens["kind"]).tolist())
    assert SCALAR_TYPE["tensor_id"] in set(np.asarray(tokens["scalar_type"]).tolist())
    assert SCALAR_TYPE["index_id"] in set(np.asarray(tokens["scalar_type"]).tolist())
```

- [ ] **Step 2: Run the focused test and confirm it fails**

Run:

```bash
uv run pytest tests/direct_optimizer/test_converter.py -q
```

Expected: FAIL because `tokens.py` and `target_text_to_computation` are missing.

- [ ] **Step 3: Implement structured token helpers**

Create `python/gristmill_symbolics/direct_optimizer/tokens.py` with this public surface:

```python
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

TOKEN_FIELDS = ("kind", "keyword", "scalar_type", "scalar_value", "mask")
SENTINEL = -1

KIND = {"PAD": 0, "BOS": 1, "EOS": 2, "KEYWORD": 3, "SCALAR": 4}
KEYWORD = {
    "range": 0,
    "id": 1,
    "size": 2,
    "tensor": 3,
    "symmetry": 4,
    "action": 5,
    "perm": 6,
    "endtensor": 7,
    "endsymmetry": 8,
    "def": 9,
    "base": 10,
    "ext": 11,
    "term": 12,
    "coeff": 13,
    "numer": 14,
    "denom": 15,
    "sum": 16,
    "factor": 17,
    "index": 18,
    "endfactor": 19,
    "endterm": 20,
    "enddef": 21,
}
SCALAR_TYPE = {
    "range_id": 0,
    "tensor_id": 1,
    "index_id": 2,
    "dim_size": 3,
    "coeff_num": 4,
    "coeff_den": 5,
    "sym_action": 6,
    "axis": 7,
}
SYM_ACTION_VALUE = {"Identity": 0, "Negate": 1}
VALUE_SYM_ACTION = {value: key for key, value in SYM_ACTION_VALUE.items()}


@dataclass(frozen=True)
class LogicalToken:
    kind: str
    keyword: str | None = None
    scalar_type: str | None = None
    scalar_value: int | str | None = None
```

Implement these functions with the exact names and signatures:

- `text_to_logical_tokens(text: str) -> list[LogicalToken]`
- `logical_tokens_to_text(tokens: list[LogicalToken]) -> str`
- `encode_text(text: str) -> dict[str, np.ndarray]`
- `decode_token_row_to_text(tokens: Mapping[str, Any]) -> str`
- `pad_tokens(tokens: Mapping[str, Any], *, length: int) -> dict[str, np.ndarray]`
- `repeat_token_row(tokens: Mapping[str, Any], *, batch_size: int) -> dict[str, np.ndarray]`
- `validate_scalar_bounds(tokens: Mapping[str, Any], *, scalar_value_min: int, scalar_value_max: int) -> None`

Rules:

- A scalar text token must have the form `<scalar_type>:<integer_or_sym_action>`.
- `sym_action:Identity` and `sym_action:Negate` encode to integer scalar values `0` and `1`.
- `logical_tokens_to_text` groups grammar-valid token sequences into canonical DSL record lines and raises `ValueError` for incomplete records.
- `pad_tokens` raises `ValueError` when `len(tokens["kind"]) > length`.
- `mask` is `True` for real tokens and `False` for padding.

- [ ] **Step 4: Implement target reconstruction in the converter**

Add `target_text_to_computation` to `python/gristmill_symbolics/direct_optimizer/converter.py`:

```python
def target_text_to_computation(
    x: TensorComputation,
    target_text: str,
) -> TensorComputation:
    definitions = target_text_to_definitions(target_text)
    x_snapshot = x.snapshot()
    tensors = list(x_snapshot["tensors"])
    known_tensors = {int(tensor["id"]) for tensor in tensors}
    generated_bases = {int(definition["base"]) for definition in definitions}
    for base in sorted(generated_bases - known_tensors):
        tensors.append({"id": base, "symmetry": []})
        known_tensors.add(base)
    for definition in definitions:
        for term in definition["terms"]:
            for factor in term["factors"]:
                tensor_id = int(factor["tensor"])
                if tensor_id not in known_tensors:
                    raise ValueError(f"factor references unknown tensor_id:{tensor_id}")
    snapshot = {
        "ranges": list(x_snapshot["ranges"]),
        "tensors": tensors,
        "definitions": definitions,
    }
    try:
        return TensorComputation.from_json_string(json.dumps(snapshot))
    except Exception as exc:
        raise ValueError("target reconstruction failed TensorComputation validation") from exc
```

- [ ] **Step 5: Run the focused test and confirm it passes**

Run:

```bash
uv run pytest tests/direct_optimizer/test_converter.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add python/gristmill_symbolics/direct_optimizer/converter.py python/gristmill_symbolics/direct_optimizer/tokens.py python/tests/direct_optimizer/test_converter.py
git commit -m "feat: add direct optimizer reconstruction and tokens"
```

## Task 4: Processed Dataset Builder, Weights, Split, And JSONL

**Files:**
- Create: `python/gristmill_symbolics/direct_optimizer/dataset.py`
- Modify: `python/tests/direct_optimizer/test_dataset.py`

- [ ] **Step 1: Write failing processed-dataset tests**

Create `python/tests/direct_optimizer/test_dataset.py`:

```python
import json
import math

import pytest

from gristmill_symbolics.direct_optimizer.dataset import (
    BuildConfig,
    SplitConfig,
    build_processed_dataset,
    read_processed_jsonl,
    read_raw_candidates_jsonl,
    split_raw_candidates,
    write_processed_jsonl,
    write_raw_candidates_jsonl,
)
from tests.direct_optimizer.fixtures import source_comp_json, two_candidate_jsons


def _raw_records():
    low, high = two_candidate_jsons()
    return [
        {
            "input_computation": source_comp_json(),
            "candidate_computation": low,
            "outputs": [1, 3],
            "initial_log_flops": 4.0,
            "candidate_log_flops": 1.0,
        },
        {
            "input_computation": json.loads(source_comp_json()),
            "candidate_computation": json.loads(high),
            "outputs": [1, 3],
            "initial_log_flops": 4.0,
            "candidate_log_flops": 3.0,
        },
    ]


def test_build_processed_dataset_preserves_ordered_outputs_and_cost_weights():
    rows = build_processed_dataset(_raw_records(), BuildConfig(beta=1.0, verify=False))

    assert [row["outputs"] for row in rows] == [[1, 3], [1, 3]]
    assert {row["input_key"] for row in rows} == {rows[0]["input_key"]}
    assert rows[0]["candidate_key"] != rows[1]["candidate_key"]
    assert math.isclose(sum(row["weight"] for row in rows), 1.0)
    cheaper = min(rows, key=lambda row: row["candidate_log_flops"])
    expensive = max(rows, key=lambda row: row["candidate_log_flops"])
    assert cheaper["weight"] > expensive["weight"]


def test_build_processed_dataset_treats_different_output_orders_as_different_groups():
    records = _raw_records()
    records.append({**records[0], "outputs": [3, 1]})

    rows = build_processed_dataset(records, BuildConfig(beta=1.0, verify=False))

    assert len({row["input_key"] for row in rows}) == 2


@pytest.mark.parametrize("outputs", [[1, 1], [True], [], ["1"]])
def test_build_processed_dataset_rejects_invalid_outputs(outputs):
    record = {**_raw_records()[0], "outputs": outputs}

    assert build_processed_dataset([record], BuildConfig(beta=1.0, verify=False)) == []


def test_build_processed_dataset_deduplicates_candidate_per_input_group():
    records = [_raw_records()[0], {**_raw_records()[0], "candidate_log_flops": 0.5}]

    rows = build_processed_dataset(records, BuildConfig(beta=1.0, verify=False))

    assert len(rows) == 1
    assert rows[0]["candidate_log_flops"] == 0.5
    assert rows[0]["weight"] == 1.0


def test_split_raw_candidates_is_deterministic_and_preserves_records():
    records = [{"id": index} for index in range(10)]

    first = split_raw_candidates(records, SplitConfig(0.6, 0.2, 0.2, seed=7))
    second = split_raw_candidates(records, SplitConfig(0.6, 0.2, 0.2, seed=7))

    assert first == second
    assert sorted(row["id"] for split in first for row in split) == list(range(10))


def test_jsonl_helpers_round_trip_raw_and_processed_rows(tmp_path):
    raw_path = tmp_path / "raw.jsonl"
    processed_path = tmp_path / "processed.jsonl"
    raw = _raw_records()
    rows = build_processed_dataset(raw, BuildConfig(beta=1.0, verify=False))

    write_raw_candidates_jsonl(raw, raw_path)
    write_processed_jsonl(rows, processed_path)

    assert read_raw_candidates_jsonl(raw_path) == raw
    assert read_processed_jsonl(processed_path) == rows
```

- [ ] **Step 2: Run the dataset test and confirm it fails**

Run:

```bash
uv run pytest tests/direct_optimizer/test_dataset.py -q
```

Expected: FAIL because `dataset.py` is missing.

- [ ] **Step 3: Implement dataset configs, keys, processed rows, split, and JSONL**

Create `python/gristmill_symbolics/direct_optimizer/dataset.py` with:

```python
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np

from gristmill_symbolics import TensorComputation, equivalent_computations

from .converter import computation_to_source_text, computation_to_target_text


@dataclass(frozen=True)
class GenerationConfig:
    seed: int
    trajectories_per_input: int
    max_steps: int
    random_subsets: bool = False
    collect_intermediates: bool = True


@dataclass(frozen=True)
class BuildConfig:
    beta: float = 1.0
    verify: bool = False


@dataclass(frozen=True)
class SplitConfig:
    train_fraction: float = 0.8
    valid_fraction: float = 0.1
    test_fraction: float = 0.1
    seed: int = 0
```

Implement these functions with the exact names and signatures:

- `write_raw_candidates_jsonl(records: Sequence[dict[str, Any]], path: Path) -> None`
- `read_raw_candidates_jsonl(path: Path) -> list[dict[str, Any]]`
- `write_processed_jsonl(rows: Sequence[dict[str, Any]], path: Path) -> None`
- `read_processed_jsonl(path: Path) -> list[dict[str, Any]]`
- `split_raw_candidates(raw_records: Sequence[dict[str, Any]], config: SplitConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]`
- `build_processed_dataset(raw_records: Sequence[dict[str, Any]], config: BuildConfig) -> list[dict[str, Any]]`

Required internal helper behavior:

- `_stable_hash(payload)` returns `"sha256:" + hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()`.
- `_parse_comp(value)` accepts a JSON string or JSON object, JSON-dumps objects, and returns `TensorComputation.from_json_string(text)`.
- `_validate_outputs(value)` returns a new list, rejects empty lists, bools, non-ints, and duplicates by raising `ValueError`.
- `_finite_cost(value)` converts to float and raises `ValueError` for non-finite values.
- `BuildConfig.beta` must be finite and positive, otherwise `build_processed_dataset` raises `ValueError`.
- Malformed raw records are skipped by `build_processed_dataset`.
- With `verify=True`, rows where `equivalent_computations(input_comp, candidate_comp, outputs)` is false or raises are skipped.
- `input_key` uses `{"source_text": source_text, "outputs": outputs}`.
- `candidate_key` uses `{"target_text": target_text}`.
- Duplicate candidates per input group keep the row with the lowest `candidate_log_flops`.
- Weights are computed per input group with `exp(-beta * (cost - min_cost))` and normalized.
- `split_raw_candidates` validates finite non-negative fractions that sum to `1.0`, shuffles with `random.Random(config.seed)`, preserves records unchanged, rejects empty train, and rejects empty valid/test when their fractions are greater than zero.

- [ ] **Step 4: Run the focused dataset test and confirm it passes**

Run:

```bash
uv run pytest tests/direct_optimizer/test_dataset.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/gristmill_symbolics/direct_optimizer/dataset.py python/tests/direct_optimizer/test_dataset.py
git commit -m "feat: build direct optimizer processed datasets"
```

## Task 5: Raw Candidate Generation And Dataset CLI

**Files:**
- Modify: `python/gristmill_symbolics/direct_optimizer/dataset.py`
- Modify: `python/tests/direct_optimizer/test_dataset.py`

- [ ] **Step 1: Add failing generator and CLI tests**

Append to `python/tests/direct_optimizer/test_dataset.py`:

```python
from gristmill_symbolics import TensorComputation
from gristmill_symbolics.direct_optimizer.dataset import (
    GenerationConfig,
    generate_raw_candidates,
    main as dataset_main,
)
from tests.test_bindings import actionable_json


def test_generate_raw_candidates_is_deterministic_and_skips_initial_state():
    comp = TensorComputation.from_json_string(actionable_json())
    config = GenerationConfig(seed=3, trajectories_per_input=2, max_steps=1)

    first = generate_raw_candidates([(comp, [3])], config)
    second = generate_raw_candidates([(TensorComputation.from_json_string(actionable_json()), [3])], config)

    assert first == second
    assert first
    assert all(row["input_computation"] != row["candidate_computation"] for row in first)
    assert set(first[0]) >= {
        "input_computation",
        "candidate_computation",
        "outputs",
        "initial_log_flops",
        "candidate_log_flops",
    }


def test_dataset_build_splits_cli_processes_each_split_independently(tmp_path):
    raw_path = tmp_path / "raw.jsonl"
    train_path = tmp_path / "train.jsonl"
    valid_path = tmp_path / "valid.jsonl"
    test_path = tmp_path / "test.jsonl"
    write_raw_candidates_jsonl(_raw_records() * 4, raw_path)

    exit_code = dataset_main(
        [
            "build-splits",
            "--raw-input",
            str(raw_path),
            "--train-output",
            str(train_path),
            "--valid-output",
            str(valid_path),
            "--test-output",
            str(test_path),
            "--train-fraction",
            "0.5",
            "--valid-fraction",
            "0.25",
            "--test-fraction",
            "0.25",
            "--split-seed",
            "9",
            "--beta",
            "1.0",
        ]
    )

    assert exit_code == 0
    for path in (train_path, valid_path, test_path):
        rows = read_processed_jsonl(path)
        assert rows
        groups = {}
        for row in rows:
            groups.setdefault(row["input_key"], 0.0)
            groups[row["input_key"]] += row["weight"]
        assert all(math.isclose(total, 1.0) for total in groups.values())
```

- [ ] **Step 2: Run the focused dataset test and confirm it fails**

Run:

```bash
uv run pytest tests/direct_optimizer/test_dataset.py -q
```

Expected: FAIL because `generate_raw_candidates` and `main` are missing.

- [ ] **Step 3: Implement raw generation**

Add imports in `dataset.py`:

```python
from gristmill_symbolics import action_space_for_def, apply_decision, validate_decision
```

Implement `generate_raw_candidates(inputs: Sequence[tuple[TensorComputation, Sequence[int]]], config: GenerationConfig) -> list[dict[str, Any]]`.

Generation rules:

- Validate `trajectories_per_input > 0` and `max_steps > 0`.
- Use `random.Random(config.seed)`.
- For each input, clone the computation for each trajectory.
- At each step, collect `(def_index, space)` for definitions where `action_space_for_def(comp, def_index)` returns a space.
- Stop the trajectory when no spaces exist or a rewrite operation raises.
- Pick one action space and one candidate template with the RNG.
- Build a decision with full masks when `random_subsets=False`; when `random_subsets=True`, pick non-empty boolean masks for the selected template sides.
- Call `validate_decision(space, decision)` and `apply_decision(comp, space, decision)`.
- Emit each post-rewrite state when `collect_intermediates=True`; otherwise emit only the last changed state.
- Store `input_computation` and `candidate_computation` as JSON strings from `to_json_string()`.
- Store `outputs` as the ordered list passed by the caller.
- Store `initial_log_flops` and `candidate_log_flops`.

- [ ] **Step 4: Implement dataset CLI subcommands**

Add `main(argv=None) -> int` to `dataset.py` with subcommands:

```text
generate
build
build-splits
```

CLI parsing details:

- `generate --input seed.json --outputs 1,3 --raw-output raw.jsonl --seed 0 --trajectories 64 --max-steps 8 --collect-intermediates`
- `build --raw-input raw.jsonl --output processed.jsonl --beta 1.0 --verify`
- `build-splits --raw-input raw.jsonl --train-output train.jsonl --valid-output valid.jsonl --test-output test.jsonl --train-fraction 0.8 --valid-fraction 0.1 --test-fraction 0.1 --split-seed 0 --beta 1.0 --verify`

Use this helper for ordered output parsing:

```python
def _parse_outputs(value: str) -> list[int]:
    if "," in value:
        parts = value.split(",")
    else:
        parts = value.split()
    outputs = [int(part) for part in parts if part != ""]
    return _validate_outputs(outputs)
```

Add the entry point guard:

```python
if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run focused tests and module CLI smoke**

Run:

```bash
uv run pytest tests/direct_optimizer/test_dataset.py -q
uv run python -m gristmill_symbolics.direct_optimizer.dataset --help
```

Expected: tests PASS and CLI help exits with status `0`.

- [ ] **Step 6: Commit**

```bash
git add python/gristmill_symbolics/direct_optimizer/dataset.py python/tests/direct_optimizer/test_dataset.py
git commit -m "feat: add direct optimizer dataset generation CLI"
```

## Task 6: Model Scoring Helpers And Structured Token Collation

**Files:**
- Create: `python/gristmill_symbolics/direct_optimizer/model.py`
- Create: `python/tests/direct_optimizer/test_model.py`

- [ ] **Step 1: Write failing model helper tests**

Create `python/tests/direct_optimizer/test_model.py`:

```python
import jax
import jax.numpy as jnp
import pytest

from gristmill_symbolics.direct_optimizer.converter import computation_to_target_text
from gristmill_symbolics.direct_optimizer.model import (
    make_decoder_inputs,
    sequence_log_prob,
    token_log_probs,
)
from gristmill_symbolics.direct_optimizer.tokens import KIND, KEYWORD, SCALAR_TYPE, encode_text, pad_tokens
from tests.direct_optimizer.fixtures import source_comp


def _target_tokens(length=32):
    return pad_tokens(encode_text(computation_to_target_text(source_comp())), length=length)


def test_make_decoder_inputs_adds_bos_prefix_and_eos_label():
    target = _target_tokens(length=32)

    decoder_input, labels, mask = make_decoder_inputs(target)

    assert int(decoder_input["kind"][0]) == KIND["BOS"]
    assert int(labels["kind"][sum(target["mask"])]) == KIND["EOS"]
    assert bool(mask[sum(target["mask"])])
    assert not bool(mask[sum(target["mask"]) + 1])


def test_token_log_probs_scores_relevant_heads_only():
    target = {
        "kind": jnp.asarray([[KIND["KEYWORD"], KIND["SCALAR"], KIND["PAD"]]]),
        "keyword": jnp.asarray([[KEYWORD["def"], -1, -1]]),
        "scalar_type": jnp.asarray([[-1, SCALAR_TYPE["tensor_id"], -1]]),
        "scalar_value": jnp.asarray([[-1, 3, -1]]),
        "mask": jnp.asarray([[True, True, False]]),
    }
    logits = {
        "kind": jnp.zeros((1, 3, len(KIND))),
        "keyword": jnp.zeros((1, 3, len(KEYWORD))),
        "scalar_type": jnp.zeros((1, 3, len(SCALAR_TYPE))),
        "scalar_value": jnp.zeros((1, 3, 11)),
        "scalar_value_min": -5,
    }

    values = token_log_probs(logits, target)

    assert values.shape == (1, 3)
    assert values[0, 0] == pytest.approx(-jnp.log(len(KIND)) - jnp.log(len(KEYWORD)))
    assert values[0, 1] == pytest.approx(-jnp.log(len(KIND)) - jnp.log(len(SCALAR_TYPE)) - jnp.log(11))
    assert values[0, 2] == pytest.approx(-jnp.log(len(KIND)))


def test_sequence_log_prob_ignores_padding_mask():
    target = _target_tokens(length=32)
    decoder_input, labels, mask = make_decoder_inputs(target)
    batch_labels = {key: jnp.asarray(value[None, :]) for key, value in labels.items()}
    logits = {
        "kind": jnp.zeros((1, 32, len(KIND))),
        "keyword": jnp.zeros((1, 32, len(KEYWORD))),
        "scalar_type": jnp.zeros((1, 32, len(SCALAR_TYPE))),
        "scalar_value": jnp.zeros((1, 32, 21)),
        "scalar_value_min": -10,
    }

    seq = sequence_log_prob(logits, batch_labels, jnp.asarray(mask[None, :]))

    assert seq.shape == (1,)
    assert jnp.isfinite(seq[0])
```

- [ ] **Step 2: Run model tests and confirm they fail**

Run:

```bash
uv run pytest tests/direct_optimizer/test_model.py -q
```

Expected: FAIL because `model.py` is missing.

- [ ] **Step 3: Implement helper functions**

Create `python/gristmill_symbolics/direct_optimizer/model.py` with:

```python
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from .tokens import KIND, KEYWORD, SCALAR_TYPE, SENTINEL, TOKEN_FIELDS, pad_tokens, validate_scalar_bounds
```

Implement these functions with the exact names and signatures:

- `make_decoder_inputs(target_tokens: Mapping[str, Any]) -> tuple[dict[str, jax.Array], dict[str, jax.Array], jax.Array]`
- `token_log_probs(logits: Mapping[str, Any], target_tokens: Mapping[str, Any]) -> jax.Array`
- `sequence_log_prob(logits: Mapping[str, Any], target_tokens: Mapping[str, Any], target_mask: jax.Array) -> jax.Array`

Rules:

- `make_decoder_inputs` treats `target_tokens["mask"]` as the real target length, builds `BOS + target[:-1]` for decoder input, builds `target + EOS` for labels, and returns a boolean mask including EOS.
- `token_log_probs` always scores `kind`; scores `keyword` only where label kind is `KEYWORD`; scores `scalar_type` and shifted `scalar_value` only where label kind is `SCALAR`.
- `scalar_value` scoring uses `scalar_value_index = scalar_value - logits["scalar_value_min"]` and raises `ValueError` before JIT helpers when labels fall outside bounds.
- `sequence_log_prob` multiplies token scores by `target_mask` and sums along sequence axis.

- [ ] **Step 4: Run helper tests and confirm they pass**

Run:

```bash
uv run pytest tests/direct_optimizer/test_model.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/gristmill_symbolics/direct_optimizer/model.py python/tests/direct_optimizer/test_model.py
git commit -m "feat: add direct optimizer model scoring helpers"
```

## Task 7: Flax NNX Transformer Forward Pass

**Files:**
- Modify: `python/gristmill_symbolics/direct_optimizer/model.py`
- Modify: `python/tests/direct_optimizer/test_model.py`

- [ ] **Step 1: Add failing NNX shape and masking tests**

Append to `python/tests/direct_optimizer/test_model.py`:

```python
from flax import nnx

from gristmill_symbolics.direct_optimizer.converter import computation_to_source_text
from gristmill_symbolics.direct_optimizer.model import DirectOptimizerTransformer


def _source_tokens(length=64):
    return pad_tokens(encode_text(computation_to_source_text(source_comp())), length=length)


def _batch(row, batch_size=2):
    return {key: jnp.asarray(value[None, :]).repeat(batch_size, axis=0) for key, value in row.items()}


def test_nnx_transformer_returns_static_logits():
    model = DirectOptimizerTransformer(
        source_len=64,
        target_len=32,
        scalar_value_min=-16,
        scalar_value_max=16,
        d_model=16,
        num_layers=1,
        num_heads=2,
        rngs=nnx.Rngs(0),
    )
    target = _target_tokens(length=32)
    decoder_input, _labels, _mask = make_decoder_inputs(target)

    logits = model(_batch(_source_tokens()), _batch(decoder_input))

    assert logits["kind"].shape == (2, 32, len(KIND))
    assert logits["keyword"].shape == (2, 32, len(KEYWORD))
    assert logits["scalar_type"].shape == (2, 32, len(SCALAR_TYPE))
    assert logits["scalar_value"].shape == (2, 32, 33)
    assert logits["scalar_value_min"] == -16


def test_structured_embedder_distinguishes_same_scalar_value_by_type():
    model = DirectOptimizerTransformer(
        source_len=4,
        target_len=4,
        scalar_value_min=-16,
        scalar_value_max=16,
        d_model=8,
        num_layers=1,
        num_heads=1,
        rngs=nnx.Rngs(1),
    )
    tensor_token = {
        "kind": jnp.asarray([[KIND["SCALAR"]]]),
        "keyword": jnp.asarray([[-1]]),
        "scalar_type": jnp.asarray([[SCALAR_TYPE["tensor_id"]]]),
        "scalar_value": jnp.asarray([[3]]),
        "mask": jnp.asarray([[True]]),
    }
    index_token = {**tensor_token, "scalar_type": jnp.asarray([[SCALAR_TYPE["index_id"]]])}

    assert not bool(jnp.allclose(model.embed_tokens(tensor_token, length=1), model.embed_tokens(index_token, length=1)))
```

- [ ] **Step 2: Run the focused model test and confirm it fails**

Run:

```bash
uv run pytest tests/direct_optimizer/test_model.py -q
```

Expected: FAIL because `DirectOptimizerTransformer` is missing.

- [ ] **Step 3: Implement the NNX encoder-decoder**

Add these NNX modules to `model.py`:

- `FeedForward(nnx.Module)` - two linear layers with GELU, dropout, and residual-compatible output dimension.
- `EncoderLayer(nnx.Module)` - source self-attention, feed-forward block, masks, residuals, and layer norms.
- `DecoderLayer(nnx.Module)` - causal target self-attention, source cross-attention, feed-forward block, masks, residuals, and layer norms.
- `DirectOptimizerTransformer(nnx.Module)` with methods:
  - `__init__(self, *, source_len: int, target_len: int, scalar_value_min: int, scalar_value_max: int, d_model: int = 128, num_layers: int = 2, num_heads: int = 4, dropout: float = 0.0, init_scale: float = 0.02, rngs: nnx.Rngs)`
  - `model_kwargs(self) -> dict[str, object]`
  - `embed_tokens(self, tokens, *, length: int) -> jax.Array`
  - `__call__(self, source_tokens, decoder_input_tokens, *, deterministic: bool = True) -> dict[str, jax.Array | int]`

Implementation rules:

- Validate positive `source_len`, `target_len`, `d_model`, `num_layers`, `num_heads`, and `scalar_value_min <= scalar_value_max`.
- Store constructor kwargs as attributes and expose `model_kwargs()` returning JSON-serializable static kwargs without `rngs`.
- Use separate embeddings for `kind`, `keyword`, and `scalar_type`.
- Project normalized scalar values with an `nnx.Linear(1, d_model)`.
- Add learned positional embeddings for source and target lengths.
- Mask padded token embeddings to zero with `tokens["mask"]`.
- Use `nnx.MultiHeadAttention`, `nnx.LayerNorm`, `nnx.Linear`, GELU, residual connections, and causal decoder self-attention.
- Return logits dictionary with heads `kind`, `keyword`, `scalar_type`, `scalar_value`, plus integer `scalar_value_min`.
- Keep source and target shapes static; raise `ValueError` if token lengths do not match constructor lengths outside JIT.

- [ ] **Step 4: Run the focused model test and confirm it passes**

Run:

```bash
uv run pytest tests/direct_optimizer/test_model.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/gristmill_symbolics/direct_optimizer/model.py python/tests/direct_optimizer/test_model.py
git commit -m "feat: add direct optimizer NNX transformer"
```

## Task 8: Batched Autoregressive Sampling

**Files:**
- Modify: `python/gristmill_symbolics/direct_optimizer/model.py`
- Modify: `python/tests/direct_optimizer/test_model.py`

- [ ] **Step 1: Add failing sampling tests**

Append to `python/tests/direct_optimizer/test_model.py`:

```python
from gristmill_symbolics.direct_optimizer.model import sample_tokens


def test_sample_tokens_returns_static_padded_batch():
    model = DirectOptimizerTransformer(
        source_len=64,
        target_len=16,
        scalar_value_min=-8,
        scalar_value_max=8,
        d_model=16,
        num_layers=1,
        num_heads=2,
        rngs=nnx.Rngs(2),
    )
    source = _batch(_source_tokens(length=64), batch_size=3)

    generated, mask = sample_tokens(
        model,
        jax.random.PRNGKey(2),
        source,
        max_length=16,
        temperature=1.0,
    )

    assert generated["kind"].shape == (3, 16)
    assert generated["mask"].shape == (3, 16)
    assert mask.shape == (3, 16)


def test_sample_tokens_is_deterministic_for_fixed_rng_and_state():
    model = DirectOptimizerTransformer(
        source_len=64,
        target_len=8,
        scalar_value_min=-4,
        scalar_value_max=4,
        d_model=8,
        num_layers=1,
        num_heads=1,
        rngs=nnx.Rngs(3),
    )
    source = _batch(_source_tokens(length=64), batch_size=2)

    left, left_mask = sample_tokens(model, jax.random.PRNGKey(5), source, max_length=8, temperature=1.0)
    right, right_mask = sample_tokens(model, jax.random.PRNGKey(5), source, max_length=8, temperature=1.0)

    assert all(jnp.array_equal(left[field], right[field]) for field in left)
    assert jnp.array_equal(left_mask, right_mask)
```

- [ ] **Step 2: Run the focused model test and confirm it fails**

Run:

```bash
uv run pytest tests/direct_optimizer/test_model.py -q
```

Expected: FAIL because `sample_tokens` is missing.

- [ ] **Step 3: Implement static-shape sampling**

Add `sample_tokens(model: DirectOptimizerTransformer, rng, source_tokens, *, max_length: int, temperature: float, mask_provider=None) -> tuple[dict[str, jax.Array], jax.Array]` to `model.py`.

Sampling rules:

- Validate `0 < max_length <= model.target_len` and `temperature > 0`.
- Initialize decoder prefix with `BOS` at position zero and `PAD` elsewhere.
- Use a `jax.lax.scan` over target positions.
- At each step, call `model(source_tokens, decoder_prefix, deterministic=True)`.
- Sample kind, keyword, scalar type, and scalar value with split PRNG keys and categorical logits divided by temperature.
- Apply `mask_provider(prefix_tokens, step, source_tokens)` only when provided. The returned mask object has optional fields `kind`, `keyword`, `scalar_type`, and `scalar_value`, each additive logit mask where invalid choices are large negative values.
- Stop a row after EOS by keeping following positions padded and mask false.
- Return token arrays shaped `[batch, target_len]` and a boolean sample mask shaped `[batch, target_len]`.

- [ ] **Step 4: Run focused model tests**

Run:

```bash
uv run pytest tests/direct_optimizer/test_model.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/gristmill_symbolics/direct_optimizer/model.py python/tests/direct_optimizer/test_model.py
git commit -m "feat: add direct optimizer token sampling"
```

## Task 9: Trainer Collation And Weighted Objective

**Files:**
- Create: `python/gristmill_symbolics/direct_optimizer/trainer.py`
- Modify: `python/tests/direct_optimizer/test_trainer.py`

- [ ] **Step 1: Write failing trainer collation tests**

Create `python/tests/direct_optimizer/test_trainer.py`:

```python
import math

import jax.numpy as jnp
import pytest

from gristmill_symbolics.direct_optimizer.dataset import BuildConfig, build_processed_dataset, write_processed_jsonl
from gristmill_symbolics.direct_optimizer.trainer import (
    DirectOptimizerTrainer,
    collate_processed_rows,
    weighted_sequence_loss,
)
from tests.direct_optimizer.test_dataset import _raw_records


def _processed_rows():
    return build_processed_dataset(_raw_records() * 2, BuildConfig(beta=1.0, verify=False))


def test_collate_processed_rows_uses_static_shapes_and_drops_remainder():
    rows = _processed_rows()[:3]

    batches = collate_processed_rows(
        rows,
        batch_size=2,
        source_len=128,
        target_len=128,
        scalar_value_min=-128,
        scalar_value_max=128,
    )

    assert len(batches) == 1
    batch = batches[0]
    assert batch["source_tokens"]["kind"].shape == (2, 128)
    assert batch["decoder_input_tokens"]["kind"].shape == (2, 128)
    assert batch["target_tokens"]["kind"].shape == (2, 128)
    assert batch["target_mask"].shape == (2, 128)
    assert batch["example_weight"].shape == (2,)


def test_collate_processed_rows_rejects_dataset_smaller_than_batch_size():
    with pytest.raises(ValueError, match="fewer compatible rows than batch_size"):
        collate_processed_rows(
            _processed_rows()[:1],
            batch_size=2,
            source_len=128,
            target_len=128,
            scalar_value_min=-128,
            scalar_value_max=128,
        )


def test_weighted_sequence_loss_normalizes_by_weight_sum():
    sequence_logp = jnp.asarray([-1.0, -3.0])
    weights = jnp.asarray([0.25, 0.75])

    loss = weighted_sequence_loss(sequence_logp, weights)

    assert loss == pytest.approx(2.5)
```

- [ ] **Step 2: Run focused trainer tests and confirm they fail**

Run:

```bash
uv run pytest tests/direct_optimizer/test_trainer.py -q
```

Expected: FAIL because `trainer.py` is missing.

- [ ] **Step 3: Implement collation and weighted loss**

Create `python/gristmill_symbolics/direct_optimizer/trainer.py` with:

```python
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from .dataset import read_processed_jsonl
from .model import DirectOptimizerTransformer, make_decoder_inputs, sequence_log_prob
from .tokens import encode_text, pad_tokens, validate_scalar_bounds
```

Implement `collate_processed_rows(rows: Sequence[dict[str, Any]], *, batch_size: int, source_len: int, target_len: int, scalar_value_min: int, scalar_value_max: int) -> list[dict[str, Any]]`.

Implement `weighted_sequence_loss` exactly as:

```python
def weighted_sequence_loss(sequence_logp, example_weight, *, epsilon: float = 1.0e-8):
    weights = jnp.asarray(example_weight, dtype=jnp.float32)
    nll = -jnp.asarray(sequence_logp, dtype=jnp.float32)
    return jnp.sum(weights * nll) / jnp.maximum(jnp.sum(weights), epsilon)
```

Collation rules:

- Validate required processed fields: `source_text`, `target_text`, `weight`, `input_key`, `candidate_key`, and `candidate_log_flops`.
- Encode source and target with direct optimizer `tokens.encode_text`.
- Build decoder inputs, labels, and target mask with `make_decoder_inputs`.
- Pad source to `source_len` and target-side arrays to `target_len`.
- Reject overlong source, overlong target, scalar values outside bounds, non-finite weights, and negative weights as incompatible rows.
- Drop final partial batches.
- Raise `ValueError("fewer compatible rows than batch_size")` when no full batch can be made.

- [ ] **Step 4: Run focused trainer tests and confirm they pass**

Run:

```bash
uv run pytest tests/direct_optimizer/test_trainer.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/gristmill_symbolics/direct_optimizer/trainer.py python/tests/direct_optimizer/test_trainer.py
git commit -m "feat: collate direct optimizer training batches"
```

## Task 10: NNX Train/Eval Steps And Epoch Loop

**Files:**
- Modify: `python/gristmill_symbolics/direct_optimizer/trainer.py`
- Modify: `python/tests/direct_optimizer/test_trainer.py`

- [ ] **Step 1: Add failing train/eval tests**

Append to `python/tests/direct_optimizer/test_trainer.py`:

```python
import jax
from flax import nnx
from gristmill_symbolics.direct_optimizer.model import DirectOptimizerTransformer


def _tiny_model():
    return DirectOptimizerTransformer(
        source_len=128,
        target_len=128,
        scalar_value_min=-128,
        scalar_value_max=128,
        d_model=16,
        num_layers=1,
        num_heads=2,
        rngs=nnx.Rngs(0),
    )


def test_train_step_changes_model_state_and_returns_finite_loss():
    trainer = DirectOptimizerTrainer(batch_size=2, learning_rate=1.0e-3)
    model = _tiny_model()
    optimizer = trainer.init_optimizer(model)
    batch = collate_processed_rows(_processed_rows(), batch_size=2, source_len=128, target_len=128, scalar_value_min=-128, scalar_value_max=128)[0]
    before = [leaf.copy() for leaf in jax.tree_util.tree_leaves(nnx.state(model)) if hasattr(leaf, "copy")]

    metrics = trainer.train_step(model, optimizer, batch)

    after = [leaf for leaf in jax.tree_util.tree_leaves(nnx.state(model)) if hasattr(leaf, "shape")]
    assert math.isfinite(float(metrics["train_loss"]))
    assert any(not bool(jnp.array_equal(left, right)) for left, right in zip(before, after, strict=False))


def test_eval_step_returns_finite_loss_without_mutating_model():
    trainer = DirectOptimizerTrainer(batch_size=2, learning_rate=1.0e-3)
    model = _tiny_model()
    batch = collate_processed_rows(_processed_rows(), batch_size=2, source_len=128, target_len=128, scalar_value_min=-128, scalar_value_max=128)[0]
    before = nnx.state(model)

    metrics = trainer.eval_step(model, batch, metric_name="valid_loss")

    assert math.isfinite(float(metrics["valid_loss"]))
    assert jax.tree_util.tree_structure(before) == jax.tree_util.tree_structure(nnx.state(model))
```

- [ ] **Step 2: Run focused trainer tests and confirm they fail**

Run:

```bash
uv run pytest tests/direct_optimizer/test_trainer.py -q
```

Expected: FAIL because `DirectOptimizerTrainer` lacks optimizer and step methods.

- [ ] **Step 3: Implement `DirectOptimizerTrainer`**

Add to `trainer.py`:

```python
class DirectOptimizerTrainer:
    def __init__(
        self,
        *,
        batch_size: int,
        learning_rate: float = 1.0e-3,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1.0e-8,
    ):
        self.batch_size = _positive_int("batch_size", batch_size)
        self.learning_rate = _positive_float("learning_rate", learning_rate)
        self.b1 = _adam_beta("b1", b1)
        self.b2 = _adam_beta("b2", b2)
        self.eps = _positive_float("eps", eps)

    def constructor_kwargs(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "b1": self.b1,
            "b2": self.b2,
            "eps": self.eps,
        }

    def init_optimizer(self, model: DirectOptimizerTransformer) -> nnx.Optimizer:
        tx = optax.adam(self.learning_rate, b1=self.b1, b2=self.b2, eps=self.eps)
        return nnx.Optimizer(model, tx)

    @staticmethod
    @nnx.jit
    def train_step(model: DirectOptimizerTransformer, optimizer: nnx.Optimizer, batch: dict[str, Any]) -> dict[str, jax.Array]:
        return _train_step(model, optimizer, batch)

    @staticmethod
    @nnx.jit
    def eval_step(model: DirectOptimizerTransformer, batch: dict[str, Any], *, metric_name: str = "valid_loss") -> dict[str, jax.Array]:
        return _eval_step(model, batch, metric_name)
```

Train step rules:

- Use `nnx.value_and_grad` on a loss function that calls `model(batch["source_tokens"], batch["decoder_input_tokens"], deterministic=False)`, `sequence_log_prob`, and `weighted_sequence_loss`.
- Call `optimizer.update(grads)`.
- Return `{"train_loss": loss}`.

Eval step rules:

- Use `deterministic=True`.
- Do not mutate model or optimizer state.
- Return `{metric_name: loss}`.

Constructor validation:

- `batch_size` positive int.
- optimizer floats finite, `learning_rate > 0`, `eps > 0`, `0 <= b1 < 1`, `0 <= b2 < 1`.

- [ ] **Step 4: Run focused trainer tests**

Run:

```bash
uv run pytest tests/direct_optimizer/test_trainer.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/gristmill_symbolics/direct_optimizer/trainer.py python/tests/direct_optimizer/test_trainer.py
git commit -m "feat: train direct optimizer model steps"
```

## Task 11: Direct Optimizer Checkpoint Helper

**Files:**
- Create: `python/gristmill_symbolics/direct_optimizer/checkpoint.py`
- Create: `python/tests/direct_optimizer/test_checkpoint.py`
- Modify: `python/gristmill_symbolics/direct_optimizer/trainer.py`

- [ ] **Step 1: Write failing checkpoint tests**

Create `python/tests/direct_optimizer/test_checkpoint.py`:

```python
import json

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from gristmill_symbolics.direct_optimizer.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    load_checkpoint,
    load_model_for_inference,
    save_checkpoint,
)
from gristmill_symbolics.direct_optimizer.model import DirectOptimizerTransformer
from gristmill_symbolics.direct_optimizer.trainer import DirectOptimizerTrainer


def _model(**overrides):
    kwargs = {
        "source_len": 32,
        "target_len": 32,
        "scalar_value_min": -16,
        "scalar_value_max": 16,
        "d_model": 8,
        "num_layers": 1,
        "num_heads": 1,
    }
    kwargs.update(overrides)
    return DirectOptimizerTransformer(**kwargs, rngs=nnx.Rngs(0))


def test_checkpoint_round_trips_model_optimizer_and_metadata(tmp_path):
    model = _model()
    trainer = DirectOptimizerTrainer(batch_size=2, learning_rate=1.0e-3)
    optimizer = trainer.init_optimizer(model)

    save_checkpoint(
        tmp_path,
        model=model,
        optimizer=optimizer,
        trainer=trainer,
        epoch=2,
        updates=5,
        last_train_loss=1.25,
        last_valid_loss=1.5,
    )
    loaded = load_checkpoint(tmp_path)

    assert loaded.metadata["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert loaded.metadata["epoch"] == 2
    assert loaded.metadata["updates"] == 5
    assert loaded.metadata["model_kwargs"] == model.model_kwargs()
    assert jax.tree_util.tree_structure(nnx.state(loaded.model)) == jax.tree_util.tree_structure(nnx.state(model))
    assert jax.tree_util.tree_structure(nnx.state(loaded.optimizer)) == jax.tree_util.tree_structure(nnx.state(optimizer))


def test_checkpoint_rejects_incompatible_static_model_kwargs(tmp_path):
    model = _model()
    trainer = DirectOptimizerTrainer(batch_size=2)
    optimizer = trainer.init_optimizer(model)
    save_checkpoint(tmp_path, model=model, optimizer=optimizer, trainer=trainer, epoch=0, updates=0, last_train_loss=0.0)

    with pytest.raises(ValueError, match="source_len"):
        load_checkpoint(tmp_path, expected_model_kwargs={**model.model_kwargs(), "source_len": 64})


def test_load_model_for_inference_ignores_optimizer_state(tmp_path):
    model = _model()
    trainer = DirectOptimizerTrainer(batch_size=2)
    optimizer = trainer.init_optimizer(model)
    save_checkpoint(tmp_path, model=model, optimizer=optimizer, trainer=trainer, epoch=0, updates=0, last_train_loss=0.0)

    loaded_model, metadata = load_model_for_inference(tmp_path)

    assert isinstance(loaded_model, DirectOptimizerTransformer)
    assert metadata["model_kwargs"] == model.model_kwargs()
```

- [ ] **Step 2: Run checkpoint tests and confirm they fail**

Run:

```bash
uv run pytest tests/direct_optimizer/test_checkpoint.py -q
```

Expected: FAIL because `checkpoint.py` is missing.

- [ ] **Step 3: Implement Orbax-backed checkpoint helper**

Create `python/gristmill_symbolics/direct_optimizer/checkpoint.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from flax import nnx
import optax
import orbax.checkpoint as ocp

from .converter import CONVERTER_SCHEMA_VERSION
from .model import DirectOptimizerTransformer


CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DirectOptimizerCheckpoint:
    model: DirectOptimizerTransformer
    optimizer: nnx.Optimizer | None
    metadata: dict[str, Any]
```

Implement these functions with the exact names and signatures:

- `save_checkpoint(path: str | Path, *, model: DirectOptimizerTransformer, optimizer: nnx.Optimizer, trainer: Any, epoch: int, updates: int, last_train_loss: float, last_valid_loss: float | None = None) -> None`
- `load_checkpoint(path: str | Path, *, expected_model_kwargs: dict[str, Any] | None = None) -> DirectOptimizerCheckpoint`
- `load_model_for_inference(path: str | Path) -> tuple[DirectOptimizerTransformer, dict[str, Any]]`

Checkpoint layout:

```text
checkpoint_dir/
  metadata.json
  model_state/
  optimizer_state/
```

Metadata fields:

```python
{
    "schema_version": CHECKPOINT_SCHEMA_VERSION,
    "converter_schema_version": CONVERTER_SCHEMA_VERSION,
    "model_kwargs": model.model_kwargs(),
    "trainer_kwargs": trainer.constructor_kwargs(),
    "epoch": epoch,
    "updates": updates,
    "last_train_loss": last_train_loss,
    "last_valid_loss": last_valid_loss,
}
```

Implementation rules:

- Store JSON metadata with sorted keys.
- Save `nnx.state(model)` through `ocp.StandardCheckpointer().save(path / "model_state", state, force=True)`.
- Save `nnx.state(optimizer)` through `ocp.StandardCheckpointer().save(path / "optimizer_state", state, force=True)`.
- On load, instantiate `DirectOptimizerTransformer(**model_kwargs, rngs=nnx.Rngs(0))`, restore model state with `target=nnx.state(model)`, and call `nnx.update(model, restored_state)`.
- Recreate `optimizer = nnx.Optimizer(model, optax.adam(**adam_kwargs_from_trainer_kwargs))` when `trainer_kwargs` exists, then restore optimizer state. Do not import `direct_optimizer.trainer` from `checkpoint.py`.
- `load_model_for_inference` calls the model restore path and returns no optimizer.
- Reject unsupported `schema_version` and mismatched `converter_schema_version`.
- `_validate_model_kwargs` compares `source_len`, `target_len`, `scalar_value_min`, `scalar_value_max`, `d_model`, `num_layers`, and `num_heads`, raising `ValueError` naming the mismatched key.

- [ ] **Step 4: Assert trainer kwargs are included in checkpoint metadata**

Extend `test_checkpoint_round_trips_model_optimizer_and_metadata` with:

```python
assert loaded.metadata["trainer_kwargs"]["batch_size"] == 2
```

- [ ] **Step 5: Run checkpoint and trainer tests**

Run:

```bash
uv run pytest tests/direct_optimizer/test_checkpoint.py tests/direct_optimizer/test_trainer.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add python/gristmill_symbolics/direct_optimizer/checkpoint.py python/gristmill_symbolics/direct_optimizer/trainer.py python/tests/direct_optimizer/test_checkpoint.py
git commit -m "feat: add direct optimizer checkpoints"
```

## Task 12: Training Loop And Train CLI

**Files:**
- Create: `python/gristmill_symbolics/direct_optimizer/train.py`
- Modify: `python/gristmill_symbolics/direct_optimizer/trainer.py`
- Modify: `python/tests/direct_optimizer/test_trainer.py`

- [ ] **Step 1: Add failing train loop and CLI tests**

Append to `python/tests/direct_optimizer/test_trainer.py`:

```python
from gristmill_symbolics.direct_optimizer.checkpoint import load_checkpoint
from gristmill_symbolics.direct_optimizer.train import main as train_main


def test_train_cli_runs_one_tiny_epoch_and_writes_checkpoint(tmp_path, capsys):
    dataset_path = tmp_path / "train.jsonl"
    checkpoint_path = tmp_path / "ckpt"
    write_processed_jsonl(_processed_rows() * 2, dataset_path)

    exit_code = train_main(
        [
            "--train-dataset",
            str(dataset_path),
            "--checkpoint-out",
            str(checkpoint_path),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--learning-rate",
            "0.001",
            "--source-len",
            "128",
            "--target-len",
            "128",
            "--scalar-value-min",
            "-128",
            "--scalar-value-max",
            "128",
            "--d-model",
            "16",
            "--num-layers",
            "1",
            "--num-heads",
            "2",
            "--seed",
            "0",
        ]
    )

    output = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    loaded = load_checkpoint(checkpoint_path)
    assert exit_code == 0
    assert math.isfinite(output["train_loss"])
    assert loaded.metadata["epoch"] == 1
    assert loaded.metadata["updates"] > 0


def test_trainer_does_not_split_processed_datasets(tmp_path):
    train_path = tmp_path / "train.jsonl"
    valid_path = tmp_path / "valid.jsonl"
    write_processed_jsonl(_processed_rows() * 2, train_path)
    write_processed_jsonl(_processed_rows() * 2, valid_path)

    exit_code = train_main(
        [
            "--train-dataset",
            str(train_path),
            "--valid-dataset",
            str(valid_path),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--source-len",
            "128",
            "--target-len",
            "128",
            "--scalar-value-min",
            "-128",
            "--scalar-value-max",
            "128",
            "--d-model",
            "16",
            "--num-layers",
            "1",
            "--num-heads",
            "2",
        ]
    )

    assert exit_code == 0
```

- [ ] **Step 2: Run focused trainer tests and confirm they fail**

Run:

```bash
uv run pytest tests/direct_optimizer/test_trainer.py -q
```

Expected: FAIL because train loop and CLI are missing.

- [ ] **Step 3: Implement epoch training helpers**

Add `train_epochs(*, train_rows: Sequence[dict[str, Any]], valid_rows: Sequence[dict[str, Any]] | None, test_rows: Sequence[dict[str, Any]] | None, model: DirectOptimizerTransformer, trainer: DirectOptimizerTrainer, optimizer: nnx.Optimizer, epochs: int, source_len: int, target_len: int, scalar_value_min: int, scalar_value_max: int, seed: int, checkpoint_out: Path | None, start_epoch: int = 0, start_updates: int = 0) -> dict[str, float]` to `trainer.py`.

Loop rules:

- Collate train, valid, and test rows separately with fixed shape.
- Shuffle train batch order each epoch with `np.random.default_rng(seed + epoch)`.
- Average train, valid, and test losses by arithmetic mean over full batches.
- Save checkpoint after every epoch when `checkpoint_out` is provided.
- Print JSON metrics from the CLI after each epoch.
- Never generate, split, verify, regroup, dedupe, or reweight rows inside trainer code.

- [ ] **Step 4: Implement train CLI**

Create `python/gristmill_symbolics/direct_optimizer/train.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from flax import nnx

from .checkpoint import load_checkpoint
from .dataset import read_processed_jsonl
from .model import DirectOptimizerTransformer
from .trainer import DirectOptimizerTrainer, train_epochs
```

CLI requirements:

- Flags exactly match the trainer spec: `--train-dataset`, `--valid-dataset`, `--test-dataset`, `--checkpoint-out`, `--checkpoint-in`, `--epochs`, `--batch-size`, `--learning-rate`, `--source-len`, `--target-len`, `--scalar-value-min`, `--scalar-value-max`, `--d-model`, `--num-layers`, `--num-heads`, `--seed`.
- `--checkpoint-in` restores model, optimizer, epoch, and updates from the direct optimizer checkpoint.
- Constructor/static-shape flags must match checkpoint kwargs when provided; mismatches raise parser errors.
- Fresh training requires all static model flags.
- Add entry point guard:

```python
if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run focused tests and CLI help**

Run:

```bash
uv run pytest tests/direct_optimizer/test_trainer.py tests/direct_optimizer/test_checkpoint.py -q
uv run python -m gristmill_symbolics.direct_optimizer.train --help
```

Expected: tests PASS and CLI help exits with status `0`.

- [ ] **Step 6: Commit**

```bash
git add python/gristmill_symbolics/direct_optimizer/train.py python/gristmill_symbolics/direct_optimizer/trainer.py python/tests/direct_optimizer/test_trainer.py
git commit -m "feat: add direct optimizer training CLI"
```

## Task 13: Verifier-Gated Sampler API

**Files:**
- Create: `python/gristmill_symbolics/direct_optimizer/sample.py`
- Create: `python/tests/direct_optimizer/test_sampler.py`

- [ ] **Step 1: Write failing sampler API tests**

Create `python/tests/direct_optimizer/test_sampler.py`:

```python
import json

import jax
import jax.numpy as jnp
import pytest

from gristmill_symbolics.direct_optimizer.converter import computation_to_target_text
from gristmill_symbolics.direct_optimizer.sample import optimize_with_model
from gristmill_symbolics.direct_optimizer.tokens import encode_text, pad_tokens
from tests.direct_optimizer.fixtures import source_comp


class FakeModel:
    target_len = 64

    def __init__(self, rows):
        self.rows = list(rows)
        self.calls = 0


def _token_row(text: str, length: int = 64):
    return pad_tokens(encode_text(text), length=length)


def _stack(rows):
    return {
        field: jnp.stack([jnp.asarray(row[field]) for row in rows], axis=0)
        for field in rows[0]
    }


def fake_sample_tokens(model, rng, source_tokens, *, max_length, temperature, mask_provider=None):
    batch_size = source_tokens["kind"].shape[0]
    start = model.calls * batch_size
    model.calls += 1
    rows = model.rows[start : start + batch_size]
    while len(rows) < batch_size:
        rows.append(_token_row("def base tensor_id:99\nenddef"))
    stacked = _stack(rows)
    return stacked, stacked["mask"]


def test_sampler_rejects_invalid_and_non_equivalent_candidates(monkeypatch):
    valid_text = computation_to_target_text(source_comp())
    invalid_text = "def base tensor_id:1\nterm\nenddef"
    model = FakeModel([_token_row(invalid_text), _token_row(valid_text)])
    monkeypatch.setattr("gristmill_symbolics.direct_optimizer.sample.sample_tokens", fake_sample_tokens)

    candidate, metrics = optimize_with_model(
        model,
        None,
        source_comp(),
        [1],
        num_samples=2,
        sample_batch_size=2,
        source_len=128,
        target_len=64,
        temperature=1.0,
        seed=0,
    )

    assert candidate is not None
    assert metrics["total_samples"] == 2
    assert metrics["parse_failures"] == 1
    assert metrics["valid_samples"] == 1
    assert metrics["best_log_flops"] is not None


def test_sampler_ignores_padded_extra_rows(monkeypatch):
    valid_text = computation_to_target_text(source_comp())
    model = FakeModel([_token_row(valid_text), _token_row(valid_text), _token_row(valid_text), _token_row("def base tensor_id:99\nenddef")])
    monkeypatch.setattr("gristmill_symbolics.direct_optimizer.sample.sample_tokens", fake_sample_tokens)

    _candidate, metrics = optimize_with_model(
        model,
        None,
        source_comp(),
        [1],
        num_samples=3,
        sample_batch_size=2,
        source_len=128,
        target_len=64,
        temperature=1.0,
        seed=0,
    )

    assert metrics["total_samples"] == 3
    assert metrics["valid_samples"] == 3
    assert metrics["parse_failures"] == 0
```

- [ ] **Step 2: Run sampler tests and confirm they fail**

Run:

```bash
uv run pytest tests/direct_optimizer/test_sampler.py -q
```

Expected: FAIL because `sample.py` is missing.

- [ ] **Step 3: Implement `optimize_with_model`**

Create `python/gristmill_symbolics/direct_optimizer/sample.py` with:

```python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from gristmill_symbolics import TensorComputation, equivalent_computations

from .converter import computation_to_source_text, target_text_to_computation, target_text_to_definitions
from .model import DirectOptimizerTransformer, sample_tokens
from .tokens import decode_token_row_to_text, encode_text, pad_tokens, repeat_token_row
```

Implement `optimize_with_model(model, params, input_computation: TensorComputation, outputs: list[int], *, num_samples: int, sample_batch_size: int, source_len: int, target_len: int, temperature: float, seed: int) -> tuple[TensorComputation | None, dict[str, Any]]`.

Sampler rules:

- Validate `num_samples`, `sample_batch_size`, `source_len`, `target_len`, and `temperature` are positive.
- Validate ordered outputs as integers and not bools; preserve order.
- Convert source computation to source DSL and encode/pad to `source_len`.
- Repeat the padded source row to `[sample_batch_size, source_len]`.
- Run `ceil(num_samples / sample_batch_size)` model batches and ignore generated rows beyond `num_samples`.
- Decode generated token row to target DSL; `ValueError` increments `decode_failures`.
- Call `target_text_to_definitions`; `ValueError` increments `parse_failures`.
- Call `target_text_to_computation`; `ValueError` increments `reconstruction_failures`.
- Call `equivalent_computations(input_computation, candidate, outputs)`; false or verifier errors increment `verifier_failures`.
- Compute `candidate.log_total_flops()` for verified candidates.
- Return the verified candidate with the lowest log flops, or `None`.
- Metrics dictionary keys: `total_samples`, `decode_failures`, `parse_failures`, `reconstruction_failures`, `verifier_failures`, `valid_samples`, `best_log_flops`.

- [ ] **Step 4: Run sampler tests and confirm they pass**

Run:

```bash
uv run pytest tests/direct_optimizer/test_sampler.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/gristmill_symbolics/direct_optimizer/sample.py python/tests/direct_optimizer/test_sampler.py
git commit -m "feat: add verifier-gated direct optimizer sampler"
```

## Task 14: Checkpoint-Backed Sampler And Sample CLI

**Files:**
- Modify: `python/gristmill_symbolics/direct_optimizer/sample.py`
- Modify: `python/tests/direct_optimizer/test_sampler.py`
- Modify: `python/gristmill_symbolics/direct_optimizer/__init__.py`

- [ ] **Step 1: Add failing checkpoint-backed sampler and CLI tests**

Append to `python/tests/direct_optimizer/test_sampler.py`:

```python
from flax import nnx

from gristmill_symbolics.direct_optimizer.checkpoint import save_checkpoint
from gristmill_symbolics.direct_optimizer.model import DirectOptimizerTransformer
from gristmill_symbolics.direct_optimizer.sample import main as sample_main, optimize_from_checkpoint
from gristmill_symbolics.direct_optimizer.trainer import DirectOptimizerTrainer


def test_optimize_from_checkpoint_rejects_static_shape_mismatch(tmp_path):
    model = DirectOptimizerTransformer(
        source_len=128,
        target_len=64,
        scalar_value_min=-128,
        scalar_value_max=128,
        d_model=8,
        num_layers=1,
        num_heads=1,
        rngs=nnx.Rngs(4),
    )
    trainer = DirectOptimizerTrainer(batch_size=2)
    optimizer = trainer.init_optimizer(model)
    save_checkpoint(tmp_path, model=model, optimizer=optimizer, trainer=trainer, epoch=0, updates=0, last_train_loss=0.0)

    with pytest.raises(ValueError, match="source_len"):
        optimize_from_checkpoint(
            tmp_path,
            source_comp(),
            [1],
            num_samples=1,
            sample_batch_size=1,
            source_len=64,
            target_len=64,
            temperature=1.0,
            seed=0,
        )


def test_sample_cli_writes_output_for_valid_monkeypatched_candidate(tmp_path, monkeypatch, capsys):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "optimized.json"
    ckpt_path = tmp_path / "ckpt"
    source_comp().write_json(input_path)
    model = DirectOptimizerTransformer(
        source_len=128,
        target_len=64,
        scalar_value_min=-128,
        scalar_value_max=128,
        d_model=8,
        num_layers=1,
        num_heads=1,
        rngs=nnx.Rngs(5),
    )
    trainer = DirectOptimizerTrainer(batch_size=1)
    save_checkpoint(ckpt_path, model=model, optimizer=trainer.init_optimizer(model), trainer=trainer, epoch=0, updates=0, last_train_loss=0.0)
    valid = _token_row(computation_to_target_text(source_comp()))
    fake = FakeModel([valid])
    monkeypatch.setattr("gristmill_symbolics.direct_optimizer.sample.load_model_for_inference", lambda path: (fake, {"model_kwargs": model.model_kwargs()}))
    monkeypatch.setattr("gristmill_symbolics.direct_optimizer.sample.sample_tokens", fake_sample_tokens)

    exit_code = sample_main(
        [
            "--checkpoint",
            str(ckpt_path),
            "--input",
            str(input_path),
            "--outputs",
            "1",
            "--samples",
            "1",
            "--sample-batch-size",
            "1",
            "--temperature",
            "1.0",
            "--output",
            str(output_path),
        ]
    )

    metrics = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert metrics["valid_samples"] == 1
    assert output_path.exists()
```

- [ ] **Step 2: Run sampler tests and confirm they fail**

Run:

```bash
uv run pytest tests/direct_optimizer/test_sampler.py -q
```

Expected: FAIL because `optimize_from_checkpoint`, `main`, and final package exports are missing.

- [ ] **Step 3: Implement checkpoint-backed inference and CLI**

Add to `sample.py`:

```python
from .checkpoint import load_model_for_inference
```

Implement `optimize_from_checkpoint(checkpoint_path: str | Path, input_computation: TensorComputation, outputs: list[int], *, num_samples: int, sample_batch_size: int, source_len: int | None = None, target_len: int | None = None, temperature: float = 1.0, seed: int = 0) -> tuple[TensorComputation | None, dict[str, Any]]`.

Rules:

- Load model and metadata with `load_model_for_inference`.
- Use checkpoint `source_len` and `target_len` when caller passes `None`.
- If caller-provided `source_len` or `target_len` differs from metadata, raise `ValueError` naming the mismatched key.
- Ignore optimizer state, epoch, updates, and losses.

Implement CLI:

```text
python -m gristmill_symbolics.direct_optimizer.sample \
  --checkpoint direct_optimizer_ckpt \
  --input input_computation.json \
  --outputs 1,3 \
  --samples 64 \
  --sample-batch-size 8 \
  --temperature 1.0 \
  --output optimized.json
```

CLI behavior:

- Read input with `TensorComputation.load_json`.
- Parse `--outputs` as comma-separated ordered ints.
- Write selected candidate with `candidate.write_json(output_path)` when a valid candidate exists.
- Print metrics JSON to stdout.
- Return `1` when no valid candidate exists.
- Add entry point guard:

```python
if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Update package exports**

Replace `python/gristmill_symbolics/direct_optimizer/__init__.py` with:

```python
"""Self-contained direct optimizer package."""

from .model import DirectOptimizerTransformer
from .sample import optimize_from_checkpoint, optimize_with_model

__all__ = (
    "DirectOptimizerTransformer",
    "optimize_from_checkpoint",
    "optimize_with_model",
)
```

Update `python/tests/direct_optimizer/test_layout.py` package import assertion:

```python
def test_direct_optimizer_package_imports():
    package = importlib.import_module("gristmill_symbolics.direct_optimizer")

    assert package.__all__ == (
        "DirectOptimizerTransformer",
        "optimize_from_checkpoint",
        "optimize_with_model",
    )
```

- [ ] **Step 5: Run sampler and layout tests**

Run:

```bash
uv run pytest tests/direct_optimizer/test_sampler.py tests/direct_optimizer/test_layout.py -q
uv run python -m gristmill_symbolics.direct_optimizer.sample --help
```

Expected: tests PASS and CLI help exits with status `0`.

- [ ] **Step 6: Commit**

```bash
git add python/gristmill_symbolics/direct_optimizer/__init__.py python/gristmill_symbolics/direct_optimizer/sample.py python/tests/direct_optimizer/test_layout.py python/tests/direct_optimizer/test_sampler.py
git commit -m "feat: add direct optimizer checkpoint sampling CLI"
```

## Task 15: Boundary, Integration, And Regression Verification

**Files:**
- Modify: `python/tests/direct_optimizer/test_layout.py`
- Modify only direct optimizer files if this task reveals a direct optimizer issue.

- [ ] **Step 1: Strengthen forbidden import tests**

Extend `python/tests/direct_optimizer/test_layout.py` with module-specific boundary tests:

```python
def test_trainer_does_not_import_symbolic_or_sampler_boundaries():
    modules = _imported_modules(PACKAGE / "trainer.py")

    assert "gristmill_symbolics" not in modules
    assert "gristmill_symbolics.direct_optimizer.sample" not in modules


def test_sampler_does_not_import_trainer_module():
    modules = _imported_modules(PACKAGE / "sample.py")

    assert "gristmill_symbolics.direct_optimizer.trainer" not in modules


def test_direct_optimizer_does_not_register_existing_cli_checkpoint():
    modules = _imported_modules(PACKAGE / "checkpoint.py")

    assert "gristmill_symbolics.cli.checkpoint" not in modules
    assert "gristmill_symbolics.direct_optimizer.trainer" not in modules
```

- [ ] **Step 2: Run direct optimizer test suite**

Run:

```bash
uv run pytest tests/direct_optimizer -q
```

Expected: PASS.

- [ ] **Step 3: Run required existing layout regression**

Run:

```bash
uv run pytest tests/test_model_trainer_cli_layout.py -q
```

Expected: PASS. This proves the new package did not reintroduce public `policy` or `reinforce` paths and did not disturb the existing model/trainer/CLI surface.

- [ ] **Step 4: Run focused Python training regression from AGENTS.md**

Run:

```bash
uv run pytest tests/model/transformer_action_selector/test_model_protocol.py -q
uv run pytest tests/trainer/reinforce/test_trainer_protocol.py -q
uv run pytest tests/cli/test_checkpoint.py tests/cli/test_checkpoint_schema.py -q
```

Expected: PASS. If protocol test filenames differ in this branch, run the closest existing equivalents under `python/tests/model/transformer_action_selector`, `python/tests/trainer/reinforce`, and `python/tests/cli`.

- [ ] **Step 5: Run broad verification**

Run:

```bash
uv run pytest -q
cargo test
```

Expected: all Python and Rust tests PASS.

- [ ] **Step 6: Commit final guard updates**

```bash
git add python/tests/direct_optimizer/test_layout.py
git commit -m "test: verify direct optimizer package boundaries"
```

If Step 2 through Step 5 required direct optimizer fixes, include only the touched direct optimizer files and tests in this commit.

## Final Self-Review Checklist

- Spec coverage: converter is covered by Tasks 2-3; dataset generation, raw split, independent processed split building, weighting, and CLI are covered by Tasks 4-5; NNX model, scoring, masking, static shapes, and sampling hook are covered by Tasks 6-8; trainer collation, objective, train/eval, no internal splitting, and train CLI are covered by Tasks 9-12; direct checkpoint helper is covered by Task 11; verifier-gated sampler and sample CLI are covered by Tasks 13-14; package boundaries and final verification are covered by Task 15.
- Placeholder scan: passed. Every task names files, commands, expected outcomes, and a commit command.
- Type consistency: public names are consistent across tasks: `DirectOptimizerTransformer`, `DirectOptimizerTrainer`, `GenerationConfig`, `BuildConfig`, `SplitConfig`, `save_checkpoint`, `load_checkpoint`, `load_model_for_inference`, `optimize_with_model`, and `optimize_from_checkpoint`.
- Boundary consistency: no task modifies existing action-selector, REINFORCE, or `gristmill_symbolics.cli.checkpoint` files.
