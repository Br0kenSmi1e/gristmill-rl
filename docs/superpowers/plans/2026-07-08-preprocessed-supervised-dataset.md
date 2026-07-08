# Preprocessed Supervised Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a strict JSONL-to-fixed-arrays dataset boundary for definitions-only supervised training.

**Architecture:** Add one focused Python module, `gristmill_symbolics.supervised_dataset`, that validates weighted `TensorComputation` pairs, tokenizes only definitions, writes `.npz` arrays plus JSON metadata, loads them back, and yields fixed-shape batches. The preprocessed dataset is a plain dictionary of arrays plus metadata, matching the existing batch contract style. Keep decoder-input construction private to preprocessing. Do not add CLI, checkpoint, validation, or training-loop code.

**Tech Stack:** Python stdlib `json` and `pathlib`; NumPy arrays and `.npz`; existing `TensorComputation` binding; existing `FlatDefinitionTokenizer`.

**Plan revision:** After implementation review, the public dataset container was simplified from a dataclass to a plain dictionary. Any earlier task snippets that mention `PreprocessedSupervisedDataset` as a dataclass are superseded by this revision.

---

### Task 1: Preprocess In-Memory Rows To Fixed Arrays

**Files:**
- Create: `python/gristmill_symbolics/supervised_dataset.py`
- Test: `python/tests/test_supervised_dataset.py`

- [ ] **Step 1: Write failing tests for row preprocessing and target construction**

Add `python/tests/test_supervised_dataset.py`:

```python
import json

import numpy as np
import pytest

from gristmill_symbolics import TensorComputation
from gristmill_symbolics.supervised_dataset import (
    SupervisedDatasetError,
    preprocess_supervised_rows,
)
from gristmill_symbolics.tokenizer import FlatDefinitionTokenizer


def _tokenizer() -> FlatDefinitionTokenizer:
    return FlatDefinitionTokenizer(
        max_range_id=2,
        max_tensor_id=3,
        max_index_id=4,
        coeff_nums=(-1, 1, 2),
        coeff_dens=(1, 2),
    )


def _computation_json(*, base: int = 1, coeff: tuple[int, int] = (1, 1)):
    return {
        "ranges": [{"id": 0, "size": 3}],
        "tensors": [
            {"id": 0, "symmetry": []},
            {"id": base, "symmetry": []},
        ],
        "definitions": [
            {
                "base": base,
                "ext_indices": [{"id": 0, "range": 0}],
                "terms": [
                    {
                        "coeff": list(coeff),
                        "sum_indices": [],
                        "factors": [{"tensor": 0, "indices": [0]}],
                    }
                ],
            }
        ],
    }


def _row(*, weight: float = 1.5):
    return {
        "source": _computation_json(base=1, coeff=(1, 1)),
        "target": _computation_json(base=2, coeff=(2, 1)),
        "weight": weight,
    }


def test_preprocess_supervised_rows_builds_fixed_arrays():
    tokenizer = _tokenizer()
    dataset = preprocess_supervised_rows(
        [_row(weight=1.5), _row(weight=0.5)],
        tokenizer,
        source_len=12,
        target_len=12,
    )

    assert dataset.source_ids.shape == (2, 12)
    assert dataset.decoder_input_ids.shape == (2, 12)
    assert dataset.target_ids.shape == (2, 12)
    assert dataset.target_mask.shape == (2, 12)
    assert dataset.example_weight.shape == (2,)
    assert dataset.source_ids.dtype == np.int32
    assert dataset.decoder_input_ids.dtype == np.int32
    assert dataset.target_ids.dtype == np.int32
    assert dataset.target_mask.dtype == np.bool_
    assert dataset.example_weight.dtype == np.float32
    assert dataset.metadata["num_examples"] == 2
    assert dataset.metadata["source_len"] == 12
    assert dataset.metadata["target_len"] == 12
    assert dataset.metadata["vocab_size"] == tokenizer.vocab_size


def test_preprocess_supervised_rows_builds_bos_eos_pad_and_target_mask():
    tokenizer = _tokenizer()
    row = _row(weight=1.0)
    target = TensorComputation.from_json_string(json.dumps(row["target"]))
    content_ids = tokenizer.encode_definitions(target.snapshot()["definitions"])

    dataset = preprocess_supervised_rows(
        [row],
        tokenizer,
        source_len=12,
        target_len=len(content_ids) + 3,
    )

    expected_decoder = np.full(
        (len(content_ids) + 3,),
        tokenizer.pad_token_id,
        dtype=np.int32,
    )
    expected_decoder[: len(content_ids) + 1] = [
        tokenizer.bos_token_id,
        *content_ids,
    ]
    expected_target = np.full(
        (len(content_ids) + 3,),
        tokenizer.pad_token_id,
        dtype=np.int32,
    )
    expected_target[: len(content_ids) + 1] = [
        *content_ids,
        tokenizer.eos_token_id,
    ]
    expected_mask = np.zeros((len(content_ids) + 3,), dtype=np.bool_)
    expected_mask[: len(content_ids) + 1] = True

    assert np.array_equal(dataset.decoder_input_ids[0], expected_decoder)
    assert np.array_equal(dataset.target_ids[0], expected_target)
    assert np.array_equal(dataset.target_mask[0], expected_mask)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd python
uv run pytest tests/test_supervised_dataset.py -q
```

Expected: FAIL during import because `gristmill_symbolics.supervised_dataset` does not exist.

- [ ] **Step 3: Implement minimal preprocessing module**

Create `python/gristmill_symbolics/supervised_dataset.py`:

```python
from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import TensorComputation
from .tokenizer import FlatDefinitionTokenizer, TokenizerError

__all__ = (
    "PreprocessedSupervisedDataset",
    "SupervisedDatasetError",
    "preprocess_supervised_rows",
)


class SupervisedDatasetError(ValueError):
    """Raised when a supervised dataset row or file is invalid."""


@dataclass(frozen=True)
class PreprocessedSupervisedDataset:
    source_ids: np.ndarray
    decoder_input_ids: np.ndarray
    target_ids: np.ndarray
    target_mask: np.ndarray
    example_weight: np.ndarray
    metadata: dict[str, object]

    @property
    def num_examples(self) -> int:
        return int(self.example_weight.shape[0])


def preprocess_supervised_rows(
    rows: Iterable[Mapping[str, object]],
    tokenizer: FlatDefinitionTokenizer,
    *,
    source_len: int,
    target_len: int,
) -> PreprocessedSupervisedDataset:
    source_rows: list[np.ndarray] = []
    decoder_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    weights: list[float] = []

    for row_number, row in enumerate(rows, start=1):
        source_ids, decoder_ids, target_ids, target_mask, weight = _preprocess_row(
            row,
            tokenizer,
            source_len=source_len,
            target_len=target_len,
            row_number=row_number,
        )
        source_rows.append(source_ids)
        decoder_rows.append(decoder_ids)
        target_rows.append(target_ids)
        mask_rows.append(target_mask)
        weights.append(weight)

    metadata = _metadata(
        tokenizer,
        source_len=source_len,
        target_len=target_len,
        num_examples=len(weights),
    )
    return PreprocessedSupervisedDataset(
        source_ids=_stack_or_empty(source_rows, source_len, np.int32),
        decoder_input_ids=_stack_or_empty(decoder_rows, target_len, np.int32),
        target_ids=_stack_or_empty(target_rows, target_len, np.int32),
        target_mask=_stack_or_empty(mask_rows, target_len, np.bool_),
        example_weight=np.asarray(weights, dtype=np.float32),
        metadata=metadata,
    )


def _preprocess_row(
    row: Mapping[str, object],
    tokenizer: FlatDefinitionTokenizer,
    *,
    source_len: int,
    target_len: int,
    row_number: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    if not isinstance(row, Mapping):
        raise SupervisedDatasetError(f"row {row_number}: row must be an object")
    for key in ("source", "target", "weight"):
        if key not in row:
            raise SupervisedDatasetError(f"row {row_number}: missing {key!r}")

    weight = _parse_weight(row["weight"], row_number=row_number)
    source = _parse_computation(row["source"], row_number=row_number, key="source")
    target = _parse_computation(row["target"], row_number=row_number, key="target")

    try:
        source_ids = tokenizer.encode_definitions_padded(
            source.snapshot()["definitions"],
            length=source_len,
        )
        content_ids = tokenizer.encode_definitions(target.snapshot()["definitions"])
    except (KeyError, TypeError, TokenizerError) as exc:
        raise SupervisedDatasetError(f"row {row_number}: {exc}") from exc

    decoder_ids, target_ids, target_mask = _target_arrays(
        content_ids,
        tokenizer,
        target_len=target_len,
        row_number=row_number,
    )
    return (
        np.asarray(source_ids, dtype=np.int32),
        decoder_ids,
        target_ids,
        target_mask,
        weight,
    )


def _parse_weight(value: object, *, row_number: int) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SupervisedDatasetError(
            f"row {row_number}: weight must be a finite non-negative number"
        )
    weight = float(value)
    if not math.isfinite(weight) or weight < 0.0:
        raise SupervisedDatasetError(
            f"row {row_number}: weight must be a finite non-negative number"
        )
    return weight


def _parse_computation(
    value: object,
    *,
    row_number: int,
    key: str,
) -> TensorComputation:
    try:
        return TensorComputation.from_json_string(json.dumps(value))
    except Exception as exc:
        raise SupervisedDatasetError(
            f"row {row_number}: invalid {key} TensorComputation: {exc}"
        ) from exc


def _target_arrays(
    content_ids: list[int],
    tokenizer: FlatDefinitionTokenizer,
    *,
    target_len: int,
    row_number: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(content_ids) + 1 > target_len:
        raise SupervisedDatasetError(
            "row "
            f"{row_number}: target content length {len(content_ids)} needs "
            f"target_len at least {len(content_ids) + 1}, got {target_len}"
        )

    decoder = np.full((target_len,), tokenizer.pad_token_id, dtype=np.int32)
    target = np.full((target_len,), tokenizer.pad_token_id, dtype=np.int32)
    mask = np.zeros((target_len,), dtype=np.bool_)

    decoder_tokens = [tokenizer.bos_token_id, *content_ids]
    target_tokens = [*content_ids, tokenizer.eos_token_id]
    decoder[: len(decoder_tokens)] = decoder_tokens
    target[: len(target_tokens)] = target_tokens
    mask[: len(target_tokens)] = True
    return decoder, target, mask


def _stack_or_empty(
    rows: list[np.ndarray],
    length: int,
    dtype: np.dtype,
) -> np.ndarray:
    if rows:
        return np.stack(rows).astype(dtype, copy=False)
    return np.empty((0, length), dtype=dtype)


def _metadata(
    tokenizer: FlatDefinitionTokenizer,
    *,
    source_len: int,
    target_len: int,
    num_examples: int,
) -> dict[str, object]:
    return {
        "format": "gristmill_symbolics.preprocessed_supervised_dataset.v1",
        "source_len": source_len,
        "target_len": target_len,
        "num_examples": num_examples,
        "vocab_size": tokenizer.vocab_size,
        "tokenizer": {
            "max_range_id": tokenizer.max_range_id,
            "max_tensor_id": tokenizer.max_tensor_id,
            "max_index_id": tokenizer.max_index_id,
            "coeff_nums": list(tokenizer.coeff_nums),
            "coeff_dens": list(tokenizer.coeff_dens),
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        },
    }
```

- [ ] **Step 4: Run tests to verify Task 1 passes**

Run:

```bash
cd python
uv run pytest tests/test_supervised_dataset.py -q
```

Expected: PASS for the two tests in this task.

### Task 2: JSONL Reading, Saving, And Loading

**Files:**
- Modify: `python/gristmill_symbolics/supervised_dataset.py`
- Modify: `python/tests/test_supervised_dataset.py`

- [ ] **Step 1: Add failing tests for JSONL preprocessing, metadata, and load round trip**

Append to `python/tests/test_supervised_dataset.py`:

```python
from gristmill_symbolics.supervised_dataset import (
    load_preprocessed_supervised_dataset,
    preprocess_supervised_jsonl,
    save_preprocessed_supervised_dataset,
)


def test_preprocess_supervised_jsonl_writes_npz_and_metadata(tmp_path):
    tokenizer = _tokenizer()
    input_path = tmp_path / "rows.jsonl"
    arrays_path = tmp_path / "dataset.npz"
    metadata_path = tmp_path / "dataset.metadata.json"
    input_path.write_text(json.dumps(_row(weight=2.0)) + "\n")

    dataset = preprocess_supervised_jsonl(
        input_path,
        arrays_path,
        metadata_path,
        tokenizer,
        source_len=12,
        target_len=12,
    )
    loaded = load_preprocessed_supervised_dataset(arrays_path, metadata_path)

    assert arrays_path.exists()
    assert metadata_path.exists()
    assert np.array_equal(loaded.source_ids, dataset.source_ids)
    assert np.array_equal(loaded.decoder_input_ids, dataset.decoder_input_ids)
    assert np.array_equal(loaded.target_ids, dataset.target_ids)
    assert np.array_equal(loaded.target_mask, dataset.target_mask)
    assert np.array_equal(loaded.example_weight, dataset.example_weight)
    assert loaded.metadata["num_examples"] == 1
    assert loaded.metadata["tokenizer"]["coeff_nums"] == [-1, 1, 2]


def test_save_and_load_preprocessed_supervised_dataset_round_trips(tmp_path):
    tokenizer = _tokenizer()
    dataset = preprocess_supervised_rows(
        [_row(weight=1.0), _row(weight=3.0)],
        tokenizer,
        source_len=12,
        target_len=12,
    )
    arrays_path = tmp_path / "dataset.npz"
    metadata_path = tmp_path / "dataset.metadata.json"

    save_preprocessed_supervised_dataset(dataset, arrays_path, metadata_path)
    loaded = load_preprocessed_supervised_dataset(arrays_path, metadata_path)

    assert np.array_equal(loaded.source_ids, dataset.source_ids)
    assert np.array_equal(loaded.decoder_input_ids, dataset.decoder_input_ids)
    assert np.array_equal(loaded.target_ids, dataset.target_ids)
    assert np.array_equal(loaded.target_mask, dataset.target_mask)
    assert np.array_equal(loaded.example_weight, dataset.example_weight)
    assert loaded.metadata == dataset.metadata
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd python
uv run pytest tests/test_supervised_dataset.py -q
```

Expected: FAIL because `preprocess_supervised_jsonl`, `save_preprocessed_supervised_dataset`, and `load_preprocessed_supervised_dataset` are not defined or exported.

- [ ] **Step 3: Implement JSONL, save, and load functions**

Modify `python/gristmill_symbolics/supervised_dataset.py`:

```python
__all__ = (
    "PreprocessedSupervisedDataset",
    "SupervisedDatasetError",
    "load_preprocessed_supervised_dataset",
    "preprocess_supervised_jsonl",
    "preprocess_supervised_rows",
    "save_preprocessed_supervised_dataset",
)
```

Add these functions after `preprocess_supervised_rows`:

```python
def preprocess_supervised_jsonl(
    input_path: str | Path,
    arrays_path: str | Path,
    metadata_path: str | Path,
    tokenizer: FlatDefinitionTokenizer,
    *,
    source_len: int,
    target_len: int,
) -> PreprocessedSupervisedDataset:
    dataset = preprocess_supervised_rows(
        _read_jsonl_rows(input_path),
        tokenizer,
        source_len=source_len,
        target_len=target_len,
    )
    save_preprocessed_supervised_dataset(dataset, arrays_path, metadata_path)
    return dataset


def save_preprocessed_supervised_dataset(
    dataset: PreprocessedSupervisedDataset,
    arrays_path: str | Path,
    metadata_path: str | Path,
) -> None:
    np.savez_compressed(
        arrays_path,
        source_ids=dataset.source_ids,
        decoder_input_ids=dataset.decoder_input_ids,
        target_ids=dataset.target_ids,
        target_mask=dataset.target_mask,
        example_weight=dataset.example_weight,
    )
    Path(metadata_path).write_text(
        json.dumps(dataset.metadata, indent=2, sort_keys=True) + "\n"
    )


def load_preprocessed_supervised_dataset(
    arrays_path: str | Path,
    metadata_path: str | Path,
) -> PreprocessedSupervisedDataset:
    metadata = json.loads(Path(metadata_path).read_text())
    with np.load(arrays_path) as arrays:
        return PreprocessedSupervisedDataset(
            source_ids=arrays["source_ids"].astype(np.int32, copy=False),
            decoder_input_ids=arrays["decoder_input_ids"].astype(np.int32, copy=False),
            target_ids=arrays["target_ids"].astype(np.int32, copy=False),
            target_mask=arrays["target_mask"].astype(np.bool_, copy=False),
            example_weight=arrays["example_weight"].astype(np.float32, copy=False),
            metadata=metadata,
        )


def _read_jsonl_rows(input_path: str | Path):
    for row_number, line in enumerate(Path(input_path).read_text().splitlines(), start=1):
        if not line.strip():
            raise SupervisedDatasetError(f"row {row_number}: empty JSONL row")
        try:
            yield json.loads(line)
        except json.JSONDecodeError as exc:
            raise SupervisedDatasetError(
                f"row {row_number}: invalid JSON: {exc.msg}"
            ) from exc
```

- [ ] **Step 4: Run tests to verify Task 2 passes**

Run:

```bash
cd python
uv run pytest tests/test_supervised_dataset.py -q
```

Expected: PASS for Task 1 and Task 2 tests.

### Task 3: Strict Error Handling

**Files:**
- Modify: `python/tests/test_supervised_dataset.py`
- Modify: `python/gristmill_symbolics/supervised_dataset.py` only if these tests expose message or row-number gaps

- [ ] **Step 1: Add failing tests for strict invalid rows and over-length rows**

Append to `python/tests/test_supervised_dataset.py`:

```python
@pytest.mark.parametrize(
    ("line", "match"),
    [
        ("not json\n", "row 1: invalid JSON"),
        ("\n", "row 1: empty JSONL row"),
        (json.dumps({"target": _computation_json(), "weight": 1.0}) + "\n", "row 1: missing 'source'"),
        (json.dumps({"source": _computation_json(), "weight": 1.0}) + "\n", "row 1: missing 'target'"),
        (json.dumps({"source": _computation_json(), "target": _computation_json()}) + "\n", "row 1: missing 'weight'"),
        (json.dumps({**_row(), "weight": -1.0}) + "\n", "row 1: weight must be"),
        (json.dumps({**_row(), "weight": float("inf")}) + "\n", "row 1: weight must be"),
    ],
)
def test_preprocess_supervised_jsonl_fails_strictly(tmp_path, line, match):
    input_path = tmp_path / "rows.jsonl"
    input_path.write_text(line)

    with pytest.raises(SupervisedDatasetError, match=match):
        preprocess_supervised_jsonl(
            input_path,
            tmp_path / "dataset.npz",
            tmp_path / "dataset.metadata.json",
            _tokenizer(),
            source_len=12,
            target_len=12,
        )


def test_preprocess_supervised_rows_reports_source_over_length():
    with pytest.raises(SupervisedDatasetError, match="row 1: .*exceeds length"):
        preprocess_supervised_rows(
            [_row()],
            _tokenizer(),
            source_len=2,
            target_len=12,
        )


def test_preprocess_supervised_rows_reports_target_over_length():
    with pytest.raises(SupervisedDatasetError, match="row 1: target content length"):
        preprocess_supervised_rows(
            [_row()],
            _tokenizer(),
            source_len=12,
            target_len=2,
        )


def test_preprocess_supervised_rows_reports_second_row_number():
    rows = [
        _row(weight=1.0),
        {**_row(weight=1.0), "weight": -1.0},
    ]

    with pytest.raises(SupervisedDatasetError, match="row 2: weight must be"):
        preprocess_supervised_rows(
            rows,
            _tokenizer(),
            source_len=12,
            target_len=12,
        )
```

- [ ] **Step 2: Run tests to verify failures or current pass**

Run:

```bash
cd python
uv run pytest tests/test_supervised_dataset.py -q
```

Expected: Either PASS if Task 1 and 2 already provide exact strict behavior, or FAIL with a precise message mismatch.

- [ ] **Step 3: Fix any strict behavior gaps**

If source over-length is not wrapped with `row 1`, keep the `except TokenizerError` block in `_preprocess_row` exactly as shown in Task 1:

```python
except (KeyError, TypeError, TokenizerError) as exc:
    raise SupervisedDatasetError(f"row {row_number}: {exc}") from exc
```

- [ ] **Step 4: Run tests to verify Task 3 passes**

Run:

```bash
cd python
uv run pytest tests/test_supervised_dataset.py -q
```

Expected: PASS for all supervised dataset tests.

### Task 4: Fixed-Shape Batch Iterator

**Files:**
- Modify: `python/gristmill_symbolics/supervised_dataset.py`
- Modify: `python/tests/test_supervised_dataset.py`

- [ ] **Step 1: Add failing tests for fixed-shape batching**

Append to `python/tests/test_supervised_dataset.py`:

```python
from gristmill_symbolics.supervised_dataset import iter_supervised_batches


def test_iter_supervised_batches_preserves_fixed_batch_shapes():
    dataset = preprocess_supervised_rows(
        [_row(weight=1.0), _row(weight=2.0), _row(weight=3.0)],
        _tokenizer(),
        source_len=12,
        target_len=12,
    )

    batches = list(iter_supervised_batches(dataset, batch_size=2))

    assert len(batches) == 1
    batch = batches[0]
    assert batch["source_ids"].shape == (2, 12)
    assert batch["decoder_input_ids"].shape == (2, 12)
    assert batch["target_ids"].shape == (2, 12)
    assert batch["target_mask"].shape == (2, 12)
    assert batch["example_weight"].shape == (2,)
    assert np.array_equal(batch["example_weight"], np.asarray([1.0, 2.0], dtype=np.float32))


def test_iter_supervised_batches_can_shuffle_with_numpy_generator():
    dataset = preprocess_supervised_rows(
        [_row(weight=1.0), _row(weight=2.0), _row(weight=3.0), _row(weight=4.0)],
        _tokenizer(),
        source_len=12,
        target_len=12,
    )
    rng = np.random.default_rng(0)

    batches = list(iter_supervised_batches(dataset, batch_size=2, shuffle=True, rng=rng))

    assert len(batches) == 2
    seen = np.concatenate([batch["example_weight"] for batch in batches])
    assert sorted(seen.tolist()) == [1.0, 2.0, 3.0, 4.0]
    assert not np.array_equal(seen, np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd python
uv run pytest tests/test_supervised_dataset.py -q
```

Expected: FAIL because `iter_supervised_batches` is not defined or exported.

- [ ] **Step 3: Implement fixed-shape batch iterator**

Modify `python/gristmill_symbolics/supervised_dataset.py`:

```python
__all__ = (
    "PreprocessedSupervisedDataset",
    "SupervisedDatasetError",
    "iter_supervised_batches",
    "load_preprocessed_supervised_dataset",
    "preprocess_supervised_jsonl",
    "preprocess_supervised_rows",
    "save_preprocessed_supervised_dataset",
)
```

Add after the load function:

```python
def iter_supervised_batches(
    dataset: PreprocessedSupervisedDataset,
    *,
    batch_size: int,
    shuffle: bool = False,
    rng: np.random.Generator | None = None,
):
    if batch_size <= 0:
        raise SupervisedDatasetError("batch_size must be positive")

    order = np.arange(dataset.num_examples)
    if shuffle:
        rng = rng or np.random.default_rng()
        rng.shuffle(order)

    stop = dataset.num_examples - (dataset.num_examples % batch_size)
    for start in range(0, stop, batch_size):
        indices = order[start : start + batch_size]
        yield {
            "source_ids": dataset.source_ids[indices],
            "decoder_input_ids": dataset.decoder_input_ids[indices],
            "target_ids": dataset.target_ids[indices],
            "target_mask": dataset.target_mask[indices],
            "example_weight": dataset.example_weight[indices],
        }
```

- [ ] **Step 4: Run tests to verify Task 4 passes**

Run:

```bash
cd python
uv run pytest tests/test_supervised_dataset.py -q
```

Expected: PASS for all supervised dataset tests.

### Task 5: Focused And Broad Verification

**Files:**
- Verify: `python/gristmill_symbolics/supervised_dataset.py`
- Verify: `python/tests/test_supervised_dataset.py`
- Verify existing flat-token tests remain green

- [ ] **Step 1: Run focused supervised dataset tests**

Run:

```bash
cd python
uv run pytest tests/test_supervised_dataset.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run existing Python tests**

Run:

```bash
cd python
uv run pytest -q
```

Expected: all Python tests pass.

- [ ] **Step 3: Run Rust tests**

Run:

```bash
cargo test
```

Expected: all Rust tests pass.

- [ ] **Step 4: Check whitespace**

Run:

```bash
git diff --check
```

Expected: no output and exit code 0.

- [ ] **Step 5: Review scope before commit**

Run:

```bash
git status --short
git diff --stat
```

Expected changed files:
- `docs/superpowers/specs/2026-07-08-preprocessed-supervised-dataset-story.md`
- `docs/superpowers/plans/2026-07-08-preprocessed-supervised-dataset.md`
- `python/gristmill_symbolics/supervised_dataset.py`
- `python/tests/test_supervised_dataset.py`

No CLI, checkpoint, validation-loop, sampling, reward, or tokenizer redesign files should be changed.
