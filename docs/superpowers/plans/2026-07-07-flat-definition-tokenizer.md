# Flat Definition Tokenizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a definition-level flat tokenizer that produces fixed-vocabulary integer IDs, round-trips TensorDef snapshot dicts, and returns right-padded NumPy arrays for ML input.

**Architecture:** Add one public Python submodule, `gristmill_symbolics.tokenizer`, containing `FlatDefinitionTokenizer` and `TokenizerError`. Keep the package root rewrite-binding exports unchanged. The tokenizer owns its configured vocabulary bounds and coefficient sets, performs strict Python parsing for raw streams, and uses NumPy only at the padded encode/decode boundary.

**Tech Stack:** Python 3.11, NumPy, pytest, uv.

---

## File Structure

- Create `python/gristmill_symbolics/tokenizer.py`: public tokenizer class, tokenizer-specific error type, fixed-vocabulary construction, raw encode/decode, padded NumPy encode/decode, and token-name lookup.
- Create `python/tests/test_flat_definition_tokenizer.py`: focused tokenizer tests for vocabulary names, raw round-trip, strict raw validation, right padding, padded decode validation, and overlong sequence rejection.
- Leave `python/gristmill_symbolics/__init__.py` unchanged so the existing thin rewrite-binding root export test remains valid.

## Task 1: Raw Tokenizer Tests

**Files:**
- Create: `python/tests/test_flat_definition_tokenizer.py`

- [ ] **Step 1: Write the failing raw tokenizer tests**

Create `python/tests/test_flat_definition_tokenizer.py` with:

```python
import pytest

from gristmill_symbolics.tokenizer import (
    FlatDefinitionTokenizer,
    TokenizerError,
)


def _tokenizer() -> FlatDefinitionTokenizer:
    return FlatDefinitionTokenizer(
        max_range_id=3,
        max_tensor_id=4,
        max_index_id=5,
        coeff_nums=(-1, 1, 2),
        coeff_dens=(1, 2, 3),
    )


def _definition() -> dict[str, object]:
    return {
        "base": 3,
        "ext_indices": [
            {"id": 0, "range": 0},
            {"id": 1, "range": 0},
        ],
        "terms": [
            {
                "coeff": {"numer": 1, "denom": 1},
                "sum_indices": [{"id": 2, "range": 0}],
                "factors": [
                    {"tensor": 0, "indices": [0, 2]},
                    {"tensor": 1, "indices": [2, 1]},
                ],
            },
            {
                "coeff": {"numer": -1, "denom": 2},
                "sum_indices": [],
                "factors": [{"tensor": 2, "indices": [0]}],
            },
        ],
    }


def test_token_names_are_inspectable_for_configured_vocabulary():
    tokenizer = _tokenizer()

    ids = tokenizer.encode_definition(_definition())

    assert tokenizer.pad_token_id == 0
    assert [tokenizer.token_name(token_id) for token_id in ids] == [
        "def_start",
        "tensorid3",
        "indexid0",
        "rangeid0",
        "indexid1",
        "rangeid0",
        "coeff_num1",
        "coeff_den1",
        "indexid2",
        "rangeid0",
        "tensorid0",
        "indexid0",
        "indexid2",
        "tensorid1",
        "indexid2",
        "indexid1",
        "coeff_num-1",
        "coeff_den2",
        "tensorid2",
        "indexid0",
        "def_end",
    ]


def test_definition_round_trips_through_raw_integer_tokens():
    tokenizer = _tokenizer()
    definition = _definition()

    ids = tokenizer.encode_definition(definition)
    decoded = tokenizer.decode_definition(ids)

    assert all(type(token_id) is int for token_id in ids)
    assert decoded == definition


def test_encode_rejects_unsupported_values_and_malformed_definitions():
    tokenizer = _tokenizer()
    definition = _definition()

    unsupported_tensor = dict(definition, base=9)
    with pytest.raises(TokenizerError, match="base.*outside supported range"):
        tokenizer.encode_definition(unsupported_tensor)

    unsupported_coeff = {
        **definition,
        "terms": [
            {
                **definition["terms"][0],
                "coeff": {"numer": 7, "denom": 1},
            }
        ],
    }
    with pytest.raises(TokenizerError, match="coeff_num7"):
        tokenizer.encode_definition(unsupported_coeff)

    missing_terms = {
        "base": 3,
        "ext_indices": [],
    }
    with pytest.raises(TokenizerError, match="missing key.*terms"):
        tokenizer.encode_definition(missing_terms)

    malformed_coeff = {
        **definition,
        "terms": [
            {
                **definition["terms"][0],
                "coeff": [1, 1],
            }
        ],
    }
    with pytest.raises(TokenizerError, match="coeff.*dict"):
        tokenizer.encode_definition(malformed_coeff)


def test_decode_rejects_malformed_raw_streams():
    tokenizer = _tokenizer()
    valid = tokenizer.encode_definition(_definition())

    with pytest.raises(TokenizerError, match="def_start"):
        tokenizer.decode_definition(valid[1:])

    with pytest.raises(TokenizerError, match="raw token stream cannot contain pad"):
        tokenizer.decode_definition([tokenizer.pad_token_id, *valid])

    with pytest.raises(TokenizerError, match="unknown token id"):
        tokenizer.decode_definition([10_000])

    missing_external_range = valid.copy()
    del missing_external_range[3]
    with pytest.raises(TokenizerError, match="external index rangeid"):
        tokenizer.decode_definition(missing_external_range)

    denominator_where_term_should_start = valid.copy()
    denominator_where_term_should_start[6] = valid[7]
    with pytest.raises(TokenizerError, match="coeff_num or def_end"):
        tokenizer.decode_definition(denominator_where_term_should_start)
```

- [ ] **Step 2: Run the raw tokenizer tests to verify they fail**

Run from `python/`:

```bash
uv run pytest tests/test_flat_definition_tokenizer.py -q
```

Expected: FAIL during import with `ModuleNotFoundError: No module named 'gristmill_symbolics.tokenizer'`.

## Task 2: Raw Tokenizer Implementation

**Files:**
- Create: `python/gristmill_symbolics/tokenizer.py`
- Test: `python/tests/test_flat_definition_tokenizer.py`

- [ ] **Step 1: Implement raw fixed-vocabulary encode/decode**

Create `python/gristmill_symbolics/tokenizer.py` with:

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral


__all__ = ("FlatDefinitionTokenizer", "TokenizerError")


class TokenizerError(ValueError):
    """Raised when a definition or token stream cannot be tokenized."""


@dataclass(frozen=True)
class _TokenSpec:
    name: str
    kind: str
    value: int | None = None


class FlatDefinitionTokenizer:
    def __init__(
        self,
        *,
        max_range_id: int,
        max_tensor_id: int,
        max_index_id: int,
        coeff_nums: Sequence[int],
        coeff_dens: Sequence[int],
    ):
        self.max_range_id = _nonnegative_int("max_range_id", max_range_id)
        self.max_tensor_id = _nonnegative_int("max_tensor_id", max_tensor_id)
        self.max_index_id = _nonnegative_int("max_index_id", max_index_id)
        self.coeff_nums = _unique_int_tuple("coeff_nums", coeff_nums)
        self.coeff_dens = _unique_positive_int_tuple("coeff_dens", coeff_dens)

        self._token_specs: list[_TokenSpec] = []
        self._token_ids: dict[str, int] = {}
        self._add_token("pad", "pad")
        self._add_token("def_start", "def_start")
        self._add_token("def_end", "def_end")
        for value in range(self.max_range_id + 1):
            self._add_token(f"rangeid{value}", "rangeid", value)
        for value in range(self.max_tensor_id + 1):
            self._add_token(f"tensorid{value}", "tensorid", value)
        for value in range(self.max_index_id + 1):
            self._add_token(f"indexid{value}", "indexid", value)
        for value in self.coeff_nums:
            self._add_token(f"coeff_num{value}", "coeff_num", value)
        for value in self.coeff_dens:
            self._add_token(f"coeff_den{value}", "coeff_den", value)

    @property
    def pad_token_id(self) -> int:
        return self._token_ids["pad"]

    @property
    def vocab_size(self) -> int:
        return len(self._token_specs)

    def token_name(self, token_id: int) -> str:
        return self._spec_for_token_id("token_id", token_id).name

    def encode_definition(self, definition: Mapping[str, object]) -> list[int]:
        definition = _mapping_with_keys(
            "definition",
            definition,
            ("base", "ext_indices", "terms"),
        )
        ids = [
            self._token_ids["def_start"],
            self._tensor_token_id("base", definition["base"]),
        ]
        for index in _sequence("definition.ext_indices", definition["ext_indices"]):
            ids.extend(self._encode_index("external index", index))
        for term in _sequence("definition.terms", definition["terms"]):
            ids.extend(self._encode_term(term))
        ids.append(self._token_ids["def_end"])
        return ids

    def decode_definition(self, ids: Sequence[int]) -> dict[str, object]:
        specs = self._token_specs_for_stream(ids, allow_pad=False)
        pos = 0

        def peek() -> _TokenSpec | None:
            if pos >= len(specs):
                return None
            return specs[pos]

        def consume(kind: str, description: str) -> _TokenSpec:
            nonlocal pos
            if pos >= len(specs):
                raise TokenizerError(
                    f"expected {description}, reached end of token stream"
                )
            spec = specs[pos]
            if spec.kind != kind:
                raise TokenizerError(
                    f"expected {description}, got {spec.name}"
                )
            pos += 1
            return spec

        consume("def_start", "def_start")
        base = consume("tensorid", "base tensorid").value
        ext_indices: list[dict[str, int]] = []
        while (spec := peek()) is not None and spec.kind == "indexid":
            index_id = consume("indexid", "external indexid").value
            range_id = consume("rangeid", "external index rangeid").value
            ext_indices.append({"id": index_id, "range": range_id})

        terms: list[dict[str, object]] = []
        while (spec := peek()) is not None and spec.kind == "coeff_num":
            numer = consume("coeff_num", "coeff_num").value
            denom = consume("coeff_den", "coeff_den").value
            sum_indices: list[dict[str, int]] = []
            while (spec := peek()) is not None and spec.kind == "indexid":
                next_pos = pos + 1
                if (
                    next_pos >= len(specs)
                    or specs[next_pos].kind != "rangeid"
                ):
                    raise TokenizerError(
                        "sum index must be encoded as indexid/rangeid before factors"
                    )
                index_id = consume("indexid", "sum indexid").value
                range_id = consume("rangeid", "sum index rangeid").value
                sum_indices.append({"id": index_id, "range": range_id})

            factors: list[dict[str, object]] = []
            while (spec := peek()) is not None and spec.kind == "tensorid":
                tensor_id = consume("tensorid", "factor tensorid").value
                factor_indices: list[int] = []
                while (spec := peek()) is not None and spec.kind == "indexid":
                    factor_indices.append(consume("indexid", "factor indexid").value)
                factors.append({"tensor": tensor_id, "indices": factor_indices})

            terms.append(
                {
                    "coeff": {"numer": numer, "denom": denom},
                    "sum_indices": sum_indices,
                    "factors": factors,
                }
            )

        if (spec := peek()) is not None and spec.kind != "def_end":
            raise TokenizerError(
                f"expected coeff_num or def_end, got {spec.name}"
            )
        consume("def_end", "def_end")
        if (spec := peek()) is not None:
            raise TokenizerError(f"unexpected token after def_end: {spec.name}")

        return {
            "base": base,
            "ext_indices": ext_indices,
            "terms": terms,
        }

    def _add_token(
        self,
        name: str,
        kind: str,
        value: int | None = None,
    ) -> None:
        if name in self._token_ids:
            raise TokenizerError(f"duplicate token name {name!r}")
        self._token_ids[name] = len(self._token_specs)
        self._token_specs.append(_TokenSpec(name=name, kind=kind, value=value))

    def _token_id(self, name: str) -> int:
        try:
            return self._token_ids[name]
        except KeyError as exc:
            raise TokenizerError(f"unsupported token {name}") from exc

    def _spec_for_token_id(self, name: str, token_id: int) -> _TokenSpec:
        token_id = _strict_int(name, token_id)
        if token_id < 0 or token_id >= len(self._token_specs):
            raise TokenizerError(f"unknown token id {token_id}")
        return self._token_specs[token_id]

    def _token_specs_for_stream(
        self,
        ids: Sequence[int],
        *,
        allow_pad: bool,
    ) -> list[_TokenSpec]:
        if isinstance(ids, (str, bytes)):
            raise TokenizerError("token stream must be a sequence of integer IDs")
        try:
            raw_ids = list(ids)
        except TypeError as exc:
            raise TokenizerError(
                "token stream must be a sequence of integer IDs"
            ) from exc
        if not raw_ids:
            raise TokenizerError("token stream must not be empty")

        specs: list[_TokenSpec] = []
        for index, token_id in enumerate(raw_ids):
            spec = self._spec_for_token_id(f"token[{index}]", token_id)
            if not allow_pad and spec.kind == "pad":
                raise TokenizerError("raw token stream cannot contain pad")
            specs.append(spec)
        return specs

    def _encode_term(self, term: object) -> list[int]:
        term = _mapping_with_keys(
            "term",
            term,
            ("coeff", "sum_indices", "factors"),
        )
        coeff = _mapping_with_keys(
            "term.coeff",
            term["coeff"],
            ("numer", "denom"),
        )
        ids = [
            self._coeff_num_token_id(coeff["numer"]),
            self._coeff_den_token_id(coeff["denom"]),
        ]
        for index in _sequence("term.sum_indices", term["sum_indices"]):
            ids.extend(self._encode_index("sum index", index))
        for factor in _sequence("term.factors", term["factors"]):
            ids.extend(self._encode_factor(factor))
        return ids

    def _encode_index(self, name: str, index: object) -> list[int]:
        index = _mapping_with_keys(name, index, ("id", "range"))
        return [
            self._index_token_id(f"{name}.id", index["id"]),
            self._range_token_id(f"{name}.range", index["range"]),
        ]

    def _encode_factor(self, factor: object) -> list[int]:
        factor = _mapping_with_keys("factor", factor, ("tensor", "indices"))
        ids = [self._tensor_token_id("factor.tensor", factor["tensor"])]
        for index in _sequence("factor.indices", factor["indices"]):
            ids.append(self._index_token_id("factor index", index))
        return ids

    def _range_token_id(self, name: str, value: object) -> int:
        value = _bounded_int(name, value, upper=self.max_range_id)
        return self._token_id(f"rangeid{value}")

    def _tensor_token_id(self, name: str, value: object) -> int:
        value = _bounded_int(name, value, upper=self.max_tensor_id)
        return self._token_id(f"tensorid{value}")

    def _index_token_id(self, name: str, value: object) -> int:
        value = _bounded_int(name, value, upper=self.max_index_id)
        return self._token_id(f"indexid{value}")

    def _coeff_num_token_id(self, value: object) -> int:
        value = _strict_int("coeff.numer", value)
        if value not in self.coeff_nums:
            raise TokenizerError(f"unsupported coefficient numerator coeff_num{value}")
        return self._token_id(f"coeff_num{value}")

    def _coeff_den_token_id(self, value: object) -> int:
        value = _strict_int("coeff.denom", value)
        if value not in self.coeff_dens:
            raise TokenizerError(
                f"unsupported coefficient denominator coeff_den{value}"
            )
        return self._token_id(f"coeff_den{value}")


def _strict_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TokenizerError(f"{name} must be an integer")
    return int(value)


def _nonnegative_int(name: str, value: object) -> int:
    value = _strict_int(name, value)
    if value < 0:
        raise TokenizerError(f"{name} must be nonnegative")
    return value


def _bounded_int(name: str, value: object, *, upper: int) -> int:
    value = _strict_int(name, value)
    if value < 0 or value > upper:
        raise TokenizerError(f"{name} value {value} is outside supported range 0..{upper}")
    return value


def _unique_int_tuple(name: str, values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TokenizerError(f"{name} must be a sequence of integers")
    result = tuple(_strict_int(f"{name}[{index}]", value) for index, value in enumerate(values))
    if not result:
        raise TokenizerError(f"{name} must not be empty")
    if len(set(result)) != len(result):
        raise TokenizerError(f"{name} contains duplicate values")
    return result


def _unique_positive_int_tuple(name: str, values: Sequence[int]) -> tuple[int, ...]:
    result = _unique_int_tuple(name, values)
    for value in result:
        if value <= 0:
            raise TokenizerError(f"{name} values must be positive")
    return result


def _mapping_with_keys(
    name: str,
    value: object,
    expected_keys: tuple[str, ...],
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TokenizerError(f"{name} must be a dict")
    actual = set(value)
    expected = set(expected_keys)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise TokenizerError(f"{name} missing key(s): {', '.join(missing)}")
    if extra:
        raise TokenizerError(f"{name} has unsupported key(s): {', '.join(extra)}")
    return value


def _sequence(name: str, value: object) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TokenizerError(f"{name} must be a list")
    return value
```

- [ ] **Step 2: Run the raw tokenizer tests**

Run from `python/`:

```bash
uv run pytest tests/test_flat_definition_tokenizer.py -q
```

Expected: PASS for the four raw tokenizer tests.

- [ ] **Step 3: Run the existing binding smoke tests**

Run from `python/`:

```bash
uv run pytest tests/test_bindings.py -q
```

Expected: PASS. The root `gristmill_symbolics.__all__` remains unchanged.

- [ ] **Step 4: Commit the raw tokenizer**

Run from the repository worktree root:

```bash
git add python/gristmill_symbolics/tokenizer.py python/tests/test_flat_definition_tokenizer.py
git commit -m "feat: add flat definition tokenizer"
```

## Task 3: Padded NumPy Tokenizer Tests

**Files:**
- Modify: `python/tests/test_flat_definition_tokenizer.py`

- [ ] **Step 1: Add NumPy import and padded-tokenizer tests**

Modify the top of `python/tests/test_flat_definition_tokenizer.py` to include NumPy:

```python
import numpy as np
import pytest
```

Append these tests to `python/tests/test_flat_definition_tokenizer.py`:

```python

def test_padded_encode_uses_right_padding_and_round_trips():
    tokenizer = _tokenizer()
    definition = _definition()
    raw = tokenizer.encode_definition(definition)

    ids, mask = tokenizer.encode_definition_padded(definition, max_len=len(raw) + 3)

    assert ids.dtype == np.int32
    assert mask.dtype == np.bool_
    np.testing.assert_array_equal(ids[: len(raw)], np.asarray(raw, dtype=np.int32))
    np.testing.assert_array_equal(
        ids[len(raw) :],
        np.full((3,), tokenizer.pad_token_id, dtype=np.int32),
    )
    np.testing.assert_array_equal(mask[: len(raw)], np.ones((len(raw),), dtype=np.bool_))
    np.testing.assert_array_equal(mask[len(raw) :], np.zeros((3,), dtype=np.bool_))
    assert tokenizer.decode_definition_padded(ids, mask) == definition


def test_padded_encode_rejects_overlong_definition():
    tokenizer = _tokenizer()
    definition = _definition()
    raw = tokenizer.encode_definition(definition)

    with pytest.raises(TokenizerError, match="exceeds max_len"):
        tokenizer.encode_definition_padded(definition, max_len=len(raw) - 1)


def test_padded_decode_rejects_invalid_padding_and_mask():
    tokenizer = _tokenizer()
    ids, mask = tokenizer.encode_definition_padded(_definition(), max_len=32)

    non_right_padded_mask = mask.copy()
    non_right_padded_mask[0] = False
    non_right_padded_mask[1] = True
    with pytest.raises(TokenizerError, match="right-padding"):
        tokenizer.decode_definition_padded(ids, non_right_padded_mask)

    non_pad_tail = ids.copy()
    non_pad_tail[-1] = ids[0]
    with pytest.raises(TokenizerError, match="pad_token_id"):
        tokenizer.decode_definition_padded(non_pad_tail, mask)

    with pytest.raises(TokenizerError, match="same shape"):
        tokenizer.decode_definition_padded(ids, mask[:-1])

    with pytest.raises(TokenizerError, match="boolean"):
        tokenizer.decode_definition_padded(ids, mask.astype(np.int32))
```

- [ ] **Step 2: Run the tokenizer tests to verify padded API fails**

Run from `python/`:

```bash
uv run pytest tests/test_flat_definition_tokenizer.py -q
```

Expected: FAIL with `AttributeError: 'FlatDefinitionTokenizer' object has no attribute 'encode_definition_padded'`.

## Task 4: Padded NumPy Tokenizer Implementation

**Files:**
- Modify: `python/gristmill_symbolics/tokenizer.py`
- Test: `python/tests/test_flat_definition_tokenizer.py`

- [ ] **Step 1: Import NumPy**

Add this import near the top of `python/gristmill_symbolics/tokenizer.py`:

```python
import numpy as np
```

- [ ] **Step 2: Add padded encode/decode methods**

Add these methods inside `FlatDefinitionTokenizer`, immediately after `decode_definition`:

```python
    def encode_definition_padded(
        self,
        definition: Mapping[str, object],
        *,
        max_len: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        max_len = _positive_int("max_len", max_len)
        raw_ids = self.encode_definition(definition)
        if len(raw_ids) > max_len:
            raise TokenizerError(
                f"encoded definition length {len(raw_ids)} exceeds max_len {max_len}"
            )

        ids = np.full((max_len,), self.pad_token_id, dtype=np.int32)
        mask = np.zeros((max_len,), dtype=np.bool_)
        ids[: len(raw_ids)] = np.asarray(raw_ids, dtype=np.int32)
        mask[: len(raw_ids)] = True
        return ids, mask

    def decode_definition_padded(
        self,
        ids: Sequence[int],
        mask: Sequence[bool],
    ) -> dict[str, object]:
        ids_array = np.asarray(ids)
        mask_array = np.asarray(mask)
        if ids_array.ndim != 1:
            raise TokenizerError("padded ids must be a 1D array")
        if mask_array.ndim != 1:
            raise TokenizerError("padding mask must be a 1D array")
        if ids_array.shape != mask_array.shape:
            raise TokenizerError("padded ids and mask must have the same shape")
        if not np.issubdtype(ids_array.dtype, np.integer):
            raise TokenizerError("padded ids must have integer dtype")
        if mask_array.dtype != np.bool_:
            raise TokenizerError("padding mask must have boolean dtype")

        mask_list = mask_array.tolist()
        first_padding = len(mask_list)
        for index, active in enumerate(mask_list):
            if not active:
                first_padding = index
                break
        if any(mask_list[first_padding:]):
            raise TokenizerError("padding mask must describe right-padding")

        padding_ids = ids_array[first_padding:]
        if padding_ids.size and not bool(np.all(padding_ids == self.pad_token_id)):
            raise TokenizerError("masked padding ids must equal pad_token_id")

        raw_ids = ids_array[:first_padding].astype(np.int64).tolist()
        return self.decode_definition(raw_ids)
```

- [ ] **Step 3: Add positive integer validation helper**

Add this helper below `_nonnegative_int`:

```python

def _positive_int(name: str, value: object) -> int:
    value = _strict_int(name, value)
    if value <= 0:
        raise TokenizerError(f"{name} must be positive")
    return value
```

- [ ] **Step 4: Run the tokenizer tests**

Run from `python/`:

```bash
uv run pytest tests/test_flat_definition_tokenizer.py -q
```

Expected: PASS for raw and padded tokenizer tests.

- [ ] **Step 5: Commit padded tokenization**

Run from the repository worktree root:

```bash
git add python/gristmill_symbolics/tokenizer.py python/tests/test_flat_definition_tokenizer.py
git commit -m "feat: add padded definition tokenization"
```

## Task 5: Final Verification

**Files:**
- Verify: `python/gristmill_symbolics/tokenizer.py`
- Verify: `python/tests/test_flat_definition_tokenizer.py`
- Verify: `python/tests/test_bindings.py`

- [ ] **Step 1: Run focused tokenizer tests**

Run from `python/`:

```bash
uv run pytest tests/test_flat_definition_tokenizer.py -q
```

Expected: PASS.

- [ ] **Step 2: Run all current Python tests**

Run from `python/`:

```bash
uv run pytest -q
```

Expected: PASS for the binding tests and tokenizer tests.

- [ ] **Step 3: Inspect git status**

Run from the repository worktree root:

```bash
git status --short --branch
```

Expected: clean `refactor/python-ml-rebuild` branch after the two implementation commits.
