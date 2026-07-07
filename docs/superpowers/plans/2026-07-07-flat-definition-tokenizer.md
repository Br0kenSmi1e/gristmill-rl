# Flat Definition Tokenizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a definition-level flat tokenizer that produces fixed-vocabulary integer IDs and round-trips TensorDef snapshot dicts, individually, as raw concatenated sequences, or as right-padded concatenated definition sequences.

**Architecture:** Add one public Python submodule, `gristmill_symbolics.tokenizer`, containing `FlatDefinitionTokenizer` and `TokenizerError`. Keep the package root rewrite-binding exports unchanged. The tokenizer stores vocabulary configuration as provided, builds its fixed vocabulary from those values, performs strict parsing for raw token streams, assumes snapshot-like encode input, reserves `pad` as token id 0, includes thin list-based right-padding wrappers for definition sequences, and leaves model-facing batching, masks, array conversion, TensorComputation wrapping, and full schema validation outside this layer.

**Tech Stack:** Python 3.11, pytest, uv.

---

Scope note: Task 4 supersedes any earlier initial-implementation snippets that
show exact dictionary-key or list-only validation in `encode_definition`. The
current accepted boundary is vocabulary and raw token-stream validation in the
tokenizer, with full snapshot schema validation deferred to CLI/data-loading
work.

Scope note: Task 5 supersedes any earlier constructor snippets that show
coercion, deduplication, or validation of constructor configuration. The current
accepted boundary stores constructor inputs directly and lets invalid
configuration fail naturally during vocabulary construction or tokenizer use.

Scope note: Task 6 adds raw definition-sequence encode/decode. This is not a
TensorComputation-level state API; it only concatenates already snapshot-like
definition dicts and splits concatenated raw token streams on def_start/def_end.

Scope note: Task 7 adds thin padded definition-sequence wrappers. This is not a
model-facing batch collation API; it only right-pads `encode_definitions` output
to a requested list length and strips trailing pad tokens before
`decode_definitions`.

## File Structure

- Create `python/gristmill_symbolics/tokenizer.py`: public tokenizer class, tokenizer-specific error type, fixed-vocabulary construction, raw single-definition and definition-sequence encode/decode, thin padded sequence wrappers, reserved pad token, and token-name lookup.
- Create `python/tests/test_flat_definition_tokenizer.py`: focused tokenizer tests for vocabulary names, raw round-trip, sequence round-trip, padded sequence wrappers, token-stream grammar validation, vocabulary rejection, permissive snapshot-like encode input, and single-definition raw-only API surface.
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
from operator import index as _index


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
        self.max_range_id = max_range_id
        self.max_tensor_id = max_tensor_id
        self.max_index_id = max_index_id
        self.coeff_nums = coeff_nums
        self.coeff_dens = coeff_dens

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
        ids = [
            self._token_ids["def_start"],
            self._tensor_token_id("base", definition["base"]),
        ]
        for index in definition["ext_indices"]:
            ids.extend(self._encode_index("external index", index))
        for term in definition["terms"]:
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
        token_id = _integer(name, token_id)
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
        coeff = term["coeff"]
        ids = [
            self._coeff_num_token_id(coeff["numer"]),
            self._coeff_den_token_id(coeff["denom"]),
        ]
        for index in term["sum_indices"]:
            ids.extend(self._encode_index("sum index", index))
        for factor in term["factors"]:
            ids.extend(self._encode_factor(factor))
        return ids

    def _encode_index(self, name: str, index: object) -> list[int]:
        return [
            self._index_token_id(f"{name}.id", index["id"]),
            self._range_token_id(f"{name}.range", index["range"]),
        ]

    def _encode_factor(self, factor: object) -> list[int]:
        ids = [self._tensor_token_id("factor.tensor", factor["tensor"])]
        for index in factor["indices"]:
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
        value = _integer("coeff.numer", value)
        if value not in self.coeff_nums:
            raise TokenizerError(f"unsupported coefficient numerator coeff_num{value}")
        return self._token_id(f"coeff_num{value}")

    def _coeff_den_token_id(self, value: object) -> int:
        value = _integer("coeff.denom", value)
        if value not in self.coeff_dens:
            raise TokenizerError(
                f"unsupported coefficient denominator coeff_den{value}"
            )
        return self._token_id(f"coeff_den{value}")


def _integer(name: str, value: object) -> int:
    try:
        return _index(value)
    except TypeError as exc:
        raise TokenizerError(f"{name} must be an integer") from exc


def _bounded_int(name: str, value: object, *, upper: int) -> int:
    value = _integer(name, value)
    if value < 0 or value > upper:
        raise TokenizerError(f"{name} value {value} is outside supported range 0..{upper}")
    return value
```

- [ ] **Step 2: Run the raw tokenizer tests**

Run from `python/`:

```bash
uv run pytest tests/test_flat_definition_tokenizer.py -q
```

Expected: PASS for the raw tokenizer tests.

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

## Task 3: Raw-Only API Revision

**Files:**
- Modify: `python/gristmill_symbolics/tokenizer.py`
- Modify: `python/tests/test_flat_definition_tokenizer.py`
- Modify: `docs/superpowers/specs/2026-07-07-flat-definition-tokenizer-story.md`
- Modify: `docs/superpowers/plans/2026-07-07-flat-definition-tokenizer.md`

- [ ] **Step 1: Add a single-definition raw-only API regression test**

Add a focused test asserting that `FlatDefinitionTokenizer` does not expose
`encode_definition_padded` or `decode_definition_padded`, while the existing
vocabulary test continues to assert `pad_token_id == 0`.

- [ ] **Step 2: Run the tokenizer tests to verify the old API is caught**

Run from `python/`:

```bash
uv run pytest tests/test_flat_definition_tokenizer.py -q
```

Expected before implementation: FAIL because the padded methods still exist.

- [ ] **Step 3: Remove padded encode/decode implementation**

Remove `encode_definition_padded`, `decode_definition_padded`, the NumPy import,
the now-unused positive integer helper, and single-definition padded tests. Do
not remove the reserved `pad` token or `pad_token_id`.

- [ ] **Step 4: Update story and plan docs**

Revise this story to describe raw single-definition encode/decode only. Record
that single-definition padding and model batching are outside this tokenizer,
while `pad` remains reserved for future batching.

- [ ] **Step 5: Run the tokenizer tests**

Run from `python/`:

```bash
uv run pytest tests/test_flat_definition_tokenizer.py -q
```

Expected: PASS for raw tokenizer tests.

## Task 4: Validation Boundary Revision

**Files:**
- Modify: `python/gristmill_symbolics/tokenizer.py`
- Modify: `python/tests/test_flat_definition_tokenizer.py`
- Modify: `docs/superpowers/specs/2026-07-07-flat-definition-tokenizer-story.md`
- Modify: `docs/superpowers/plans/2026-07-07-flat-definition-tokenizer.md`

- [ ] **Step 1: Add a permissive encode regression test**

Add a focused test showing that `encode_definition` accepts snapshot-like
mappings with extra keys and iterable fields such as tuples/ranges, then
normalizes through raw decode.

- [ ] **Step 2: Run the tokenizer tests to verify schema policing is caught**

Run from `python/`:

```bash
uv run pytest tests/test_flat_definition_tokenizer.py -q
```

Expected before implementation: FAIL because the current helper rejects an
extra snapshot key before tokenization.

- [ ] **Step 3: Remove schema-level encode helpers**

Remove exact-key and list-only helper usage from `encode_definition` and its
encode helpers. Keep vocabulary bounds and raw token-stream grammar checks.

- [ ] **Step 4: Update story and plan docs**

Revise this story to state that full TensorDef snapshot schema validation is a
future CLI/data-loading concern, not a tokenizer responsibility.

- [ ] **Step 5: Run the tokenizer tests**

Run from `python/`:

```bash
uv run pytest tests/test_flat_definition_tokenizer.py -q
```

Expected: PASS for raw tokenizer tests.

## Task 5: Constructor Configuration Revision

**Files:**
- Modify: `python/gristmill_symbolics/tokenizer.py`
- Modify: `python/tests/test_flat_definition_tokenizer.py`
- Modify: `docs/superpowers/specs/2026-07-07-flat-definition-tokenizer-story.md`
- Modify: `docs/superpowers/plans/2026-07-07-flat-definition-tokenizer.md`

- [ ] **Step 1: Add a constructor-preservation regression test**

Add a focused test showing that `FlatDefinitionTokenizer` stores
`max_range_id`, `max_tensor_id`, `max_index_id`, `coeff_nums`, and
`coeff_dens` exactly as provided.

- [ ] **Step 2: Run the tokenizer tests to verify constructor coercion is caught**

Run from `python/`:

```bash
uv run pytest tests/test_flat_definition_tokenizer.py -q
```

Expected before implementation: FAIL because the current constructor coerces
integer-like bounds and tuple-converts coefficient sequences.

- [ ] **Step 3: Remove constructor coercion and validation**

Assign constructor arguments directly to instance attributes. Keep
tokenization-time checks in `_integer` and `_bounded_int`.

- [ ] **Step 4: Update story and plan docs**

Revise this story to state that constructor argument validation and
normalization are outside the tokenizer boundary.

- [ ] **Step 5: Run the tokenizer tests**

Run from `python/`:

```bash
uv run pytest tests/test_flat_definition_tokenizer.py -q
```

Expected: PASS for raw tokenizer tests.

## Task 6: Definition Sequence Encode/Decode

**Files:**
- Modify: `python/gristmill_symbolics/tokenizer.py`
- Modify: `python/tests/test_flat_definition_tokenizer.py`
- Modify: `docs/superpowers/specs/2026-07-07-flat-definition-tokenizer-story.md`
- Modify: `docs/superpowers/plans/2026-07-07-flat-definition-tokenizer.md`

- [ ] **Step 1: Add sequence encode/decode tests**

Add focused tests showing that `encode_definitions` concatenates
`encode_definition` outputs, `decode_definitions` splits the concatenated stream
on def_start/def_end, empty sequences round-trip as empty lists, and malformed
or non-sequence concatenated streams are rejected.

- [ ] **Step 2: Run the tokenizer tests to verify the methods are missing**

Run from `python/`:

```bash
uv run pytest tests/test_flat_definition_tokenizer.py -q
```

Expected before implementation: FAIL because `FlatDefinitionTokenizer` has no
`encode_definitions` method.

- [ ] **Step 3: Add the sequence methods**

Implement `encode_definitions(definitions)` by extending a list with each
`encode_definition(definition)` result. Implement `decode_definitions(ids)` by
returning `[]` for empty input, validating the raw stream, slicing complete
def_start/def_end spans, and delegating each slice to `decode_definition`.

- [ ] **Step 4: Update story and plan docs**

Revise this story to include raw definition-sequence round-trip while keeping
TensorComputation-level state APIs out of scope.

- [ ] **Step 5: Run the tokenizer tests**

Run from `python/`:

```bash
uv run pytest tests/test_flat_definition_tokenizer.py -q
```

Expected: PASS for raw tokenizer tests.

## Task 7: Padded Definition Sequence Encode Wrapper

**Files:**
- Modify: `python/gristmill_symbolics/tokenizer.py`
- Modify: `python/tests/test_flat_definition_tokenizer.py`
- Modify: `docs/superpowers/specs/2026-07-07-flat-definition-tokenizer-story.md`
- Modify: `docs/superpowers/plans/2026-07-07-flat-definition-tokenizer.md`

- [ ] **Step 1: Add padded sequence wrapper tests**

Add focused tests showing that `encode_definitions_padded` calls the raw
sequence encoder and right-pads with `pad_token_id` to the requested length,
and that overlong encode requests are rejected. Generated-sequence decode with
BOS/EOS is covered by the later tokenizer generated-sequence boundary story.

- [ ] **Step 2: Run the tokenizer tests to verify the methods are missing**

Run from `python/`:

```bash
uv run pytest tests/test_flat_definition_tokenizer.py -q
```

Expected before implementation: FAIL because `FlatDefinitionTokenizer` has no
`encode_definitions_padded` method.

- [ ] **Step 3: Add the padded sequence encode wrapper**

Implement `encode_definitions_padded(definitions, *, length)` by delegating to
`encode_definitions`, rejecting raw sequences longer than `length`, and
right-padding the returned list with `pad_token_id`.

- [ ] **Step 4: Update story and plan docs**

Revise this story to include the padded definition-sequence encode wrapper while
keeping single-definition padded APIs, model batching, masks, NumPy conversion,
and TensorComputation-level state APIs out of scope.

- [ ] **Step 5: Run the tokenizer tests**

Run from `python/`:

```bash
uv run pytest tests/test_flat_definition_tokenizer.py -q
```

Expected: PASS for raw and padded-wrapper tokenizer tests.

## Task 8: Final Verification

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

Expected: clean `refactor/python-ml-rebuild` branch after the implementation
and revision commits.
