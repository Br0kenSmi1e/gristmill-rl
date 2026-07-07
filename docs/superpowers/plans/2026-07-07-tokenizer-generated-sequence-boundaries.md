# Tokenizer Generated Sequence Boundaries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add global BOS/EOS tokens and generated-sequence decode framing to the flat definition tokenizer.

**Architecture:** Keep the tokenizer as the only production module touched. BOS/EOS become fixed vocabulary tokens after PAD, raw definition encode/decode remains content-only, and generated decode validates `bos content eos pad...` before delegating content to `decode_definitions`.

**Tech Stack:** Python 3.11, pytest, uv.

---

## File Structure

- Modify `python/gristmill_symbolics/tokenizer.py`: add BOS/EOS vocabulary entries and properties, remove `decode_definitions_padded`, and add `decode_definitions_generated`.
- Modify `python/tests/test_flat_definition_tokenizer.py`: update vocabulary expectations and add generated decode behavior tests.
- Modify `docs/superpowers/specs/2026-07-07-flat-definition-tokenizer-story.md`: note that generated decode replaces padded decode after BOS/EOS.
- Modify `docs/superpowers/plans/2026-07-07-flat-definition-tokenizer.md`: remove stale padded-decode wording from the previous tokenizer plan.

## Task 1: BOS/EOS Vocabulary Contract

**Files:**
- Modify: `python/tests/test_flat_definition_tokenizer.py`
- Modify: `python/gristmill_symbolics/tokenizer.py`

- [ ] **Step 1: Write the failing vocabulary test**

Update `test_token_names_are_inspectable_for_configured_vocabulary` so it first
asserts:

```python
assert tokenizer.pad_token_id == 0
assert tokenizer.bos_token_id == 1
assert tokenizer.eos_token_id == 2
assert [tokenizer.token_name(token_id) for token_id in range(5)] == [
    "pad",
    "bos",
    "eos",
    "def_start",
    "def_end",
]
```

Leave the existing encoded-name assertion content-only, beginning with
`def_start` and ending with `def_end`.

- [ ] **Step 2: Run the focused test and verify RED**

Run from `python/`:

```bash
uv run pytest tests/test_flat_definition_tokenizer.py -q
```

Expected before implementation: FAIL because `bos_token_id` does not exist and
token IDs 1/2 are currently `def_start`/`def_end`.

- [ ] **Step 3: Implement BOS/EOS vocabulary**

In `FlatDefinitionTokenizer.__init__`, add:

```python
self._add_token("pad", "pad")
self._add_token("bos", "bos")
self._add_token("eos", "eos")
self._add_token("def_start", "def_start")
self._add_token("def_end", "def_end")
```

Add properties:

```python
@property
def bos_token_id(self) -> int:
    return self._token_ids["bos"]

@property
def eos_token_id(self) -> int:
    return self._token_ids["eos"]
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run from `python/`:

```bash
uv run pytest tests/test_flat_definition_tokenizer.py -q
```

Expected: vocabulary tests pass or remaining failures point to the generated
decode API change in Task 2.

## Task 2: Generated Definition Sequence Decode

**Files:**
- Modify: `python/tests/test_flat_definition_tokenizer.py`
- Modify: `python/gristmill_symbolics/tokenizer.py`
- Modify: `docs/superpowers/specs/2026-07-07-flat-definition-tokenizer-story.md`

- [ ] **Step 1: Write failing generated decode tests**

Replace tests that call `decode_definitions_padded` with tests for:

```python
generated = [
    tokenizer.bos_token_id,
    *raw,
    tokenizer.eos_token_id,
    tokenizer.pad_token_id,
]
assert tokenizer.decode_definitions_generated(generated) == definitions
assert tokenizer.decode_definitions_generated([
    tokenizer.bos_token_id,
    tokenizer.eos_token_id,
    tokenizer.pad_token_id,
]) == []
assert not hasattr(tokenizer, "decode_definitions_padded")
```

Add rejection checks for non-sequence input, missing BOS, missing EOS, PAD before
EOS, nested BOS in content, and non-PAD after EOS.

- [ ] **Step 2: Run the focused tests and verify RED**

Run from `python/`:

```bash
uv run pytest tests/test_flat_definition_tokenizer.py -q
```

Expected before implementation: FAIL because `decode_definitions_generated` is
missing and `decode_definitions_padded` still exists.

- [ ] **Step 3: Implement generated decode**

Remove `decode_definitions_padded` and add:

```python
def decode_definitions_generated(
    self,
    ids: Sequence[int],
) -> list[dict[str, object]]:
    if isinstance(ids, (str, bytes, Mapping)) or not isinstance(ids, Sequence):
        raise TokenizerError("token stream must be a sequence of integer IDs")
    specs = [self._spec_for_token_id(f"token[{i}]", token_id)
             for i, token_id in enumerate(ids)]
    if not specs or specs[0].kind != "bos":
        raise TokenizerError("generated token stream must start with bos")

    content_ids: list[int] = []
    saw_eos = False
    for pos, spec in enumerate(specs[1:], start=1):
        if saw_eos:
            if spec.kind != "pad":
                raise TokenizerError("generated token stream must contain only pad after eos")
            continue
        if spec.kind == "eos":
            saw_eos = True
            continue
        if spec.kind == "pad":
            raise TokenizerError("generated token stream cannot contain pad before eos")
        if spec.kind == "bos":
            raise TokenizerError("generated token stream cannot contain nested bos")
        content_ids.append(ids[pos])

    if not saw_eos:
        raise TokenizerError("generated token stream must contain eos")
    if not content_ids:
        return []
    return self.decode_definitions(content_ids)
```

- [ ] **Step 4: Update the earlier tokenizer story doc**

Revise the flat tokenizer story to describe BOS/EOS and
`decode_definitions_generated`, and remove references to
`decode_definitions_padded`.

- [ ] **Step 5: Run focused tests**

Run from `python/`:

```bash
uv run pytest tests/test_flat_definition_tokenizer.py -q
```

Expected: PASS.

## Task 3: Final Verification

**Files:**
- Verify: `python/gristmill_symbolics/tokenizer.py`
- Verify: `python/tests/test_flat_definition_tokenizer.py`
- Verify: `docs/superpowers/specs/2026-07-07-tokenizer-generated-sequence-boundaries-story.md`
- Verify: `docs/superpowers/plans/2026-07-07-tokenizer-generated-sequence-boundaries.md`
- Verify: `docs/superpowers/plans/2026-07-07-flat-definition-tokenizer.md`

- [ ] **Step 1: Run focused tokenizer tests**

Run from `python/`:

```bash
uv run pytest tests/test_flat_definition_tokenizer.py -q
```

Expected: PASS.

- [ ] **Step 2: Run all Python tests**

Run from `python/`:

```bash
uv run pytest -q
```

Expected: PASS.

- [ ] **Step 3: Run Rust tests**

Run from the worktree root:

```bash
cargo test
```

Expected: PASS.

- [ ] **Step 4: Commit**

Run from the worktree root:

```bash
git add docs/superpowers/plans/2026-07-07-tokenizer-generated-sequence-boundaries.md docs/superpowers/plans/2026-07-07-flat-definition-tokenizer.md docs/superpowers/specs/2026-07-07-tokenizer-generated-sequence-boundaries-story.md docs/superpowers/specs/2026-07-07-flat-definition-tokenizer-story.md python/gristmill_symbolics/tokenizer.py python/tests/test_flat_definition_tokenizer.py
git commit -m "feat: add generated sequence boundary tokens"
```
