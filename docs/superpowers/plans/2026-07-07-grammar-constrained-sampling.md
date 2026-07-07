# Grammar-Constrained Flat Token Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fixed-shape JAX sampler that generates grammar-constrained flat token IDs and returns the log probability of the sampled sequence.

**Architecture:** Create a focused `sampling.py` module that depends on `FlatDefinitionGrammar` but not on a concrete model class. The sampler initializes a BOS/PAD prefix, carries grammar state through `jax.lax.scan`, samples from grammar-masked logits, records sampled-token logp, and keeps inactive rows padded after EOS.

**Tech Stack:** Python 3.11, JAX, pytest, uv.

---

## File Structure

- Create `python/gristmill_symbolics/sampling.py`: public `sample_token_ids(...)` helper returning a plain 3-tuple.
- Create `python/tests/test_flat_token_sampling.py`: behavior, EOS/PAD, max-length, JIT, and gradient tests.
- Do not modify tokenizer, grammar, scoring, model core, trainer, dataset, CLI, or checkpoint files.

## Task 1: Add Behavior Tests

**Files:**
- Create: `python/tests/test_flat_token_sampling.py`

- [ ] **Step 1: Write deterministic sequence tests**

Add this test file:

```python
import jax
import jax.numpy as jnp

from gristmill_symbolics.grammar import FlatDefinitionGrammar
from gristmill_symbolics.sampling import sample_token_ids
from gristmill_symbolics.tokenizer import FlatDefinitionTokenizer


def _tokenizer() -> FlatDefinitionTokenizer:
    return FlatDefinitionTokenizer(
        max_range_id=2,
        max_tensor_id=3,
        max_index_id=4,
        coeff_nums=(-1, 1, 2),
        coeff_dens=(1, 2),
    )


def _id(tokenizer: FlatDefinitionTokenizer, kind: str, offset: int = 0) -> int:
    return tokenizer.token_ids_for_kind(kind)[offset]


def _scripted_model(tokenizer: FlatDefinitionTokenizer, choices: list[int]):
    def model(source_ids, decoder_input_ids, *, deterministic=True):
        assert deterministic is True
        batch_size = source_ids.shape[0]
        target_len = decoder_input_ids.shape[1]
        logits = jnp.full(
            (batch_size, target_len, tokenizer.vocab_size),
            -1000.0,
            dtype=jnp.float32,
        )
        for position, token_id in enumerate(choices):
            logits = logits.at[:, position, token_id].set(1000.0)
        return logits

    return model


def _assert_grammar_valid_prefix(grammar: FlatDefinitionGrammar, row: jax.Array):
    state = grammar.initial_state(())
    for token_id in list(row):
        mask = grammar.allowed_by_state[state]
        assert bool(mask[int(token_id)])
        state = grammar.advance_state(
            state,
            jnp.asarray(token_id, dtype=jnp.int32),
        )


def test_sample_token_ids_generates_grammar_valid_sequence_with_logp():
    tokenizer = _tokenizer()
    grammar = FlatDefinitionGrammar(tokenizer)
    choices = [
        _id(tokenizer, "def_start"),
        _id(tokenizer, "tensorid"),
        _id(tokenizer, "def_end"),
        tokenizer.eos_token_id,
    ]
    source_ids = jnp.asarray([[1, 2, 0], [1, 0, 0]], dtype=jnp.int32)

    result = sample_token_ids(
        _scripted_model(tokenizer, choices),
        jax.random.key(0),
        source_ids,
        grammar,
        target_len=6,
    )
    assert type(result) is tuple
    generated_ids, token_log_probs, sequence_log_prob = result

    expected = jnp.asarray(
        [
            [
                tokenizer.bos_token_id,
                choices[0],
                choices[1],
                choices[2],
                choices[3],
                tokenizer.pad_token_id,
            ],
            [
                tokenizer.bos_token_id,
                choices[0],
                choices[1],
                choices[2],
                choices[3],
                tokenizer.pad_token_id,
            ],
        ],
        dtype=jnp.int32,
    )
    assert generated_ids.shape == (2, 6)
    assert token_log_probs.shape == (2, 6)
    assert sequence_log_prob.shape == (2,)
    assert jnp.array_equal(generated_ids, expected)
    assert jnp.allclose(token_log_probs[:, 0], 0.0)
    assert jnp.allclose(token_log_probs[:, 5], 0.0)
    assert jnp.allclose(
        sequence_log_prob,
        jnp.sum(token_log_probs, axis=-1),
    )
    for row in generated_ids:
        _assert_grammar_valid_prefix(grammar, row)
```

- [ ] **Step 2: Run the new test and verify it fails before implementation**

Run:

```bash
cd python
uv run pytest tests/test_flat_token_sampling.py::test_sample_token_ids_generates_grammar_valid_sequence_with_logp -q
```

Expected: FAIL because `gristmill_symbolics.sampling` does not exist yet.

## Task 2: Add Edge, JIT, and Gradient Tests

**Files:**
- Modify: `python/tests/test_flat_token_sampling.py`

- [ ] **Step 1: Add max-length, JIT, and gradient coverage**

Append these tests:

```python
def test_sample_token_ids_does_not_force_eos_at_max_length():
    tokenizer = _tokenizer()
    grammar = FlatDefinitionGrammar(tokenizer)
    choices = [
        _id(tokenizer, "def_start"),
        _id(tokenizer, "tensorid"),
        _id(tokenizer, "def_end"),
        _id(tokenizer, "def_start"),
        _id(tokenizer, "tensorid", 1),
    ]
    source_ids = jnp.asarray([[1, 0, 0]], dtype=jnp.int32)

    generated_ids, _token_log_probs, _sequence_log_prob = sample_token_ids(
        _scripted_model(tokenizer, choices),
        jax.random.key(1),
        source_ids,
        grammar,
        target_len=6,
    )

    assert generated_ids[0, 0] == tokenizer.bos_token_id
    assert tokenizer.eos_token_id not in set(map(int, generated_ids[0]))
    assert generated_ids[0, -1] == choices[-1]
    _assert_grammar_valid_prefix(grammar, generated_ids[0])


def test_sample_token_ids_is_jittable_for_fixed_shapes():
    tokenizer = _tokenizer()
    grammar = FlatDefinitionGrammar(tokenizer)
    choices = [
        _id(tokenizer, "def_start"),
        _id(tokenizer, "tensorid"),
        _id(tokenizer, "def_end"),
        tokenizer.eos_token_id,
    ]
    model = _scripted_model(tokenizer, choices)
    source_ids = jnp.asarray([[1, 2, 0]], dtype=jnp.int32)

    @jax.jit
    def run(rng, source):
        return sample_token_ids(
            model,
            rng,
            source,
            grammar,
            target_len=6,
        )[0]

    generated = run(jax.random.key(2), source_ids)

    assert generated.shape == (1, 6)
    assert generated[0, 0] == tokenizer.bos_token_id
    assert generated[0, 4] == tokenizer.eos_token_id
    assert generated[0, 5] == tokenizer.pad_token_id


def test_sampled_logp_is_differentiable_to_logits():
    tokenizer = _tokenizer()
    grammar = FlatDefinitionGrammar(tokenizer)
    source_ids = jnp.asarray([[1, 0, 0]], dtype=jnp.int32)
    logits = jnp.zeros((1, 4, tokenizer.vocab_size), dtype=jnp.float32)

    def score(x):
        def model(_source_ids, _decoder_input_ids, *, deterministic=True):
            assert deterministic is True
            return x

        _generated_ids, _token_log_probs, sequence_log_prob = sample_token_ids(
            model,
            jax.random.key(3),
            source_ids,
            grammar,
            target_len=4,
        )
        return sequence_log_prob[0]

    grad = jax.grad(score)(logits)

    assert grad.shape == logits.shape
    assert bool(jnp.any(jnp.abs(grad) > 0.0))
```

- [ ] **Step 2: Run the full sampling test file and verify it fails before implementation**

Run:

```bash
cd python
uv run pytest tests/test_flat_token_sampling.py -q
```

Expected: FAIL because the sampler module is still missing.

## Task 3: Implement the Sampler

**Files:**
- Create: `python/gristmill_symbolics/sampling.py`

- [ ] **Step 1: Create the sampler module**

Add this file:

```python
from __future__ import annotations

import jax
import jax.numpy as jnp

from .grammar import FlatDefinitionGrammar

__all__ = ("sample_token_ids",)


def sample_token_ids(
    model: Callable[..., jax.Array],
    rng: jax.Array,
    source_ids: jax.Array,
    grammar: FlatDefinitionGrammar,
    *,
    target_len: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    batch_size = source_ids.shape[0]
    generated_ids = jnp.full(
        (batch_size, target_len),
        grammar.pad_token_id,
        dtype=jnp.int32,
    )
    generated_ids = generated_ids.at[:, 0].set(grammar.bos_token_id)
    token_log_probs = jnp.zeros((batch_size, target_len), dtype=jnp.float32)

    grammar_state = grammar.initial_state((batch_size,))
    grammar_state = grammar.advance_state(
        grammar_state,
        jnp.full((batch_size,), grammar.bos_token_id, dtype=jnp.int32),
    )
    alive = jnp.ones((batch_size,), dtype=bool)

    def step(carry, step_index):
        step_rng, prefix, active, state, logp = carry
        step_rng, sample_key = jax.random.split(step_rng)

        logits = model(source_ids, prefix, deterministic=True)
        step_logits = logits[:, step_index, :]
        valid_next = jnp.take(grammar.allowed_by_state, state, axis=0)
        masked_logits = jnp.where(valid_next, step_logits, -jnp.inf)
        step_log_probs = jax.nn.log_softmax(masked_logits, axis=-1)

        sampled = jax.random.categorical(sample_key, masked_logits).astype(jnp.int32)
        next_id = jnp.where(active, sampled, grammar.pad_token_id)
        selected_logp = jnp.take_along_axis(
            step_log_probs,
            sampled[:, None],
            axis=-1,
        )[:, 0]
        next_logp = jnp.where(active, selected_logp, 0.0)

        next_position = step_index + 1
        prefix = prefix.at[:, next_position].set(next_id)
        logp = logp.at[:, next_position].set(next_logp)

        next_state = grammar.advance_state(state, next_id)
        state = jnp.where(active, next_state, state)
        active = active & (next_id != grammar.eos_token_id)

        return (step_rng, prefix, active, state, logp), None

    (_rng, generated_ids, _alive, _state, token_log_probs), _ = jax.lax.scan(
        step,
        (rng, generated_ids, alive, grammar_state, token_log_probs),
        jnp.arange(target_len - 1, dtype=jnp.int32),
    )
    return generated_ids, token_log_probs, jnp.sum(token_log_probs, axis=-1)
```

- [ ] **Step 2: Run the sampling tests**

Run:

```bash
cd python
uv run pytest tests/test_flat_token_sampling.py -q
```

Expected: PASS.

## Task 4: Focused Regression Verification

**Files:**
- No code changes expected.

- [ ] **Step 1: Run related Python tests**

Run:

```bash
cd python
uv run pytest tests/test_flat_token_sampling.py tests/test_constrained_logp.py tests/test_flat_definition_grammar.py tests/test_flat_seq2seq_transformer.py -q
```

Expected: PASS.

- [ ] **Step 2: Run broad Python and Rust tests**

Run:

```bash
cd python
uv run pytest -q
```

Expected: PASS.

Run from the repository root:

```bash
cargo test
```

Expected: PASS.

- [ ] **Step 3: Check whitespace and scope**

Run:

```bash
git diff --check
git diff --stat
```

Expected: `git diff --check` prints no errors. The diff should be limited to the story, plan, sampler module, and sampling tests.

## Self-Review

- Spec coverage: Tasks cover fixed-shape sampling, grammar state carry, EOS/PAD behavior, max-length without EOS, JIT, and gradient flow from sampled logp.
- Placeholder scan: No placeholder tasks or deferred behavior remain.
- Type consistency: The plan consistently uses `sample_token_ids`, `generated_ids`, `token_log_probs`, and `sequence_log_prob`.
