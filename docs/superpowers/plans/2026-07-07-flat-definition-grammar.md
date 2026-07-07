# Flat Definition Grammar Mask Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable JAX-friendly grammar mask for flat definition token streams.

**Architecture:** Extend `FlatDefinitionTokenizer` with public token-kind lookup helpers, then add a focused grammar module that builds fixed-size category and allowed-token tables from the tokenizer. Runtime grammar methods use JAX arrays and `jax.lax.scan` so fixed-shape padded training and sampling prefixes can be constrained without dynamic variable-length arrays.

**Tech Stack:** Python 3.11, JAX, pytest, uv.

---

## File Structure

- Modify `python/gristmill_symbolics/tokenizer.py`: add `token_kind()` and `token_ids_for_kind()` public helpers.
- Create `python/gristmill_symbolics/grammar.py`: define `FlatDefinitionGrammar` and state/category constants.
- Create `python/tests/test_flat_definition_grammar.py`: focused grammar tests.
- Modify `python/tests/test_flat_definition_tokenizer.py`: add tokenizer helper coverage.

## Task 1: Tokenizer Helper Tests

- [ ] Add tests showing `token_kind()` returns `"pad"`, `"bos"`, `"eos"`, `"def_start"`, `"def_end"`, and configured scalar token kinds.
- [ ] Add tests showing `token_ids_for_kind("tensorid")`, `"indexid"`, and coefficient kinds return stable tuples of token IDs.
- [ ] Run `uv run pytest python/tests/test_flat_definition_tokenizer.py -q` and verify the new tests fail before implementation.
- [ ] Implement the two tokenizer helpers.
- [ ] Re-run `uv run pytest python/tests/test_flat_definition_tokenizer.py -q` and verify it passes.

## Task 2: Grammar FSM Tests

- [ ] Add tests for legal next-token families after `bos`, `def_start`, base tensor, coefficient numerator, coefficient denominator, factor tensor, `eos`, and invalid prefixes.
- [ ] Add tests for `valid_next_masks_for_decoder_input()` teacher-forcing alignment on a fixed padded generated sequence.
- [ ] Add tests for `valid_next_mask_from_prefix()` on fixed-shape prefixes.
- [ ] Add tests for scalar and batched `advance_state()`.
- [ ] Run `uv run pytest python/tests/test_flat_definition_grammar.py -q` and verify the tests fail before implementation.

## Task 3: Grammar Implementation

- [ ] Create `python/gristmill_symbolics/grammar.py`.
- [ ] Define compact category and state constants.
- [ ] Build `category_by_id` from `tokenizer.token_kind(token_id)`.
- [ ] Build `allowed_by_state` with rows for each grammar state.
- [ ] Implement `initial_state(batch_shape)` with fixed-shape `jnp.full`.
- [ ] Implement `advance_state(state, token_id)` with `jnp.where` transitions.
- [ ] Implement `valid_next_masks_for_decoder_input(decoder_input_ids)` with `jax.lax.scan`.
- [ ] Implement `valid_next_mask_from_prefix(prefix_ids)` with `jax.lax.scan`.
- [ ] Re-run `uv run pytest python/tests/test_flat_definition_grammar.py -q`.

## Task 4: Verification

- [ ] Run `uv run pytest python/tests/test_flat_definition_tokenizer.py python/tests/test_flat_definition_grammar.py -q`.
- [ ] Run `uv run pytest -q`.
- [ ] Run `cargo test`.
- [ ] Review `git diff --stat` and `git diff` for scope control.
