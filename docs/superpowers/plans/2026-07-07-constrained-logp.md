# Differentiable Grammar-Constrained LogP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add JAX-friendly grammar-constrained log-probability helpers for flat token logits.

**Architecture:** Create a focused scoring module that consumes logits, decoder input IDs, labels, a label mask, and `FlatDefinitionGrammar`. The helpers derive valid-next masks from the grammar, mask logits before `log_softmax`, select label log-probabilities, zero padding positions, and keep gradients flowing through valid active logits.

**Tech Stack:** Python 3.11, JAX, pytest, uv.

---

## File Structure

- Create `python/gristmill_symbolics/scoring.py`: `constrained_token_log_probs` and `constrained_sequence_log_prob`.
- Create `python/tests/test_constrained_logp.py`: focused behavior, JIT, and gradient tests.
- Do not modify model, trainer, sampler, dataset, CLI, or checkpoint files.

## Task 1: Behavior Tests

- [ ] Add tests for active valid labels matching `jax.nn.log_softmax` over grammar-valid logits.
- [ ] Add tests for masked positions returning `0.0`.
- [ ] Add tests for active grammar-invalid labels returning a large negative logp.
- [ ] Add tests for sequence logp summing token logp per example.
- [ ] Run `uv run pytest tests/test_constrained_logp.py -q` from `python/` and verify it fails before implementation because the module is missing.

## Task 2: JAX Tests

- [ ] Add a `jax.jit` test for `constrained_sequence_log_prob`.
- [ ] Add a gradient test showing valid active logits receive nonzero gradients.
- [ ] Add a gradient test showing invalid selected labels do not create gradient on that invalid token.
- [ ] Re-run `uv run pytest tests/test_constrained_logp.py -q` and verify tests fail before implementation.

## Task 3: Implementation

- [ ] Create `python/gristmill_symbolics/scoring.py`.
- [ ] Implement `constrained_token_log_probs(logits, decoder_input_ids, labels, label_mask, grammar)`.
- [ ] Implement `constrained_sequence_log_prob(logits, decoder_input_ids, labels, label_mask, grammar)`.
- [ ] Use `FlatDefinitionGrammar.valid_next_masks_for_decoder_input`.
- [ ] Mask invalid logits with `jnp.where(valid_next, logits, -jnp.inf)` and
  use `jax.nn.log_softmax`.
- [ ] Use `jnp.take_along_axis` for label selection.
- [ ] Return `0.0` for masked label positions.
- [ ] Return a large negative logp for active grammar-invalid labels.
- [ ] Re-run `uv run pytest tests/test_constrained_logp.py -q`.

## Task 4: Verification

- [ ] Run `uv run pytest tests/test_constrained_logp.py tests/test_flat_definition_grammar.py -q`.
- [ ] Run `uv run pytest -q`.
- [ ] Run `cargo test`.
- [ ] Run `git diff --check`.
- [ ] Review `git diff --stat` and `git diff` for scope control.
