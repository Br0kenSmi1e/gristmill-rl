# Shared Next-Token Scoring Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make scoring and sampling share one grammar-constrained next-token log-probability helper.

**Architecture:** Add a small helper in `scoring.py` that converts raw logits and a valid-next mask into constrained log-probs. Sequence scoring will call it for `[B, T, V]` tensors, and sampling will call it for each `[B, V]` step before categorical sampling.

**Tech Stack:** Python 3.11, JAX, pytest, uv.

---

## File Structure

- Modify `python/gristmill_symbolics/scoring.py`: add `constrained_next_token_log_probs` and use it from sequence scoring.
- Modify `python/gristmill_symbolics/sampling.py`: use `constrained_next_token_log_probs` for each sampled step.
- Modify `python/tests/test_constrained_logp.py`: add focused helper tests.
- Modify `python/tests/test_flat_token_sampling.py`: no behavior changes expected, existing tests verify sampler behavior.

## Task 1: Add Failing Helper Tests

**Files:**
- Modify: `python/tests/test_constrained_logp.py`

- [ ] **Step 1: Add import and focused tests**

Add `constrained_next_token_log_probs` to the scoring imports and add tests
that verify invalid logits are excluded and all-invalid rows are finite.

- [ ] **Step 2: Run targeted tests**

Run:

```bash
cd python
uv run pytest tests/test_constrained_logp.py::test_constrained_next_token_log_probs_masks_invalid_logits -q
```

Expected: FAIL because `constrained_next_token_log_probs` does not exist yet.

## Task 2: Implement Shared Helper And Refactor Scoring

**Files:**
- Modify: `python/gristmill_symbolics/scoring.py`

- [ ] **Step 1: Implement helper**

Add `constrained_next_token_log_probs(logits, valid_next)` and export it.

- [ ] **Step 2: Use helper in `constrained_token_log_probs`**

Replace local mask/log-softmax logic with the helper, keeping label gathering and
invalid-label behavior unchanged.

- [ ] **Step 3: Run scoring tests**

Run:

```bash
cd python
uv run pytest tests/test_constrained_logp.py -q
```

Expected: PASS.

## Task 3: Refactor Sampling To Use Shared Helper

**Files:**
- Modify: `python/gristmill_symbolics/sampling.py`

- [ ] **Step 1: Import and call helper**

Import `constrained_next_token_log_probs` from `scoring.py`. Replace the local
mask/log-softmax block with the helper. Sample using the constrained log-probs.

- [ ] **Step 2: Run sampling tests**

Run:

```bash
cd python
uv run pytest tests/test_flat_token_sampling.py -q
```

Expected: PASS.

## Task 4: Verification

- [ ] **Step 1: Run focused tests**

Run:

```bash
cd python
uv run pytest tests/test_constrained_logp.py tests/test_flat_token_sampling.py tests/test_flat_definition_grammar.py tests/test_flat_seq2seq_transformer.py -q
```

Expected: PASS.

- [ ] **Step 2: Run broad checks**

Run:

```bash
cd python
uv run pytest -q
```

Expected: PASS.

Run from repo root:

```bash
cargo test
```

Expected: PASS.

Run:

```bash
git diff --check
```

Expected: no output.

## Self-Review

- Spec coverage: Tasks cover helper creation, sequence scoring reuse, sampling reuse, and verification.
- Placeholder scan: No placeholders or deferred behavior.
- Type consistency: The helper is consistently named `constrained_next_token_log_probs`.
