# Weighted Supervised NLL Totals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a supervised objective helper that returns additive weighted NLL totals for prebuilt flat token batches.

**Architecture:** Create a focused objective module above `scoring.py`. It consumes logits and prebuilt target-side batch arrays, calls `constrained_sequence_log_prob`, and returns `(weighted_nll_sum, weight_sum)` for later micro-batch accumulation.

**Tech Stack:** Python 3.11, JAX, pytest, uv.

---

## File Structure

- Create `python/gristmill_symbolics/supervised.py`: `weighted_supervised_nll_totals`.
- Create `python/tests/test_weighted_supervised_nll.py`: focused objective tests.
- Do not modify tokenizer, grammar, sampling, model core, dataset, optimizer, CLI, or checkpoint files.

## Task 1: Add Failing Objective Tests

**Files:**
- Create: `python/tests/test_weighted_supervised_nll.py`

- [ ] **Step 1: Add tests**

Add tests for weighted NLL math, masked target positions, zero total weight,
JIT, and gradient flow to logits.

- [ ] **Step 2: Run targeted test**

Run:

```bash
cd python
uv run pytest tests/test_weighted_supervised_nll.py::test_weighted_supervised_nll_totals_matches_manual_sequence_nll -q
```

Expected: FAIL because `gristmill_symbolics.supervised` does not exist yet.

## Task 2: Implement Objective Helper

**Files:**
- Create: `python/gristmill_symbolics/supervised.py`

- [ ] **Step 1: Implement `weighted_supervised_nll_totals`**

The helper signature should be:

```python
def weighted_supervised_nll_totals(
    logits,
    decoder_input_ids,
    target_ids,
    target_mask,
    example_weight,
    grammar,
):
    ...
```

Return only:

```python
weighted_nll_sum, weight_sum
```

- [ ] **Step 2: Run objective tests**

Run:

```bash
cd python
uv run pytest tests/test_weighted_supervised_nll.py -q
```

Expected: PASS.

## Task 3: Verification

- [ ] **Step 1: Run focused tests**

Run:

```bash
cd python
uv run pytest tests/test_weighted_supervised_nll.py tests/test_constrained_logp.py tests/test_flat_token_sampling.py tests/test_flat_definition_grammar.py tests/test_flat_seq2seq_transformer.py -q
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

- Spec coverage: Tasks cover weighted sum math, additive weight sum, masking,
  zero-weight behavior, JIT, and gradient flow.
- Placeholder scan: No placeholders or deferred behavior.
- Type consistency: The helper is consistently named
  `weighted_supervised_nll_totals`.
