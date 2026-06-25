# Reusable Batched Policy JIT API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reusable module-level batched JAX policy callables and wire rollout to reuse them instead of rebuilding transforms inside the rollout loop.

**Architecture:** Create a focused `policy/batched.py` module that owns the stable `jit(vmap(...))` and `jit(vmap(value_and_grad(...)))` boundaries. Export the callables from `gristmill_symbolics.policy`, then update streamed rollout to call those module-level APIs while keeping rollout state, padding, masking, metrics, validation, and Rust action application unchanged.

**Tech Stack:** Python 3.11, JAX, NumPy, pytest, `uv`, existing PyO3-backed `gristmill_symbolics` package.

---

## Baseline And Constraints

Approved spec:

`docs/superpowers/specs/2026-06-25-reusable-batched-policy-jit-api-design.md`

Current design commit:

```bash
git log --oneline -1
# 4f2db57 docs: design reusable batched policy jit api
```

Do not touch these unrelated untracked files unless the user explicitly changes scope:

```text
.superpowers/
docs/superpowers/plans/2026-06-11-reinforce-phase-1-row-env.md
docs/superpowers/plans/2026-06-11-reinforce-phase-2-policy-model.md
docs/superpowers/plans/2026-06-12-reinforce-phase-3-training.md
```

This work is Python-only. Rust row-environment code, scalar policy semantics,
and the full-attention memory wall are out of scope.

## File Structure

Create:

- `python/gristmill_symbolics/policy/batched.py`: reusable batched policy callables composed once at import time.
- `python/tests/test_policy_batched.py`: parity tests for the four new batched callables.

Modify:

- `python/gristmill_symbolics/policy/__init__.py`: export the four new batched APIs.
- `python/gristmill_symbolics/reinforce/rollout.py`: replace inline `jax.vmap(...)` and `jax.value_and_grad(...)` construction with the new module-level batched callables.
- `python/tests/test_policy_package.py`: extend expected policy exports.
- `python/tests/test_reinforce_streaming.py`: update monkeypatch tests to patch rollout's batched imports.

Do not modify:

- `python/gristmill_symbolics/policy/api.py`: scalar sampling and scoring stay unchanged.
- `python/gristmill_symbolics/policy/model.py`: no model or attention redesign.
- Rust source files: legality checks and row action application stay in Rust.

---

### Task 1: Add Package Export Tests

**Files:**
- Modify: `python/tests/test_policy_package.py`
- Test: `python/tests/test_policy_package.py`

- [ ] **Step 1: Write the failing export expectation**

In `python/tests/test_policy_package.py`, update `EXPECTED_POLICY_EXPORTS` so it includes the four new names after the scalar policy APIs:

```python
EXPECTED_POLICY_EXPORTS = (
    "ACTION_TOKEN_FIELDS",
    "SENTINEL",
    "STATE_TOKEN_FIELDS",
    "ActionChoiceTree",
    "PolicyConfig",
    "action_choice_to_python",
    "make_action_choice",
    "pad_token_tree",
    "stack_token_trees",
    "tokenize_state_snapshot",
    "tokenize_action_space_snapshot",
    "init_policy_params",
    "sample_target",
    "score_target",
    "sample_action",
    "score_action",
    "batched_sample_target",
    "batched_score_target_grad",
    "batched_sample_action",
    "batched_score_action_grad",
)
```

- [ ] **Step 2: Run the package export tests and verify failure**

Run:

```bash
cd python
uv run pytest tests/test_policy_package.py -q
```

Expected: FAIL because `gristmill_symbolics.policy.__all__` does not yet include the four batched names and star import does not bind them.

- [ ] **Step 3: Keep the failing export test for the implementation task**

Do not commit yet. Leave `python/tests/test_policy_package.py` modified so Task 3 can commit the failing test and passing implementation together.

Run:

```bash
git status --short
```

Expected: `python/tests/test_policy_package.py` is modified.

---

### Task 2: Add Batched Policy Parity Tests

**Files:**
- Create: `python/tests/test_policy_batched.py`
- Test: `python/tests/test_policy_batched.py`

- [ ] **Step 1: Create the failing batched API test module**

Create `python/tests/test_policy_batched.py` with this complete content:

```python
import copy

import jax
import jax.numpy as jnp

from gristmill_symbolics.policy import (
    PolicyConfig,
    batched_sample_action,
    batched_sample_target,
    batched_score_action_grad,
    batched_score_target_grad,
    init_policy_params,
    sample_action,
    sample_target,
    score_action,
    score_target,
    stack_token_trees,
    tokenize_action_space_snapshot,
    tokenize_state_snapshot,
)
from tests.policy_fixtures import (
    actionable_action_space_snapshot,
    actionable_state_snapshot,
)


def _params():
    return init_policy_params(
        PolicyConfig(d_model=16),
        jax.random.PRNGKey(0),
    )


def _state_tree():
    return tokenize_state_snapshot(actionable_state_snapshot())


def _action_tree():
    return tokenize_action_space_snapshot(actionable_action_space_snapshot())


def _two_row_state_batch():
    return stack_token_trees([_state_tree(), _state_tree()])


def _two_row_action_batch():
    return stack_token_trees([_action_tree(), _action_tree()])


def _wide_action_space_snapshot(*, candidates=10, side_terms=6):
    snapshot = copy.deepcopy(actionable_action_space_snapshot())
    template = snapshot["candidate_templates"][0]
    widened = []
    for _ in range(candidates):
        candidate = copy.deepcopy(template)
        for side_name in ("left_definition", "right_definition"):
            terms = candidate[side_name]["terms"]
            candidate[side_name]["terms"] = [
                copy.deepcopy(terms[index % len(terms)]) for index in range(side_terms)
            ]
        widened.append(candidate)
    snapshot["candidate_templates"] = widened
    return snapshot


def _slice_tree(tree, index):
    return jax.tree_util.tree_map(lambda value: value[index], tree)


def _floating_leaves(tree):
    return [
        leaf
        for leaf in jax.tree_util.tree_leaves(tree)
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating)
    ]


def _tree_allclose(left, right, *, atol=1.0e-5):
    left_leaves = _floating_leaves(left)
    right_leaves = _floating_leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        assert jnp.allclose(left_leaf, right_leaf, atol=atol, rtol=atol)


def _assert_choice_equal(left, right):
    assert set(left) == set(right)
    for key in left:
        assert jnp.array_equal(left[key], right[key])


def test_batched_sample_target_matches_existing_vmap_call():
    params = _params()
    state_tokens, state_mask = _two_row_state_batch()
    def_mask = jnp.asarray([[True], [False]])
    keys = jax.random.split(jax.random.PRNGKey(10), 2)

    actual = batched_sample_target(
        params,
        state_tokens,
        state_mask,
        def_mask,
        keys,
    )
    expected = jax.vmap(sample_target, in_axes=(None, 0, 0, 0, 0))(
        params,
        state_tokens,
        state_mask,
        def_mask,
        keys,
    )

    assert jnp.array_equal(actual, expected)


def test_batched_score_target_grad_matches_existing_vmap_call():
    params = _params()
    state_tokens, state_mask = _two_row_state_batch()
    def_mask = jnp.asarray([[False], [True]])
    choices = jnp.asarray([-1, 0], dtype=jnp.int32)

    actual_logps, actual_grads = batched_score_target_grad(
        params,
        state_tokens,
        state_mask,
        def_mask,
        choices,
    )
    expected_logps, expected_grads = jax.vmap(
        jax.value_and_grad(score_target, argnums=0),
        in_axes=(None, 0, 0, 0, 0),
    )(
        params,
        state_tokens,
        state_mask,
        def_mask,
        choices,
    )

    assert actual_logps.shape == (2,)
    assert jnp.allclose(actual_logps, expected_logps, atol=1.0e-5)
    _tree_allclose(actual_grads, expected_grads)
    for leaf in _floating_leaves(actual_grads):
        assert leaf.shape[0] == 2


def test_batched_sample_action_matches_existing_vmap_call():
    params = _params()
    state_tokens, state_mask = _two_row_state_batch()
    action_tokens, action_mask = _two_row_action_batch()
    selected_defs = jnp.asarray([0, 0], dtype=jnp.int32)
    keys = jax.random.split(jax.random.PRNGKey(25), 2)

    actual = batched_sample_action(
        params,
        state_tokens,
        state_mask,
        selected_defs,
        action_tokens,
        action_mask,
        keys,
    )
    expected = jax.vmap(
        sample_action,
        in_axes=(None, 0, 0, 0, 0, 0, 0),
    )(
        params,
        state_tokens,
        state_mask,
        selected_defs,
        action_tokens,
        action_mask,
        keys,
    )

    for index in range(2):
        _assert_choice_equal(_slice_tree(actual, index), _slice_tree(expected, index))


def test_batched_score_action_grad_matches_existing_vmap_call():
    params = _params()
    state_tokens, state_mask = _two_row_state_batch()
    action_tokens, action_mask = _two_row_action_batch()
    selected_defs = jnp.asarray([0, 0], dtype=jnp.int32)
    keys = jax.random.split(jax.random.PRNGKey(20), 2)
    choices = batched_sample_action(
        params,
        state_tokens,
        state_mask,
        selected_defs,
        action_tokens,
        action_mask,
        keys,
    )

    actual_logps, actual_grads = batched_score_action_grad(
        params,
        state_tokens,
        state_mask,
        selected_defs,
        action_tokens,
        action_mask,
        choices,
    )
    expected_logps, expected_grads = jax.vmap(
        jax.value_and_grad(score_action, argnums=0),
        in_axes=(None, 0, 0, 0, 0, 0, 0),
    )(
        params,
        state_tokens,
        state_mask,
        selected_defs,
        action_tokens,
        action_mask,
        choices,
    )

    assert actual_logps.shape == (2,)
    assert jnp.allclose(actual_logps, expected_logps, atol=1.0e-5)
    _tree_allclose(actual_grads, expected_grads)
    for leaf in _floating_leaves(actual_grads):
        assert leaf.shape[0] == 2


def test_batched_sample_action_uses_local_padding_width_not_model_config():
    params = _params()
    state_tokens, state_mask = _two_row_state_batch()
    small_action = _action_tree()
    wide_action = tokenize_action_space_snapshot(
        _wide_action_space_snapshot(candidates=10, side_terms=6)
    )
    action_tokens, action_mask = stack_token_trees([small_action, wide_action])
    selected_defs = jnp.asarray([0, 0], dtype=jnp.int32)
    keys = jax.random.split(jax.random.PRNGKey(31), 2)

    choices = batched_sample_action(
        params,
        state_tokens,
        state_mask,
        selected_defs,
        action_tokens,
        action_mask,
        keys,
    )
    logp, _grad = batched_score_action_grad(
        params,
        state_tokens,
        state_mask,
        selected_defs,
        action_tokens,
        action_mask,
        choices,
    )

    assert choices["left_mask"].shape == choices["left_valid_mask"].shape
    assert choices["right_mask"].shape == choices["right_valid_mask"].shape
    assert int(jnp.sum(choices["left_valid_mask"][1])) == 6
    assert int(jnp.sum(choices["right_valid_mask"][1])) == 6
    assert bool(jnp.all(jnp.isfinite(logp)))
```

- [ ] **Step 2: Run the new tests and verify failure**

Run:

```bash
cd python
uv run pytest tests/test_policy_batched.py -q
```

Expected: FAIL during import because `batched_sample_target`, `batched_score_target_grad`, `batched_sample_action`, and `batched_score_action_grad` are not exported yet.

- [ ] **Step 3: Keep the failing batched parity tests for the implementation task**

Do not commit yet. Leave `python/tests/test_policy_batched.py` untracked so Task 3 can commit the failing test and passing implementation together.

Run:

```bash
git status --short
```

Expected: `python/tests/test_policy_batched.py` is untracked, and `python/tests/test_policy_package.py` remains modified from Task 1.

---

### Task 3: Implement Reusable Batched Policy APIs

**Files:**
- Create: `python/gristmill_symbolics/policy/batched.py`
- Modify: `python/gristmill_symbolics/policy/__init__.py`
- Test: `python/tests/test_policy_package.py`
- Test: `python/tests/test_policy_batched.py`

- [ ] **Step 1: Add the batched policy module**

Create `python/gristmill_symbolics/policy/batched.py` with this complete content:

```python
from __future__ import annotations

import jax

from .api import sample_action, sample_target, score_action, score_target

batched_sample_target = jax.jit(
    jax.vmap(sample_target, in_axes=(None, 0, 0, 0, 0))
)

batched_score_target_grad = jax.jit(
    jax.vmap(
        jax.value_and_grad(score_target, argnums=0),
        in_axes=(None, 0, 0, 0, 0),
    )
)

batched_sample_action = jax.jit(
    jax.vmap(sample_action, in_axes=(None, 0, 0, 0, 0, 0, 0))
)

batched_score_action_grad = jax.jit(
    jax.vmap(
        jax.value_and_grad(score_action, argnums=0),
        in_axes=(None, 0, 0, 0, 0, 0, 0),
    )
)

__all__ = (
    "batched_sample_target",
    "batched_score_target_grad",
    "batched_sample_action",
    "batched_score_action_grad",
)
```

- [ ] **Step 2: Export the batched APIs from the policy package**

In `python/gristmill_symbolics/policy/__init__.py`, add this import after the scalar API import:

```python
from .api import sample_action, sample_target, score_action, score_target
from .batched import (
    batched_sample_action,
    batched_sample_target,
    batched_score_action_grad,
    batched_score_target_grad,
)
```

Then update `__all__` so the four new names appear after `"score_action"`:

```python
    "sample_target",
    "score_target",
    "sample_action",
    "score_action",
    "batched_sample_target",
    "batched_score_target_grad",
    "batched_sample_action",
    "batched_score_action_grad",
)
```

- [ ] **Step 3: Run package and batched API tests**

Run:

```bash
cd python
uv run pytest tests/test_policy_package.py tests/test_policy_batched.py -q
```

Expected: PASS. If the first JAX compile makes the run slow, wait for the command to finish rather than interrupting it.

- [ ] **Step 4: Commit the batched policy API tests and implementation**

Run:

```bash
git add \
  python/tests/test_policy_package.py \
  python/tests/test_policy_batched.py \
  python/gristmill_symbolics/policy/batched.py \
  python/gristmill_symbolics/policy/__init__.py
git commit -m "feat: add reusable batched policy jit calls"
```

Expected: commit succeeds with the export test, batched parity tests, and policy API implementation staged.

---

### Task 4: Wire Rollout To The Reusable Batched APIs

**Files:**
- Modify: `python/gristmill_symbolics/reinforce/rollout.py`
- Test: `python/tests/test_reinforce_streaming.py`

- [ ] **Step 1: Update rollout imports**

In `python/gristmill_symbolics/reinforce/rollout.py`, replace the policy import block with:

```python
from gristmill_symbolics.policy import (
    action_choice_to_python,
    batched_sample_action,
    batched_sample_target,
    batched_score_action_grad,
    batched_score_target_grad,
    stack_token_trees,
    tokenize_action_space_snapshot,
    tokenize_state_snapshot,
)
```

This removes direct rollout imports of `sample_action`, `sample_target`,
`score_action`, and `score_target`.

- [ ] **Step 2: Replace target policy inline transforms**

In `_collect_streamed_rollout_gradients`, replace this target sampling block:

```python
        target_choices = jax.vmap(sample_target, in_axes=(None, 0, 0, 0, 0))(
            policy.params,
            state_tokens_batch,
            state_mask_batch,
            target_def_mask_batch,
            jnp.stack(target_keys, axis=0),
        )
        target_logps, target_grads = jax.vmap(
            jax.value_and_grad(score_target, argnums=0),
            in_axes=(None, 0, 0, 0, 0),
        )(
            policy.params,
            state_tokens_batch,
            state_mask_batch,
            target_def_mask_batch,
            target_choices,
        )
```

with:

```python
        target_choices = batched_sample_target(
            policy.params,
            state_tokens_batch,
            state_mask_batch,
            target_def_mask_batch,
            jnp.stack(target_keys, axis=0),
        )
        target_logps, target_grads = batched_score_target_grad(
            policy.params,
            state_tokens_batch,
            state_mask_batch,
            target_def_mask_batch,
            target_choices,
        )
```

- [ ] **Step 3: Replace action policy inline transforms**

In `_collect_streamed_rollout_gradients`, replace this action sampling block:

```python
            action_choices = jax.vmap(sample_action, in_axes=(None, 0, 0, 0, 0, 0, 0))(
                policy.params,
                action_state_tokens,
                action_state_mask,
                selected,
                action_tokens_batch,
                action_mask_batch,
                stacked_action_keys,
            )
            action_logps, action_grads = jax.vmap(
                jax.value_and_grad(score_action, argnums=0),
                in_axes=(None, 0, 0, 0, 0, 0, 0),
            )(
                policy.params,
                action_state_tokens,
                action_state_mask,
                selected,
                action_tokens_batch,
                action_mask_batch,
                action_choices,
            )
```

with:

```python
            action_choices = batched_sample_action(
                policy.params,
                action_state_tokens,
                action_state_mask,
                selected,
                action_tokens_batch,
                action_mask_batch,
                stacked_action_keys,
            )
            action_logps, action_grads = batched_score_action_grad(
                policy.params,
                action_state_tokens,
                action_state_mask,
                selected,
                action_tokens_batch,
                action_mask_batch,
                action_choices,
            )
```

- [ ] **Step 4: Run rollout tests and observe monkeypatch failures**

Run:

```bash
cd python
uv run pytest tests/test_reinforce_streaming.py -q
```

Expected: FAIL in tests that monkeypatch `rollout_module.sample_target`, because rollout now calls `rollout_module.batched_sample_target`.

- [ ] **Step 5: Keep rollout wiring for the test repair task**

Do not commit yet because rollout tests are expected to be red until Task 5 updates the monkeypatches.

Run:

```bash
git status --short
```

Expected: `python/gristmill_symbolics/reinforce/rollout.py` is modified.

---

### Task 5: Update Rollout Monkeypatch Tests

**Files:**
- Modify: `python/tests/test_reinforce_streaming.py`
- Test: `python/tests/test_reinforce_streaming.py`

- [ ] **Step 1: Update the stop-sampling monkeypatch**

In `test_static_rollout_skips_dummy_only_action_application`, replace the scalar patch helper and monkeypatch:

```python
    def sample_stop(
        _params,
        _state_tokens,
        _state_token_mask,
        _def_mask,
        _rng,
    ):
        return jnp.asarray(-1, dtype=jnp.int32)

    monkeypatch.setattr(rollout_module, "sample_target", sample_stop)
```

with this batched helper:

```python
    def sample_stop(
        _params,
        _state_tokens,
        _state_token_mask,
        def_mask,
        _rng,
    ):
        return jnp.full((def_mask.shape[0],), -1, dtype=jnp.int32)

    monkeypatch.setattr(rollout_module, "batched_sample_target", sample_stop)
```

- [ ] **Step 2: Update the first-target monkeypatch**

In `test_static_rollout_preserves_physical_target_rows_after_lower_row_stops`, replace the scalar patch helper and monkeypatch:

```python
    def sample_first_target_when_present(
        _params,
        _state_tokens,
        _state_token_mask,
        def_mask,
        _rng,
    ):
        return jnp.where(jnp.any(def_mask), 0, -1).astype(jnp.int32)

    monkeypatch.setattr(
        rollout_module,
        "sample_target",
        sample_first_target_when_present,
    )
```

with this batched helper:

```python
    def sample_first_target_when_present(
        _params,
        _state_tokens,
        _state_token_mask,
        def_mask,
        _rng,
    ):
        return jnp.where(jnp.any(def_mask, axis=1), 0, -1).astype(jnp.int32)

    monkeypatch.setattr(
        rollout_module,
        "batched_sample_target",
        sample_first_target_when_present,
    )
```

- [ ] **Step 3: Run focused rollout tests**

Run:

```bash
cd python
uv run pytest \
  tests/test_reinforce_streaming.py::test_static_rollout_skips_dummy_only_action_application \
  tests/test_reinforce_streaming.py::test_static_rollout_preserves_physical_target_rows_after_lower_row_stops \
  tests/test_reinforce_streaming.py::test_streamed_rollout_accumulates_one_step_sampled_score_gradients \
  tests/test_reinforce_streaming.py::test_static_rollout_matches_dynamic_streamed_mixed_batch \
  -q
```

Expected: PASS.

- [ ] **Step 4: Run all rollout tests**

Run:

```bash
cd python
uv run pytest tests/test_reinforce_streaming.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit rollout wiring and test updates**

Run:

```bash
git add \
  python/gristmill_symbolics/reinforce/rollout.py \
  python/tests/test_reinforce_streaming.py
git commit -m "refactor: reuse batched policy calls in rollout"
```

Expected: commit succeeds with rollout wiring and rollout monkeypatch test updates staged.

---

### Task 6: Full Python Verification And Compile-Oriented Smoke Check

**Files:**
- Test: `python/tests/test_policy_package.py`
- Test: `python/tests/test_policy_batched.py`
- Test: `python/tests/test_policy_vmap.py`
- Test: `python/tests/test_reinforce_streaming.py`
- Test: all Python tests

- [ ] **Step 1: Run focused policy and rollout tests**

Run:

```bash
cd python
uv run pytest \
  tests/test_policy_package.py \
  tests/test_policy_batched.py \
  tests/test_policy_vmap.py \
  tests/test_reinforce_streaming.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run the full Python suite**

Run:

```bash
cd python
uv run pytest -q
```

Expected: PASS.

- [ ] **Step 3: Run a small compile-log smoke command**

Run:

```bash
cd python
uv run python -c 'from pathlib import Path; from tests.policy_fixtures import actionable_json; Path("/tmp/gristmill-actionable.json").write_text(actionable_json())'
JAX_LOG_COMPILES=1 uv run python -m gristmill_symbolics.reinforce.train \
  --input /tmp/gristmill-actionable.json \
  --updates 1 \
  --batch-size 1 \
  --max-steps 1 \
  --seed 42 \
  --static-policy-batch \
  --state-token-pad-to 512 \
  --action-token-pad-to 512 \
  --definition-pad-to 8 \
  > /tmp/gristmill-batched-policy-smoke.jsonl \
  2> /tmp/gristmill-batched-policy-smoke.stderr
```

Expected: command exits `0`, `/tmp/gristmill-batched-policy-smoke.jsonl` contains one training JSON line, and `/tmp/gristmill-batched-policy-smoke.stderr` contains JAX compile messages. This smoke command is not the CCSD profiling acceptance run; it only verifies the CLI path still works with static policy batching and the new batched wrappers.

- [ ] **Step 4: Inspect final worktree status**

Run:

```bash
git status --short
```

Expected: only the pre-existing unrelated untracked files remain:

```text
?? .superpowers/
?? docs/superpowers/plans/2026-06-11-reinforce-phase-1-row-env.md
?? docs/superpowers/plans/2026-06-11-reinforce-phase-2-policy-model.md
?? docs/superpowers/plans/2026-06-12-reinforce-phase-3-training.md
```

---

### Task 7: Optional CCSD Profiling Evidence On CUDA Host

**Files:**
- No source files
- Output: `/tmp/ccsd-profile/batched-policy-jit/`

Run this task only on the CUDA machine with `../tmp/working_eqn.json` available and the CUDA JAX environment preserved. Use `uv run --no-sync` if a plain `uv run` would sync away the CUDA JAX install.

- [ ] **Step 1: Run the issue #20 static-shape profile for updates=1**

Run:

```bash
cd python
RUN=/tmp/ccsd-profile/batched-policy-jit/updates1
mkdir -p "$RUN"

JAX_LOG_COMPILES=1 \
/usr/bin/time -v uv run --no-sync python \
  -m gristmill_symbolics.reinforce.train \
  --input ../tmp/working_eqn.json \
  --updates 1 \
  --batch-size 2 \
  --max-steps 64 \
  --seed 42 \
  --static-policy-batch \
  --state-token-pad-to 3072 \
  --action-token-pad-to 4096 \
  --definition-pad-to 128 \
  > "$RUN/stdout.jsonl" \
  2> "$RUN/stderr.log"
```

Expected: command exits `0`, `stdout.jsonl` ends with one training JSON line, and `stderr.log` contains compile logs plus `/usr/bin/time -v` output.

- [ ] **Step 2: Count compile lines for updates=1**

Run:

```bash
grep -c '^Compiling' /tmp/ccsd-profile/batched-policy-jit/updates1/stderr.log
```

Expected: a materially lower number than the pre-change `updates=1` count of `2130` from issue #20.

- [ ] **Step 3: Run updates=2 and updates=3 profiles**

Run:

```bash
cd python
for UPDATES in 2 3; do
  RUN="/tmp/ccsd-profile/batched-policy-jit/updates${UPDATES}"
  mkdir -p "$RUN"
  JAX_LOG_COMPILES=1 \
  /usr/bin/time -v uv run --no-sync python \
    -m gristmill_symbolics.reinforce.train \
    --input ../tmp/working_eqn.json \
    --updates "$UPDATES" \
    --batch-size 2 \
    --max-steps 64 \
    --seed 42 \
    --static-policy-batch \
    --state-token-pad-to 3072 \
    --action-token-pad-to 4096 \
    --definition-pad-to 128 \
    > "$RUN/stdout.jsonl" \
    2> "$RUN/stderr.log"
done
```

Expected: both commands exit `0`.

- [ ] **Step 4: Summarize compile counts**

Run:

```bash
for UPDATES in 1 2 3; do
  COUNT=$(grep -c '^Compiling' "/tmp/ccsd-profile/batched-policy-jit/updates${UPDATES}/stderr.log")
  printf 'updates=%s compiles=%s\n' "$UPDATES" "$COUNT"
done
```

Expected: counts no longer scale with the old `128 * updates` repeated `jit(scan)` signatures. Compare against issue #20:

```text
updates=1  compiles=2130
updates=2  compiles=3103
updates=3  compiles=4433
```

- [ ] **Step 5: Check repeated scan signatures**

Run:

```bash
for UPDATES in 1 2 3; do
  echo "updates=$UPDATES"
  rg 'Compiling jit\(scan\).*bool\[4096,2\].*float32\[4096,2\]' \
    "/tmp/ccsd-profile/batched-policy-jit/updates${UPDATES}/stderr.log" \
    | wc -l
done
```

Expected: counts are far below the old `128`, `256`, and `384` pattern reported in issue #20.

- [ ] **Step 6: Report profiling evidence**

Add the compile counts, final training JSON lines, wall times from `/usr/bin/time -v`, and repeated scan signature counts to the PR or issue comment. State explicitly that this PR does not change the known full-attention memory wall.
