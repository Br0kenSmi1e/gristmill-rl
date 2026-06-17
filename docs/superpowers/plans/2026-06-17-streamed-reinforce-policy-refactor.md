# Streamed REINFORCE Policy Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fixed-cap policy/action model and rollout-table trainer with a breaking streamed REINFORCE implementation that samples choices only, scores through explicit log-probability APIs, and accumulates per-trajectory score-function gradients during rollout.

**Architecture:** Milestone 1 cleans the policy stack first: remove global action caps from config/params, make scalar sample functions return choices only, and use local padding plus `vmap` for target/action batches. Milestone 2 removes the table-objective training path and replaces it with a streamed rollout collector that immediately scores each sampled decision with `value_and_grad(score_*, argnums=0)`, accumulates per-sample `sum_t logp_t` and `sum_t grad_logp_t`, then applies one Optax update from `-mean_i stop(A_i) * G_i`.

**Tech Stack:** Python 3.11, JAX, Optax, pytest, uv/maturin, Rust row rewrite bindings.

---

## Baseline

Worktree:

```bash
/Users/longli/rcode/gristmill-symbolics/.worktrees/refactor-streamed-reinforce-policy
```

Branch:

```bash
refactor/streamed-reinforce-policy
```

Baseline commands already run in the fresh worktree:

```bash
cd /Users/longli/rcode/gristmill-symbolics/.worktrees/refactor-streamed-reinforce-policy
cargo test
cd python
uv run pytest -q
```

Baseline result:

```text
cargo test: passed
uv run pytest -q: 144 passed in 81.06s
```

## File Structure

Milestone 1 policy files:

- Modify: `python/gristmill_symbolics/policy/types.py:15` removes `PolicyConfig.max_candidates` and `PolicyConfig.max_side_terms`; keeps `ActionChoiceTree` semantics.
- Modify: `python/gristmill_symbolics/policy/model.py:33` removes cap-shaped action params and initializes shared scalar action biases.
- Modify: `python/gristmill_symbolics/policy/api.py:90` changes `sample_target` to return a choice only; `api.py:103-107` replaces param-derived action widths/candidate ranges with local input-derived ranges; `api.py:204-299` removes per-slot biases; `api.py:504-715` updates action validation/scoring/sampling for local capacities and choice-only sampling.
- Keep: `python/gristmill_symbolics/policy/tree.py:44` and `tree.py:64` as the local padding/stacking adapter unless implementation reveals a real need for a tiny action-choice padding helper.
- Modify tests: `python/tests/test_policy_model.py`, `python/tests/test_policy_tree.py`, `python/tests/test_policy_target.py`, `python/tests/test_policy_action.py`, `python/tests/test_policy_vmap.py`, `python/tests/test_policy_jit_grad.py`, and `python/tests/test_policy_package.py`.

Milestone 2 reinforce files:

- Modify: `python/gristmill_symbolics/reinforce/types.py:83` deletes `RolloutTable`, `ScoreOutputs`, and `LossDiagnostics`; `types.py:115` expands `UpdateMetrics` with explicit reward objective and surrogate diagnostics.
- Modify: `python/gristmill_symbolics/reinforce/objective.py:22` keeps `compute_rewards` and `compute_advantages`; deletes `_reinforce_loss_value`, `reinforce_loss`, `score_rollout`, and table scoring helpers.
- Rewrite: `python/gristmill_symbolics/reinforce/rollout.py:45-612` keeps `make_rng_grid` and selected mask/tree helpers, removes table assembly, and adds private streamed rollout/gradient helpers.
- Modify: `python/gristmill_symbolics/reinforce/train_state.py:106` changes `train_update` to return exactly `(TrainState, UpdateMetrics)` and to consume the streamed rollout result.
- Modify: `python/gristmill_symbolics/reinforce/train.py:47` updates CLI default `PolicyConfig` and two-value `train_update` call.
- Modify: `python/gristmill_symbolics/reinforce/__init__.py:3` removes old table-objective exports.
- Modify: `python/gristmill_symbolics/reinforce/checkpoint.py:27` only as needed for `UpdateMetrics`/`PolicyConfig` shape changes; do not add old-checkpoint migration.
- Rewrite/delete tests tied to the old public table API: `python/tests/test_reinforce_rollout.py`, `python/tests/test_reinforce_objective.py`, `python/tests/test_reinforce_train.py`, `python/tests/test_reinforce_package.py`, `python/tests/test_reinforce_checkpoint.py`, and `python/tests/test_reinforce_cli.py`.
- Add: `python/tests/test_reinforce_streaming.py` for streamed `sum_t grad_logp_t`, advantage weighting, and reward-objective metric semantics.

## Milestone 1: Policy Model Cleanup

### Task 1: Make PolicyConfig And Params Shape-Independent

**Files:**

- Modify: `python/tests/test_policy_model.py:10`
- Modify: `python/tests/test_policy_tree.py:232`
- Modify: `python/gristmill_symbolics/policy/types.py:15`
- Modify: `python/gristmill_symbolics/policy/model.py:33`

- [ ] **Step 1: Write failing config/param tests**

Replace the old fixed-cap assertions with explicit absence checks and shared scalar biases:

```python
def test_init_policy_params_shapes_and_stop_bias():
    config = PolicyConfig(d_model=16, id_vocab_size=32)
    params = init_policy_params(config, jax.random.PRNGKey(0))

    assert params["field_embeddings"]["token_kind"].shape[1] == 16
    assert set(params["action"]) == {
        "candidate_w",
        "candidate_bias",
        "left_w",
        "left_bias",
        "right_w",
        "right_bias",
        "left_context_w",
    }
    assert params["action"]["candidate_bias"].shape == ()
    assert params["action"]["left_bias"].shape == ()
    assert params["action"]["right_bias"].shape == ()
    assert params["target"]["stop_bias"].shape == ()
    assert float(params["target"]["stop_bias"]) == -20.0
```

Update `test_init_policy_params_supports_configured_attention_layer_count` to construct:

```python
config = PolicyConfig(d_model=8, num_attention_layers=8, id_vocab_size=16)
```

Update `test_policy_config_defaults_match_phase_2_small_model` to assert:

```python
assert not hasattr(config, "max_candidates")
assert not hasattr(config, "max_side_terms")
```

- [ ] **Step 2: Run the focused failing tests**

Run:

```bash
cd /Users/longli/rcode/gristmill-symbolics/.worktrees/refactor-streamed-reinforce-policy/python
uv run pytest tests/test_policy_model.py tests/test_policy_tree.py -q
```

Expected: FAIL on `PolicyConfig` constructor/assertions until code is changed.

- [ ] **Step 3: Remove config caps and initialize shared biases**

In `python/gristmill_symbolics/policy/types.py`, make `PolicyConfig`:

```python
@dataclass(frozen=True)
class PolicyConfig:
    d_model: int = 32
    num_attention_layers: int = 1
    id_vocab_size: int = 128
    init_scale: float = 0.02
    stop_bias_init: float = -20.0
```

In `python/gristmill_symbolics/policy/model.py`, replace the action param block with:

```python
"action": {
    "candidate_w": _normal(next(keys), (d,), config.init_scale),
    "candidate_bias": jnp.asarray(0.0, dtype=jnp.float32),
    "left_w": _normal(next(keys), (d,), config.init_scale),
    "left_bias": jnp.asarray(0.0, dtype=jnp.float32),
    "right_w": _normal(next(keys), (d,), config.init_scale),
    "right_bias": jnp.asarray(0.0, dtype=jnp.float32),
    "left_context_w": _normal(next(keys), (d,), config.init_scale),
},
```

Do not change `_FIXED_PARAM_KEY_COUNT`; the number of random action weight vectors stays four.

- [ ] **Step 4: Run focused tests again**

Run:

```bash
cd /Users/longli/rcode/gristmill-symbolics/.worktrees/refactor-streamed-reinforce-policy/python
uv run pytest tests/test_policy_model.py tests/test_policy_tree.py -q
```

Expected: PASS for model/tree tests, with reinforce tests still allowed to fail later in this milestone.

- [ ] **Step 5: Commit milestone 1 config cleanup**

```bash
git add python/gristmill_symbolics/policy/types.py python/gristmill_symbolics/policy/model.py python/tests/test_policy_model.py python/tests/test_policy_tree.py
git commit -m "refactor: remove fixed policy action caps"
```

### Task 2: Make Sampling Return Choices Only

**Files:**

- Modify: `python/tests/test_policy_target.py:52`
- Modify: `python/tests/test_policy_action.py:41`
- Modify: `python/tests/test_policy_vmap.py:80`
- Modify: `python/tests/test_policy_jit_grad.py:46`
- Modify: `python/gristmill_symbolics/policy/api.py:90`
- Modify: `python/gristmill_symbolics/policy/api.py:638`

- [ ] **Step 1: Write failing choice-only target tests**

Update target sampling tests so log-probability is obtained by replay scoring:

```python
def test_target_all_masked_definitions_keep_stop_legal_when_stop_logit_is_tiny():
    params = _params()
    params = {
        **params,
        "target": {
            **params["target"],
            "stop_bias": jnp.asarray(-1.0e35, dtype=jnp.float32),
        },
    }
    state_tokens, state_mask = _state()
    def_mask = jnp.asarray([False])

    logp = score_target(
        params, state_tokens, state_mask, def_mask, jnp.asarray(-1, dtype=jnp.int32)
    )
    choice = sample_target(
        params, state_tokens, state_mask, def_mask, jax.random.PRNGKey(1)
    )

    assert float(logp) == pytest.approx(0.0)
    assert int(choice) == -1
    assert float(score_target(params, state_tokens, state_mask, def_mask, choice)) == pytest.approx(0.0)
```

Update `test_target_sampling_never_returns_masked_definition`:

```python
choice = sample_target(params, state_tokens, state_mask, def_mask, jax.random.PRNGKey(1))
logp = score_target(params, state_tokens, state_mask, def_mask, choice)

assert int(choice) == -1
assert float(logp) == pytest.approx(0.0)
```

Update `test_target_sampling_is_deterministic_for_same_rng`:

```python
left = sample_target(params, state_tokens, state_mask, def_mask, jax.random.PRNGKey(123))
right = sample_target(params, state_tokens, state_mask, def_mask, jax.random.PRNGKey(123))

assert int(left) == int(right)
assert float(score_target(params, state_tokens, state_mask, def_mask, left)) == pytest.approx(
    float(score_target(params, state_tokens, state_mask, def_mask, right))
)
```

- [ ] **Step 2: Write failing choice-only action tests**

Update `_sample()` in `python/tests/test_policy_action.py` so it returns `choice` only:

```python
choice = sample_action(
    params,
    state_tokens,
    state_mask,
    jnp.asarray(0, dtype=jnp.int32),
    action_tokens,
    action_mask,
    jax.random.PRNGKey(1),
)
return params, state_tokens, state_mask, action_tokens, action_mask, choice
```

Rename `test_action_sample_returns_padded_choice_tree_and_finite_logp` to `test_action_sample_returns_choice_tree_and_replay_scores_finite_logp`, and compute replay explicitly:

```python
replay = score_action(
    params,
    state_tokens,
    state_mask,
    jnp.asarray(0, dtype=jnp.int32),
    action_tokens,
    action_mask,
    choice,
)
assert replay.shape == ()
assert bool(jnp.isfinite(replay))
```

Replace `test_action_score_replays_sampled_logp` with deterministic replay:

```python
def test_action_score_replays_sampled_choice_deterministically():
    params, state_tokens, state_mask, action_tokens, action_mask, choice = _sample()

    left = score_action(
        params,
        state_tokens,
        state_mask,
        jnp.asarray(0, dtype=jnp.int32),
        action_tokens,
        action_mask,
        choice,
    )
    right = score_action(
        params,
        state_tokens,
        state_mask,
        jnp.asarray(0, dtype=jnp.int32),
        action_tokens,
        action_mask,
        choice,
    )

    assert float(left) == pytest.approx(float(right))
```

Update all destructuring from `(choice, _)` to `choice`.

- [ ] **Step 3: Update vmap and jit/grad tests for choice-only sampling**

In `python/tests/test_policy_vmap.py`, change target sample vmap:

```python
vmapped_choices = jax.vmap(sample_target, in_axes=(None, 0, 0, 0, 0))(
    params, state_tokens, state_mask, def_mask, keys
)
scalar_choices = jnp.asarray(
    [
        sample_target(
            params,
            _slice_tree(state_tokens, index),
            state_mask[index],
            def_mask[index],
            keys[index],
        )
        for index in range(2)
    ]
)
vmapped_logp = jax.vmap(score_target, in_axes=(None, 0, 0, 0, 0))(
    params, state_tokens, state_mask, def_mask, vmapped_choices
)
scalar_logp = jax.vmap(score_target, in_axes=(None, 0, 0, 0, 0))(
    params, state_tokens, state_mask, def_mask, scalar_choices
)
```

Change action sample vmap similarly:

```python
vmapped_choices = jax.vmap(sample_action, in_axes=(None, 0, 0, 0, 0, 0, 0))(
    params, state_tokens, state_mask, selected_defs, action_tokens, action_mask, keys
)
vmapped_logp = jax.vmap(score_action, in_axes=(None, 0, 0, 0, 0, 0, 0))(
    params, state_tokens, state_mask, selected_defs, action_tokens, action_mask, vmapped_choices
)
```

In `python/tests/test_policy_jit_grad.py`, update `_sampled_action`:

```python
choice = sample_action(
    params,
    state_tokens,
    state_mask,
    jnp.asarray(0, dtype=jnp.int32),
    action_tokens,
    action_mask,
    jax.random.PRNGKey(1),
)
return state_tokens, state_mask, action_tokens, action_mask, choice
```

- [ ] **Step 4: Run focused failing tests**

Run:

```bash
cd /Users/longli/rcode/gristmill-symbolics/.worktrees/refactor-streamed-reinforce-policy/python
uv run pytest tests/test_policy_target.py tests/test_policy_action.py tests/test_policy_vmap.py tests/test_policy_jit_grad.py -q
```

Expected: FAIL because `sample_target` and `sample_action` still return `(choice, logp)`.

- [ ] **Step 5: Implement choice-only sampling**

In `python/gristmill_symbolics/policy/api.py`, change `sample_target` to:

```python
def sample_target(params, state_tokens, state_token_mask, def_mask, rng):
    logits = _target_logits(params, state_tokens, state_token_mask, def_mask)
    legal = jnp.concatenate([jnp.asarray([True]), def_mask.astype(jnp.bool_)], axis=0)
    masked_logits = _mask_illegal_logits(logits, legal)
    sampled_index = jax.random.categorical(rng, masked_logits)
    return jnp.where(sampled_index == 0, -1, sampled_index - 1).astype(jnp.int32)
```

Change the final line of `sample_action` to:

```python
return {
    "candidate_index": candidate,
    "left_mask": left_mask,
    "left_valid_mask": left_valid,
    "right_mask": right_mask,
    "right_valid_mask": right_valid,
}
```

Do not add helper wrappers that return log-probabilities.

- [ ] **Step 6: Run focused tests again**

Run:

```bash
cd /Users/longli/rcode/gristmill-symbolics/.worktrees/refactor-streamed-reinforce-policy/python
uv run pytest tests/test_policy_target.py tests/test_policy_action.py tests/test_policy_vmap.py tests/test_policy_jit_grad.py -q
```

Expected: action tests may still fail on fixed `left_position_bias` references until Task 3; target tests should pass.

- [ ] **Step 7: Commit choice-only sampling**

```bash
git add python/gristmill_symbolics/policy/api.py python/tests/test_policy_target.py python/tests/test_policy_action.py python/tests/test_policy_vmap.py python/tests/test_policy_jit_grad.py
git commit -m "refactor: make policy sampling choice-only"
```

### Task 3: Replace Fixed Action Widths With Local Input Capacities

**Files:**

- Modify: `python/gristmill_symbolics/policy/api.py:103-107`
- Modify: `python/gristmill_symbolics/policy/api.py:204-299`
- Modify: `python/gristmill_symbolics/policy/api.py:352-410`
- Modify: `python/gristmill_symbolics/policy/api.py:504-715`
- Modify: `python/tests/test_policy_action.py:97`
- Modify: `python/tests/test_policy_vmap.py:21`

- [ ] **Step 1: Add wide action-space fixture helpers to action tests**

In `python/tests/test_policy_action.py`, add imports:

```python
import copy
from gristmill_symbolics.policy import stack_token_trees
```

Add helper functions:

```python
def _wide_action_space_snapshot(*, candidates=10, side_terms=6):
    snapshot = copy.deepcopy(_action_space().snapshot())
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


def _wide_action_space_tokens(*, candidates=10, side_terms=6):
    return tokenize_action_space_snapshot(
        _wide_action_space_snapshot(candidates=candidates, side_terms=side_terms)
    )
```

- [ ] **Step 2: Write failing dynamic capacity action test**

Add:

```python
def test_action_sample_scores_candidate_and_side_counts_above_removed_caps():
    params = _params()
    state_tokens, state_mask = _state()
    action_tokens, action_mask = _wide_action_space_tokens(candidates=10, side_terms=6)

    choice = sample_action(
        params,
        state_tokens,
        state_mask,
        jnp.asarray(0, dtype=jnp.int32),
        action_tokens,
        action_mask,
        jax.random.PRNGKey(7),
    )
    logp = score_action(
        params,
        state_tokens,
        state_mask,
        jnp.asarray(0, dtype=jnp.int32),
        action_tokens,
        action_mask,
        choice,
    )

    assert int(choice["candidate_index"]) < 10
    assert choice["left_mask"].shape == (action_mask.shape[0],)
    assert choice["right_mask"].shape == (action_mask.shape[0],)
    assert int(jnp.sum(choice["left_valid_mask"])) == 6
    assert int(jnp.sum(choice["right_valid_mask"])) == 6
    assert bool(jnp.isfinite(logp))
```

This test must not set or reference `PolicyConfig.max_candidates`, `PolicyConfig.max_side_terms`, `candidate_slot_bias`, `left_position_bias`, or `right_position_bias`.

- [ ] **Step 3: Write failing local action vmap padding test**

In `python/tests/test_policy_vmap.py`, add a local wide snapshot helper or import it from a shared test helper if Task 3 created one there. Add:

```python
def test_vmap_sample_action_uses_local_padding_width_not_model_config():
    params = _params()
    state_tokens, state_mask = _two_row_state_batch()
    small_action = _action_tree()
    wide_action = tokenize_action_space_snapshot(_wide_action_space_snapshot(candidates=10, side_terms=6))
    action_tokens, action_mask = stack_token_trees([small_action, wide_action])
    selected_defs = jnp.asarray([0, 0], dtype=jnp.int32)
    keys = jax.random.split(jax.random.PRNGKey(31), 2)

    choices = jax.vmap(sample_action, in_axes=(None, 0, 0, 0, 0, 0, 0))(
        params, state_tokens, state_mask, selected_defs, action_tokens, action_mask, keys
    )
    logp = jax.vmap(score_action, in_axes=(None, 0, 0, 0, 0, 0, 0))(
        params, state_tokens, state_mask, selected_defs, action_tokens, action_mask, choices
    )

    assert choices["left_mask"].shape == (2, action_mask.shape[1])
    assert choices["right_mask"].shape == (2, action_mask.shape[1])
    assert bool(jnp.all(jnp.isfinite(logp)))
```

- [ ] **Step 4: Run focused failing tests**

Run:

```bash
cd /Users/longli/rcode/gristmill-symbolics/.worktrees/refactor-streamed-reinforce-policy/python
uv run pytest tests/test_policy_action.py tests/test_policy_vmap.py -q
```

Expected: FAIL because action code still derives capacity from removed params.

- [ ] **Step 5: Replace action capacity helpers**

In `python/gristmill_symbolics/policy/api.py`, replace `_action_width` and `_candidate_indices` with:

```python
def _action_capacity(action_space_tokens):
    return int(action_space_tokens["token_kind"].shape[0])


def _candidate_indices(action_space_tokens):
    return jnp.arange(_action_capacity(action_space_tokens), dtype=jnp.int32)
```

Update all callers:

```python
indices = _candidate_indices(action_space_tokens)
width = _action_capacity(action_space_tokens)
```

For concrete validation, use the local action capacity:

```python
candidate_count = _action_capacity(action_space_tokens)
```

For side-mask shape validation:

```python
width = _action_capacity(action_space_tokens)
```

- [ ] **Step 6: Replace per-slot biases with shared scalar biases**

Update `_candidate_logits`:

```python
logits = jax.vmap(
    lambda embedding: jnp.dot(
        embedding + context, params["action"]["candidate_w"]
    )
    + params["action"]["candidate_bias"]
)(candidate_embeddings)
```

Update `_left_logits`:

```python
def _left_logits(params, context, candidate_embedding, left_embeddings):
    return jax.vmap(
        lambda embedding: jnp.dot(
            embedding + candidate_embedding + context, params["action"]["left_w"]
        )
        + params["action"]["left_bias"]
    )(left_embeddings)
```

Update `_right_logits`:

```python
def _right_logits(params, context, candidate_embedding, right_embeddings, left_summary):
    context_bias = jnp.dot(left_summary, params["action"]["left_context_w"])
    return jax.vmap(
        lambda embedding: jnp.dot(
            embedding + candidate_embedding + context, params["action"]["right_w"]
        )
        + params["action"]["right_bias"]
        + context_bias
    )(right_embeddings)
```

- [ ] **Step 7: Update final-bit constraint test**

Replace the old per-position bias override in `test_action_final_bit_constraint_prevents_empty_side_masks` with scalar biases:

```python
params = {
    **params,
    "action": {
        **params["action"],
        "left_w": jnp.zeros_like(params["action"]["left_w"]),
        "right_w": jnp.zeros_like(params["action"]["right_w"]),
        "left_context_w": jnp.zeros_like(params["action"]["left_context_w"]),
        "left_bias": jnp.asarray(-100.0, dtype=jnp.float32),
        "right_bias": jnp.asarray(-100.0, dtype=jnp.float32),
    },
}
```

- [ ] **Step 8: Run all policy tests**

Run:

```bash
cd /Users/longli/rcode/gristmill-symbolics/.worktrees/refactor-streamed-reinforce-policy/python
uv run pytest tests/test_policy_model.py tests/test_policy_tree.py tests/test_policy_target.py tests/test_policy_action.py tests/test_policy_vmap.py tests/test_policy_jit_grad.py tests/test_policy_tokenize_state.py tests/test_policy_tokenize_action.py tests/test_policy_package.py -q
```

Expected: PASS. Reinforce tests may still fail until Milestone 2.

- [ ] **Step 9: Run milestone 1 cleanup search**

Run:

```bash
cd /Users/longli/rcode/gristmill-symbolics/.worktrees/refactor-streamed-reinforce-policy
rg -n "RolloutTable|ScoreOutputs|LossDiagnostics|collect_rollout_batch|score_rollout|reinforce_loss|sampled_target_logp|sampled_action_logp|max_candidates|max_side_terms|candidate_slot_bias|left_position_bias|right_position_bias" python tests src
```

Expected after Milestone 1: `RolloutTable`, `ScoreOutputs`, `LossDiagnostics`, `collect_rollout_batch`, `score_rollout`, `reinforce_loss`, `sampled_target_logp`, and `sampled_action_logp` may still appear in reinforce live code/tests. No live policy code or policy tests should mention `max_candidates`, `max_side_terms`, `candidate_slot_bias`, `left_position_bias`, or `right_position_bias`.

- [ ] **Step 10: Commit local action capacity cleanup**

```bash
git add python/gristmill_symbolics/policy/api.py python/tests/test_policy_action.py python/tests/test_policy_vmap.py
git commit -m "refactor: use local action capacities in policy"
```

## Milestone 2: Streamed REINFORCE Trainer

### Task 4: Remove Old Public Table-Objective API From Tests First

**Files:**

- Modify: `python/tests/test_reinforce_package.py:1`
- Modify: `python/gristmill_symbolics/reinforce/types.py:83`
- Modify: `python/gristmill_symbolics/reinforce/__init__.py:1`

- [ ] **Step 1: Rewrite package export tests to reject old API**

In `python/tests/test_reinforce_package.py`, remove imports of `LossDiagnostics`, `RolloutTable`, and `ScoreOutputs`. Replace old export assertions with:

```python
def test_reinforce_package_exports_streamed_training_contracts():
    config = PolicyConfig(d_model=8)
    state = PolicyState(config=config, params={})

    assert state.config is config
    assert state.params == {}
    assert RolloutConfig(batch_size=2, max_steps=3).seed == 0
    assert RewardConfig().kind == "log_flops_improvement"
    assert BaselineConfig().standardize is False
    assert LossConfig().require_scored_terms is True
    assert OptimizerConfig().learning_rate == pytest.approx(1.0e-3)
    assert issubclass(TrainingError, RuntimeError)
    assert reinforce.CheckpointData is CheckpointData
    assert reinforce.FinalColumnMetrics is FinalColumnMetrics
    assert reinforce.TrainState is TrainState
    assert reinforce.UpdateMetrics is UpdateMetrics

    for removed in (
        "RolloutTable",
        "ScoreOutputs",
        "LossDiagnostics",
        "collect_rollout_batch",
        "score_rollout",
        "reinforce_loss",
    ):
        assert not hasattr(reinforce, removed)
```

- [ ] **Step 2: Run failing package test**

Run:

```bash
cd /Users/longli/rcode/gristmill-symbolics/.worktrees/refactor-streamed-reinforce-policy/python
uv run pytest tests/test_reinforce_package.py -q
```

Expected: FAIL until exports/types are removed.

- [ ] **Step 3: Remove old dataclasses and exports**

In `python/gristmill_symbolics/reinforce/types.py`, delete:

```python
@dataclass(frozen=True)
class RolloutTable: ...

@dataclass(frozen=True)
class ScoreOutputs: ...

@dataclass(frozen=True)
class LossDiagnostics: ...
```

In `python/gristmill_symbolics/reinforce/__init__.py`, remove imports and `__all__` entries for:

```text
collect_rollout_batch
LossDiagnostics
RolloutTable
ScoreOutputs
reinforce_loss
score_rollout
```

Keep `make_rng_grid`, `compute_rewards`, and `compute_advantages`.

- [ ] **Step 4: Run package test again**

Run:

```bash
cd /Users/longli/rcode/gristmill-symbolics/.worktrees/refactor-streamed-reinforce-policy/python
uv run pytest tests/test_reinforce_package.py -q
```

Expected: PASS for package exports; broader reinforce tests still fail until the training path is rewritten.

- [ ] **Step 5: Commit public API removal**

```bash
git add python/gristmill_symbolics/reinforce/types.py python/gristmill_symbolics/reinforce/__init__.py python/tests/test_reinforce_package.py
git commit -m "refactor: remove rollout table public API"
```

### Task 5: Add Streamed Rollout Gradient Collector

**Files:**

- Rewrite: `python/gristmill_symbolics/reinforce/rollout.py:45-612`
- Add: `python/tests/test_reinforce_streaming.py`
- Modify: `python/tests/test_reinforce_rollout.py:1`

- [ ] **Step 1: Replace old rollout tests with RNG-only public test**

Keep `test_make_rng_grid_uses_step_sample_decision_kind_axes` in `python/tests/test_reinforce_rollout.py`. Remove tests that call `collect_rollout_batch`.

- [ ] **Step 2: Write streamed one-step gradient test**

Create `python/tests/test_reinforce_streaming.py` with:

```python
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gristmill_symbolics import RewriteState, TensorComputation, validate_decision
from gristmill_symbolics.policy import (
    PolicyConfig,
    action_choice_to_python,
    init_policy_params,
    sample_action,
    sample_target,
    score_action,
    score_target,
    tokenize_action_space_snapshot,
    tokenize_state_snapshot,
)
from gristmill_symbolics.reinforce.rollout import (
    _collect_streamed_rollout_gradients,
    make_rng_grid,
)
from gristmill_symbolics.reinforce.types import (
    DECISION_ACTION,
    DECISION_TARGET,
    PolicyState,
    RolloutConfig,
)
from tests.policy_fixtures import actionable_json


def _policy(*, stop_bias_init=-20.0):
    config = PolicyConfig(d_model=8, stop_bias_init=stop_bias_init)
    return PolicyState(
        config=config,
        params=init_policy_params(config, jax.random.PRNGKey(0)),
    )


def _state_from_json(text):
    return RewriteState.from_computation(TensorComputation.from_json_string(text))


def _floating_leaves(tree):
    return [
        leaf
        for leaf in jax.tree_util.tree_leaves(tree)
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating)
    ]


def _tree_allclose(left, right, *, atol=1.0e-5):
    for left_leaf, right_leaf in zip(
        _floating_leaves(left), _floating_leaves(right), strict=True
    ):
        assert jnp.allclose(left_leaf, right_leaf, atol=atol, rtol=atol)


def test_streamed_one_step_accumulates_target_and_action_logp_grads():
    policy = _policy(stop_bias_init=-20.0)
    state = _state_from_json(actionable_json())
    config = RolloutConfig(batch_size=1, max_steps=1, seed=5)
    root = jax.random.PRNGKey(5)

    result = _collect_streamed_rollout_gradients(
        policy,
        [state],
        config,
        update_index=0,
        root_key=root,
    )

    rng_grid = make_rng_grid(root, update_index=0, max_steps=1, batch_size=1)
    state_tokens, state_mask = tokenize_state_snapshot(state.snapshot())
    def_mask = jnp.asarray(state.definition_mask(), dtype=jnp.bool_)
    target_choice = sample_target(
        policy.params,
        state_tokens,
        state_mask,
        def_mask,
        rng_grid[0, 0, DECISION_TARGET],
    )
    target_value_and_grad = jax.value_and_grad(score_target, argnums=0)
    target_logp, target_grad = target_value_and_grad(
        policy.params,
        state_tokens,
        state_mask,
        def_mask,
        target_choice,
    )
    space = state.action_space_for_def(int(np.asarray(target_choice)))
    assert space is not None
    action_tokens, action_mask = tokenize_action_space_snapshot(space.snapshot())
    action_choice = sample_action(
        policy.params,
        state_tokens,
        state_mask,
        target_choice,
        action_tokens,
        action_mask,
        rng_grid[0, 0, DECISION_ACTION],
    )
    action_value_and_grad = jax.value_and_grad(score_action, argnums=0)
    action_logp, action_grad = action_value_and_grad(
        policy.params,
        state_tokens,
        state_mask,
        target_choice,
        action_tokens,
        action_mask,
        action_choice,
    )

    expected_grad = jax.tree_util.tree_map(lambda left, right: left + right, target_grad, action_grad)
    assert result.trajectory_logp.shape == (1,)
    np.testing.assert_allclose(
        np.asarray(result.trajectory_logp[0]),
        np.asarray(target_logp + action_logp),
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    _tree_allclose(
        jax.tree_util.tree_map(lambda leaf: leaf[0], result.trajectory_grad_logp),
        expected_grad,
    )
```

- [ ] **Step 3: Write multi-step accumulation test**

Add a scalar direct-rollout oracle that does not use `RolloutTable`:

```python
def _two_actionable_json():
    import json

    data = json.loads(actionable_json())
    data["tensors"].append({"id": 4, "symmetry": []})
    data["definitions"].append({**data["definitions"][0], "base": 4})
    return json.dumps(data)


def _direct_scalar_rollout_logp_and_grad(policy, initial_state, *, max_steps, root_key):
    state = initial_state
    total_logp = jnp.asarray(0.0, dtype=jnp.float32)
    total_grad = jax.tree_util.tree_map(jnp.zeros_like, policy.params)
    rng_grid = make_rng_grid(root_key, update_index=0, max_steps=max_steps, batch_size=1)
    for step in range(max_steps):
        state_tokens, state_mask = tokenize_state_snapshot(state.snapshot())
        def_mask = jnp.asarray(state.definition_mask(), dtype=jnp.bool_)
        target_choice = sample_target(
            policy.params, state_tokens, state_mask, def_mask, rng_grid[step, 0, DECISION_TARGET]
        )
        target_logp, target_grad = jax.value_and_grad(score_target, argnums=0)(
            policy.params, state_tokens, state_mask, def_mask, target_choice
        )
        total_logp = total_logp + target_logp
        total_grad = jax.tree_util.tree_map(lambda left, right: left + right, total_grad, target_grad)
        if int(np.asarray(target_choice)) == -1:
            break
        space = state.action_space_for_def(int(np.asarray(target_choice)))
        if space is None:
            continue
        action_tokens, action_mask = tokenize_action_space_snapshot(space.snapshot())
        action_choice = sample_action(
            policy.params,
            state_tokens,
            state_mask,
            target_choice,
            action_tokens,
            action_mask,
            rng_grid[step, 0, DECISION_ACTION],
        )
        action_logp, action_grad = jax.value_and_grad(score_action, argnums=0)(
            policy.params,
            state_tokens,
            state_mask,
            target_choice,
            action_tokens,
            action_mask,
            action_choice,
        )
        total_logp = total_logp + action_logp
        total_grad = jax.tree_util.tree_map(lambda left, right: left + right, total_grad, action_grad)
        py_choice = action_choice_to_python(action_choice)
        decision = {
            "candidate_index": py_choice["candidate_index"],
            "left_mask": [
                keep
                for keep, valid in zip(py_choice["left_mask"], py_choice["left_valid_mask"])
                if valid
            ],
            "right_mask": [
                keep
                for keep, valid in zip(py_choice["right_mask"], py_choice["right_valid_mask"])
                if valid
            ],
        }
        validate_decision(space, decision)
        state.apply_validated_decision(space, decision)
    return total_logp, total_grad


def test_streamed_multi_step_accumulates_sum_t_grad_logp():
    policy = _policy(stop_bias_init=-20.0)
    state = _state_from_json(_two_actionable_json())
    root = jax.random.PRNGKey(16)
    result = _collect_streamed_rollout_gradients(
        policy,
        [state],
        RolloutConfig(batch_size=1, max_steps=2, seed=16),
        update_index=0,
        root_key=root,
    )
    expected_logp, expected_grad = _direct_scalar_rollout_logp_and_grad(
        policy, state, max_steps=2, root_key=root
    )

    np.testing.assert_allclose(np.asarray(result.trajectory_logp[0]), np.asarray(expected_logp), rtol=1.0e-5, atol=1.0e-5)
    _tree_allclose(
        jax.tree_util.tree_map(lambda leaf: leaf[0], result.trajectory_grad_logp),
        expected_grad,
    )
```

- [ ] **Step 4: Run failing streaming tests**

Run:

```bash
cd /Users/longli/rcode/gristmill-symbolics/.worktrees/refactor-streamed-reinforce-policy/python
uv run pytest tests/test_reinforce_rollout.py tests/test_reinforce_streaming.py -q
```

Expected: FAIL because `_collect_streamed_rollout_gradients` does not exist and old rollout imports still refer to `RolloutTable`.

- [ ] **Step 5: Implement streamed result dataclass and gradient helpers**

In `python/gristmill_symbolics/reinforce/rollout.py`, define:

```python
@dataclass(frozen=True)
class _StreamedRolloutResult:
    final: FinalColumnMetrics
    trajectory_logp: jax.Array
    trajectory_grad_logp: dict[str, object]
    valid_action_count: int
    stop_count: int
    empty_action_space_count: int
    finished_count: int
    target_score_count: int
    action_score_count: int
    target_logp_sum: float
    action_logp_sum: float
```

Add helpers:

```python
def _zero_trajectory_grad(params, batch_size: int):
    return jax.tree_util.tree_map(
        lambda leaf: jnp.zeros((batch_size, *leaf.shape), dtype=leaf.dtype),
        params,
    )


def _scatter_add_grad(accum, sample_indices, step_grad):
    sample_indices = jnp.asarray(sample_indices, dtype=jnp.int32)
    return jax.tree_util.tree_map(
        lambda acc_leaf, grad_leaf: acc_leaf.at[sample_indices].add(grad_leaf),
        accum,
        step_grad,
    )


def _sum_logp(values) -> float:
    if values.size == 0:
        return 0.0
    return float(np.asarray(jnp.sum(values)))
```

Add JAX helpers:

```python
_batched_target_value_and_grad = jax.jit(
    jax.vmap(jax.value_and_grad(score_target, argnums=0), in_axes=(None, 0, 0, 0, 0))
)

_batched_action_value_and_grad = jax.jit(
    jax.vmap(
        jax.value_and_grad(score_action, argnums=0),
        in_axes=(None, 0, 0, 0, 0, 0, 0),
    )
)
```

If module-level jitted callables cause import-time issues with tests, implement them as private functions returning the `jax.jit(jax.vmap(...))` callable and cache nothing.

- [ ] **Step 6: Implement `_sample_targets_for_active` as choice-only local batching**

Use the existing local padding structure, but return choices only:

```python
def _sample_targets_for_active(policy, state_items, target_def_masks, keys):
    if not state_items:
        return jnp.zeros((0,), dtype=jnp.int32)
    state_tokens, state_token_mask = stack_token_trees(state_items)
    def_length = _max_mask_length(target_def_masks)
    target_def_mask = jnp.stack(
        [_pad_bool_mask(mask, def_length) for mask in target_def_masks], axis=0
    )
    return jax.vmap(sample_target, in_axes=(None, 0, 0, 0, 0))(
        policy.params,
        state_tokens,
        state_token_mask,
        target_def_mask,
        jnp.stack(keys, axis=0),
    )
```

Add `_score_targets_for_active` returning `(target_logp, target_grad, state_tokens, state_token_mask, target_def_mask)` so action sampling can reuse the same `state_tokens` rows:

```python
def _score_targets_for_active(policy, state_items, target_def_masks, target_choices):
    state_tokens, state_token_mask = stack_token_trees(state_items)
    def_length = _max_mask_length(target_def_masks)
    target_def_mask = jnp.stack(
        [_pad_bool_mask(mask, def_length) for mask in target_def_masks], axis=0
    )
    target_logp, target_grad = _batched_target_value_and_grad(
        policy.params,
        state_tokens,
        state_token_mask,
        target_def_mask,
        target_choices,
    )
    return target_logp, target_grad, state_tokens, state_token_mask, target_def_mask
```

- [ ] **Step 7: Implement `_sample_actions_for_non_empty` as choice-only local batching**

Use current-state token rows gathered from the active target batch:

```python
def _sample_actions_for_non_empty(policy, state_tokens, state_token_mask, active_positions, selected_def_indices, action_items, keys):
    if not action_items:
        return {
            "candidate_index": jnp.zeros((0,), dtype=jnp.int32),
            "left_mask": jnp.zeros((0, 0), dtype=jnp.bool_),
            "left_valid_mask": jnp.zeros((0, 0), dtype=jnp.bool_),
            "right_mask": jnp.zeros((0, 0), dtype=jnp.bool_),
            "right_valid_mask": jnp.zeros((0, 0), dtype=jnp.bool_),
        }
    gathered_state_tokens = jax.tree_util.tree_map(lambda value: value[jnp.asarray(active_positions)], state_tokens)
    gathered_state_mask = state_token_mask[jnp.asarray(active_positions)]
    action_tokens, action_token_mask = stack_token_trees(action_items)
    selected = jnp.asarray(selected_def_indices, dtype=jnp.int32)
    return jax.vmap(sample_action, in_axes=(None, 0, 0, 0, 0, 0, 0))(
        policy.params,
        gathered_state_tokens,
        gathered_state_mask,
        selected,
        action_tokens,
        action_token_mask,
        jnp.stack(keys, axis=0),
    )
```

Add `_score_actions_for_non_empty` with the same gathered state rows and `_batched_action_value_and_grad`.

- [ ] **Step 8: Implement `_collect_streamed_rollout_gradients`**

Port the control flow from old `collect_rollout_batch` but delete `_SampleRecord`, `_assemble_rollout`, sampled logp fields, and dummy table rows. Keep:

```python
initial_log_flops = np.asarray(
    [state.log_total_flops() for state in initial_states], dtype=np.float64
)
row = RewriteStateRow.from_states(initial_states)
rng_grid = make_rng_grid(...)
active = [True] * config.batch_size
stopped = [False] * config.batch_size
exact_empty_def_masks = [None] * config.batch_size
trajectory_logp = jnp.zeros((config.batch_size,), dtype=jnp.float32)
trajectory_grad_logp = _zero_trajectory_grad(policy.params, config.batch_size)
```

Each target batch does:

```python
target_choices = _sample_targets_for_active(...)
target_logp, target_grad, batched_state_tokens, batched_state_mask, batched_def_mask = _score_targets_for_active(...)
trajectory_logp = trajectory_logp.at[jnp.asarray(active_indices)].add(target_logp)
trajectory_grad_logp = _scatter_add_grad(trajectory_grad_logp, active_indices, target_grad)
```

Each non-empty action batch does:

```python
action_choices = _sample_actions_for_non_empty(...)
action_logp, action_grad = _score_actions_for_non_empty(...)
trajectory_logp = trajectory_logp.at[jnp.asarray(non_empty_samples)].add(action_logp)
trajectory_grad_logp = _scatter_add_grad(trajectory_grad_logp, non_empty_samples, action_grad)
```

Row validation/application uses `action_choice_to_python(action_choice)` for each sampled action. Counts update in Python integers:

```python
target_score_count += len(active_indices)
action_score_count += len(non_empty_samples)
valid_action_count += len(non_empty_samples)
stop_count += ...
empty_action_space_count += ...
finished_count += ...
target_logp_sum += _sum_logp(target_logp)
action_logp_sum += _sum_logp(action_logp)
```

Return:

```python
return _StreamedRolloutResult(
    final=FinalColumnMetrics(...),
    trajectory_logp=trajectory_logp,
    trajectory_grad_logp=trajectory_grad_logp,
    valid_action_count=valid_action_count,
    stop_count=stop_count,
    empty_action_space_count=empty_action_space_count,
    finished_count=finished_count,
    target_score_count=target_score_count,
    action_score_count=action_score_count,
    target_logp_sum=target_logp_sum,
    action_logp_sum=action_logp_sum,
)
```

- [ ] **Step 9: Run streamed rollout tests**

Run:

```bash
cd /Users/longli/rcode/gristmill-symbolics/.worktrees/refactor-streamed-reinforce-policy/python
uv run pytest tests/test_reinforce_rollout.py tests/test_reinforce_streaming.py -q
```

Expected: PASS after fixing any mechanical direct-oracle issues. No test may import or construct `RolloutTable`.

- [ ] **Step 10: Commit streamed collector**

```bash
git add python/gristmill_symbolics/reinforce/rollout.py python/tests/test_reinforce_rollout.py python/tests/test_reinforce_streaming.py
git commit -m "refactor: stream rollout policy gradients"
```

### Task 6: Compute REINFORCE Gradients And Reward Metrics In train_update

**Files:**

- Modify: `python/gristmill_symbolics/reinforce/types.py:115`
- Modify: `python/gristmill_symbolics/reinforce/train_state.py:1-168`
- Modify: `python/tests/test_reinforce_train.py:64`
- Modify: `python/tests/test_reinforce_streaming.py`

- [ ] **Step 1: Add explicit metric fields**

In `UpdateMetrics`, insert:

```python
reward_stderr: float
objective_loss_mean: float
objective_loss_stderr: float
surrogate_loss: float
```

Keep:

```python
loss: float
```

`loss` must equal `objective_loss_mean` for CLI compatibility.

- [ ] **Step 2: Add private gradient reduction helper**

In `train_state.py`, add:

```python
def _reinforce_grad_loss(trajectory_grad_logp, advantage):
    stopped_advantage = jax.lax.stop_gradient(jnp.asarray(advantage, dtype=jnp.float32))

    def reduce_leaf(grad_leaf):
        scale = stopped_advantage.reshape(
            (stopped_advantage.shape[0],) + (1,) * (grad_leaf.ndim - 1)
        )
        return -jnp.mean(scale * grad_leaf, axis=0)

    return jax.tree_util.tree_map(reduce_leaf, trajectory_grad_logp)


def _surrogate_loss(trajectory_logp, advantage):
    stopped_advantage = jax.lax.stop_gradient(jnp.asarray(advantage, dtype=jnp.float32))
    return -jnp.mean(stopped_advantage * trajectory_logp)
```

- [ ] **Step 3: Test advantage-weighted gradient reduction**

In `python/tests/test_reinforce_streaming.py`, add:

```python
from gristmill_symbolics.reinforce.train_state import _reinforce_grad_loss, _surrogate_loss


def test_reinforce_grad_loss_is_negative_mean_advantage_times_trajectory_grad():
    trajectory_grad = {
        "leaf": jnp.asarray(
            [
                [1.0, 2.0],
                [3.0, 5.0],
                [-7.0, 11.0],
            ],
            dtype=jnp.float32,
        )
    }
    advantage = np.asarray([2.0, -1.0, 0.5], dtype=np.float64)

    grad_loss = _reinforce_grad_loss(trajectory_grad, advantage)

    expected = -jnp.mean(
        jnp.asarray(advantage, dtype=jnp.float32)[:, None] * trajectory_grad["leaf"],
        axis=0,
    )
    assert jnp.allclose(grad_loss["leaf"], expected)
```

Add surrogate test:

```python
def test_surrogate_loss_uses_trajectory_logp_diagnostic_only():
    logp = jnp.asarray([-1.0, -2.0, -4.0], dtype=jnp.float32)
    advantage = np.asarray([2.0, -1.0, 0.5], dtype=np.float64)

    assert float(_surrogate_loss(logp, advantage)) == pytest.approx(
        float(-jnp.mean(jnp.asarray(advantage, dtype=jnp.float32) * logp))
    )
```

- [ ] **Step 4: Run failing train/streaming tests**

Run:

```bash
cd /Users/longli/rcode/gristmill-symbolics/.worktrees/refactor-streamed-reinforce-policy/python
uv run pytest tests/test_reinforce_streaming.py tests/test_reinforce_train.py -q
```

Expected: FAIL until `train_update` uses streamed gradients and returns two values.

- [ ] **Step 5: Rewrite `train_update`**

In `train_state.py`, remove imports of `_reinforce_loss_value`, `reinforce_loss`, `score_rollout`, and `collect_rollout_batch`. Import:

```python
from .rollout import _collect_streamed_rollout_gradients
```

Change `train_update` flow:

```python
streamed = _collect_streamed_rollout_gradients(
    state.policy,
    initial_states,
    rollout_config,
    update_index=state.update_index,
    root_key=state.root_key,
)
reward = compute_rewards(streamed.final, reward_config)
advantage = compute_advantages(reward, baseline_config)
grads = _reinforce_grad_loss(streamed.trajectory_grad_logp, advantage)
surrogate_loss = _surrogate_loss(streamed.trajectory_logp, advantage)
```

Apply Optax once:

```python
optimizer = make_optimizer(state.optimizer_config)
updates, opt_state = optimizer.update(grads, state.opt_state, state.policy.params)
new_params = optax.apply_updates(state.policy.params, updates)
```

Validate finite params and params changed exactly as before.

Compute reward-objective metrics:

```python
reward_mean = float(np.mean(reward))
reward_std = float(np.std(reward))
reward_stderr = float(reward_std / np.sqrt(rollout_config.batch_size))
objective_loss_mean = -reward_mean
objective_loss_stderr = reward_stderr
```

Compute logp means from streamed sums:

```python
target_logp_mean = (
    streamed.target_logp_sum / streamed.target_score_count
    if streamed.target_score_count
    else 0.0
)
action_logp_mean = (
    streamed.action_logp_sum / streamed.action_score_count
    if streamed.action_score_count
    else 0.0
)
```

If `loss_config.require_scored_terms` and both score counts are zero, raise:

```python
raise TrainingError("no scored policy terms in rollout batch")
```

Return exactly:

```python
return new_state, metrics
```

- [ ] **Step 6: Update train tests for two-value return and reward objective**

In `python/tests/test_reinforce_train.py`, update all calls:

```python
new_state, metrics = train_update(...)
```

Remove table assertions and add:

```python
assert metrics.loss == pytest.approx(metrics.objective_loss_mean)
assert metrics.objective_loss_mean == pytest.approx(-metrics.reward_mean)
assert metrics.objective_loss_stderr == pytest.approx(metrics.reward_stderr)
assert np.isfinite(metrics.surrogate_loss)
```

Keep counts:

```python
assert metrics.target_score_count >= metrics.action_score_count
assert metrics.valid_action_count >= 1
```

- [ ] **Step 7: Run train and streaming tests**

Run:

```bash
cd /Users/longli/rcode/gristmill-symbolics/.worktrees/refactor-streamed-reinforce-policy/python
uv run pytest tests/test_reinforce_streaming.py tests/test_reinforce_train.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit streamed train update**

```bash
git add python/gristmill_symbolics/reinforce/types.py python/gristmill_symbolics/reinforce/train_state.py python/tests/test_reinforce_train.py python/tests/test_reinforce_streaming.py
git commit -m "refactor: train from streamed reinforce gradients"
```

### Task 7: Delete Table Objective Helpers And Update Remaining Reinforce Tests

**Files:**

- Modify: `python/gristmill_symbolics/reinforce/objective.py:1-203`
- Modify: `python/gristmill_symbolics/reinforce/train.py:47`
- Modify: `python/gristmill_symbolics/reinforce/checkpoint.py:27`
- Modify: `python/tests/test_reinforce_objective.py:1`
- Modify: `python/tests/test_reinforce_checkpoint.py:36`
- Modify: `python/tests/test_reinforce_cli.py:10`

- [ ] **Step 1: Shrink objective tests to rewards and advantages**

In `python/tests/test_reinforce_objective.py`, keep only:

```python
test_compute_rewards_uses_float64_log_flops_improvement
test_compute_rewards_rejects_non_1d_log_flops
test_compute_rewards_rejects_metric_shape_mismatch
test_compute_advantages_batch_mean_and_optional_standardization
```

Delete tests for `ScoreOutputs`, `reinforce_loss`, `_reinforce_loss_value`, `score_rollout`, and `collect_rollout_batch`.

- [ ] **Step 2: Delete table objective code**

In `objective.py`, keep imports:

```python
import numpy as np

from .types import BaselineConfig, FinalColumnMetrics, RewardConfig, TrainingError
```

Keep only:

```python
compute_rewards(...)
compute_advantages(...)
```

Delete:

```text
_masked_mean
_reinforce_loss_value
reinforce_loss
_slice_tree_2d
_slice_action_choice_2d
_finite_or_raise
score_rollout
```

- [ ] **Step 3: Update CLI for new config and return shape**

In `python/gristmill_symbolics/reinforce/train.py`, change initial policy config:

```python
PolicyConfig(d_model=8)
```

Change update call:

```python
train_state, metrics = train_update(
    train_state,
    initial_states,
    rollout_config,
    reward_config,
    baseline_config,
    loss_config,
)
```

- [ ] **Step 4: Update checkpoint tests for new config and metrics**

In `python/tests/test_reinforce_checkpoint.py`, remove `max_candidates` and `max_side_terms` from every `PolicyConfig(...)`.

Add new fields to the `UpdateMetrics(...)` fixture:

```python
reward_stderr=0.17677669529663687,
objective_loss_mean=-1.5,
objective_loss_stderr=0.17677669529663687,
surrogate_loss=-0.125,
```

Keep:

```python
loss=-1.5
```

because `loss` now means `objective_loss_mean`.

- [ ] **Step 5: Update CLI tests for objective metrics**

In `python/tests/test_reinforce_cli.py`, add assertions:

```python
assert "reward_stderr" in line
assert "objective_loss_mean" in line
assert "objective_loss_stderr" in line
assert "surrogate_loss" in line
assert line["loss"] == pytest.approx(line["objective_loss_mean"])
```

- [ ] **Step 6: Run remaining reinforce tests**

Run:

```bash
cd /Users/longli/rcode/gristmill-symbolics/.worktrees/refactor-streamed-reinforce-policy/python
uv run pytest tests/test_reinforce_objective.py tests/test_reinforce_checkpoint.py tests/test_reinforce_cli.py tests/test_reinforce_package.py tests/test_reinforce_rollout.py tests/test_reinforce_streaming.py tests/test_reinforce_train.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit objective/helper cleanup**

```bash
git add python/gristmill_symbolics/reinforce/objective.py python/gristmill_symbolics/reinforce/train.py python/gristmill_symbolics/reinforce/checkpoint.py python/tests/test_reinforce_objective.py python/tests/test_reinforce_checkpoint.py python/tests/test_reinforce_cli.py
git commit -m "refactor: remove table objective helpers"
```

### Task 8: Final Cleanup, Search, And Full Verification

**Files:**

- Verify all live Python code/tests
- Verify docs/spec mentions only

- [ ] **Step 1: Run full Python test suite**

Run:

```bash
cd /Users/longli/rcode/gristmill-symbolics/.worktrees/refactor-streamed-reinforce-policy/python
uv run pytest -q
```

Expected: PASS.

- [ ] **Step 2: Run Rust test suite**

Run:

```bash
cd /Users/longli/rcode/gristmill-symbolics/.worktrees/refactor-streamed-reinforce-policy
cargo test
```

Expected: PASS.

- [ ] **Step 3: Run hard cleanup search for live code/tests**

Run:

```bash
cd /Users/longli/rcode/gristmill-symbolics/.worktrees/refactor-streamed-reinforce-policy
rg -n "RolloutTable|ScoreOutputs|LossDiagnostics|collect_rollout_batch|score_rollout|reinforce_loss|sampled_target_logp|sampled_action_logp|max_candidates|max_side_terms|candidate_slot_bias|left_position_bias|right_position_bias" python tests src
```

Expected: no output. Any hit in live code/tests must be removed unless it names the removed API in a negative export/removal assertion; prefer avoiding even negative string mentions in live tests by storing removed names as split strings only if a cleanup search must be strictly empty.

- [ ] **Step 4: Run documentation-only search**

Run:

```bash
cd /Users/longli/rcode/gristmill-symbolics/.worktrees/refactor-streamed-reinforce-policy
rg -n "RolloutTable|ScoreOutputs|LossDiagnostics|collect_rollout_batch|score_rollout|reinforce_loss|sampled_target_logp|sampled_action_logp|max_candidates|max_side_terms|candidate_slot_bias|left_position_bias|right_position_bias" docs
```

Expected: hits in historical specs/plans are acceptable. Do not edit historical committed specs unless the user asks.

- [ ] **Step 5: Inspect final diff**

Run:

```bash
cd /Users/longli/rcode/gristmill-symbolics/.worktrees/refactor-streamed-reinforce-policy
git status --short
git diff --stat
git diff -- python/gristmill_symbolics/policy python/gristmill_symbolics/reinforce python/tests
```

Expected: no compatibility wrappers, no old table-objective path, no public policy-gradient batch class, no microbatching, and no edits outside the planned Python policy/reinforce/tests plus this plan file.

- [ ] **Step 6: Commit final cleanup if needed**

If Step 3 or Step 5 required cleanup edits:

```bash
git add python/gristmill_symbolics python/tests
git commit -m "test: align streamed reinforce cleanup checks"
```

## Self-Review Checklist

- Spec coverage: Milestone 1 tasks remove policy caps/slot biases, make sample APIs choice-only, preserve score APIs as sole logp APIs, add local `stack_token_trees` plus `vmap` tests, and preserve action-state row reuse in the rollout plan. Milestone 2 tasks remove table-objective public APIs, stream `value_and_grad(score_*, argnums=0)` during rollout, accumulate `trajectory_logp` and `trajectory_grad_logp`, compute `-mean_i stop(A_i) * G_i`, report reward-based metrics, keep `surrogate_loss`, and avoid public `PolicyGradientBatch`/microbatching.
- Placeholder scan: no step uses unresolved placeholder language. Implementation code snippets name concrete files, functions, and commands.
- Type consistency: `sample_target` returns `jax.Array`; `sample_action` returns `ActionChoiceTree`; `train_update` returns `tuple[TrainState, UpdateMetrics]`; `UpdateMetrics.loss` equals `objective_loss_mean`; private streamed collector returns `_StreamedRolloutResult`.
- Cleanup policy: after Milestone 1, old reinforce symbols may remain only because Milestone 2 has not run. After Milestone 2, cleanup search over `python tests src` must be empty for the listed old symbols unless an implementation reviewer explicitly decides that negative tests justify a string hit; the preferred final state is zero live-code/test hits.
