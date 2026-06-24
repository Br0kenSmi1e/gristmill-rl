# Static Rollout Shapes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in static-shape rollout mode that pads JAX policy-call inputs without changing default rollout behavior or policy semantics.

**Architecture:** Keep the existing streamed rollout as the single owner of rollout state and metrics. Add static-mode branches inside `python/gristmill_symbolics/reinforce/rollout.py` that build full physical-batch JAX inputs, pad state/action/definition dimensions to configured limits, mask dummy rows before accumulation, and leave Python row-environment updates restricted to real active samples. Expose the mode through `RolloutConfig` and CLI flags only.

**Tech Stack:** Python 3.11, JAX, NumPy, Optax, PyO3 bindings, pytest, `uv`, Rust `cargo test`.

---

## Baseline And Constraints

Worktree:

`/Users/longli/rcode/gristmill-symbolics/.worktrees/perf-reinforce-static-rollout-shapes`

Branch:

`perf-reinforce-static-rollout-shapes`

Baseline already verified before writing this plan:

```bash
cd python
uv sync --locked
uv run pytest -q
# 142 passed in 86.62s

cd ..
cargo test
# 129 Rust tests passed across unit/integration targets
```

Do not touch unrelated untracked files in the main checkout:

```text
.superpowers/
docs/superpowers/plans/2026-06-11-reinforce-phase-1-row-env.md
docs/superpowers/plans/2026-06-11-reinforce-phase-2-policy-model.md
docs/superpowers/plans/2026-06-12-reinforce-phase-3-training.md
```

## File Structure

Modify:

- `python/gristmill_symbolics/reinforce/types.py`: add static rollout fields to `RolloutConfig` and validate static-mode requirements.
- `python/gristmill_symbolics/reinforce/train.py`: add CLI flags and wire them into new `RolloutConfig` fields for new runs.
- `python/gristmill_symbolics/reinforce/rollout.py`: add padding validation helpers, dummy policy rows, row-gradient masking, and static target/action policy input paths.
- `python/tests/test_reinforce_package.py`: cover config defaults and validation.
- `python/tests/test_reinforce_cli.py`: cover CLI static flag wiring.
- `python/tests/test_reinforce_streaming.py`: cover static shape behavior, too-small pads, parity, metrics, and dummy row finiteness.

Do not modify:

- `python/gristmill_symbolics/policy/model.py`: no model redesign.
- `python/gristmill_symbolics/policy/api.py`: no policy behavior changes.
- Rust row-environment code: legality and action application stay in Rust.

---

### Task 1: Config And CLI Surface

**Files:**
- Modify: `python/gristmill_symbolics/reinforce/types.py`
- Modify: `python/gristmill_symbolics/reinforce/train.py`
- Test: `python/tests/test_reinforce_package.py`
- Test: `python/tests/test_reinforce_cli.py`

- [ ] **Step 1: Add failing config validation tests**

In `python/tests/test_reinforce_package.py`, extend `test_reinforce_package_exports_streamed_training_contracts` with:

```python
    assert RolloutConfig(batch_size=2, max_steps=3).state_token_pad_to is None
    assert RolloutConfig(batch_size=2, max_steps=3).action_token_pad_to is None
    assert RolloutConfig(batch_size=2, max_steps=3).definition_pad_to is None
    assert RolloutConfig(batch_size=2, max_steps=3).static_policy_batch is False
```

Add these tests after `test_rollout_config_validation_requires_integer_values`:

```python
@pytest.mark.parametrize(
    ("kwargs", "field_name"),
    [
        ({"state_token_pad_to": 0}, "state_token_pad_to"),
        ({"state_token_pad_to": True}, "state_token_pad_to"),
        ({"action_token_pad_to": 0}, "action_token_pad_to"),
        ({"action_token_pad_to": False}, "action_token_pad_to"),
        ({"definition_pad_to": 0}, "definition_pad_to"),
        ({"definition_pad_to": 1.5}, "definition_pad_to"),
        ({"static_policy_batch": 1}, "static_policy_batch"),
    ],
)
def test_rollout_config_validation_rejects_invalid_static_shape_fields(
    kwargs, field_name
):
    config = RolloutConfig(batch_size=1, max_steps=1, **kwargs)

    with pytest.raises(TrainingError, match=field_name):
        validate_rollout_config(config)


@pytest.mark.parametrize(
    "missing_field",
    [
        "state_token_pad_to",
        "action_token_pad_to",
        "definition_pad_to",
    ],
)
def test_rollout_config_validation_requires_all_static_pads(missing_field):
    kwargs = {
        "state_token_pad_to": 64,
        "action_token_pad_to": 64,
        "definition_pad_to": 4,
        "static_policy_batch": True,
    }
    kwargs[missing_field] = None

    with pytest.raises(TrainingError, match=missing_field):
        validate_rollout_config(RolloutConfig(batch_size=1, max_steps=1, **kwargs))
```

- [ ] **Step 2: Add failing CLI wiring tests**

In `python/tests/test_reinforce_cli.py`, add this test after `test_train_cli_completes_one_update_and_writes_checkpoint`:

```python
def test_train_cli_wires_static_rollout_flags_to_checkpoint(tmp_path, capsys):
    input_path = tmp_path / "actionable.json"
    checkpoint_path = tmp_path / "checkpoint.pkl"
    input_path.write_text(actionable_json())

    exit_code = main(
        [
            "--input",
            str(input_path),
            "--updates",
            "1",
            "--batch-size",
            "1",
            "--max-steps",
            "1",
            "--seed",
            "22",
            "--static-policy-batch",
            "--state-token-pad-to",
            "256",
            "--action-token-pad-to",
            "256",
            "--definition-pad-to",
            "4",
            "--checkpoint-out",
            str(checkpoint_path),
        ]
    )

    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    checkpoint = load_checkpoint(checkpoint_path)
    assert exit_code == 0
    assert line["batch_size"] == 1
    assert checkpoint.rollout_config.static_policy_batch is True
    assert checkpoint.rollout_config.state_token_pad_to == 256
    assert checkpoint.rollout_config.action_token_pad_to == 256
    assert checkpoint.rollout_config.definition_pad_to == 4
```

Add this parser rejection test near `test_train_cli_rejects_non_positive_updates`:

```python
@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--state-token-pad-to", "0"),
        ("--action-token-pad-to", "-1"),
        ("--definition-pad-to", "0"),
    ],
)
def test_train_cli_rejects_non_positive_static_pad_flags(tmp_path, flag, value):
    input_path = tmp_path / "actionable.json"
    input_path.write_text(actionable_json())

    with pytest.raises(SystemExit) as exc_info:
        main(["--input", str(input_path), flag, value])

    assert exc_info.value.code == 2
```

- [ ] **Step 3: Run the new tests and verify they fail**

Run:

```bash
cd python
uv run pytest \
  tests/test_reinforce_package.py::test_reinforce_package_exports_streamed_training_contracts \
  tests/test_reinforce_package.py::test_rollout_config_validation_rejects_invalid_static_shape_fields \
  tests/test_reinforce_package.py::test_rollout_config_validation_requires_all_static_pads \
  tests/test_reinforce_cli.py::test_train_cli_wires_static_rollout_flags_to_checkpoint \
  tests/test_reinforce_cli.py::test_train_cli_rejects_non_positive_static_pad_flags \
  -q
```

Expected: FAIL because `RolloutConfig` has no static fields and the CLI does not know the flags.

- [ ] **Step 4: Implement `RolloutConfig` fields and validation**

In `python/gristmill_symbolics/reinforce/types.py`, replace `RolloutConfig` with:

```python
@dataclass(frozen=True)
class RolloutConfig:
    batch_size: int
    max_steps: int
    seed: int = 0
    state_token_pad_to: int | None = None
    action_token_pad_to: int | None = None
    definition_pad_to: int | None = None
    static_policy_batch: bool = False
```

Add these helpers above `validate_rollout_config`:

```python
def _validate_optional_positive_int(name: str, value: int | None) -> None:
    if value is None:
        return
    if type(value) is not int:
        raise TrainingError(f"{name} must be an int or None")
    if value <= 0:
        raise TrainingError(f"{name} must be positive")
```

Extend `validate_rollout_config` after the existing `seed` checks:

```python
    _validate_optional_positive_int("state_token_pad_to", config.state_token_pad_to)
    _validate_optional_positive_int("action_token_pad_to", config.action_token_pad_to)
    _validate_optional_positive_int("definition_pad_to", config.definition_pad_to)
    if type(config.static_policy_batch) is not bool:
        raise TrainingError("static_policy_batch must be a bool")
    if config.static_policy_batch:
        if config.state_token_pad_to is None:
            raise TrainingError(
                "static_policy_batch requires state_token_pad_to"
            )
        if config.action_token_pad_to is None:
            raise TrainingError(
                "static_policy_batch requires action_token_pad_to"
            )
        if config.definition_pad_to is None:
            raise TrainingError(
                "static_policy_batch requires definition_pad_to"
            )
```

- [ ] **Step 5: Implement CLI flags**

In `python/gristmill_symbolics/reinforce/train.py`, add parser flags after `--seed`:

```python
    parser.add_argument("--static-policy-batch", action="store_true")
    parser.add_argument("--state-token-pad-to", type=_positive_int)
    parser.add_argument("--action-token-pad-to", type=_positive_int)
    parser.add_argument("--definition-pad-to", type=_positive_int)
```

When constructing `RolloutConfig` for a new run, include:

```python
            state_token_pad_to=args.state_token_pad_to,
            action_token_pad_to=args.action_token_pad_to,
            definition_pad_to=args.definition_pad_to,
            static_policy_batch=args.static_policy_batch,
```

Do not apply CLI rollout flags when `--checkpoint-in` is used; keep the existing checkpoint config semantics.

- [ ] **Step 6: Run focused tests and commit**

Run:

```bash
cd python
uv run pytest \
  tests/test_reinforce_package.py::test_reinforce_package_exports_streamed_training_contracts \
  tests/test_reinforce_package.py::test_rollout_config_validation_rejects_invalid_static_shape_fields \
  tests/test_reinforce_package.py::test_rollout_config_validation_requires_all_static_pads \
  tests/test_reinforce_cli.py::test_train_cli_wires_static_rollout_flags_to_checkpoint \
  tests/test_reinforce_cli.py::test_train_cli_rejects_non_positive_static_pad_flags \
  -q
```

Expected: PASS.

Commit:

```bash
git add python/gristmill_symbolics/reinforce/types.py python/gristmill_symbolics/reinforce/train.py python/tests/test_reinforce_package.py python/tests/test_reinforce_cli.py
git commit -m "feat: add static rollout config flags"
```

---

### Task 2: Padding, Dummy Rows, And Gradient Mask Helpers

**Files:**
- Modify: `python/gristmill_symbolics/reinforce/rollout.py`
- Test: `python/tests/test_reinforce_streaming.py`

- [ ] **Step 1: Add failing helper tests**

In `python/tests/test_reinforce_streaming.py`, extend the rollout import to include:

```python
    _dummy_action_policy_item,
    _dummy_state_policy_item,
    _mask_tree_rows,
    _stack_bool_masks,
```

Add these tests after `_tree_row`:

```python
def test_stack_bool_masks_can_pad_to_static_width():
    stacked = _stack_bool_masks(
        [
            jnp.asarray([True], dtype=jnp.bool_),
            jnp.asarray([False, True], dtype=jnp.bool_),
        ],
        pad_to=3,
    )

    assert stacked.shape == (2, 3)
    assert stacked.tolist() == [[True, False, False], [False, True, False]]


def test_stack_bool_masks_rejects_static_width_that_is_too_small():
    with pytest.raises(
        TrainingError,
        match="definition mask length 2 exceeds definition_pad_to 1",
    ):
        _stack_bool_masks(
            [jnp.asarray([True, False], dtype=jnp.bool_)],
            pad_to=1,
        )


def test_mask_tree_rows_zeroes_inactive_rows_without_changing_active_rows():
    grads = {
        "leaf": jnp.asarray(
            [
                [1.0, 2.0],
                [3.0, 5.0],
                [7.0, 11.0],
            ],
            dtype=jnp.float32,
        )
    }

    masked = _mask_tree_rows(
        grads,
        jnp.asarray([True, False, True], dtype=jnp.bool_),
    )

    assert masked["leaf"].tolist() == [[1.0, 2.0], [0.0, 0.0], [7.0, 11.0]]


def test_dummy_action_policy_inputs_score_finite_values():
    policy = _policy()
    state_tokens, state_mask = _dummy_state_policy_item()
    action_tokens, action_mask = _dummy_action_policy_item()
    selected = jnp.asarray(0, dtype=jnp.int32)
    action_choice = sample_action(
        policy.params,
        state_tokens,
        state_mask,
        selected,
        action_tokens,
        action_mask,
        jax.random.PRNGKey(99),
    )

    logp, grad = jax.value_and_grad(score_action, argnums=0)(
        policy.params,
        state_tokens,
        state_mask,
        selected,
        action_tokens,
        action_mask,
        action_choice,
    )

    assert np.isfinite(float(np.asarray(logp)))
    for leaf in _floating_leaves(grad):
        assert bool(jnp.all(jnp.isfinite(leaf)))
```

- [ ] **Step 2: Run helper tests and verify they fail**

Run:

```bash
cd python
uv run pytest \
  tests/test_reinforce_streaming.py::test_stack_bool_masks_can_pad_to_static_width \
  tests/test_reinforce_streaming.py::test_stack_bool_masks_rejects_static_width_that_is_too_small \
  tests/test_reinforce_streaming.py::test_mask_tree_rows_zeroes_inactive_rows_without_changing_active_rows \
  tests/test_reinforce_streaming.py::test_dummy_action_policy_inputs_score_finite_values \
  -q
```

Expected: FAIL because helpers do not exist and `_stack_bool_masks` has no `pad_to`.

- [ ] **Step 3: Add static padding validation helpers**

In `python/gristmill_symbolics/reinforce/rollout.py`, add these helpers above `_stack_bool_masks`:

```python
def _raise_static_pad_too_small(
    *, dimension: str, config_field: str, observed: int, configured: int
) -> None:
    raise TrainingError(
        f"{dimension} length {observed} exceeds {config_field} {configured}"
    )


def _validate_static_pad_limit(
    *,
    dimension: str,
    config_field: str,
    observed: int,
    configured: int | None,
) -> None:
    if configured is not None and observed > configured:
        _raise_static_pad_too_small(
            dimension=dimension,
            config_field=config_field,
            observed=observed,
            configured=configured,
        )


def _stack_token_trees_for_policy(
    items: list[tuple[TokenTree, jax.Array]],
    *,
    pad_to: int | None,
    dimension: str,
    config_field: str,
):
    if pad_to is not None:
        for _tokens, mask in items:
            _validate_static_pad_limit(
                dimension=dimension,
                config_field=config_field,
                observed=int(mask.shape[0]),
                configured=pad_to,
            )
    return stack_token_trees(items, pad_to=pad_to)
```

Replace `_stack_bool_masks` with:

```python
def _stack_bool_masks(masks: list[jax.Array], pad_to: int | None = None) -> jax.Array:
    length = int(pad_to) if pad_to is not None else _max_mask_length(masks)
    if pad_to is not None:
        for mask in masks:
            _validate_static_pad_limit(
                dimension="definition mask",
                config_field="definition_pad_to",
                observed=int(mask.shape[0]),
                configured=length,
            )
    return jnp.stack([_pad_bool_mask(mask, length) for mask in masks], axis=0)
```

- [ ] **Step 4: Add dummy policy rows**

In `python/gristmill_symbolics/reinforce/rollout.py`, add these constants above `_collect_streamed_rollout_gradients`:

```python
_DUMMY_STATE_SNAPSHOT = {
    "ranges": [],
    "tensors": [{"id": 0, "symmetry": []}],
    "definitions": [
        {
            "base": 0,
            "ext_indices": [],
            "terms": [],
        }
    ],
}

_DUMMY_TERM = {
    "coeff": {"numer": 1, "denom": 1},
    "sum_indices": [],
    "factors": [{"tensor": 0, "indices": []}],
}

_DUMMY_DEFINITION = {
    "base": 0,
    "ext_indices": [],
    "terms": [_DUMMY_TERM],
}

_DUMMY_ACTION_SPACE_SNAPSHOT = {
    "def_index": 0,
    "candidate_templates": [
        {
            "left_definition": _DUMMY_DEFINITION,
            "right_definition": _DUMMY_DEFINITION,
            "rewritten_definition": _DUMMY_DEFINITION,
        }
    ],
}
```

Add these helpers below `make_rng_grid`:

```python
def _dummy_state_policy_item() -> tuple[TokenTree, jax.Array]:
    return tokenize_state_snapshot(_DUMMY_STATE_SNAPSHOT)


def _dummy_definition_mask() -> jax.Array:
    return jnp.zeros((1,), dtype=jnp.bool_)


def _dummy_action_policy_item() -> tuple[TokenTree, jax.Array]:
    return tokenize_action_space_snapshot(_DUMMY_ACTION_SPACE_SNAPSHOT)
```

- [ ] **Step 5: Add row-gradient masking helper**

In `python/gristmill_symbolics/reinforce/rollout.py`, add below `_scatter_add_grad`:

```python
def _mask_tree_rows(tree, row_mask: jax.Array):
    row_mask = jnp.asarray(row_mask, dtype=jnp.bool_)

    def mask_leaf(leaf):
        leaf = jnp.asarray(leaf)
        scale = row_mask.astype(leaf.dtype).reshape(
            (row_mask.shape[0],) + (1,) * (leaf.ndim - 1)
        )
        return leaf * scale

    return jax.tree_util.tree_map(mask_leaf, tree)
```

- [ ] **Step 6: Run helper tests and commit**

Run:

```bash
cd python
uv run pytest \
  tests/test_reinforce_streaming.py::test_stack_bool_masks_can_pad_to_static_width \
  tests/test_reinforce_streaming.py::test_stack_bool_masks_rejects_static_width_that_is_too_small \
  tests/test_reinforce_streaming.py::test_mask_tree_rows_zeroes_inactive_rows_without_changing_active_rows \
  tests/test_reinforce_streaming.py::test_dummy_action_policy_inputs_score_finite_values \
  -q
```

Expected: PASS.

Commit:

```bash
git add python/gristmill_symbolics/reinforce/rollout.py python/tests/test_reinforce_streaming.py
git commit -m "feat: add static rollout padding helpers"
```

---

### Task 3: Static Target Policy Path

**Files:**
- Modify: `python/gristmill_symbolics/reinforce/rollout.py`
- Test: `python/tests/test_reinforce_streaming.py`

- [ ] **Step 1: Add static config test helper**

In `python/tests/test_reinforce_streaming.py`, add this helper after `_tree_row`:

```python
def _static_config(**overrides):
    kwargs = {
        "batch_size": 1,
        "max_steps": 1,
        "seed": 5,
        "state_token_pad_to": 512,
        "action_token_pad_to": 512,
        "definition_pad_to": 8,
        "static_policy_batch": True,
    }
    kwargs.update(overrides)
    return RolloutConfig(**kwargs)
```

- [ ] **Step 2: Add failing static target pad tests**

In `python/tests/test_reinforce_streaming.py`, add these tests after `test_streamed_rollout_rejects_non_floating_param_leaves`:

```python
def test_static_rollout_rejects_too_small_state_token_pad():
    policy = _policy()

    with pytest.raises(
        TrainingError,
        match="state token length .* exceeds state_token_pad_to 1",
    ):
        _collect_streamed_rollout_gradients(
            policy,
            [_state_from_json(actionable_json())],
            _static_config(state_token_pad_to=1),
            update_index=0,
            root_key=jax.random.PRNGKey(5),
        )


def test_static_rollout_rejects_too_small_definition_pad():
    policy = _policy()

    with pytest.raises(
        TrainingError,
        match="definition mask length 2 exceeds definition_pad_to 1",
    ):
        _collect_streamed_rollout_gradients(
            policy,
            [_state_from_json(_two_actionable_json())],
            _static_config(definition_pad_to=1),
            update_index=0,
            root_key=jax.random.PRNGKey(5),
        )
```

- [ ] **Step 3: Run tests and verify they fail**

Run:

```bash
cd python
uv run pytest \
  tests/test_reinforce_streaming.py::test_static_rollout_rejects_too_small_state_token_pad \
  tests/test_reinforce_streaming.py::test_static_rollout_rejects_too_small_definition_pad \
  -q
```

Expected: FAIL because static mode still uses dynamic target input batching.

- [ ] **Step 4: Implement static target batch selection**

In `_collect_streamed_rollout_gradients`, after `target_logp_sum = 0.0`, add:

```python
    static_policy_batch = config.static_policy_batch
    static_state_pad_to = config.state_token_pad_to if static_policy_batch else None
    static_definition_pad_to = (
        config.definition_pad_to if static_policy_batch else None
    )
    static_action_pad_to = config.action_token_pad_to if static_policy_batch else None
```

Inside the `for step in range(config.max_steps):` loop, replace:

```python
        if not active_indices:
            continue
```

with:

```python
        if not active_indices and not static_policy_batch:
            continue
```

After initializing `replay_exact_empty_samples`, add:

```python
        target_policy_samples = (
            list(range(config.batch_size)) if static_policy_batch else active_indices
        )
```

Change the target input loop from:

```python
        for sample in active_indices:
```

to:

```python
        for sample in target_policy_samples:
```

At the top of that loop, add:

```python
            if not active[sample]:
                state_items.append(_dummy_state_policy_item())
                target_def_masks.append(_dummy_definition_mask())
                target_keys.append(rng_grid[step, sample, DECISION_TARGET])
                continue
```

Replace target stacking:

```python
        state_tokens_batch, state_mask_batch = stack_token_trees(state_items)
        target_def_mask_batch = _stack_bool_masks(target_def_masks)
```

with:

```python
        state_tokens_batch, state_mask_batch = _stack_token_trees_for_policy(
            state_items,
            pad_to=static_state_pad_to,
            dimension="state token",
            config_field="state_token_pad_to",
        )
        target_def_mask_batch = _stack_bool_masks(
            target_def_masks,
            pad_to=static_definition_pad_to,
        )
```

After the `target_logps, target_grads = ...` call, add:

```python
        if static_policy_batch:
            target_active_mask = jnp.asarray(active, dtype=jnp.bool_)
            target_choices = jnp.where(target_active_mask, target_choices, -1)
            target_logps = jnp.where(target_active_mask, target_logps, 0.0)
            target_grads = _mask_tree_rows(target_grads, target_active_mask)
```

Change target accumulation:

```python
        trajectory_logp = trajectory_logp.at[jnp.asarray(active_indices)].add(
            target_logps
        )
        trajectory_grad_logp = _scatter_add_grad(
            trajectory_grad_logp, active_indices, target_grads
        )
```

to:

```python
        target_scatter_samples = target_policy_samples
        trajectory_logp = trajectory_logp.at[jnp.asarray(target_scatter_samples)].add(
            target_logps
        )
        trajectory_grad_logp = _scatter_add_grad(
            trajectory_grad_logp, target_scatter_samples, target_grads
        )
```

Keep metric semantics unchanged:

```python
        target_score_count += len(active_indices)
        target_logp_sum += float(np.asarray(jnp.sum(target_logps)))
```

Change `active_position_by_sample` to map target policy positions:

```python
        active_position_by_sample = {
            sample: position for position, sample in enumerate(target_policy_samples)
        }
```

The later loops still iterate over `active_indices`, so inactive rows do not update Python state.

- [ ] **Step 5: Run static target tests and commit**

Run:

```bash
cd python
uv run pytest \
  tests/test_reinforce_streaming.py::test_static_rollout_rejects_too_small_state_token_pad \
  tests/test_reinforce_streaming.py::test_static_rollout_rejects_too_small_definition_pad \
  tests/test_reinforce_streaming.py::test_streamed_rollout_accumulates_one_step_sampled_score_gradients \
  tests/test_reinforce_streaming.py::test_streamed_rollout_supports_batched_mixed_action_counts \
  -q
```

Expected: PASS.

Commit:

```bash
git add python/gristmill_symbolics/reinforce/rollout.py python/tests/test_reinforce_streaming.py
git commit -m "feat: pad static target rollout inputs"
```

---

### Task 4: Static Action Policy Path

**Files:**
- Modify: `python/gristmill_symbolics/reinforce/rollout.py`
- Test: `python/tests/test_reinforce_streaming.py`

- [ ] **Step 1: Add failing action pad test**

In `python/tests/test_reinforce_streaming.py`, add this test after `test_static_rollout_rejects_too_small_definition_pad`:

```python
def test_static_rollout_rejects_too_small_action_token_pad():
    policy = _policy()

    with pytest.raises(
        TrainingError,
        match="action token length .* exceeds action_token_pad_to 1",
    ):
        _collect_streamed_rollout_gradients(
            policy,
            [_state_from_json(actionable_json())],
            _static_config(action_token_pad_to=1),
            update_index=0,
            root_key=jax.random.PRNGKey(5),
        )
```

- [ ] **Step 2: Run the action pad test and verify it fails**

Run:

```bash
cd python
uv run pytest \
  tests/test_reinforce_streaming.py::test_static_rollout_rejects_too_small_action_token_pad \
  -q
```

Expected: FAIL because action inputs still use dynamic `stack_token_trees(action_items)`.

- [ ] **Step 3: Implement static action batch construction**

In `_collect_streamed_rollout_gradients`, replace:

```python
        if non_empty_samples:
            action_state_tokens = _take_tree_rows(
                state_tokens_batch, non_empty_active_positions
            )
            action_state_mask = state_mask_batch[jnp.asarray(non_empty_active_positions)]
            action_tokens_batch, action_mask_batch = stack_token_trees(action_items)
            selected = jnp.asarray(selected_def_indices, dtype=jnp.int32)
```

with:

```python
        if non_empty_samples or static_policy_batch:
            if static_policy_batch:
                non_empty_sample_set = set(non_empty_samples)
                non_empty_position_by_sample = {
                    sample: position
                    for position, sample in enumerate(non_empty_samples)
                }
                target_position_by_sample = active_position_by_sample
                action_policy_samples = list(range(config.batch_size))
                action_state_items: list[tuple[TokenTree, jax.Array]] = []
                action_policy_items: list[tuple[TokenTree, jax.Array]] = []
                selected_def_policy_indices: list[int] = []
                action_keys_for_policy: list[jax.Array] = []
                for sample in action_policy_samples:
                    if sample in non_empty_sample_set:
                        target_position = target_position_by_sample[sample]
                        action_state_items.append(
                            (
                                _slice_tree(state_tokens_batch, target_position),
                                state_mask_batch[target_position],
                            )
                        )
                        action_position = non_empty_position_by_sample[sample]
                        action_policy_items.append(action_items[action_position])
                        selected_def_policy_indices.append(
                            selected_def_indices[action_position]
                        )
                    else:
                        action_state_items.append(_dummy_state_policy_item())
                        action_policy_items.append(_dummy_action_policy_item())
                        selected_def_policy_indices.append(0)
                    action_keys_for_policy.append(
                        rng_grid[step, sample, DECISION_ACTION]
                    )

                action_state_tokens, action_state_mask = _stack_token_trees_for_policy(
                    action_state_items,
                    pad_to=static_state_pad_to,
                    dimension="state token",
                    config_field="state_token_pad_to",
                )
                action_tokens_batch, action_mask_batch = _stack_token_trees_for_policy(
                    action_policy_items,
                    pad_to=static_action_pad_to,
                    dimension="action token",
                    config_field="action_token_pad_to",
                )
                selected = jnp.asarray(selected_def_policy_indices, dtype=jnp.int32)
                stacked_action_keys = jnp.stack(action_keys_for_policy, axis=0)
                action_position_by_sample = {
                    sample: sample for sample in range(config.batch_size)
                }
            else:
                action_policy_samples = non_empty_samples
                action_state_tokens = _take_tree_rows(
                    state_tokens_batch, non_empty_active_positions
                )
                action_state_mask = state_mask_batch[
                    jnp.asarray(non_empty_active_positions)
                ]
                action_tokens_batch, action_mask_batch = stack_token_trees(action_items)
                selected = jnp.asarray(selected_def_indices, dtype=jnp.int32)
                stacked_action_keys = jnp.stack(action_keys, axis=0)
                action_position_by_sample = {
                    sample: position
                    for position, sample in enumerate(non_empty_samples)
                }
```

In the `sample_action` call, replace:

```python
                jnp.stack(action_keys, axis=0),
```

with:

```python
                stacked_action_keys,
```

After the `action_logps, action_grads = ...` call, add:

```python
            if static_policy_batch:
                action_active_mask = jnp.asarray(
                    [sample in set(non_empty_samples) for sample in action_policy_samples],
                    dtype=jnp.bool_,
                )
                action_logps = jnp.where(action_active_mask, action_logps, 0.0)
                action_grads = _mask_tree_rows(action_grads, action_active_mask)
```

Replace action accumulation:

```python
            trajectory_logp = trajectory_logp.at[jnp.asarray(non_empty_samples)].add(
                action_logps
            )
            trajectory_grad_logp = _scatter_add_grad(
                trajectory_grad_logp, non_empty_samples, action_grads
            )
```

with:

```python
            trajectory_logp = trajectory_logp.at[jnp.asarray(action_policy_samples)].add(
                action_logps
            )
            trajectory_grad_logp = _scatter_add_grad(
                trajectory_grad_logp, action_policy_samples, action_grads
            )
```

Keep metric semantics unchanged:

```python
            action_score_count += len(non_empty_samples)
            action_logp_sum += float(np.asarray(jnp.sum(action_logps)))
```

Replace action choice conversion:

```python
            for position, sample in enumerate(non_empty_samples):
                action_choices_for_row[sample] = action_choice_to_python(
                    _slice_tree(action_choices, position)
                )
```

with:

```python
            for sample in non_empty_samples:
                action_choices_for_row[sample] = action_choice_to_python(
                    _slice_tree(action_choices, action_position_by_sample[sample])
                )
```

This keeps Rust validation and row application restricted to `non_empty_samples`.

- [ ] **Step 4: Run action path tests and commit**

Run:

```bash
cd python
uv run pytest \
  tests/test_reinforce_streaming.py::test_static_rollout_rejects_too_small_action_token_pad \
  tests/test_reinforce_streaming.py::test_dummy_action_policy_inputs_score_finite_values \
  tests/test_reinforce_streaming.py::test_streamed_rollout_accumulates_one_step_sampled_score_gradients \
  tests/test_reinforce_streaming.py::test_streamed_rollout_supports_batched_mixed_action_counts \
  -q
```

Expected: PASS.

Commit:

```bash
git add python/gristmill_symbolics/reinforce/rollout.py python/tests/test_reinforce_streaming.py
git commit -m "feat: pad static action rollout inputs"
```

---

### Task 5: Static Rollout Parity And Metrics Semantics

**Files:**
- Modify: `python/gristmill_symbolics/reinforce/rollout.py`
- Test: `python/tests/test_reinforce_streaming.py`
- Test: `python/tests/test_reinforce_train.py`

- [ ] **Step 1: Add failing/static parity tests**

In `python/tests/test_reinforce_streaming.py`, add these tests after `test_streamed_rollout_accumulates_multi_step_scalar_oracle`:

```python
def test_static_rollout_matches_scalar_oracle_on_one_sample():
    policy = _policy()
    root = jax.random.PRNGKey(17)
    config = _static_config(max_steps=2, seed=17)

    result = _collect_streamed_rollout_gradients(
        policy,
        [_state_from_json(_two_actionable_json())],
        config,
        update_index=0,
        root_key=root,
    )
    expected_logp, expected_grad = _scalar_rollout_oracle(
        policy,
        _state_from_json(_two_actionable_json()),
        config,
        update_index=0,
        root_key=root,
    )

    assert jnp.allclose(result.trajectory_logp[0], expected_logp, atol=1.0e-5)
    _tree_allclose(_tree_row(result.trajectory_grad_logp, 0), expected_grad)
    assert result.target_score_count >= 1
    assert result.action_score_count >= 1
```

Add these tests after `test_streamed_rollout_supports_batched_mixed_action_counts`:

```python
def test_static_rollout_matches_dynamic_streamed_mixed_batch():
    policy = _policy()
    root = jax.random.PRNGKey(23)
    initial_states = [
        _state_from_json(actionable_json()),
        _state_from_json(exact_empty_json()),
        _state_from_json(_two_actionable_json()),
    ]
    dynamic_config = RolloutConfig(batch_size=3, max_steps=2, seed=23)
    static_config = _static_config(batch_size=3, max_steps=2, seed=23)

    dynamic = _collect_streamed_rollout_gradients(
        policy,
        [_state_from_json(actionable_json()), _state_from_json(exact_empty_json()), _state_from_json(_two_actionable_json())],
        dynamic_config,
        update_index=0,
        root_key=root,
    )
    static = _collect_streamed_rollout_gradients(
        policy,
        initial_states,
        static_config,
        update_index=0,
        root_key=root,
    )

    assert np.allclose(static.final.initial_log_flops, dynamic.final.initial_log_flops)
    assert np.allclose(static.final.final_log_flops, dynamic.final.final_log_flops)
    assert static.final.stopped.tolist() == dynamic.final.stopped.tolist()
    assert static.final.max_steps.tolist() == dynamic.final.max_steps.tolist()
    assert jnp.allclose(static.trajectory_logp, dynamic.trajectory_logp, atol=1.0e-5)
    _tree_allclose(static.trajectory_grad_logp, dynamic.trajectory_grad_logp)
    assert static.valid_action_count == dynamic.valid_action_count
    assert static.stop_count == dynamic.stop_count
    assert static.empty_action_space_count == dynamic.empty_action_space_count
    assert static.finished_count == dynamic.finished_count
    assert static.target_score_count == dynamic.target_score_count
    assert static.action_score_count == dynamic.action_score_count
    assert static.target_logp_sum == pytest.approx(dynamic.target_logp_sum, abs=1.0e-5)
    assert static.action_logp_sum == pytest.approx(dynamic.action_logp_sum, abs=1.0e-5)


def test_static_rollout_inactive_rows_do_not_increase_metrics_or_apply_counts():
    policy = _policy(stop_bias_init=100.0)

    result = _collect_streamed_rollout_gradients(
        policy,
        [_state_from_json(actionable_json())],
        _static_config(max_steps=3, seed=31),
        update_index=0,
        root_key=jax.random.PRNGKey(31),
    )

    assert result.stop_count == 1
    assert result.finished_count == 2
    assert result.valid_action_count == 0
    assert result.target_score_count == 1
    assert result.action_score_count == 0
    assert result.target_logp_sum == pytest.approx(float(result.trajectory_logp[0]))
    assert result.action_logp_sum == pytest.approx(0.0)
    assert np.isfinite(np.asarray(result.trajectory_logp)).all()
    for leaf in _floating_leaves(result.trajectory_grad_logp):
        assert bool(jnp.all(jnp.isfinite(leaf)))
```

In `test_static_rollout_matches_dynamic_streamed_mixed_batch`, keep the repeated dynamic-state construction. It avoids sharing mutable `RewriteState` instances between dynamic and static runs.

- [ ] **Step 2: Add train-update static behavior test**

In `python/tests/test_reinforce_train.py`, add this test after `test_multi_sample_update_reports_finite_loss_and_core_metrics`:

```python
def test_train_update_accepts_static_rollout_config_and_preserves_metrics():
    state = init_train_state(
        PolicyConfig(d_model=8, stop_bias_init=-20.0),
        OptimizerConfig(learning_rate=1.0e-2),
        seed=29,
    )

    new_state, metrics = train_update(
        state,
        _mixed_initial_states(),
        RolloutConfig(
            batch_size=2,
            max_steps=2,
            seed=29,
            static_policy_batch=True,
            state_token_pad_to=512,
            action_token_pad_to=512,
            definition_pad_to=8,
        ),
    )

    assert new_state.update_index == 1
    assert metrics.batch_size == 2
    assert metrics.max_steps == 2
    assert np.isfinite(metrics.loss)
    assert np.isfinite(metrics.surrogate_loss)
    assert metrics.target_score_count >= metrics.action_score_count
    assert metrics.valid_action_count >= 1
```

- [ ] **Step 3: Run parity tests and verify failures**

Run:

```bash
cd python
uv run pytest \
  tests/test_reinforce_streaming.py::test_static_rollout_matches_scalar_oracle_on_one_sample \
  tests/test_reinforce_streaming.py::test_static_rollout_matches_dynamic_streamed_mixed_batch \
  tests/test_reinforce_streaming.py::test_static_rollout_inactive_rows_do_not_increase_metrics_or_apply_counts \
  tests/test_reinforce_train.py::test_train_update_accepts_static_rollout_config_and_preserves_metrics \
  -q
```

Expected: FAIL if static target/action masking, metric counts, or dummy rows are incomplete.

- [ ] **Step 4: Fix static masking and count semantics**

In `python/gristmill_symbolics/reinforce/rollout.py`, verify these invariants in the implementation and adjust code until the tests pass:

```python
target_score_count += len(active_indices)
target_logp_sum += float(np.asarray(jnp.sum(target_logps)))
```

`target_logps` must already be zeroed for inactive rows in static mode.

```python
action_score_count += len(non_empty_samples)
action_logp_sum += float(np.asarray(jnp.sum(action_logps)))
```

`action_logps` must already be zeroed for dummy rows in static mode.

Only these loops may update Python rollout state:

```python
for position, sample in enumerate(active_indices):
    ...

for sample in active_indices:
    ...

for sample in non_empty_samples:
    ...
```

Do not iterate over `range(config.batch_size)` for Rust validation, row application, stop counts, exact-empty replay updates, valid-action counts, or empty-action-space counts.

- [ ] **Step 5: Run parity tests and commit**

Run:

```bash
cd python
uv run pytest \
  tests/test_reinforce_streaming.py::test_static_rollout_matches_scalar_oracle_on_one_sample \
  tests/test_reinforce_streaming.py::test_static_rollout_matches_dynamic_streamed_mixed_batch \
  tests/test_reinforce_streaming.py::test_static_rollout_inactive_rows_do_not_increase_metrics_or_apply_counts \
  tests/test_reinforce_train.py::test_train_update_accepts_static_rollout_config_and_preserves_metrics \
  -q
```

Expected: PASS.

Commit:

```bash
git add python/gristmill_symbolics/reinforce/rollout.py python/tests/test_reinforce_streaming.py python/tests/test_reinforce_train.py
git commit -m "test: verify static rollout semantics"
```

---

### Task 6: Final Verification And Profiling Notes

**Files:**
- Read: `docs/superpowers/specs/2026-06-24-static-rollout-shapes-design.md`
- Read: `python/gristmill_symbolics/reinforce/rollout.py`
- Read: `python/tests/test_reinforce_streaming.py`

- [ ] **Step 1: Run static rollout focused tests**

Run:

```bash
cd python
uv run pytest \
  tests/test_reinforce_package.py \
  tests/test_reinforce_cli.py \
  tests/test_reinforce_streaming.py \
  tests/test_reinforce_train.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run policy regression tests**

Run:

```bash
cd python
uv run pytest \
  tests/test_policy_tree.py \
  tests/test_policy_target.py \
  tests/test_policy_action.py \
  tests/test_policy_jit_grad.py \
  tests/test_policy_vmap.py \
  -q
```

Expected: PASS.

- [ ] **Step 3: Run full Python suite**

Run:

```bash
cd python
uv run pytest -q
```

Expected: PASS.

- [ ] **Step 4: Run Rust suite**

Run:

```bash
cargo test
```

Expected: PASS.

- [ ] **Step 5: Inspect diff for accidental behavior changes**

Run:

```bash
git diff main...HEAD -- python/gristmill_symbolics/reinforce python/tests
```

Confirm:

- Dynamic rollout uses `stack_token_trees(..., pad_to=None)` behavior and existing active/non-empty batching when `static_policy_batch` is false.
- Static mode is the only path using configured pad sizes.
- Static pad failures raise `TrainingError` and name the dimension plus observed/configured lengths.
- Metrics count only active target rows, real non-empty action rows, and real applied actions.
- No policy model or policy scoring logic changed.

- [ ] **Step 6: Optional profiling branch, no optimization yet**

Only after mergeable correctness work is complete, create a profiling branch on top of this branch and cherry-pick timer commit `241bdc3`:

```bash
git checkout -b perf-reinforce-static-rollout-shapes-profile
git cherry-pick 241bdc3
```

On the RTX 4060 Ti machine, run:

```bash
GRISTMILL_PROFILE_ROLLOUT=1 \
GRISTMILL_PROFILE_ROLLOUT_SYNC=1 \
JAX_LOG_COMPILES=1 \
uv run --no-sync python -m gristmill_symbolics.reinforce.train \
  --input "$INPUT" \
  --updates 1 \
  --batch-size 1 \
  --max-steps 64 \
  --seed 42 \
  --static-policy-batch \
  --state-token-pad-to 3072 \
  --action-token-pad-to 4096 \
  --definition-pad-to "$DEF_PAD"
```

Record total wall time, `backend_compile_and_load` time, `grep -c "Compiling" stderr.log`, rollout phase totals, max state/action token lengths, host RSS, and GPU memory. Do not optimize based on speculation; only act on profiling evidence in a separate follow-up.

- [ ] **Step 7: Final commit if needed**

If Task 6 found only test or comment adjustments:

```bash
git add python/gristmill_symbolics/reinforce python/tests
git commit -m "test: cover static rollout verification"
```

If there are no additional changes, do not create an empty commit.

---

## Self-Review

Spec coverage:

- Opt-in mode only: Task 1 adds defaults and CLI flags off by default; Tasks 3 and 4 branch on `config.static_policy_batch`.
- No model redesign: No changes planned in `policy/model.py` or `policy/api.py`.
- No default behavior change: Dynamic path remains the current active/non-empty batching path.
- No speculative optimization: Task 6 profiling is observational only.
- Pads state tokens, action tokens, definition masks, target batch rows, and action batch rows: Tasks 2, 3, and 4 cover all dimensions.
- CLI flags: Task 1 adds `--static-policy-batch`, `--state-token-pad-to`, `--action-token-pad-to`, and `--definition-pad-to`.
- Fail-fast `TrainingError`: Tasks 2, 3, and 4 add explicit too-small pad checks.
- Metrics semantics: Task 5 asserts count and logp semantics.
- Tests before implementation: Every implementation task starts with failing tests.

Placeholder scan:

- No placeholder markers are present.
- Each task has exact files, test code, implementation code, commands, and expected outcomes.

Type consistency:

- New config names match the spec exactly: `static_policy_batch`, `state_token_pad_to`, `action_token_pad_to`, `definition_pad_to`.
- Helper names are consistent across tests and implementation: `_stack_token_trees_for_policy`, `_stack_bool_masks`, `_mask_tree_rows`, `_dummy_state_policy_item`, `_dummy_action_policy_item`.
