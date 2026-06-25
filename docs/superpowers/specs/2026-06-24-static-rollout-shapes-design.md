# Static Rollout Shapes Design

## Problem

The CCSD `batch_size=1, max_steps=64` profile in issue #20 shows that small
batch runtime is dominated by JAX/XLA compilation rather than Rust row work or
Python tokenization. The rollout loop currently feeds JAX policy calls with
step-dependent shapes:

- the number of active samples can shrink as samples stop;
- the number of non-empty action samples can vary by target choice;
- state token length changes as rewrites mutate the expression;
- action token length changes with the selected action space;
- definition-mask length can change as the row state changes.

JAX specializes compiled executables to concrete array shapes, so these varying
dimensions can cause repeated compilation within a single update.

## Goal

Add an opt-in static-shape rollout mode that stabilizes the JAX policy-call
input shapes used during REINFORCE rollout. The mode is intended to reduce
compile churn enough to make small-batch profiling and training more useful.

## Non-Goals

- Do not redesign the policy model or attention implementation.
- Do not reduce the full-attention memory requirement.
- Do not change default rollout behavior.
- Do not make `batch_size=8` memory-safe in this PR.
- Do not move Rust rewrite legality or action application into JAX.

## User-Facing Interface

Extend `RolloutConfig` with optional static-shape controls:

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

Add matching CLI flags:

```text
--state-token-pad-to N
--action-token-pad-to N
--definition-pad-to N
--static-policy-batch
```

All new options are off by default. Existing tests and existing invocations use
the current dynamic per-step shapes unless these flags are set.

When `static_policy_batch=True`, all three pad sizes are required:
`state_token_pad_to`, `action_token_pad_to`, and `definition_pad_to`.

## Shape Contract

When static-shape mode is enabled, each JAX policy work unit should see fixed
shapes for a run:

```text
target state tokens:       [batch_size, state_token_pad_to]
target definition masks:   [batch_size, definition_pad_to]
target RNG keys:           [batch_size, 2]

action state tokens:       [batch_size, state_token_pad_to]
selected definition index: [batch_size]
action tokens:             [batch_size, action_token_pad_to]
action RNG keys:           [batch_size, 2]
```

If a real state token tree, action token tree, or definition mask exceeds its
configured limit, rollout must fail fast with a `TrainingError` that names the
dimension and observed/configured lengths.

## Target Policy Path

The dynamic rollout currently builds target policy inputs only for
`active_indices`. Static-shape mode builds target inputs for every sample in the
physical batch on every step.

For active samples:

- use the real state snapshot tokens;
- use the real definition mask, including exact-empty replay handling;
- use the sample's target RNG key.

For inactive samples:

- use a valid dummy state row padded to the static state length;
- use an all-false definition mask so only STOP remains legal in
  `score_target`;
- use the sample's target RNG key, though the result is ignored.

After target sampling and scoring, inactive rows are masked out:

```python
target_choices = jnp.where(active_mask, target_choices, -1)
target_logps = jnp.where(active_mask, target_logps, 0.0)
target_grads = mask_tree_rows(target_grads, active_mask)
```

Only active samples update Python rollout state, query action spaces, or stop.

## Action Policy Path

The dynamic rollout currently builds action policy inputs only for
`non_empty_samples`. Static-shape mode builds action policy inputs for every
sample in the physical batch on every step.

For real non-empty action samples:

- use the selected state row from the target batch;
- use the selected target definition index;
- use real action-space tokens;
- use the sample's action RNG key.

For all other rows:

- use a valid dummy state row;
- use selected definition index `0`;
- use valid dummy action-space tokens;
- use the sample's action RNG key, though the result is ignored.

Dummy action rows must be valid enough for both `sample_action` and
`score_action`; invalid dummy rows would introduce non-finite masked-out values
and make debugging harder. The dummy action space is generated from a small
Python snapshot with one candidate, one left term, and one right term, then
tokenized through `tokenize_action_space_snapshot` instead of hand-writing token
arrays.

After action sampling and scoring, non-action rows are masked out:

```python
action_logps = jnp.where(action_active_mask, action_logps, 0.0)
action_grads = mask_tree_rows(action_grads, action_active_mask)
```

Only real non-empty action samples are converted to Python action choices,
validated by Rust, and applied to the row environment.

## Padding

Use the existing policy tree padding helper where possible:

```python
stack_token_trees(items, pad_to=...)
```

Apply it to:

- target/state token batches with `state_token_pad_to`;
- action token batches with `action_token_pad_to`.

Extend `_stack_bool_masks` to accept `pad_to` and use it for
`definition_pad_to`.

The initial CCSD profiling recommendation is:

```text
state_token_pad_to=3072
action_token_pad_to=4096
```

`definition_pad_to` is required when `static_policy_batch` is enabled. Choose it
after inspecting observed definition-mask lengths or compile logs.

## Gradient Masking

Masking must happen before gradients are added into `trajectory_grad_logp`.
The row mask is a length-`batch_size` boolean vector. Each gradient leaf has
shape:

```text
[batch_size, *param_leaf_shape]
```

Broadcast the mask over every non-batch dimension:

```python
scale = mask.astype(grad_leaf.dtype).reshape(
    (mask.shape[0],) + (1,) * (grad_leaf.ndim - 1)
)
masked_leaf = grad_leaf * scale
```

This preserves the existing streamed-gradient accumulation semantics while
ensuring dummy/inactive rows contribute exactly zero.

## Metrics And Environment Semantics

Static-shape mode must not change rollout metrics semantics:

- `target_score_count` counts real active target scoring events only.
- `action_score_count` counts real non-empty action scoring events only.
- `valid_action_count` counts real applied actions only.
- `stop_count`, `empty_action_space_count`, `finished_count`, and
  `max_steps_count` keep their current meanings.

Dummy rows are a JAX shape-stabilization detail and must not affect metrics,
Rust validation, action application, or final row state.

## Testing Strategy

Add tests before implementation:

- default rollout behavior remains unchanged when static flags are unset;
- static-shape rollout matches the existing scalar oracle on a one-sample
  actionable fixture;
- static-shape rollout matches current streamed behavior on mixed
  actionable/exact-empty batches;
- too-small `state_token_pad_to` raises a clear `TrainingError`;
- too-small `action_token_pad_to` raises a clear `TrainingError`;
- too-small `definition_pad_to` raises a clear `TrainingError`;
- inactive target rows and dummy action rows do not increase score or apply
  counts;
- dummy action rows produce finite masked-out logp/grad values.

## Profiling Strategy

Implement the mergeable static-shape change on a branch based on `main`, then
create a profiling branch on top by cherry-picking the existing profiling timer
commit `241bdc3`.

Run the CCSD profile on the RTX 4060 Ti machine with:

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

Compare against the issue #20 batch1 profile:

- total wall time;
- `backend_compile_and_load` cumulative time;
- `grep -c "Compiling" stderr.log`;
- rollout phase totals;
- max state/action token lengths;
- host RSS and GPU memory.

## Acceptance Criteria

- Existing dynamic behavior remains the default.
- Static-shape mode passes rollout and train tests.
- Static-shape mode fails fast on undersized static limits.
- Static-shape mode preserves rollout metrics and final behavior on small
  deterministic fixtures.
- CCSD batch1 profiling shows substantially fewer compiles or substantially less
  compile time.
- Any added padding overhead is reported separately from compile-time reduction.
