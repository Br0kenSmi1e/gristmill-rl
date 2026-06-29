# Current Transformer Model Adapter Design

Status: planned
Depends on:
- `2026-06-26-reinforce-policy-refactor-suite-overview-design.md`
- `2026-06-27-expression-model-trainer-protocols-partial-design.md`
Feeds implementation plan: yes

## Summary

This spec defines how the current target/action transformer rollout fits behind
the top-level model protocol:

```python
out_row, logp, grad_logp, metrics = model.sample_with_logp_grad(
    params,
    rng,
    row,
    config,
)
```

The adapter is intentionally narrow. It preserves the current symbolic rewrite
and target/action policy behavior while changing the public boundary to a
row-native model call. It does not introduce a new proposal result object, a new
tokenizer contract, or a trainer implementation plan.

The adapter is static-shape only. Dynamic per-step policy shapes are deprecated
for this boundary because they make the rollout code harder to reason about and
perform too poorly for the intended training path.

## Accepted Boundary Decisions

### Input Row

`row` is a `RewriteStateRow`.

The adapter does not accept `Sequence[RewriteState]` and does not define a small
row wrapper. Constructing a `RewriteStateRow` from individual `RewriteState`
objects is a caller-side convenience and belongs outside the model protocol.

### Row Mutability

The input row is consumed by the adapter. During rollout, validated actions are
applied through the existing Rust/PyO3 row API, mutating the row in place.

`out_row` is the final `RewriteStateRow`, normally the same object that was
passed in. Callers that need the original row state must compute or copy what
they need before calling the model.

This keeps current behavior and avoids adding row clone or snapshot
reconstruction machinery to this spec.

### Output Row

`out_row` is a bare `RewriteStateRow`.

The adapter does not return final snapshots, final costs, or a richer result
object. The protocol return shape stays exactly:

```text
RewriteStateRow, per-sample logp, per-sample grad_logp, metrics mapping
```

### Cost And Reward Ownership

Flop calculation is not model responsibility.

The trainer or reward layer owns cost evaluation:

```text
initial_log_flops = row.log_total_flops()
out_row, logp, grad_logp, model_metrics = model.sample_with_logp_grad(...)
final_log_flops = out_row.log_total_flops()
reward / advantage / optimizer update
```

Because the adapter consumes the row, callers must read initial costs before
passing the row into `sample_with_logp_grad`.

### Metrics

The adapter exposes only one public metric:

```python
metrics = {
    "stopped": stopped,
}
```

Where `stopped` is a NumPy boolean array with shape `[batch_size]`. A row is
marked stopped when that sample selected the target STOP choice during rollout.

The following current rollout counters are intentionally not part of the model
boundary:

- `valid_action_count`;
- `stop_count`;
- `empty_action_space_count`;
- `finished_count`;
- `target_score_count`;
- `action_score_count`;
- `target_logp_sum`;
- `action_logp_sum`;
- `max_steps`.

Reasons:

- `stop_count` is derivable from `stopped`.
- `max_steps` is derivable from `~stopped` under the current lifecycle, where
  STOP is the only terminal event.
- `finished_count` is rollout-loop bookkeeping.
- target/action score counts and log-probability sums are legacy diagnostics,
  not training inputs.
- valid-action and exact-empty counts are useful debug details, but they expose
  backend-specific rollout internals.

If implementation temporarily retains these values for private tests or debug
logging, they must not become trainer-facing protocol requirements.

## Static-Shape-Only Contract

The current transformer adapter requires global static rollout shapes. The
configuration accepted by this adapter must provide concrete positive values
for:

```text
batch_size
max_steps
state_token_pad_to
definition_pad_to
action_token_pad_to
```

There is no adapter-level dynamic-shape path. Every target and action policy
call uses the full physical batch.

The old `static_policy_batch` switch is not part of this adapter boundary.
Static shape is mandatory, not an optional mode.

Target policy call shapes:

```text
state tokens:       [batch_size, state_token_pad_to]
state token mask:   [batch_size, state_token_pad_to]
definition mask:    [batch_size, definition_pad_to]
target RNG keys:    [batch_size, ...]
```

Action policy call shapes:

```text
state tokens:       [batch_size, state_token_pad_to]
state token mask:   [batch_size, state_token_pad_to]
selected def index: [batch_size]
action tokens:      [batch_size, action_token_pad_to]
action token mask:  [batch_size, action_token_pad_to]
action RNG keys:    [batch_size, ...]
```

Inactive rows, stopped rows, and rows without a real non-empty action space use
valid dummy policy inputs. Their sampled and scored log-probability rows and
gradient rows are masked to zero before trajectory accumulation. Dummy rows must
not affect `out_row`, `logp`, `grad_logp`, or `metrics["stopped"]`.

If any real state tokens, definition mask, or action-space tokens exceed the
configured static pad, the adapter raises `TrainingError` with an error message
that names the dimension and observed/configured lengths.

Dynamic-shape rollout is deprecated for this adapter. It should not be optimized
or treated as the target architecture. A later implementation may remove or
bypass dynamic-path code while extracting the static path.

## Mapping From Current Streamed Rollout

The adapter behavior is a direct extraction of the static branch of today's
`_collect_streamed_rollout_gradients` into a model method.

Current loop behavior remains:

```text
for each step:
  read row snapshots and definition masks
  tokenize current state snapshots
  build full-batch static target policy inputs
  sample target choices
  score target choices with value_and_grad(score_target)
  mask inactive rows to zero
  handle STOP choices
  query Rust row action spaces for active non-STOP samples
  preserve exact-empty replay behavior
  tokenize real non-empty action spaces
  build full-batch static action policy inputs
  sample action choices
  score actions with value_and_grad(score_action)
  mask non-action rows to zero
  validate real action choices through Rust
  apply validated actions through the Rust row API
```

Protocol outputs map as:

- accumulated current `trajectory_logp` becomes `logp`;
- accumulated current `trajectory_grad_logp` becomes `grad_logp`;
- the final mutable row becomes `out_row`;
- the current `stopped` vector becomes `metrics["stopped"]`.

`logp` has shape `[batch_size]` and is the per-sample sum of all sampled/scored
target and action log-probability terms. `grad_logp` is a pytree matching
`params`, where every trainable leaf has a leading batch axis
`[batch_size, *param_leaf_shape]`.

## RNG Ownership

The model protocol receives `rng` directly. The adapter should derive the
per-step, per-sample target/action key grid from this key.

Update-index folding belongs outside the model protocol. A stateful trainer may
fold an update index into its root key before calling `sample_with_logp_grad`,
but the adapter itself does not require `update_index` or `root_key`
parameters.

For fixed params, input row, config, and `rng`, the adapter should preserve the
current sampled target/action behavior within the static-shape path.

## Model-Private Internals

The following are private implementation details of the current transformer
adapter:

- tokenizer calls and token tree structures;
- state token padding;
- definition-mask padding;
- action-space token padding;
- dummy state rows and dummy action rows;
- target choice semantics, including STOP as `-1`;
- action choice semantics, including candidate index and left/right side masks;
- action-space snapshot tokenization;
- exact-empty replay state;
- target/action RNG splitting per step and sample;
- use of `batched_sample_target`, `batched_score_target_grad`,
  `batched_sample_action`, and `batched_score_action_grad`;
- target/action split diagnostics and debug counters.

The trainer must not inspect token arrays, token masks, definition masks,
action-space snapshots, target/action choices, exact-empty replay state, or
dummy rows. It only consumes:

```text
out_row
logp
grad_logp
metrics["stopped"]
```

Rust/PyO3 remains the authority for action-space generation, rewrite validation,
rewrite application, and final row mutation.

## Helper Surface

This refactor should keep most existing rollout helpers intact. The intended
implementation shape is to extract and simplify the existing static path, not to
rewrite the helper layer.

Helpers that should remain usable with little or no semantic change include:

- dummy policy input helpers;
- trajectory-gradient zeroing and scatter-add helpers;
- row masking helpers;
- tree row slicing and gathering helpers;
- static pad validation helpers;
- token-tree and boolean-mask stacking helpers;
- exact-empty mask helpers;
- reusable batched policy APIs in `policy.batched`.

Expected simplifications are local:

- helper signatures can stop accepting optional pad lengths where static pads
  are required;
- `_max_mask_length` and active-subset dynamic stacking become unnecessary for
  this adapter boundary;
- `make_rng_grid` or its replacement should derive keys from the protocol
  `rng`, not from `root_key` plus `update_index`;
- `_collect_streamed_rollout_gradients` can become the body of the adapter or a
  private static-only helper taking `(params, rng, row, config)`.

The policy model, tokenizer semantics, Rust row APIs, and target/action
probability calculations should remain essentially unchanged.

## Behavior Preserved

The adapter must preserve:

- symbolic rewrite semantics;
- Rust/PyO3 rewrite legality and application authority;
- current target/action probability semantics in `policy.api`;
- replay scoring through `score_target` and `score_action`;
- exact-empty replay behavior;
- static dummy-row masking semantics;
- trajectory log-probability accumulation;
- per-sample gradient accumulation;
- finite masked-out dummy-row logp and gradient behavior;
- static pad failure behavior for too-small configured dimensions.

## Intentionally Deferred

This spec does not address:

- tokenizer or padding performance improvements beyond requiring static shapes;
- candidate and side-term shape-axis split;
- fused attention or other memory-efficient attention implementations;
- trainer refactor details;
- revised public training metric dataclasses;
- replacing target/action rollout with a true end-to-end expression model;
- public proposal trace or proposal result objects;
- row clone/copy semantics;
- preserving legacy target/action log-probability mean diagnostics.

## Acceptance Criteria

This design is accepted when it clearly establishes that the current transformer
adapter:

- implements `sample_with_logp_grad(params, rng, row, config)`;
- accepts and consumes a `RewriteStateRow`;
- returns a bare final `RewriteStateRow`;
- returns per-sample trajectory `logp`;
- returns per-sample `grad_logp`;
- exposes only `metrics["stopped"]`;
- requires global static padding;
- keeps tokenizer, padding, target/action choices, dummy rows, and exact-empty
  replay private to the model;
- leaves cost/reward computation to the trainer or reward layer;
- preserves current symbolic rewrite and policy behavior.
