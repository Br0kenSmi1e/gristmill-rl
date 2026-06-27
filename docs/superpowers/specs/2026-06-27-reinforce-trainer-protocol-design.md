# REINFORCE Trainer Protocol Design

Status: planned
Depends on:
- `2026-06-26-reinforce-policy-refactor-suite-overview-design.md`
- `2026-06-27-expression-model-trainer-protocols-partial-design.md`
- `2026-06-27-current-transformer-model-adapter-design.md`
Feeds implementation plan: yes

## Summary

This spec defines the clean REINFORCE trainer protocol against the new model
protocol:

```python
new_params, new_opt_state, metrics = trainer.update(
    params,
    opt_state,
    batch,
    model,
    rng,
    config,
)
```

The trainer re-implements the current `train_update` objective and optimizer
behavior behind a smaller boundary. It does not preserve legacy rollout tables,
legacy rollout counters, or the old `train_update` API shape as required public
contracts.

For current REINFORCE, the trainer receives a batch of initial
`RewriteState` samples, computes reward-relevant initial costs, builds a
`RewriteStateRow`, and calls the model:

```python
out_row, logp, grad_logp, model_metrics = model.sample_with_logp_grad(
    params,
    rng,
    row,
    model_config,
)
```

The trainer uses only `out_row`, `logp`, and `grad_logp` for training.
`model_metrics` are optional diagnostics and must not affect objective,
gradient, or optimizer behavior.

## Accepted Boundary Decisions

### Trainer Batch

For the current REINFORCE trainer, `batch` is a `Sequence[RewriteState]`.

The trainer owns:

- validating that `len(batch) == config.batch_size`;
- computing `initial_log_flops` from the input states before row construction;
- building `RewriteStateRow.from_states(batch)`;
- passing that row to the model;
- computing `final_log_flops` from the returned `out_row`.

The model remains row-native. It receives a `RewriteStateRow`, may mutate it
during rollout, and returns the final `RewriteStateRow`.

This keeps the caller-facing training batch close to the current workflow while
still giving the model adapter the row-native boundary accepted in the model
adapter spec.

### Model Outputs Used By Trainer

The trainer depends on exactly these model outputs:

```text
out_row
logp
grad_logp
```

`logp` is the per-sample trajectory log probability with shape `[batch_size]`.
`grad_logp` is a params-shaped pytree whose trainable leaves have a leading
sample axis `[batch_size, *param_leaf_shape]`.

The trainer must validate enough shape and finiteness information to fail early
when these outputs cannot produce a valid update. It must not inspect model
internals such as token IDs, masks, padding, target choices, action choices,
exact-empty replay state, dummy rows, action-space snapshots, or Rust validation
details.

### Model Metrics

Model metrics are reporting-only diagnostics. The trainer may return them under
a separate namespace such as:

```python
metrics["model"] = model_metrics
```

Trainer behavior must not depend on model metrics. In particular, the trainer
must not require:

- `valid_action_count`;
- `stop_count`;
- `empty_action_space_count`;
- `finished_count`;
- `max_steps_count`;
- `target_score_count`;
- `action_score_count`;
- `target_logp_mean`;
- `action_logp_mean`;
- token or padding summaries.

Those values were useful during earlier rollout development, but requiring them
at the trainer boundary would leak current target/action rollout internals into
every future model backend.

## Config Ownership

The old `RolloutConfig` should be split or replaced. The target design has
separate trainer and model configuration.

Trainer configuration owns objective and update semantics:

```text
batch_size
reward_config
baseline_config
optimizer_config
```

No separate loss config is required for this spec. The old loss config only
controlled target/action scored-term validation, and that coupling is removed
below. A future trainer may add an explicit loss option when there is a real
choice to configure.

Model configuration owns rollout/model execution:

```text
max_steps
state_token_pad_to
definition_pad_to
action_token_pad_to
```

For the current transformer model adapter, static padding is mandatory.
`static_policy_batch` is not part of the clean protocol because static batching
is not a mode at this boundary.

RNG ownership is split as follows:

- the run state owns `root_key` and `update_index`;
- the update runner folds `update_index` into `root_key` to derive the update
  `rng`;
- `trainer.update` receives that `rng`;
- the trainer passes `rng` to `model.sample_with_logp_grad`;
- the model derives per-step and per-sample rollout keys internally.

The model adapter does not receive `root_key` or `update_index`.

## Mapping From Current `train_update`

The current implementation is:

```text
TrainState + Sequence[RewriteState] + rollout/reward/baseline/loss configs
  -> _collect_streamed_rollout_gradients(...)
  -> reward / advantage
  -> weighted grad_logp reduction
  -> optimizer update
  -> new TrainState + UpdateMetrics
```

The new implementation target is:

```text
params + opt_state + Sequence[RewriteState] + model + rng + config
  -> compute initial flops
  -> build RewriteStateRow
  -> model.sample_with_logp_grad(...)
  -> compute final flops from out_row
  -> reward / advantage
  -> weighted grad_logp reduction
  -> optimizer update
  -> new_params + new_opt_state + metrics
```

The behavior currently inside `_collect_streamed_rollout_gradients` moves behind
the model protocol. The trainer no longer owns rollout control.

## Trainer-Owned Behavior

The trainer owns:

- batch length validation;
- initial flop capture;
- row construction;
- final flop capture;
- reward calculation;
- baseline and advantage calculation;
- REINFORCE gradient reduction;
- surrogate loss diagnostic calculation;
- optimizer construction and update;
- updated-parameter finite validation;
- `params_changed` calculation;
- trainer metrics.

Reward remains:

```text
reward_i = initial_log_flops_i - final_log_flops_i
```

Advantage remains the current batch-baseline behavior:

```text
advantage_i = reward_i - mean_j reward_j
```

If advantage standardization is enabled, it uses the same semantics as the
current `compute_advantages` implementation.

The optimizer gradient remains:

```text
grad_loss = -mean_i stop_gradient(advantage_i) * grad_logp_i
```

The surrogate loss diagnostic remains:

```text
surrogate_loss = -mean_i stop_gradient(advantage_i) * logp_i
```

The optimizer is applied once per update.

The trainer validates updated params after applying optimizer updates and raises
`TrainingError` if any floating updated parameter is non-finite.

## Model-Owned Behavior

The model owns:

- target sampling;
- action sampling;
- target and action replay scoring;
- tokenization;
- padding;
- dummy policy rows;
- exact-empty replay;
- Rust row action-space queries;
- Rust action validation;
- Rust action application;
- final row mutation;
- trajectory log-probability accumulation;
- per-sample `grad_logp` accumulation;
- backend-private traces and diagnostics.

The current transformer model adapter must preserve current target/action
sampling, replay scoring, exact-empty replay, dummy masking, static pad failure,
and Rust-backed mutation behavior. Those are model responsibilities and are not
re-specified as trainer logic.

## Removed Legacy Couplings

The old `LossConfig.require_scored_terms` behavior depends on target/action
score counters. That check should not survive as a trainer requirement because
the clean model protocol does not expose target/action scored-term counts.

The replacement safety checks are protocol-level output validations:

- `logp` has shape `[batch_size]`;
- floating `logp` values are finite;
- `grad_logp` matches the params pytree structure;
- each floating gradient leaf has leading dimension `batch_size`;
- floating gradient leaves are finite;
- reward, advantage, and surrogate loss are finite.

The trainer should not infer whether a model rollout had target or action
decisions.

## Metrics

Trainer metrics should be compact. Required trainer metrics are:

```text
reward_mean
reward_std
objective_loss_mean
surrogate_loss
final_flops_best
params_changed
```

The update runner may add these useful log fields:

```text
update_index
batch_size
```

`objective_loss_mean` remains:

```text
objective_loss_mean = -reward_mean
```

`final_flops_best` is:

```text
min_i final_log_flops_i
```

The following legacy metrics are intentionally not part of the target trainer
metric schema:

- `initial_log_flops_mean`;
- `final_log_flops_mean`;
- `reward_stderr`;
- `advantage_mean`;
- `advantage_std`;
- `objective_loss_stderr`;
- rollout counters;
- target/action score counts;
- target/action log-probability means.

The trainer may include model diagnostics separately, but model diagnostics are
not trainer metrics and are not required for training correctness.

## Train State And Checkpoints

`TrainState` should map directly onto protocol state:

```text
params
opt_state
root_key
update_index
```

Model configuration belongs to the model or model config stored with the run.
Trainer configuration belongs to trainer/run config. Optimizer configuration is
trainer configuration, not mutable training state.

Checkpoint data should store:

```text
schema_version
train_state
model_config
trainer_config
recent_metrics
```

The trainer/checkpoint schema should not include tokenizer schema version in
this spec. Tokenizer and padding are model-private.

## Behavior Preserved

This spec preserves:

- symbolic rewrite semantics;
- Rust/PyO3 rewrite legality and application authority;
- reward as initial log flops minus final log flops;
- batch-baseline advantage behavior;
- optional advantage standardization behavior;
- `grad_loss = -mean_i stop(advantage_i) * grad_logp_i`;
- `surrogate_loss = -mean_i stop(advantage_i) * logp_i`;
- one optimizer update per rollout batch;
- finite updated-parameter validation;
- current transformer target/action probability semantics behind the model
  adapter;
- exact-empty replay behavior behind the model adapter;
- static dummy-row masking semantics behind the model adapter;
- static pad failure behavior behind the model adapter.

## Intentionally Deferred

This spec does not address:

- tokenizer or padding optimization;
- candidate and side-term shape-axis split;
- fused attention;
- a true seq2seq model;
- a new proposal result object;
- a public trace object;
- preserving legacy rollout counters;
- preserving the old `train_update` call shape as a required API;
- dynamic-shape rollout support in the current transformer adapter;
- checkpoint migration from old schemas.

## Testing Strategy

Trainer tests should use a fake model returning known `out_row`, `logp`, and
`grad_logp` values. Those tests should verify:

- batch length validation;
- initial and final flop use in reward;
- reward and advantage behavior;
- optional advantage standardization;
- REINFORCE gradient reduction;
- surrogate loss diagnostic;
- optimizer update;
- non-finite updated-parameter rejection;
- compact trainer metric contents.

Model adapter tests should verify the extracted current transformer behavior
separately from trainer tests:

- static rollout matches current rollout behavior within tolerance;
- exact-empty replay is preserved;
- dummy rows are masked out;
- target/action replay scoring is preserved;
- static pad failures name the failed dimension and configured size;
- `out_row`, `logp`, `grad_logp`, and optional model metrics have the expected
  shapes.

Run/checkpoint tests should verify:

- `TrainState` stores params, opt state, root key, and update index;
- the runner folds update index into the root key before update;
- update index increments once per successful update;
- model and trainer configs are checkpointed at run level;
- recent compact metrics round-trip.

## Acceptance Criteria

This design is accepted when the REINFORCE trainer:

- exposes `update(params, opt_state, batch, model, rng, config)`;
- treats current REINFORCE `batch` as `Sequence[RewriteState]`;
- computes initial flops before row construction;
- builds `RewriteStateRow` and passes it to the model;
- trains only from `out_row`, `logp`, and `grad_logp`;
- keeps model metrics diagnostic-only;
- owns reward, advantage, gradient weighting, optimizer update, param
  validation, and compact metrics;
- leaves rollout, tokenization, target/action choices, Rust apply, final row,
  logp, and grad_logp behind the model;
- splits trainer config from model config;
- maps `TrainState` to params, opt state, root key, and update index;
- avoids preserving unneeded legacy counters and old API shape requirements;
- keeps symbolic rewrite semantics unchanged;
- defers optimization work and model architecture changes.
