# Expression Model And Trainer Protocols Partial Design

Status: partial
Depends on: `2026-06-26-reinforce-policy-refactor-suite-overview-design.md`
Feeds implementation plan: no

## Summary

This partial spec records the first-level protocols for expression-in /
expression-out models and trainers. It intentionally covers only the top-level
model/trainer boundary agreed so far. Tokenizer protocols, step-wise model
internals, end-to-end model internals, padding shapes, and concrete rollout
refactors are deferred to later specs.

The top-level system has two protocols:

```text
Model:   expression row -> expression row + logp + grad_logp
Trainer: params + opt_state + batch + model -> new params + new opt_state
```

Tokenizer and padding are model-private infrastructure at this level.

## Model Protocol

The model protocol is expression-row in, expression-row out. A model owns all
work needed to turn an input symbolic row into a proposed output symbolic row.

Canonical shape:

```python
class Model(Protocol):
    def sample_with_logp_grad(
        self,
        params,
        rng,
        row,
        config,
    ) -> tuple[object, jax.Array, object, Mapping[str, object]]:
        ...
```

Canonical call:

```python
out_row, logp, grad_logp, metrics = model.sample_with_logp_grad(
    params,
    rng,
    row,
    config,
)
```

The return values mean:

- `out_row`: proposed output expression row. In the current system this is a
  `RewriteStateRow` or row-like object.
- `logp`: per-sample log probability, shape `[batch_size]`.
- `grad_logp`: pytree matching `params`, where every trainable leaf has leading
  batch axis `[batch_size, *param_leaf_shape]`.
- `metrics`: model/proposal diagnostics. This may include rollout counts,
  stop counts, invalid counts, timing summaries, or backend-specific metrics.

The model owns:

- tokenizer use;
- JAX network calls;
- step-wise rollout or end-to-end decoding;
- action-space querying when the backend needs it;
- Rust validation and rewrite application;
- final row construction;
- log-probability calculation;
- per-sample `grad_logp` calculation;
- backend-private traces.

The model does not own optimizer state, reward calculation, baseline/advantage
calculation, or parameter updates.

## Trainer Protocol

The trainer protocol owns parameter updates. It is not specific to REINFORCE,
supervised learning, or any other training objective.

Canonical shape:

```python
class Trainer(Protocol):
    def update(
        self,
        params,
        opt_state,
        batch,
        model: Model,
        rng,
        config,
    ) -> tuple[object, object, Mapping[str, object]]:
        ...
```

Canonical call:

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

The return values mean:

- `new_params`: updated model parameters.
- `new_opt_state`: updated optimizer state.
- `metrics`: trainer metrics and selected model metrics.

The input values mean:

- `params`: current model parameters.
- `opt_state`: current optimizer state.
- `batch`: trainer-specific update data. For current REINFORCE this is the
  initial expression row batch. For supervised training this may be input rows
  plus target rows.
- `model`: object satisfying the model protocol required by this trainer.
- `rng`: stochastic update key or root key.
- `config`: trainer-specific configuration.

The trainer owns:

- calling the model protocol;
- computing objective-specific losses or rewards;
- reducing per-sample model gradients into a parameter gradient;
- applying optimizer updates;
- validating updated parameters;
- trainer metrics and checkpoint-facing state.

The trainer must not inspect:

- token IDs;
- token masks;
- padding layout;
- action-space internals;
- target/action choices;
- biclique left/right masks;
- seq2seq decoder traces;
- model-private traces.

## Current REINFORCE Mapping

The current `train_update` implementation can be viewed as a concrete trainer
whose model call is currently embedded in rollout code.

Current flow:

```text
initial RewriteState batch
  -> _collect_streamed_rollout_gradients(policy, initial_states, ...)
  -> trajectory_logp, trajectory_grad_logp, final costs, rollout metrics
  -> reward and advantage
  -> weighted policy gradient
  -> optax update
```

In the new boundary, `_collect_streamed_rollout_gradients` behavior belongs
behind `Model.sample_with_logp_grad`, and `train_update` keeps only trainer
responsibilities:

```text
model.sample_with_logp_grad
  -> reward / loss computation
  -> gradient reduction
  -> optimizer update
  -> trainer metrics
```

## Tokenizer Boundary At This Level

Tokenizer is not a first-level protocol in this partial spec. It is private
model infrastructure.

The current model may still use an internal tokenizer with this concrete shape:

```python
tokens, token_mask, def_mask = tokenizer.encode_batch(row, shapes)
```

Where:

```text
tokens leaves: [batch_size, state_token_pad_to]
token_mask:    [batch_size, state_token_pad_to]
def_mask:      [batch_size, definition_pad_to]
```

This tokenizer contract should be specified in a later model-internals or
tokenizer/padding spec.

## Deferred Specs

Later specs should define:

- tokenizer and padding contracts;
- current step-wise model internals;
- end-to-end model internals;
- step-wise-to-end-to-end adaptation;
- REINFORCE trainer details;
- supervised or other trainer details if needed;
- concrete config objects;
- concrete metrics schemas.

## Acceptance Criteria

This partial spec is accepted when the top-level model/trainer responsibilities
are clear enough to prevent trainer code from depending on tokenizer or
step-wise rollout internals.
