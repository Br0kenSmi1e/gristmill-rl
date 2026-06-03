# Naive REINFORCE Training Design

## Summary

Add a standalone Python package for issue #3 that trains the merged
`transformer_policy` package with a direct on-policy REINFORCE objective.

The trainer samples fresh batches of complete rewrite episodes, computes final
reward as `R = -final_log_flops`, subtracts the batch mean reward, and updates
only the policy model log probability of the sampled token choices. Environment
rollout runs in Python/Rust worker processes. Differentiable log-probability
rescoring runs in the main process with padded JAX arrays and `vmap`.

This package is a successor path to the existing `gristmill_rl` prototype. It
must not import `gristmill_rl`.

## Merged Transformer Policy Context

The merged `transformer_policy` package currently exposes:

```text
transformer_policy.types
  Token
  T
  Stage1Attempt
  PolicySample

transformer_policy.tokenize
  tokenize_tensor_def
  build_state_context
  build_action_space_context

transformer_policy.embed
  TOKEN_FEATURE_DIM
  token_features
  TokenEmbedder

transformer_policy.sequence_model
  CausalTransformerScorer.score_next(context, prefix, legal) -> jax.Array

transformer_policy.decoder
  sample_step(state, scorer, rng) -> PolicySample
  score_step(state, scorer, sample) -> float

transformer_policy.policy
  TransformerPolicy.sample_step(state, rng) -> PolicySample
  TransformerPolicy.score_step(state, sample) -> float
```

Important details for issue #3:

- `PolicySample` can contain a PyO3 `ActionSpace` handle. It is useful inside a
  rollout worker but must not be returned across process boundaries.
- `PolicySample.log_prob` is a Python float from sampling. It is useful for
  metrics only.
- `TransformerPolicy.score_step` returns a Python float and uses the object
  replay path. It validates semantics but is not a differentiable training
  objective.
- `CausalTransformerScorer.score_next` is the differentiable scoring primitive.
- `token_features` already converts `Token` records to stable float features,
  and `CausalTransformerScorer` already embeds legal-next tokens at the next
  sequence position.

Therefore issue #3 needs a small public trace/batch hook in
`transformer_policy`: enough to record token-choice events during sampling and
rescore those events in padded JAX batches without duplicating private decoder
or model internals in `reinforce_training`.

## Goals

- Add an end-to-end naive REINFORCE CLI.
- Keep the training package independent from legacy `gristmill_rl`.
- Use `transformer_policy` as the only policy distribution.
- Treat `STOP` as a normal masked policy action.
- Sample complete episode batches under frozen policy parameters.
- Parallelize episode sampling across worker processes.
- Use final reward `R = -final_log_flops`, not improvement.
- Use the batch mean reward as the baseline.
- Differentiate only through model log probabilities of sampled token choices.
- Use padded JAX event batches for production loss computation.
- Save and load model, optimizer, config, and progress checkpoints.

## Non-Goals

- Implementing AlphaZero, MCTS, replay, value targets, or a value head.
- Reusing `gristmill_rl` modules or checkpoints.
- Differentiating through Rust rewrite application, cost calculation, or action
  sampling.
- Moving rewrite legality, `RewriteState`, or action-space generation into JAX.
- Vectorizing full environment rollout with `jax.vmap`.
- Adding a learned baseline or critic.
- Keeping sampled trajectories for off-policy reuse.
- Recomputing action spaces in the main process to interpret worker traces.

## Package Boundary

Add a new top-level Python package:

```text
python/
  reinforce_training/
    __init__.py
    rollout.py
    trace.py
    objective.py
    train.py
    checkpoint.py
```

Dependency direction:

```text
reinforce_training -> transformer_policy
reinforce_training -> gristmill_symbolics

reinforce_training !-> gristmill_rl
transformer_policy !-> reinforce_training
```

`transformer_policy` owns token records, tokenization, token features, legal
next-token scoring, and masked decoding. `reinforce_training` owns episode
collection, serializable traces, reward and advantage calculation, optimizer
updates, metrics, checkpoints, and the CLI.

Update `python/pyproject.toml` during implementation:

```toml
python-packages = ["gristmill_rl", "transformer_policy", "reinforce_training"]
```

When `gristmill_rl` is removed later, it can be dropped from that list without
changing `reinforce_training`.

## Required Transformer Policy Hooks

The merged package has the right semantic API but not yet the right training
API. Issue #3 should add a narrow public hook to `transformer_policy`, likely in
`python/transformer_policy/trace.py` or `batch.py`.

The hook should expose immutable token-choice events:

```text
TokenChoiceEvent {
  sequence_tokens: tuple[Token, ...]
  legal_next_tokens: tuple[Token, ...]
  chosen_index: int
  phase: "def" | "candidate" | "left_bit" | "right_bit"
  step_index: int
}
```

`sequence_tokens` is exactly the scorer input sequence:

```text
context_tokens + decision_prefix_tokens
```

This matches the merged `CausalTransformerScorer.score_next` implementation,
which encodes the concatenated sequence and scores legal-next tokens at
`next_position = len(sequence_tokens)`.

The hook should also expose a traced sampling API:

```python
sample_step_with_events(state, scorer, rng) -> TracedPolicySample
```

where:

```text
TracedPolicySample {
  sample: PolicySample
  events: tuple[TokenChoiceEvent, ...]
}
```

The existing `TransformerPolicy.sample_step` can remain unchanged. The traced
API is for training and should reuse the same decoder logic, masks, validation,
and token order as `sample_step`. `reinforce_training` must not copy private
decoder functions such as `_stage1_legal`, `_candidate_legal`, or `_bit_legal`.

For differentiable rescoring, add a public padded event batch scorer:

```python
score_event_batch(scorer, batch: PaddedTokenChoiceBatch) -> jax.Array
```

It must produce the same logits as calling `scorer.score_next` event by event,
but operate on padded numeric arrays and use `jax.vmap` internally. It returns
logits with shape `[events, max_legal]`. It should live in `transformer_policy`
so it can use the same `TokenEmbedder`, legal-token position handling, causal
attention, and model modules as the normal scorer.

## Rollout Data Flow

Each CLI update samples a fresh on-policy batch:

```text
main process:
  freeze current policy parameters
  split RNG seeds for batch_size episodes
  dispatch episode jobs to workers
  collect serializable episode traces sorted by episode_index
  compute rewards and advantages
  rescore trace events under the same frozen parameters
  apply one optimizer update
  discard traces
```

Each worker receives only serializable inputs:

```text
input_json
policy model config
policy parameter snapshot
rollout config
episode_index
episode_seed
```

The worker reconstructs local objects:

```python
comp = TensorComputation.from_json_string(input_json)
state = RewriteState.from_computation(comp)
policy = restore_policy(policy_config, policy_parameters)
rng = np.random.default_rng(episode_seed)

for step in range(max_steps):
    traced = policy.sample_step_with_events(state, rng)
    record_serializable_step(step, state.snapshot(), traced)

    if traced.sample.stopped:
        terminal_reason = "stop"
        break

    state.step_with_space(traced.sample.action_space, traced.sample.decision)
else:
    terminal_reason = "max_steps"

final_log_flops = state.log_total_flops()
reward = -final_log_flops
```

Workers may use `PolicySample.action_space` internally to apply rewrites. They
must not return `ActionSpace`, `RewriteState`, `TensorComputation`, JAX arrays,
or NNX objects to the main process.

## Serialized Traces

The main process receives JSON-like records.

The shapes below use `Token` for readability. The worker transport format
should be plain data, for example:

```text
TokenWire {
  kind: str
  payload: tuple[tuple[str, int | float | str | bool], ...]
}
```

The main process reconstructs `Token` records before calling
`transformer_policy.token_features`.

Episode record:

```text
EpisodeTrace {
  episode_index: int
  episode_seed: int
  steps: tuple[StepTrace, ...]
  final_snapshot: dict
  final_log_flops: float
  reward: float
  terminal_reason: "stop" | "max_steps"
}
```

Step record:

```text
StepTrace {
  step_index: int
  state_snapshot: dict
  stopped: bool
  def_attempts: tuple[Stage1AttemptTrace, ...]
  def_index: int | None
  action_space_snapshot: dict | None
  decision: dict | None
  decision_tokens: tuple[Token, ...]
  token_events: tuple[TokenChoiceEvent, ...]
  sample_log_prob: float
}
```

Stage-1 attempt record:

```text
Stage1AttemptTrace {
  def_index: int
  log_prob: float
  accepted: bool
}
```

For rewrite steps, `action_space_snapshot` and `decision` are retained for
debugging and validation, but training should use `token_events`. This avoids
recomputing action spaces in the main process and avoids relying on candidate
ordering across process-local Rust objects.

The worker trace should preserve rejected cheap-mask stage-1 probes. They are
ordinary sampled token-choice events and contribute to trajectory log
probability.

## STOP Handling

`STOP` is not special-cased in the objective.

The rollout loop always calls the policy. If the sampled step is stopped, the
episode ends and the event trace includes the `STOP` token choice. If `STOP` is
the only legal token, the masked distribution is degenerate:

```text
log p(STOP | state) = 0
```

That event contributes no gradient naturally. No separate "forced terminal"
objective branch is needed.

## Parallel Sampling

Use `concurrent.futures.ProcessPoolExecutor` for `--num-workers > 1`.

Reasons:

- Rollout is CPU-bound and crosses the Python/Rust PyO3 boundary.
- Mutable `RewriteState` objects stay inside each worker.
- Processes avoid GIL and thread-safety ambiguity.
- The main process owns the trainable model and optimizer.
- `--num-workers 1` remains the deterministic single-process path.

The policy snapshot sent to workers is read-only for the entire batch. The main
process must not apply an optimizer update until every episode in the batch has
returned or failed. This keeps each update on-policy with respect to one fixed
parameter snapshot.

Episode seeds should be deterministic:

```text
episode_seed = seed + update_index * batch_size + episode_index
```

An equivalent reproducible `SeedSequence` is acceptable. The selected scheme
must be stored in checkpoint metadata so resumed training continues
deterministically.

Worker results must be sorted by `episode_index` before metrics and training
batches are computed.

## Objective

For a batch of `N` episodes:

```text
reward_i = -final_log_flops_i
baseline = mean_i(reward_i)
advantage_i = stop_gradient(reward_i - baseline)
trajectory_log_prob_i = sum_events log p_theta(chosen_event | event_sequence)
loss = -mean_i advantage_i * trajectory_log_prob_i
```

The reward, baseline, advantage, sampled actions, state transitions, and final
costs are constants for autodiff. The only differentiated term is the current
model distribution log probability of already-sampled token choices.

There is no value loss, entropy bonus, KL penalty, replay target, or learned
baseline in this first design.

## Padded JAX Rescoring

Full rollout is not a `jax.vmap` target because it mutates `RewriteState`,
calls Rust, has dynamic legal sets, and stops at different episode lengths.
Training rescoring should still be batched.

Flatten all episode traces into token-choice events, then convert them with
`transformer_policy.token_features` into padded arrays:

```text
sequence_features: [events, max_sequence_len, TOKEN_FEATURE_DIM]
sequence_mask:     [events, max_sequence_len]
legal_features:    [events, max_legal, TOKEN_FEATURE_DIM]
legal_mask:        [events, max_legal]
next_position:     [events]
chosen_index:      [events]
episode_id:        [events]
event_mask:        [events]
```

`sequence_features` represents `context_tokens + decision_prefix_tokens`.
`legal_features` represents `legal_next_tokens`; before projection, its
position column must be set to `next_position`, matching the merged
`CausalTransformerScorer._embed_legal_tokens` behavior.

Compute:

```text
logits = score_event_batch(scorer, padded_events)
masked_logits = where(legal_mask, logits, -inf)
chosen_logp = gather(log_softmax(masked_logits), chosen_index)
trajectory_logp = segment_sum(chosen_logp, episode_id)
loss = -mean(advantage * trajectory_logp)
```

The batch scorer must match single-event `score_next` numerically on unpadded
inputs. Padded sequence tokens must not affect attention. Padded legal tokens
must not affect softmax normalization.

## CLI

The CLI entry point should be:

```bash
python -m reinforce_training.train
```

Core flags:

```text
--input PATH
--updates N
--batch-size N
--max-steps N
--num-workers N
--learning-rate FLOAT
--seed INT
--checkpoint-in PATH
--checkpoint-out PATH
--checkpoint-overwrite
```

Model flags should mirror `CausalTransformerScorer`:

```text
--hidden-dim N
--num-heads N
--num-layers N
--mlp-dim N
```

Each update prints one JSON metrics line:

```json
{
  "update": 1,
  "updates": 10,
  "batch_size": 8,
  "num_workers": 4,
  "mean_reward": -12.3,
  "mean_final_log_flops": 12.3,
  "best_final_log_flops": 10.8,
  "mean_steps": 2.5,
  "stop_count": 6,
  "max_steps_count": 2,
  "loss": 0.17,
  "mean_sample_log_prob": -4.3,
  "mean_trajectory_log_prob": -4.2,
  "params_changed": true,
  "checkpoint_in": null,
  "checkpoint_out": "runs/reinforce-checkpoint"
}
```

Monitoring UI, entropy schedules, multiple input datasets, and baseline
comparisons are follow-up work.

## Checkpoints

Use Orbax for checkpoint state, following the existing repository pattern but
implemented independently in `reinforce_training.checkpoint`.

Checkpoint contents:

```text
metadata.json
state/
```

Metadata:

```text
schema_version
package = "reinforce_training"
model_class = "CausalTransformerScorer"
transformer_policy_config
optimizer = "adam"
learning_rate
rollout_config
update_count
seed
seed_scheme
user metadata
```

State:

```text
model parameters
optimizer state
```

Loading a checkpoint restores model parameters, optimizer state, policy config,
rollout config, update count, and seed progress. If user-supplied model flags
conflict with checkpoint metadata, loading fails clearly.

Saving refuses to overwrite an existing checkpoint unless
`--checkpoint-overwrite` is supplied. Overwrite publishes through a temporary
directory and preserves the existing checkpoint if state writing fails.

## Error Handling

CLI validation:

- `--updates`, `--batch-size`, `--max-steps`, and `--num-workers` must be
  positive.
- `--learning-rate` must be finite and positive.
- `--checkpoint-in` must exist when supplied.
- `--checkpoint-out` must not exist unless `--checkpoint-overwrite` is supplied.
- `--hidden-dim`, `--num-heads`, `--num-layers`, and `--mlp-dim` must satisfy
  the same validation as `CausalTransformerScorer`.

Runtime validation:

- Worker failures propagate to the main process with update index, episode
  index, and episode seed.
- Nonfinite `final_log_flops`, rewards, advantages, log-probs, or losses fail
  before checkpointing.
- Serialized traces fail validation before optimizer updates if a chosen token
  is absent from its legal set, `chosen_index` is out of range, episode IDs are
  invalid, or padding masks are inconsistent.
- Event batching fails if any episode has no token events.
- Padded batch scoring fails if batch logits contain nonfinite values before
  legal masking.

## Testing

Transformer policy hook tests:

- Traced sampling returns the same `PolicySample` shape as `sample_step`.
- Traced sampling records one `TokenChoiceEvent` per sampled token decision.
- STOP-only states produce a single event with one legal token and log-prob
  zero.
- Rejected stage-1 probes are represented as events.
- Padded event batch logits match per-event `score_next` outputs.
- Legal-token position handling matches `CausalTransformerScorer.score_next`.

Package and dependency tests:

- `reinforce_training` imports without importing `gristmill_rl`.
- All `reinforce_training` modules avoid `gristmill_rl` imports.

Rollout tests:

- A tiny actionable input samples a complete serializable episode trace.
- `STOP` at the first state ends the episode and records its event.
- A run that reaches `max_steps` records `terminal_reason = "max_steps"`.
- Worker traces contain no PyO3 handles or JAX arrays.
- `--num-workers 1` and `--num-workers 2` both return a full batch.
- Fixed seeds are deterministic for `--num-workers 1`.
- Fixed seeds are stable for `--num-workers 2` after sorting by
  `episode_index`.

Objective and trace-batch tests:

- Reward equals `-final_log_flops`.
- Batch-mean advantages sum to zero within numerical tolerance.
- Reward and advantage values are treated as stop-gradient constants.
- Padded event batching reconstructs per-episode trajectory log-probs.
- Padded legal-token masks exclude padding from softmax normalization.
- Invalid chosen indices or inconsistent masks fail clearly.
- One training update changes model parameters on a nonzero-advantage batch.

CLI and checkpoint tests:

- `python -m reinforce_training.train` completes a tiny run.
- The CLI prints JSON metrics with expected keys.
- The CLI writes a checkpoint.
- Loading a checkpoint restores model outputs and optimizer state.
- Loading a checkpoint and continuing increments update counters correctly.
- Checkpoint overwrite refusal and overwrite replacement are covered.
- Conflicting model config on checkpoint load fails clearly.

## Acceptance Criteria

- A new `reinforce_training` package is specified as a standalone successor path
  that does not import `gristmill_rl`.
- The design is based on the merged `transformer_policy` APIs.
- The design identifies the minimal additional `transformer_policy` trace and
  batch-scoring hooks needed for training.
- The CLI samples episodes in process-parallel workers.
- The objective is vanilla on-policy REINFORCE with final reward and batch-mean
  baseline.
- The design differentiates only through trajectory token log probabilities.
- Serialized traces avoid returning PyO3 handles from worker processes.
- Padded JAX event batching is the production rescoring path.
- Checkpoints contain enough state to resume training.

## References

- GitHub issue #3: https://github.com/Br0kenSmi1e/gristmill-rl/issues/3
- Merged transformer policy package: `python/transformer_policy/`
- Transformer policy design:
  `docs/superpowers/specs/2026-06-02-symbolic-tensor-embedding-design.md`
