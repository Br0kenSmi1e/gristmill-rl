# Scalable REINFORCE Refactor Design

## Summary

The current `reinforce_training` package is a useful issue #3 prototype, but it
is not a scalable training architecture. Debugging on `tmp/ccsd/working_eqn.json`
exposed three connected problems:

- memory scales with the physical JAX event batch and duplicated worker
  runtimes, so meaningful REINFORCE batch sizes are not feasible;
- random initialization plus always-legal `STOP` and sparse terminal reward
  gives little or no learning signal;
- `transformer_policy` was not designed as a training API, so the trainer had
  to change policy internals to obtain traces and batched rescoring.

This refactor should separate policy training interfaces from rollout mechanics,
support large logical batches through microbatch accumulation, and add a
cost-aware warm start before REINFORCE fine-tuning.

## Observed Failures

### Memory And Batching

The current CLI treats `--batch-size` as one physical JAX batch. Each sampled
token event is padded to the maximum sequence length and rescored through full
causal attention. On the CCSD input the state context is about 1500 tokens, so
attention memory scales roughly as:

```text
event_count * num_heads * sequence_len^2
```

That is incompatible with logical REINFORCE batches such as 1024 to 8192
episodes. Process workers make this worse because each worker reconstructs a
JAX/Flax scorer and carries its own JAX runtime memory.

### Warm Start And Reward Signal

The initial policy samples `STOP` with nontrivial probability because `STOP` is
legal at every stage-1 decision. Even when random rollouts do rewrite, the final
`log_total_flops` values are often identical or differ only around `1e-8`.
The current reward path casts final costs to `float32`, which collapses those
differences and yields zero advantages, zero loss, and unchanged parameters.

The "largest biclique" heuristic is also not reliable. On the CCSD input it can
choose a cost-increasing rewrite. A cost-aware all-true-mask greedy baseline
improves:

```text
65.28397955781062 -> 55.67160321783035
```

but Gristmill's optimized result reaches:

```text
49.23057289544251
```

The gap is a scaling gap. The greedy result is already mostly binary
contractions, but it still leaves high-rank binary contractions with three or
more summed indices. Gristmill introduces staged intermediates that reduce the
dominant binary contractions to fewer summed indices.

### Policy API Boundary

`reinforce_training` needed new `transformer_policy` hooks for traced sampling
and padded event rescoring. That was necessary for the prototype, but it shows
that the policy package does not yet expose a stable training-facing API.
Future trainer changes should not need to copy decoder internals or keep adding
ad hoc policy hooks.

## Goals

- Support large logical REINFORCE batches without materializing one giant JAX
  event batch.
- Keep physical training memory bounded through rollout and gradient
  microbatches.
- Avoid process-based JAX worker duplication by default.
- Add warm-start data from a cost-aware, rank-aware heuristic.
- Make `STOP` behavior configurable for training versus inference.
- Preserve rewards and advantages with enough precision to avoid float32 reward
  collapse.
- Expose a stable `transformer_policy` training API that can support imitation
  learning and REINFORCE without further internal rewrites.
- Keep the new path independent from deprecated `gristmill_rl`.

## Non-Goals

- Reintroducing the legacy `gristmill_rl` API.
- Implementing AlphaZero or MCTS in this refactor.
- Differentiating through rewrite application or Rust cost evaluation.
- Matching Gristmill's optimizer exactly in the first refactor.
- Vectorizing full environment rollout with JAX.

## Proposed Architecture

### Logical Batch With Microbatch Accumulation

The CLI should distinguish logical batch size from physical batch size:

```text
--batch-size                 logical episodes per optimizer update
--rollout-microbatch-size    episodes sampled before returning progress
--train-microbatch-events    max token-choice events per gradient chunk
```

One update should:

```text
sample logical batch in rollout chunks
compute rewards and baseline over the full logical batch
split trace events into training microbatches
accumulate gradients
apply one optimizer update
discard traces
```

The trainer should print progress to stderr before and after rollout chunks,
training chunks, and checkpoint writes. Metrics should continue to print as JSON
on stdout.

Process workers should default to disabled for JAX scorers. If workers remain
available, the CLI should warn or cap worker count unless an explicit
`--allow-process-workers` flag is set.

### Cost-Aware Warm Start

Warm start should be supervised imitation, not REINFORCE. Generate
demonstration trajectories from a deterministic heuristic and train the
transformer with negative log likelihood on the chosen tokens.

The heuristic should minimize post-action cost and rank risk, not biclique size:

```text
score(action) =
  post_total_log_flops
  + alpha * post_max_term_log_flops
  + beta * post_max_binary_sum_indices
  + gamma * post_high_rank_binary_term_count
```

Lower score is better. Candidate evaluation should include all-true masks plus a
bounded partial-mask search. Exhaustive mask search is not acceptable for large
candidates.

During warm start, `STOP` should be legal only when no action remains or no
candidate passes the heuristic's improvement threshold. During early REINFORCE,
the CLI should support a minimum rewrite count or a "terminal-only STOP" mode.

### Training-Ready Policy API

`transformer_policy` should expose one stable training interface. The exact name
can change, but it should cover:

```text
sample_step_with_trace(state, rng, stop_mode) -> TracedPolicySample
score_trace_events(scorer, event_batch) -> logits
encode_demonstration_step(state, decision) -> TokenChoiceEvent sequence
pad_trace_events(events, limits) -> PaddedTokenChoiceBatch
```

The trainer should not know private token legality functions such as stage-1,
candidate, or bit-token construction. `transformer_policy` owns tokenization,
legality, trace event construction, and differentiable rescoring. The training
package owns rollout scheduling, reward calculation, gradient accumulation,
checkpointing, and CLI metrics.

## Metrics And Diagnostics

The CLI should add diagnostics that make stalled training obvious:

```text
initial_log_flops
mean_final_log_flops
best_final_log_flops
mean_improvement
reward_std
advantage_std
distinct_final_log_flops
rewrite_step_count
stop_step_count
max_steps_count
event_count
max_sequence_len
max_legal_count
train_microbatch_count
params_changed
```

Rewards should stay in `float64` at least through baseline and advantage
calculation. If reward magnitudes remain too small, the trainer should support
scaled improvement rewards such as:

```text
reward = initial_log_flops - final_log_flops
```

with optional advantage normalization.

## Acceptance Criteria

- `--batch-size 1024` can run as a logical batch on the CCSD input without
  memory blow-up by using small physical microbatches.
- The CLI shows progress before the first optimizer update completes.
- Warm-start demonstrations avoid early `STOP` and produce nonzero improvement
  on `tmp/ccsd/working_eqn.json`.
- Supervised warm-start training changes model parameters and lowers imitation
  loss on a small demonstration set.
- REINFORCE reports nonzero reward or advantage variance when rollouts differ.
- `transformer_policy` has a documented training API, and
  `reinforce_training` no longer needs private decoder knowledge.

## Open Questions

- What bounded mask-search strategy gives the best cost/runtime tradeoff?
- Should the warm-start heuristic optimize total cost only, or explicitly
  penalize high-rank binary contractions?
- Should training use terminal-only `STOP` permanently, or only during warm
  start and early RL?
- Should rollout workers be reintroduced with an actor process that keeps one
  persistent model instance, or should the first scalable version stay
  single-process?
