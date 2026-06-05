# Policy And REINFORCE API Refactor Design

## Summary

Refactor the boundary between the rewrite policy model and the REINFORCE
trainer. The current prototype exposes too much transformer decoder structure
to the trainer, which makes the training package depend on policy internals and
prevents the model architecture from evolving.

The new boundary should be a two-stage rewrite policy API. The trainer knows
that rewrite decisions have target selection and action construction phases,
but it does not know how the policy parameterizes either phase. The policy
returns semantic decisions for rollout and policy-owned traces for later
differentiable scoring. The trainer executes decisions, computes rewards and
advantages, scores traces in bounded chunks, and applies REINFORCE updates.

## Goals

- Define a stable policy/trainer interface for REINFORCE.
- Preserve the meaningful two-stage structure of rewrite decisions.
- Keep transformer tokenization, legality masks, and action parameterization out
  of the trainer.
- Let the first model architecture move away from a pure next-token transformer
  decoder while still using attention.
- Support parallel rollout sampling with `--num-workers`.
- Support bounded trace scoring so logical batch size is independent from
  physical scoring size.
- Keep this refactor independent from deprecated `gristmill_rl`.

## Non-Goals

- Designing warm-start imitation training in this first API refactor.
- Designing a formal memory-budget planner.
- Requiring direct logits over the full exponential rewrite action space.
- Supporting every future action-construction architecture immediately.
- Reintroducing AlphaZero, MCTS, replay, or a value model.

## Public Decision Model

Use domain-specific names instead of generic stage numbers.

`TargetSelection` chooses whether to stop or which definition to rewrite:

```text
TargetSelection =
  STOP
  selected definition
```

`ActionConstruction` constructs the concrete rewrite action for the selected
definition:

```text
ActionConstruction =
  concrete rewrite action
```

A complete rollout decision is:

```text
RewriteDecision {
  target_selection: TargetSelection
  action_construction: ActionConstruction?  # absent for STOP
}
```

The trainer may inspect and execute `RewriteDecision`. It should not inspect the
model-specific evidence used to produce or rescore that decision.

## Policy API

The policy exposes semantic sampling and trace scoring:

```text
RewritePolicy.sample_decision(state, mode) -> PolicySample
RewritePolicy.score_traces(policy_traces) -> PolicyScoreBatch
```

`PolicySample` contains:

```text
PolicySample {
  decision: RewriteDecision
  trace: PolicyDecisionTrace
  metrics: optional policy diagnostics
}
```

`PolicyDecisionTrace` is policy-owned evidence for rescoring the same sampled
decision after rollout rewards are known. The trainer may store traces inside an
episode trace, but treats them as opaque except for high-level metadata such as:

```text
rollout step
whether target selection is present
whether action construction is present
trainable choice count
```

`PolicyScoreBatch` returns differentiable log-probability terms:

```text
PolicyLogpTerm {
  trace id
  rollout step
  kind: target_selection | action_construction
  logp
}
```

The policy may return multiple log-probability terms for one action
construction. This allows action construction to be factored or autoregressive
without exposing that representation to the trainer.

## REINFORCE Trainer API

The trainer owns rollout, rewards, advantages, optimization, metrics, and
checkpoints:

```text
ReinforceTrainer.collect_episode(policy, initial_state, mode) -> EpisodeTrace
ReinforceTrainer.collect_batch(policy, initial_states, mode) -> EpisodeBatch
ReinforceTrainer.compute_advantages(batch) -> AdvantageAssignment
ReinforceTrainer.update(policy, batch, advantages) -> UpdateResult
```

The trainer executes only semantic decisions:

```text
policy samples RewriteDecision
trainer applies RewriteDecision to the rewrite state
trainer records terminal reason and costs
```

The REINFORCE objective uses the policy's returned log-probability terms:

```text
loss = -sum(advantage_for_term * term.logp)
```

The trainer may aggregate losses and diagnostics by `target_selection` and
`action_construction`, but it does not need to know the internal action
construction roles.

## V1 Policy Architecture

The first refactor should no longer force all decisions to be next tokens in
the same sequence language as tensor tokens. Instead, the policy should use
attention to contextualize the rewrite state and action space, then decode
semantic choices directly.

The decision distribution is:

```text
p(decision | state)
  = p(target_selection | state)
  * p(action_construction | state, selected target)
```

Target selection is a direct categorical choice over:

```text
STOP + rewriteable definitions
```

Action construction v1 is autoregressive:

```text
p(action_construction | state, target)
  = p(candidate | state, target, action space)
  * p(left mask | state, target, candidate)
  * p(right mask | state, target, candidate, left mask)
```

This avoids direct logits over the full exponential action space while still
giving an exact log probability for the sampled action:

```text
logp(action_construction)
  = logp(candidate)
  + logp(left mask)
  + logp(right mask)
```

The policy may expose candidate, left-mask, and right-mask diagnostics, but the
trainer objective only requires stage-level log-probability terms.

## Scalable REINFORCE Execution

The API must support executable REINFORCE batches without treating one logical
batch as one physical scoring batch.

Keep three concepts separate:

```text
batch_size
  logical number of episodes per optimizer update

num_workers
  parallel rollout sampling workers

score_chunk_size
  bounded number of traces or decisions scored at once
```

The trainer flow is:

```text
collect a logical episode batch, optionally with num_workers
compute rewards and advantages over the full logical batch
split policy traces into bounded scoring chunks
score chunks through policy.score_traces
accumulate the REINFORCE objective across chunks
apply one optimizer update
discard rollout traces
```

Workers are for rollout sampling. Scoring and model updates remain coordinated
by the trainer so each logical batch has one reward baseline and one optimizer
update. The policy API only needs to guarantee that subsets of traces can be
scored independently.

## Deferred Warm Start

Warm start should be handled later as supervised imitation over the same
semantic decision API:

```text
demonstrator(state) -> RewriteDecision
policy encodes the demonstrated decision as PolicyDecisionTrace
trainer maximizes the returned log-probability terms
```

This first refactor should not design the demonstrator or supervised trainer.
It only needs a policy boundary that will allow them to exist later without
teaching the trainer about policy internals.

## Acceptance Criteria

- `reinforce_training` samples and executes semantic `RewriteDecision` values.
- `reinforce_training` no longer depends on transformer token grammar, decoder
  legality helpers, or action-construction internals.
- `transformer_policy` exposes a documented policy API for sampling decisions
  and scoring policy traces.
- REINFORCE loss is computed from returned `PolicyLogpTerm` values weighted by
  trainer-computed advantages.
- The trainer can report target-selection and action-construction diagnostics
  separately.
- `--num-workers` remains a rollout sampling control.
- Logical batch size, worker count, and scoring chunk size are independent.
- Trace scoring can run in bounded chunks and still produce one optimizer update
  per logical batch.
- Warm start remains explicitly deferred.
