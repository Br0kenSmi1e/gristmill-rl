# REINFORCE Scalar Step Design

## Summary

This spec defines the first scalar REINFORCE policy, rollout, and training
objective for one sample column. It is the semantic source of truth for the
parallel row-wrapper spec.

The central design is:

```text
model:
  target distribution over STOP plus definition indices
  action distribution over candidate_index, left_mask, right_mask

rollout:
  one sample at step t -> same sample at step t + 1

training:
  stored row data -> recompute logp -> REINFORCE loss
```

The scalar implementation should be complete enough to train the first model
version.

## Goals

- Define the first-version attention-context policy architecture.
- Define the target and action probability distributions.
- Define the scalar rollout order for one sample step.
- Define the width-1 row data stored for recomputation.
- Define the scalar REINFORCE objective.
- Preserve the target/action split so target selection never expands action
  spaces for every definition.
- Keep scalar behavior compatible with the row-table overview and the later
  parallel row wrapper.

## Non-Goals

- Defining row-level parallel execution.
- Choosing low-level padding values for masked row entries.
- Defining the final model architecture.
- Solving warm start or imitation learning.
- Running CCSD-scale batches.
- Differentiating through rewrite application, action-space generation, rewards,
  or sampled choices.
- Keeping the previous transformer/reinforce prototype API.

## Model Distribution

The policy has two distributions:

```text
target:
  p(STOP or def_index | TargetInput)

action:
  p(candidate_index, left_mask, right_mask | ActionInput)
```

The model samples from these distributions during rollout. Training recomputes
the log probability of the sampled choices from stored inputs and choices.

The model uses attention context and direct logits over semantic choices. It
does not use the old token-decoder grammar where policy choices are generated
as detached extra tokens.

## Target Model

Target selection decides where to act:

```text
TargetInput -> STOP or def_index
```

`TargetInput` contains:

- state token sequence;
- definition metadata or definition positions inside the state tokens;
- target legality mask;
- STOP legality.

`TargetInput` must not contain:

- action-space tokens;
- candidate information;
- left/right mask information.

The first-version target architecture is:

```text
state tokens + definition metadata + target legality mask
  -> attention context
  -> logits over STOP plus definition indices
  -> masked softmax
```

Each legal definition receives one target logit. STOP receives one target logit.
Illegal definitions are masked before sampling and scoring.

Target scoring recomputes:

```text
target_logp = log p(stored target choice | stored TargetInput)
```

The target model must not call action-space generation. This must be true by the
shape of the model input, not only by convention.

## Action Model

Action selection decides how to rewrite after target selection chooses one
definition with a non-empty action space:

```text
ActionInput -> candidate_index, left_mask, right_mask
```

`ActionInput` contains:

- state token sequence;
- selected `def_index`;
- selected definition's action-space token sequence;
- candidate positions and candidate legality;
- left/right term positions for each candidate.

The first-version action architecture is:

```text
state tokens + selected def_index + selected action-space tokens
  -> attention context
  -> candidate logits
  -> left-mask logits
  -> right-mask logits
```

The action distribution is autoregressive:

```text
candidate_index
left_mask conditioned on candidate_index
right_mask conditioned on candidate_index and left_mask
```

`candidate_index` is sampled from masked logits over the selected action space.
`left_mask` is sampled over the selected candidate's left term positions.
`right_mask` is sampled over the selected candidate's right term positions.
Invalid padded positions and invalid empty selections are excluded from the
distribution before sampling and scoring.

Action scoring recomputes:

```text
candidate_logp = log p(stored candidate_index | ActionInput)
left_mask_logp =
  log p(stored left_mask | ActionInput, stored candidate_index)
right_mask_logp =
  log p(stored right_mask | ActionInput, stored candidate_index, stored left_mask)

action_logp = candidate_logp + left_mask_logp + right_mask_logp
```

The action model only sees the action space for the selected definition. It does
not score or materialize action spaces for unselected definitions.

## Scalar Rollout

The scalar rollout step updates one sample by one step:

```text
step_sample(sample_t) -> sample_t_plus_1, data for this sample in row t
```

The active-sample order is:

```text
1. Build TargetInput from the current sample state.
2. Sample STOP or def_index from the target distribution.
3. Store target input, target choice, and target score mask.
4. If STOP, mark the sample terminal and end the step.
5. Build action space for the selected def_index only.
6. If action space is empty, refine target legality and end the step.
7. Build ActionInput from current state, selected def_index, and action space.
8. Sample candidate_index, left_mask, right_mask from the action distribution.
9. Store action input, action choice, and action score mask.
10. Apply the action to produce the next sample state.
```

There is no retry loop inside one scalar step. If a selected definition has an
empty action space, the next target selection happens in the next row.

## Step Cases

The scalar step has four cases:

```text
case                  target_score_mask    action_score_mask
already finished      false                false
STOP                  true                 false
empty action space    true                 false
valid action          true                 true
```

### Already Finished

If the input sample is already finished:

- no target choice is sampled;
- no action choice is sampled;
- no target or action logp contributes to training;
- the sample remains finished.

### STOP

If target selection chooses STOP:

- the target choice is stored and scored;
- no action choice exists;
- the sample becomes terminal.

STOP is part of the target distribution, not the action distribution.

### Empty Action Space

If target selection chooses a definition whose action space is empty:

- the target choice is stored and scored;
- no action choice exists;
- the selected definition becomes unavailable for the next target selection from
  this sample state;
- the sample continues unless another stopping rule applies.

The action model is not scored because no action distribution existed.

### Valid Action

If target selection chooses a definition with a non-empty action space:

- the target choice is stored and scored;
- the action choice is stored and scored;
- the chosen rewrite is applied to produce the next sample state.

## Stored Width-1 Row

Scalar training is the width-1 version of the row-table design:

```text
target_input[t, sample 0]
target_choice[t, sample 0]
target_score_mask[t, sample 0]

action_input[t, sample 0]
action_choice[t, sample 0]
action_score_mask[t, sample 0]
```

If a score mask is false, the corresponding input and choice are ignored by loss
and metrics. Masked entries still need safe padded values so scoring can run
without out-of-range indexing or shape errors.

Example:

```text
row 0: valid action        target mask true   action mask true
row 1: empty action space  target mask true   action mask false
row 2: valid action        target mask true   action mask true
row 3: STOP                target mask true   action mask false
```

No later finished row is required in a scalar rollout unless a test explicitly
exercises masked finished entries.

## REINFORCE Training

Rollout stores inputs and sampled choices. It may compute sampled logp for
metrics, but rollout logp is not the gradient source.

Training recomputes:

```text
target_logp[t, 0] =
  log p(stored target_choice[t, 0] | stored target_input[t, 0])

action_logp[t, 0] =
  log p(stored action_choice[t, 0] | stored action_input[t, 0])
```

After rollout, the trainer computes a scalar reward for the sample column from
the configured reward evaluator. Advantage is:

```text
advantage = reward - baseline
```

The first scalar implementation may use a baseline value of zero, so advantage
equals reward, or an externally supplied advantage in deterministic tests.

For one sample column, the REINFORCE loss is:

```text
loss =
  -advantage *
  sum over rows t (
    target_score_mask[t, 0] * target_logp[t, 0]
  + action_score_mask[t, 0] * action_logp[t, 0]
  )
```

Only recomputed target/action logp terms are differentiable. The trainer must
not backpropagate through:

- sampled choices;
- action-space generation;
- rewrite application;
- reward computation;
- baseline or advantage computation.

One scalar optimizer update may use one or more completed sample columns. If
more than one column is used, this is accumulation over independent width-1
scalar trajectories, not row-level parallel rollout.

## Public Contract

The scalar public contract is intentionally small:

```text
step_sample(sample_t) -> sample_t_plus_1, data for this sample in row t
score_target(stored TargetInput, stored target choice) -> target logp
score_action(stored ActionInput, stored action choice) -> action logp
```

The model owns target/action sampling and scoring. The trainer owns rollout
control, reward and advantage computation, recomputed logp loss, and optimizer
updates. The rewrite environment owns action-space generation and rewrite
application.

## Relationship To Rows

The scalar step defines the semantics for one sample position. The parallel row
wrapper must preserve those semantics when lifting:

```text
row t -> row t + 1
```

Row-level mechanics may organize work privately, but they must not change the
per-sample target/action distribution, stored score masks, or REINFORCE loss
meaning.

## Acceptance Criteria

- Target input construction and target sampling do not call action-space
  generation.
- Target scoring uses masked logits over STOP plus definition indices.
- Action scoring uses the autoregressive candidate, left-mask, right-mask
  distribution over the selected action space.
- Already-finished samples emit no target or action score.
- STOP emits target scoring data, emits no action scoring data, and makes the
  sample terminal.
- Empty action space emits target scoring data, emits no action scoring data,
  and makes the selected definition unavailable for the next target selection.
- Valid action emits target and action scoring data and applies one rewrite.
- Scalar training recomputes target/action logp from stored row data and sampled
  choices.
- Scalar training uses target/action score masks to include exactly the real
  target and action logp terms.
- The scalar REINFORCE loss uses recomputed logp terms weighted by the sample
  column advantage.
- The scalar design uses the overview vocabulary: sample, row, and column.
