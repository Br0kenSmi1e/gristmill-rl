# REINFORCE Attention Policy Model Design

Status: planned
Supersedes: the architecture parts of
`2026-06-02-symbolic-tensor-embedding-design.md` and
`2026-06-05-policy-reinforce-api-refactor-design.md`
Depends on: `2026-06-02-rewrite-state-api-design.md`
Feeds implementation plan: yes

## Summary

This spec defines the first runnable attention-based rewrite policy model for
REINFORCE training over the symbolic kernel.

The model tokenizes tensor expressions and selected action spaces, uses
attention to build contextual embeddings, and emits direct semantic logits for
the policy choices required by the scalar and row rollout specs:

```text
target:
  STOP or def_index

action:
  candidate_index
  left_mask as a bit sequence
  right_mask as a bit sequence
```

The model does not generate rewrite decisions as detached grammar tokens in the
same language as tensor-expression tokens. Tensor tokens are model inputs.
Target, candidate, and mask decisions are semantic heads over legal choices.

## Goals

- Define the tokenizer inputs required by target and action scoring.
- Define the attention architecture at a level concrete enough to implement.
- Define the probability distribution and logp decomposition for target,
  candidate, left-bit, and right-bit choices.
- Keep target selection independent from action-space generation.
- Decode left and right masks as legal bit sequences, not categorical subsets.
- Store immutable model inputs and choices so training can recompute logp.
- Support row scoring through padded batches and score masks.

## Non-Goals

- Choosing the final high-performance architecture.
- Implementing a graph neural network or typed contraction graph encoder.
- Canonicalizing tensor, range, or index IDs for generalization.
- Differentiating through Rust action-space generation, rewrite application, or
  reward computation.
- Defining reward, advantage, optimizer, checkpoint, or CLI behavior.
- Preserving the previous causal token-decoder policy API.

## Dependencies

This design assumes the Rust/PyO3 rewrite boundary exposes:

```text
RewriteState.definition_mask()
RewriteState.action_space_for_def(def_index)
RewriteState.step_with_space(action_space, decision)
RewriteState.snapshot()
ActionSpace.snapshot()
```

Rust remains authoritative for exact action spaces and for validating decisions.
Python policy code should not infer actionability by inspecting symbolic
expressions beyond constructing model token records from snapshots and masks
supplied by the environment.

## Public Contracts

### Model Inputs Are Token Sequences

The neural model consumes token sequences.

The rollout and trainer also need sidecar data such as legality masks and
semantic choices. Those sidecars are part of stored scoring records, not separate
symbolic inputs to the attention model.

In this spec:

- target tokens are the model-facing input for target selection;
- action tokens are the model-facing input for action selection;
- `TargetRecord` and `ActionRecord` are immutable rollout/scoring records that
  wrap token sequences plus rollout metadata needed to interpret logits.

### TargetRecord

`TargetRecord` is an immutable snapshot used for target sampling and scoring:

```text
TargetRecord {
  target_tokens
  def_mask
}
```

`target_tokens` are derived from the current symbolic tensor state. They include
definition marker tokens, so the target head can associate logits with
`def_index` values. An implementation may cache definition positions as tokenizer
sidecars for efficient pooling, but those positions are derived from the token
sequence.

`def_mask` is the current cheap/lazily-refined definition mask. It is applied to
definition logits after the target head runs. STOP is always an available target
choice and is not controlled by a separate mask.

`TargetRecord` must not contain:

- action-space tokens;
- candidate information;
- left/right term information;
- exact action-space generation results for unselected definitions.

### ActionRecord

`ActionRecord` is an immutable snapshot built only after one target definition has
been selected and Rust has returned a non-empty `ActionSpace`:

```text
ActionRecord {
  action_tokens
  selected_def_index
}
```

`action_tokens` are the model-facing token sequence for action selection. They
contain the current state context, a selected-definition marker, and the selected
definition's action-space context. Candidate and side-term boundaries are encoded
in the token sequence. An implementation may cache candidate positions and side
term positions as tokenizer sidecars for efficient heads and batching, but those
positions are derived from `action_tokens`.

All candidates encoded in `action_tokens` are legal candidates returned by Rust.
The `ActionRecord` stores enough plain data to score the sampled action later
without a live `ActionSpace` handle.

### Choices

Target choices are:

```text
TargetChoice =
  STOP
  def_index
```

Action choices are Rust-compatible:

```text
ActionChoice {
  candidate_index
  left_mask: tuple[bool, ...]
  right_mask: tuple[bool, ...]
}
```

The left and right masks are stored in the deterministic term order encoded for
the selected candidate in `ActionRecord.action_tokens`.

### Policy API

The policy-facing API should expose semantic sampling and scoring:

```text
sample_target(TargetRecord, rng) -> TargetSample
score_target(TargetRecord, TargetChoice) -> target_logp

sample_action(ActionRecord, rng) -> ActionSample
score_action(ActionRecord, ActionChoice) -> action_logp
```

The row wrapper may call batched variants:

```text
sample_target_batch(padded_target_records, rngs)
score_target_batch(padded_target_records, target_choices)
sample_action_batch(padded_action_records, rngs)
score_action_batch(padded_action_records, action_choices)
```

Batched variants must preserve the same semantics as the scalar functions.

## Tokenization

The tokenizer converts symbolic snapshots to deterministic structured tokens. It
preserves snapshot order and values:

- ranges;
- tensors;
- definitions;
- base tensor IDs;
- external indices;
- terms;
- coefficients;
- summed indices;
- factors;
- factor index lists.

The first implementation may use raw integer IDs as payloads. It should not
canonicalize IDs, reorder terms, infer graph roles, or simplify symbolic content.

State tokenization produces:

```text
STATE_START
global range/tensor metadata
DEF_START def_index=0
TensorDef(definition_0)
DEF_END
DEF_START def_index=1
TensorDef(definition_1)
DEF_END
...
STATE_END
```

Action-space tokenization produces:

```text
ACTION_SPACE_START selected_def_index=i
CAND_START candidate_index=0
LEFT_DEF_START
TensorDef(left_definition)
LEFT_DEF_END
RIGHT_DEF_START
TensorDef(right_definition)
RIGHT_DEF_END
REWRITTEN_DEF_START
TensorDef(rewritten_definition)
REWRITTEN_DEF_END
CAND_END
...
ACTION_SPACE_END
```

The tokenizer must also return spans or positions for definitions, candidates,
and candidate side terms. Heads and bit decoders should consume these positions
instead of rediscovering structure from token strings.

## Architecture

The v1 architecture has five replaceable components:

```text
structured tokenizer
token embedder
state attention encoder
selected action-space attention encoder
semantic policy heads
```

### Token Embedder

The embedder maps structured token records to vectors. It combines:

- token kind embedding;
- numeric payload embeddings or projections;
- position embeddings;
- segment/type embeddings such as state, definition, action space, candidate,
  left side, right side, and rewritten definition.

The exact embedding math is an implementation detail, but the tokenizer output
and head inputs should be stable.

### State Encoder

The state encoder attends over `state_tokens` and returns:

```text
state_token_embeddings
definition_embeddings[def_index]
global_state_embedding
```

`definition_embeddings` are gathered or pooled from definition marker spans in
the target token sequence. `global_state_embedding` may be a pooled token, an
explicit summary token, or an attention-pooled vector.

### Target Head

The target head produces logits over:

```text
STOP + every current def_index
```

Definition logits are computed from definition embeddings with global state
context. The STOP logit is computed from the global state embedding.

Before sampling or scoring, definition logits are masked by `def_mask`.

The STOP logit remains present even when definitions are legal. To avoid early
STOP dominating random initial rollouts, the first implementation should
initialize the STOP head bias to a negative value. A bias around `-20` makes STOP
effectively unavailable at initialization when any definition logit is near zero,
but STOP still receives probability one when all definition logits are masked.

The masked categorical distribution is the complete target distribution.

### Action Encoder

After a definition is selected and exact action space is generated, the action
encoder attends over:

```text
state_tokens + selected definition marker + action_space_tokens
```

It returns:

```text
action_token_embeddings
candidate_embeddings[candidate_index]
left_term_embeddings[candidate_index, term_index]
right_term_embeddings[candidate_index, term_index]
action_context_embedding
```

The implementation may reuse the state encoder outputs or recompute a combined
state/action context. Caching is an optimization, not a public contract.

### Candidate Head

The candidate head produces one logit per candidate in the selected action
space. For one scalar sample, every candidate encoded in `action_tokens` is a
legal candidate. In padded batches, padded candidate slots are masked out after
logits are produced.

The categorical distribution is:

```text
p(candidate_index | ActionRecord)
```

### Bit-Sequence Mask Decoder

Left and right masks are decoded as ordered bit sequences. For each side, the
decoder walks the selected candidate's terms in deterministic order and emits a
binary decision:

```text
KEEP or DROP
```

The left bit distribution is autoregressive:

```text
p(left_mask | context, candidate)
  = product_i p(left_bit_i | context, candidate, left_prefix_before_i)
```

The right bit distribution is autoregressive and conditions on the completed left
mask:

```text
p(right_mask | context, candidate, left_mask)
  = product_j p(right_bit_j | context, candidate, left_mask, right_prefix_before_j)
```

Bit logits are computed from side term embeddings, the action context, the
selected candidate embedding, and a compact prefix state. The prefix state may be
a small recurrent state, an attention over emitted bit embeddings, or a masked
Transformer over side-term positions. The public requirement is the
autoregressive legal-bit distribution, not the internal implementation.

## Probability And Logp

The target log probability is:

```text
target_logp =
  log p(target_choice | TargetRecord)
```

The action log probability is:

```text
action_logp =
  log p(candidate_index | ActionRecord)
+ sum_i log p(left_bit_i | ActionRecord, candidate_index, left_prefix_before_i)
+ sum_j log p(right_bit_j | ActionRecord, candidate_index, left_mask, right_prefix_before_j)
```

The full scored step contribution is:

```text
target_score_mask * target_logp
+ action_score_mask * action_logp
```

The trainer owns advantage weighting and normalization.

## Bit Decoding Constraint

Rust requires both side masks to be non-empty. The bit decoder enforces this as a
decoder constraint without turning masks into categorical subsets.

For a side with `n` terms:

- if `n == 0`, the stored action input is invalid for the current Rust kernel;
- if no previous bit is `KEEP` and this is the final bit, only `KEEP` is legal;
- otherwise both `KEEP` and `DROP` are legal.

Sampling applies this constraint before sampling the next bit. Scoring replays
stored bits through the same constraint.

When only one bit is legal, the distribution is deterministic. The implementation
may record that bit with logp zero or omit it from metric counts, but scoring and
loss must be consistent between sampling and recomputation.

## STOP Initialization

STOP is always part of the target distribution.

The first runnable implementation should make immediate STOP rare through
initialization rather than through a trainer-controlled STOP mask:

```text
stop_bias_init ~= -20
```

This keeps the target-policy interface simple:

```text
state tokens -> raw logits over STOP + defs -> apply def_mask to defs only
```

When all definitions are masked out, STOP is the only unmasked choice and is
sampled with probability one regardless of the negative bias.

## Immutable Storage

Rollout must store immutable data for every scored decision:

```text
TargetRecord
TargetChoice
target_score_mask

ActionRecord
ActionChoice
action_score_mask
```

Stored inputs must not hold live `RewriteState` or `ActionSpace` handles. PyO3
handles may be used during rollout execution, but training replay must operate on
plain immutable Python/JAX data derived from snapshots.

This requirement protects scoring from stale action-space handles and from
mutation of `RewriteState.definition_mask()` during exact-empty refinement.

## Padding And Batching

Batched target scoring pads:

- state token sequences;
- definition positions;
- target choices;
- definition masks.

Batched action scoring pads:

- state/action token sequences;
- candidates;
- side term positions;
- stored bit sequences;
- candidate padding masks for padded batches.

Padding values must be safe for the model to process and must be masked out
before logits are interpreted. Padding must not change logp, metrics, or loss for
real decisions.

## Error Handling

Policy scoring should fail with clear errors when:

- a target choice is illegal under `TargetRecord`;
- an action choice is scored with a masked action entry;
- `candidate_index` is out of range or points to a padded batch slot;
- a stored bit sequence length does not match the selected candidate side;
- a stored side mask is empty;
- an illegal bit appears during replay.

Invalid stored choices indicate a rollout/scoring contract bug and should not be
silently masked.

## Testing Requirements

Tokenizer tests:

- state tokenization is deterministic for a fixture computation;
- `definition_positions` align with tokenized definitions;
- action-space tokenization is deterministic for a fixture action space;
- candidate and side-term positions align with candidate snapshots;
- tokenization does not query action spaces for unselected definitions.

Target model tests:

- target logits include STOP and all current definitions;
- illegal definitions are masked before sampling and scoring;
- STOP remains available and uses the configured negative initial bias;
- target scoring matches manual masked-softmax logp on a small fixture;
- target record construction does not call exact action-space generation.

Action model tests:

- candidate logits include only selected action-space candidates;
- candidate scoring matches manual softmax logp for one scalar action space;
- padded candidate slots are masked only in padded batches;
- left and right masks are decoded in deterministic bit order;
- final-bit forcing prevents empty side masks;
- scoring an illegal empty mask fails;
- sampled action choices can be applied through `RewriteState.step_with_space`.

Batching tests:

- padded target batches match scalar target scoring;
- padded action batches match scalar action scoring;
- masked padded entries do not affect logits used for real choices;
- width-1 row scoring matches scalar scoring.

## Acceptance Criteria

- The policy can sample and score target choices without generating unselected
  action spaces.
- The policy can sample and score an action for one selected non-empty action
  space.
- Left and right masks are modeled as bit sequences with exact recomputed logp.
- Stored target/action inputs are immutable and sufficient for differentiable
  rescoring.
- The policy API supports scalar and batched row scoring.
- The implementation can be tested without importing deprecated `gristmill_rl`
  policy APIs.
