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
expressions beyond constructing model inputs from snapshots and masks supplied by
the environment.

## Public Contracts

### TargetInput

`TargetInput` is an immutable snapshot used for target sampling and scoring:

```text
TargetInput {
  state_tokens
  definition_positions
  target_legal_mask
  stop_legal
  sample_metadata
}
```

`state_tokens` are derived from the current symbolic tensor state.
`definition_positions` map each `def_index` to the token span or pooled embedding
used by the target head. `target_legal_mask` is the current cheap/lazily-refined
definition mask. `stop_legal` is supplied by trainer rollout config through the
scalar step.

`TargetInput` must not contain:

- action-space tokens;
- candidate information;
- left/right term information;
- exact action-space generation results for unselected definitions.

### ActionInput

`ActionInput` is an immutable snapshot built only after one target definition has
been selected and Rust has returned a non-empty `ActionSpace`:

```text
ActionInput {
  state_tokens
  selected_def_index
  selected_definition_position
  action_space_tokens
  candidate_positions
  candidate_legal_mask
  left_term_positions_by_candidate
  right_term_positions_by_candidate
  sample_metadata
}
```

The `ActionInput` stores enough information to score the sampled action later
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

The left and right masks are stored in the deterministic term order defined by
the selected candidate in `ActionInput`.

### Policy API

The policy-facing API should expose semantic sampling and scoring:

```text
sample_target(TargetInput, rng) -> TargetSample
score_target(TargetInput, TargetChoice) -> target_logp

sample_action(ActionInput, rng) -> ActionSample
score_action(ActionInput, ActionChoice) -> action_logp
```

The row wrapper may call batched variants:

```text
sample_target_batch(padded_target_inputs, rngs)
score_target_batch(padded_target_inputs, target_choices)
sample_action_batch(padded_action_inputs, rngs)
score_action_batch(padded_action_inputs, action_choices)
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

`definition_embeddings` are gathered or pooled from the token spans for each
definition. `global_state_embedding` may be a pooled token, an explicit summary
token, or an attention-pooled vector.

### Target Head

The target head produces logits over:

```text
STOP + every current def_index
```

Definition logits are computed from definition embeddings with global state
context. The STOP logit is computed from the global state embedding.

Before sampling or scoring, logits are masked by:

- `target_legal_mask` for definitions;
- `stop_legal` for STOP.

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
space. Logits are masked by `candidate_legal_mask`. The masked categorical
distribution is:

```text
p(candidate_index | ActionInput)
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
  log p(target_choice | TargetInput)
```

The action log probability is:

```text
action_logp =
  log p(candidate_index | ActionInput)
+ sum_i log p(left_bit_i | ActionInput, candidate_index, left_prefix_before_i)
+ sum_j log p(right_bit_j | ActionInput, candidate_index, left_mask, right_prefix_before_j)
```

The full scored step contribution is:

```text
target_score_mask * target_logp
+ action_score_mask * action_logp
```

The trainer owns advantage weighting and normalization.

## Bit Legality

Rust requires both side masks to be non-empty. The bit decoder enforces this
without turning masks into categorical subsets.

For a side with `n` terms:

- if `n == 0`, the stored action input is invalid for the current Rust kernel;
- if no previous bit is `KEEP` and this is the final bit, only `KEEP` is legal;
- otherwise both `KEEP` and `DROP` are legal.

Sampling masks illegal bit values before sampling. Scoring replays stored bits
through the same legality rule.

When only one bit is legal, the distribution is deterministic. The implementation
may record that bit with logp zero or omit it from metric counts, but scoring and
loss must be consistent between sampling and recomputation.

## STOP Legality

The model treats STOP as a target choice controlled by `stop_legal`.

The training spec owns STOP modes such as:

- always legal;
- terminal only;
- minimum rewrite count before legal.

The policy model does not decide which mode is active. It only masks STOP
according to `TargetInput.stop_legal`.

## Immutable Storage

Rollout must store immutable data for every scored decision:

```text
TargetInput snapshot
TargetChoice
target_score_mask

ActionInput snapshot
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
- target legality masks.

Batched action scoring pads:

- state/action token sequences;
- candidates;
- side term positions;
- stored bit sequences;
- candidate and bit legality masks.

Padding values must be safe for the model to process and must be masked out
before logits are interpreted. Padding must not change logp, metrics, or loss for
real decisions.

## Error Handling

Policy scoring should fail with clear errors when:

- a target choice is illegal under `TargetInput`;
- STOP is chosen while `stop_legal` is false;
- an action choice is scored with a masked action entry;
- `candidate_index` is out of range or illegal;
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
- illegal definitions and illegal STOP are masked before sampling and scoring;
- target scoring matches manual masked-softmax logp on a small fixture;
- target input construction does not call exact action-space generation.

Action model tests:

- candidate logits include only selected action-space candidates;
- candidate scoring matches manual masked-softmax logp;
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
