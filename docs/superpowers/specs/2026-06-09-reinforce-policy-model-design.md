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
- Store immutable model arrays and choices so training can recompute logp.
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
expressions beyond constructing model token arrays from snapshots and masks
supplied by the environment.

## Public Contracts

### Model Inputs Are Token Representations

The neural model consumes JAX-compatible token representations. This spec uses
`state_tokens` and `action_space_tokens` as short names, but they are not required
to be raw integer token-id vectors.

```text
TokenTree[T] =
  a rectangular JAX pytree whose leaves share leading token axis T
```

A `TokenTree` may be structured integer fields, projected float features, packed
numeric payloads, or another rectangular representation accepted by the token
embedder. Token padding masks remain explicit arrays because attention and
`jax.vmap` need stable rectangular shapes.

The rollout and trainer also need sidecar arrays such as definition masks and
semantic choices. Those sidecars are stored in the rollout table, not passed as
separate symbolic inputs to the attention model.

In this spec:

- target tokens are the model-facing input for target selection;
- action tokens are the model-facing input for action selection;
- rollout storage is a struct-of-arrays table containing token pytrees, masks,
  choices, and score masks.

### Target Arrays

One scalar target decision stores:

```text
state_tokens: TokenTree[T_state]
state_token_mask: bool[T_state]
def_mask: bool[D]
target_choice: int32[]  # STOP = -1, def i = i
```

`state_tokens` are derived from the current symbolic tensor state. They include
definition marker tokens, so the target head can associate logits with
`def_index` values. An implementation may cache definition positions as tokenizer
sidecars for efficient pooling, but those positions are derived from the token
sequence.

`def_mask` is the current cheap/lazily-refined definition mask. It is applied to
definition logits after the target head runs. STOP is always an available target
choice and is not controlled by a separate mask.

Target arrays must not contain:

- action-space tokens;
- candidate information;
- left/right term information;
- exact action-space generation results for unselected definitions.

### Action Arrays

One scalar action decision stores arrays built only after one target definition
has been selected and Rust has returned a non-empty `ActionSpace`:

```text
state_tokens: TokenTree[T_state]
state_token_mask: bool[T_state]
selected_def_index: int32[]
action_space_tokens: TokenTree[T_action]
action_space_token_mask: bool[T_action]
candidate_index: int32[]
left_mask: bool[L]
left_valid_mask: bool[L]
right_mask: bool[R]
right_valid_mask: bool[R]
```

`state_tokens`, `selected_def_index`, and `action_space_tokens` are the
model-facing action input. Candidate and side-term boundaries are encoded in the
action-space token sequence. An implementation may cache candidate positions and
side-term positions as tokenizer sidecars for efficient heads and vectorization,
but those positions are derived from `action_space_tokens`.

All candidates encoded in `action_space_tokens` are legal candidates returned by
Rust. The stored arrays contain enough plain data to score the sampled action
later without a live `ActionSpace` handle.

### Choices

Target choices are:

```text
TargetChoice =
  -1      # STOP
  0..D-1  # def_index
```

`target_choice` is the stored semantic choice, not the target-head logit index.
The target head may use internal logit order `[STOP, def0, def1, ...]`; scoring
maps `target_choice == -1` to logit index `0` and `target_choice >= 0` to
`target_choice + 1`.

Action choices are Rust-compatible:

```text
ActionChoice {
  candidate_index
  left_mask: tuple[bool, ...]
  right_mask: tuple[bool, ...]
}
```

The left and right masks are stored in the deterministic term order encoded for
the selected candidate in `action_space_tokens`.

### Policy API

The policy-facing API should expose scalar JAX functions. Row and batch scoring
should use `jax.vmap` over these scalar functions after padding stored arrays into
rectangular arrays.

```text
sample_target(params, state_tokens, state_token_mask, def_mask, rng)
  -> target_choice
  -> target_logp

score_target(params, state_tokens, state_token_mask, def_mask, target_choice)
  -> target_logp

sample_action(
  params,
  state_tokens,
  state_token_mask,
  selected_def_index,
  action_space_tokens,
  action_space_token_mask,
  rng,
)
  -> action_choice
  -> action_logp

score_action(
  params,
  state_tokens,
  state_token_mask,
  selected_def_index,
  action_space_tokens,
  action_space_token_mask,
  action_choice,
)
  -> action_logp
```

`state_token_mask` and `action_space_token_mask` are padding/attention masks.
They are not legality masks. `def_mask` is the target legality mask.

`target_logp` and `action_logp` are scalar JAX values. Sampled logp from rollout
is diagnostic; training recomputes differentiable logp with the `score_*`
functions.

The intended vectorized scoring shape is:

```text
target_logp_batch =
  jax.vmap(score_target, in_axes=(None, 0, 0, 0, 0))(
    params,
    state_tokens_batch,
    state_token_mask_batch,
    def_mask_batch,
    target_choice_batch,
  )

action_logp_batch =
  jax.vmap(score_action, in_axes=(None, 0, 0, 0, 0, 0, 0))(
    params,
    state_tokens_batch,
    state_token_mask_batch,
    selected_def_index_batch,
    action_space_tokens_batch,
    action_space_token_mask_batch,
    action_choice_batch,
  )
```

For `TokenTree` arguments, an `in_axes=0` entry means JAX maps every token-tree
leaf over its leading sample axis.

The policy should not expose a separate public batch API unless a later
performance design needs one.

## Tokenization

The tokenizer is a faithful serializer from symbolic snapshots to deterministic
JAX-compatible `TokenTree` values. It preserves snapshot order and values:

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

The tokenizer should not rewrite, canonicalize, optimize, reorder terms, infer
graph roles, or simplify symbolic content. If the representation includes symbol
references, generated names, or payload ids, their meaning is scoped to the
current serialized snapshot. Equal references within one snapshot should remain
equal in the token representation, but a payload value such as an intermediate
name id carries no cross-sample semantic identity.

The examples below use readable token names. The implementation may use
structured integer leaves, float feature leaves, or another rectangular JAX
pytree accepted by the token embedder.

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
faithful tokenizer
token embedder
state attention encoder
selected action-space attention encoder
semantic policy heads
```

### Token Embedder

The embedder maps `TokenTree` leaves to vectors. Depending on the tokenizer
representation, it may combine:

- token kind embedding;
- numeric payload embeddings, projections, or feature projections;
- position embeddings;
- segment/type embeddings such as state, definition, action space, candidate,
  left side, right side, and rewritten definition.

The exact embedding math is an implementation detail, but the tokenizer output
and head inputs should be stable.

### State Encoder

The state encoder attends over `state_tokens` with `state_token_mask` and
returns:

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
state_tokens + state_token_mask
+ selected definition marker
+ action_space_tokens + action_space_token_mask
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
space. For one scalar sample, every candidate encoded in `action_space_tokens` is
a legal candidate. In padded batches, padded candidate slots are masked out after
logits are produced.

The categorical distribution is:

```text
p(candidate_index |
  state_tokens,
  state_token_mask,
  selected_def_index,
  action_space_tokens,
  action_space_token_mask)
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
  log p(target_choice | state_tokens, state_token_mask, def_mask)
```

The action log probability is:

```text
action_context =
  state_tokens,
  state_token_mask,
  selected_def_index,
  action_space_tokens,
  action_space_token_mask

action_logp =
  log p(candidate_index | action_context)
+ sum_i log p(left_bit_i | action_context, candidate_index, left_prefix_before_i)
+ sum_j log p(right_bit_j | action_context, candidate_index, left_mask, right_prefix_before_j)
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
state_tokens + state_token_mask -> raw logits over STOP + defs
-> apply def_mask to defs only
```

When all definitions are masked out, STOP is the only unmasked choice and is
sampled with probability one regardless of the negative bias.

## Immutable Storage

Rollout must store immutable arrays for every scored decision:

```text
state_tokens
state_token_mask
def_mask
target_choice
target_score_mask

selected_def_index
action_space_tokens
action_space_token_mask
candidate_index
left_mask
left_valid_mask
right_mask
right_valid_mask
action_score_mask
```

Here `state_tokens` and `action_space_tokens` denote immutable token pytrees. Each
leaf is stored as a rectangular array with the same row/sample/token leading axes
as the corresponding token mask.

Stored arrays must not hold live `RewriteState` or `ActionSpace` handles. PyO3
handles may be used during rollout execution, but training replay must operate on
plain immutable Python/JAX arrays derived from snapshots.

This requirement protects scoring from stale action-space handles and from
mutation of `RewriteState.definition_mask()` during exact-empty refinement.

## Padding And Vectorization

Before using `jax.vmap`, row or training code pads scalar arrays into rectangular
arrays. Target vectorization pads:

- state token sequences;
- definition positions;
- target choices;
- definition masks.

Action vectorization pads:

- state/action token sequences;
- candidates;
- side term positions;
- stored bit sequences;
- candidate padding masks for padded batches.

Padding values must be safe for the scalar model functions to process under
`jax.vmap` and must be masked out before logits are interpreted. Padding must not
change logp, metrics, or loss for real decisions.

## Error Handling

Policy scoring should fail with clear errors when:

- a target choice is neither `-1` nor a valid `def_index`, or selects a masked
  definition;
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
- tokenization is a faithful serialization and does not rewrite, canonicalize, or
  simplify symbolic content;
- repeated snapshot references remain equal within one serialized state, while
  generated payload ids are not treated as cross-sample semantic identities;
- `definition_positions` align with tokenized definitions;
- action-space tokenization is deterministic for a fixture action space;
- candidate and side-term positions align with candidate snapshots;
- tokenization does not query action spaces for unselected definitions.

Target model tests:

- target logits include STOP and all current definitions;
- illegal definitions are masked before sampling and scoring;
- STOP remains available and uses the configured negative initial bias;
- stored `target_choice=-1` scores STOP and `target_choice=k` scores definition
  `k`;
- target scoring matches manual masked-softmax logp on a small fixture;
- target array construction does not call exact action-space generation.

Action model tests:

- candidate logits include only selected action-space candidates;
- candidate scoring matches manual softmax logp for one scalar action space;
- padded candidate slots are masked only in padded batches;
- left and right masks are decoded in deterministic bit order;
- final-bit forcing prevents empty side masks;
- scoring an illegal empty mask fails;
- sampled action choices can be applied through `RewriteState.step_with_space`.

Vectorization tests:

- `jax.vmap(score_target)` over padded arrays matches scalar target scoring;
- `jax.vmap(score_action)` over padded arrays matches scalar action scoring;
- masked padded entries do not affect logits used for real choices;
- width-1 row scoring matches scalar scoring.

## Acceptance Criteria

- The policy can sample and score target choices without generating unselected
  action spaces.
- The policy can sample and score an action for one selected non-empty action
  space.
- Left and right masks are modeled as bit sequences with exact recomputed logp.
- Stored target/action arrays are immutable and sufficient for differentiable
  rescoring.
- The policy API supports scalar sampling/scoring and `jax.vmap` row scoring.
- The implementation can be tested without importing deprecated `gristmill_rl`
  policy APIs.
