# REINFORCE Policy Model Sampling And Scoring Implementation Design

Status: planned
Phase: 2 of 3
Feeds implementation plan: yes

## Summary

This spec defines the second implementation phase for the REINFORCE system: the
model-facing data contracts, tokenizer, attention policy, sampling functions, and
differentiable scoring functions.

The policy reads immutable snapshots of symbolic states and selected action
spaces. It samples and scores semantic choices:

```text
target choice: STOP or def_index
action choice: candidate_index, left bit sequence, right bit sequence
```

The policy does not execute rewrites, generate exact action spaces, compute
rewards, or update parameters. It can be implemented and tested with fixtures or
with snapshots from the row environment.

## Goals

- Define deterministic state and action-space tokenization.
- Store immutable target/action arrays sufficient for later scoring.
- Implement a replaceable attention-based model over token pytrees.
- Implement target sampling and target scoring.
- Implement action sampling and action scoring.
- Decode left and right masks as constrained bit sequences.
- Support padded `jax.vmap` sampling and scoring for rows.
- Make immediate STOP rare through model initialization, not trainer masking.
- Fail clearly for illegal stored choices.

## Non-Goals

- Implementing Rust row environment behavior.
- Implementing rollout orchestration, rewards, loss, optimizer, metrics, or
  checkpoints.
- Differentiating through action-space generation, rewrite application, or cost.
- Preserving deprecated causal token-decoder policy APIs.
- Choosing a final high-performance architecture.
- Canonicalizing symbolic identifiers across unrelated samples.

## Model Inputs

The model consumes JAX-compatible token pytrees:

```text
TokenTree[T] =
  a rectangular JAX pytree whose leaves share leading token axis T
```

Leaves may be integer ids, numeric payloads, projected features, structural
markers, or another representation accepted by the token embedder.

Every token tree must include structural marker leaves sufficient to derive
pooling and decoding masks under JAX. Feature-only token arrays are not enough.

Common state token leaves:

```text
token_kind: int32[T]
def_index: int32[T]  # -1 outside definition tokens
```

Common action-space token leaves:

```text
token_kind: int32[T]
candidate_index: int32[T]  # -1 outside candidate tokens
side: int32[T]             # none, left, right, rewritten
term_index: int32[T]       # -1 outside side-term tokens
```

Padding masks remain explicit arrays because attention and `jax.vmap` need
stable rectangular shapes.

## Concrete TokenTree Encoding

The first implementation should represent a token position as a typed record of
fields, stored column-wise as a JAX pytree. A token is not one vocabulary id.

For one scalar state:

```text
state_tokens = {
  token_kind: int32[T],
  segment: int32[T],
  def_index: int32[T],
  term_index: int32[T],
  factor_index: int32[T],
  tensor_id: int32[T],
  range_id: int32[T],
  index_id: int32[T],
  coeff_num: int32[T],
  coeff_den: int32[T],
  position: int32[T],
}
```

For one scalar action space:

```text
action_space_tokens = {
  token_kind: int32[T],
  segment: int32[T],
  candidate_index: int32[T],
  side: int32[T],
  term_index: int32[T],
  factor_index: int32[T],
  tensor_id: int32[T],
  range_id: int32[T],
  index_id: int32[T],
  coeff_num: int32[T],
  coeff_den: int32[T],
  position: int32[T],
}
```

The exact field set may grow if implementation needs more structural markers,
but v1 should start with this columnar field-record representation. Row batches
add a leading sample axis to every leaf:

```text
state_tokens.token_kind[sample, token]
action_space_tokens.candidate_index[sample, token]
```

Sentinel values such as `-1` mark fields that are not meaningful for a token.
Sentinels must not point to real definitions, candidates, tensors, ranges,
indices, terms, factors, sides, or segments.

Ids are scoped to the current snapshot. Equal ids within one snapshot should
remain equal in the token fields so the model can learn local reference
structure. The same numeric id in unrelated samples must not be treated as the
same global semantic object.

Example factor serialization:

```text
Factor { tensor: 7, indices: [2, 5, 8] }

FACTOR_START  tensor_id=7
FACTOR_INDEX  index_id=2
FACTOR_INDEX  index_id=5
FACTOR_INDEX  index_id=8
FACTOR_END
```

The tokenizer may emit additional `DEF_START`, `TERM_START`, `COEFF`,
`SUM_INDEX`, and matching end-marker tokens as needed to preserve snapshot
structure.

## Target Arrays

One scalar target decision stores:

```text
state_tokens: TokenTree[T_state]
state_token_mask: bool[T_state]
def_mask: bool[D]
target_choice: int32[]  # STOP = -1, def i = i
target_score_mask: bool[]
```

`state_tokens` are derived from the current symbolic state snapshot. They must
not contain action-space tokens or information about unselected definitions'
exact action spaces.

`def_mask` is the current definition legality mask supplied by the environment.
The target head applies it to definition logits. STOP is always present and is
not controlled by `def_mask`.

## Action Arrays

One scalar action decision stores:

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
action_score_mask: bool[]
```

Action arrays are built only after the environment returns a non-empty action
space for one selected definition. They contain enough plain data to score the
sampled action later without a live `ActionSpace` or `ActionSpaceRow` handle.

In rollout-table storage, `state_tokens` and `state_token_mask` should be stored
once per row/sample and shared by target and action scoring. The action model API
still receives state tokens as inputs; that does not require a second physical
state-token table.

## Choices

Target choices are semantic:

```text
-1      # STOP
0..D-1  # definition index
```

If the target head uses internal logit order `[STOP, def0, def1, ...]`, scoring
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

The left and right masks are interpreted in the deterministic term order encoded
by the selected candidate in `action_space_tokens`.

## Policy API

The policy-facing API exposes scalar JAX functions:

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

`state_token_mask` and `action_space_token_mask` are padding and attention masks.
They are not legality masks. `def_mask` is the target legality mask.

Sampled rollout logp is diagnostic. Training must recompute differentiable logp
with the `score_*` functions from stored arrays and choices.

Row-level policy sampling is expected to parallelize through `jax.vmap` over the
scalar `sample_*` functions. Rollout code should provide row-aligned arrays and
one RNG key per sampled row entry, then call the vectorized sampler instead of a
Python loop over samples.

## Tokenization

The tokenizer is a faithful serializer from snapshots to deterministic token
trees. It preserves snapshot order and values. It must not rewrite, simplify,
canonicalize, or optimize symbolic content.

State tokenization serializes:

- global range and tensor metadata;
- definitions in snapshot order;
- base tensor ids;
- external indices;
- terms;
- coefficients;
- summed indices;
- factors;
- factor index lists;
- structural markers for definition membership.

Action-space tokenization serializes:

- selected definition marker;
- candidates in snapshot order;
- left definition data per candidate;
- left terms in deterministic order;
- right definition data per candidate;
- right terms in deterministic order;
- rewritten definition data per candidate;
- optional graph or incidence metadata exposed by the snapshot;
- structural markers for candidate, side, and side-term membership.

Symbol references and generated payload ids are scoped to one serialized
snapshot. Equal references within one snapshot should stay equal in the token
representation, but payload ids do not carry cross-sample semantic identity.

The first implementation may tokenize action-space snapshots in Python from
plain host data returned by the row environment. JAX must never trace through
live PyO3 handles.

## Architecture

The v1 architecture has five replaceable components:

```text
faithful tokenizer
token embedder
state attention encoder
selected action-space attention encoder
semantic policy heads
```

The token embedder maps token-tree leaves to one dense vector per token position.
It should compose field embeddings and numeric projections, for example:

```text
x[t] =
  embed_token_kind(token_kind[t])
+ embed_segment(segment[t])
+ embed_side(side[t])
+ embed_scoped_tensor_id(tensor_id[t])
+ embed_scoped_range_id(range_id[t])
+ embed_scoped_index_id(index_id[t])
+ embed_def_index(def_index[t])
+ embed_candidate_index(candidate_index[t])
+ project_numeric(coeff_num[t], coeff_den[t], position[t])
```

Sentinel field values contribute zero for that field. The output of the embedder
is a dense array:

```text
embedded_tokens[token, d_model]
```

For row batches, the output is:

```text
embedded_tokens[sample, token, d_model]
```

The state encoder attends over `state_tokens` with `state_token_mask` and
returns:

```text
state_token_embeddings
definition_embeddings[def_index]
global_state_embedding
```

Definition embeddings are pooled from masks derived from state token structural
markers.

After a definition is selected, the action encoder attends over state context,
the selected definition marker, and `action_space_tokens`. It returns:

```text
action_token_embeddings
candidate_embeddings[candidate_index]
left_term_embeddings[candidate_index, term_index]
right_term_embeddings[candidate_index, term_index]
action_context_embedding
```

The implementation may reuse state encoder outputs or recompute combined
state/action context. Caching is an optimization.

## Target Distribution

The target head produces logits over:

```text
STOP + every current def_index
```

Definition logits are masked by `def_mask` before sampling or scoring. STOP
remains available even when definitions are legal.

The first runnable implementation should initialize the STOP head bias near
`-20`. This makes immediate STOP rare when any definition logit is near zero,
while STOP still has probability one when all definitions are masked.

## Action Distribution

The action distribution factorizes as:

```text
log p(action_choice | action_context) =
  log p(candidate_index | action_context)
+ sum_i log p(left_bit_i | action_context, candidate_index, left_prefix_before_i)
+ sum_j log p(right_bit_j | action_context, candidate_index, left_mask, right_prefix_before_j)
```

Candidate logits cover only legal candidates encoded in the selected action
space. Padded candidate slots in batched arrays are masked after logits are
produced.

Left and right masks are decoded as ordered bit sequences. The bit decoder must
enforce the Rust kernel's non-empty side-mask requirement:

- if a side has zero terms, the stored action input is invalid;
- if no previous bit is `KEEP` and this is the final bit, only `KEEP` is legal;
- otherwise both `KEEP` and `DROP` are legal.

Sampling applies this constraint before sampling each bit. Scoring replays
stored bits through the same constraint. Deterministic forced bits have logp zero
or are consistently omitted from metric counts.

## Padding And Vectorization

Before `jax.vmap`, scalar arrays are padded into rectangular row arrays.

Target vectorization pads:

- state token-tree leaves;
- state token masks;
- definition masks;
- target choices for scoring;
- RNG keys for sampling.

Action vectorization pads:

- state token-tree leaves;
- action-space token-tree leaves;
- token masks;
- selected definition indices;
- candidate choices for scoring;
- left and right bit sequences;
- left and right valid masks;
- RNG keys for sampling.

Padding values must be safe for scalar model functions under `jax.vmap`.
Structural marker padding uses sentinels such as `-1` that do not point to real
definitions, candidates, sides, or terms.

Masked score entries contribute no logp, loss, or metrics. Padding must not
change logp for real decisions. Sampling masks must ensure padded entries do not
produce choices that are interpreted as real rollout decisions.

## Row Sampling And Scoring

The intended row sampling shape is:

```text
jax.vmap(sample_target, in_axes=(None, 0, 0, 0, 0))
jax.vmap(sample_action, in_axes=(None, 0, 0, 0, 0, 0, 0))
```

The intended row scoring shape is:

```text
jax.vmap(score_target, in_axes=(None, 0, 0, 0, 0))
jax.vmap(score_action, in_axes=(None, 0, 0, 0, 0, 0, 0))
```

For `TokenTree` arguments, `in_axes=0` maps every token-tree leaf over its
leading sample axis.

## Error Handling

Policy scoring should fail clearly when:

- a target choice is neither STOP nor a valid definition index;
- a target choice selects a masked definition;
- an action choice is scored for a masked action entry;
- a candidate index is out of range or points to a padded slot;
- a stored bit sequence length does not match the selected candidate side;
- a stored left or right side mask is empty;
- an illegal bit appears during constrained replay;
- required token arrays or masks are missing for a true score mask.

Invalid stored choices are contract bugs and must not be silently ignored.

## Testing Requirements

Tokenizer tests:

- state tokenization is deterministic for fixture snapshots;
- tokenization preserves snapshot order and does not simplify symbolic content;
- repeated references stay equal within one serialized snapshot;
- generated payload ids are not treated as cross-sample semantic identities;
- structural marker leaves align with definitions, candidates, sides, and terms;
- derived definition, candidate, and side-term masks align with snapshots;
- action-space tokenization from row snapshots matches scalar snapshot
  tokenization for non-empty entries;
- target array construction does not query action spaces.

Target tests:

- target logits include STOP and all current definitions;
- illegal definitions are masked before sampling and scoring;
- STOP remains available and uses the configured negative initial bias;
- `target_choice=-1` scores STOP and `target_choice=k` scores definition `k`;
- target scoring matches manual masked-softmax logp on small fixtures;
- all-masked definitions make STOP probability one.

Action tests:

- candidate logits include selected action-space candidates;
- candidate scoring matches manual softmax logp on a fixture;
- padded candidate slots are masked only in padded batches;
- left and right masks decode in deterministic bit order;
- final-bit forcing prevents empty side masks;
- scoring an illegal empty side mask fails;
- sampled action choices validate through the row environment when it is
  available.

Vectorization tests:

- `jax.vmap(sample_target)` over padded arrays returns row-aligned target choices
  and sampled logp;
- `jax.vmap(sample_action)` over padded arrays returns row-aligned action choices
  and sampled logp;
- `jax.vmap(score_target)` over padded arrays matches scalar target scoring;
- `jax.vmap(score_action)` over padded arrays matches scalar action scoring;
- masked padded entries do not affect real-choice logp;
- masked padded entries are not interpreted as real sampled choices;
- width-1 row scoring matches scalar scoring.

## Exit Criteria

Phase 2 is complete when:

- deterministic state and action-space tokenizers produce token trees with
  structural markers and masks;
- scalar target sampling and scoring work without constructing unselected action
  spaces;
- scalar action sampling and scoring work for selected non-empty action spaces;
- left and right masks are modeled as constrained bit sequences;
- stored target/action arrays are immutable plain data sufficient for rescoring;
- padded `jax.vmap` sampling returns row-aligned target/action choices and
  sampled logp;
- padded `jax.vmap` scoring matches scalar scoring;
- STOP bias initialization is implemented and tested;
- tests do not require the trainer or deprecated policy APIs.
