# Symbolic Tensor Embedding Design

## Summary

Add a v1 network representation for symbolic tensor rewrite policies.

The policy is a grammar-constrained autoregressive Transformer over faithful
tokens derived from `TensorComputation`, `TensorDef`, and `ActionSpace`
snapshots. The model samples one rewrite-path step in two calls:

```text
state context -> DEF@i | STOP
state + action-space context -> CAND@j, left mask bits, right mask bits, END
```

Rust remains authoritative for state mutation, legal definition masks, and exact
action-space generation. The Python policy layer only parameterizes the
probability distribution over legal choices.

Stage 1 must not enumerate exact action spaces for every definition before
sampling. It uses the current `RewriteState::definition_mask()` as a cheap,
lazy-refined mask and asks Rust for an exact `ActionSpace` only after a
definition has been sampled.

This design is issue #4's first slice. It replaces the current hand-built
summary feature path conceptually, but it does not specify the RL training
algorithm that will consume the policy.

## Goals

- Tokenize symbolic tensor expressions as deterministic, faithful token
  sequences.
- Add a versioned `TensorDef` token schema reusable by state and action-space
  contexts.
- Parameterize a sampleable rewrite policy with a Transformer-style
  autoregressive model.
- Include `STOP` as a first-class terminal rewrite-path action.
- Enforce legal final decisions with dynamic next-token masks derived from
  `RewriteState` and `ActionSpace`.
- Expose sampling and scoring interfaces suitable for REINFORCE or other RL
  algorithms.
- Keep vocabulary and token schema details network-internal and versioned.

## Non-Goals

- Choosing or implementing REINFORCE, MCTS, replay, rollout control, or
  objective-target construction.
- Implementing a typed graph encoder.
- Implementing DeepSets-style set pooling as the v1 representation.
- Canonicalizing, renaming, or normalizing tensor, range, or index IDs.
- Moving action legality into Python-side symbolic logic.
- Changing Rust symbolic JSON or `TensorDef` semantics.
- Adding token truncation or approximate action-space summaries.

## Policy Boundary

Issue #4 owns the policy parameterization:

```text
p(path_step | rewrite_state)
```

The semantic Python-facing operations should be:

```text
sample_step(rewrite_state, rng) -> PolicySample
score_step(rewrite_state, choice) -> log_prob
```

`sample_step` returns either `STOP` or a concrete rewrite decision:

```text
PolicySample {
  stopped: bool,
  def_index: int | None,
  action_space: ActionSpace | None,
  decision: dict | None,
  log_prob: float,
  def_attempts: list[stage-1 trace item],
  decision_tokens: optional debug data,
}
```

`def_attempts` records any stage-1 definition probes that were sampled before
the final `STOP` or accepted definition. This is needed because
`definition_mask()` can have cheap false positives.

`score_step` reruns the same masked decoding transitions against a provided
sample trace or choice and returns the sum of masked next-token log
probabilities. It raises a clear error when the final accepted choice is illegal
for the supplied state.

The policy boundary is stable. Token IDs, token order, model size, and internal
decoder details may change across token schema versions.

## Two-Stage Sampling

The rewrite path step has two choices separated by Rust environment logic:

```text
1. p(DEF@i or STOP | state)
2. p(decision | state, DEF@i, action_space_for_def(i))
```

The first call conditions on the current `RewriteState` computation:

```text
[STATE context] -> DEF@i | STOP
```

The stage-1 mask is built from the current `RewriteState::definition_mask()`:

```text
legal next tokens = STOP plus DEF@i for every i where definition_mask[i] is true
```

The policy runtime must not query exact action spaces for every true mask entry.
`STOP` is always legal. If the cheap definition mask has no true entries,
`STOP` is the only legal token.

If `STOP` is sampled, the rewrite path ends and no local action decision is
decoded.

If `DEF@i` is sampled, the caller asks Rust for only that definition:

```text
state.action_space_for_def(i)
```

If Rust returns `Some(ActionSpace)`, the definition choice is accepted and the
policy proceeds to stage 2.

If Rust returns `None`, the chosen definition was a cheap-mask false positive.
The runtime records the rejected `DEF@i` and its log probability, keeps the
refined mask update made by Rust, and restarts stage 1 on the same computation.
This preserves lazy exactness without scanning the whole state. The final
`PolicySample.log_prob` includes the rejected definition probes plus the final
accepted `DEF@i` or `STOP`.

The second call conditions on the state plus the generated local action space:

```text
[STATE context][ACTION_SPACE context] -> CAND@j -> LEFT bits -> RIGHT bits -> END
```

`CAND@j` is legal only for candidates present in the generated `ActionSpace`.
Left and right mask bits are emitted in deterministic term order for the
selected candidate.

## Token Schema Versioning

The v1 schema is:

```text
TokenSchemaVersion = 1
```

Checkpoints must save this version. Loading a checkpoint with a different token
schema version either uses an explicit migration or fails with a clear
incompatibility error.

The exact token vocabulary is not a Rust or symbolic JSON contract. It is a
network-internal representation. Future versions may change token order, token
types, payload encoding, mask encoding, or even the encoder family as long as
the semantic policy boundary remains clear.

## TensorDef Tokenization

The core reusable unit is a faithful `TensorDef` tokenizer:

```text
TensorDef snapshot -> deterministic token sequence
```

V1 tokenization preserves the existing snapshot fields and order:

```text
base
ext_indices
terms
coeff
sum_indices
factors
tensor
indices
```

For a term such as:

```json
{
  "coeff": [1, 1],
  "sum_indices": [{"id": 2, "range": 0}],
  "factors": [
    {"tensor": 0, "indices": [0, 2]},
    {"tensor": 1, "indices": [2, 1]}
  ]
}
```

the tokenizer produces a sequence equivalent to:

```text
TERM_START
COEFF_NUM value=1
COEFF_DEN value=1
SUM_INDEX id=2 range=0
FACTOR tensor=0
INDEX id=0
INDEX id=2
FACTOR tensor=1
INDEX id=2
INDEX id=1
TERM_END
```

The exact implementation may use structured token records rather than printable
strings, but the conversion must stay faithful:

- no ID canonicalization
- no graph conversion
- no inferred local index roles
- no reordering beyond the snapshot's deterministic order
- no semantic simplification

Raw integer IDs are allowed as token payloads in v1. If arbitrary IDs later
hurt generalization, a future token schema can introduce canonicalized roles or
another representation.

## Context Wrappers

State and action-space contexts compose the `TensorDef` tokenizer with
structural punctuation tokens.

State context:

```text
STATE_START
TensorDef(definition_0)
TensorDef(definition_1)
...
STATE_END
```

Action-space context:

```text
ACTION_SPACE_START def_index=i
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

Structural tokens such as `STATE_START`, `DEF_START`, `TERM_START`,
`CAND_START`, and `ACTION_SPACE_END` act like punctuation marks. They identify
scope and sequence structure. Payload tokens such as `COEFF_NUM`, `FACTOR`,
`INDEX`, and `SUM_INDEX` carry the symbolic content copied from the snapshot.

## Decision Tokens

Decision tokens are separate from symbolic context tokens:

```text
STOP
DEF@i
CAND@j
LEFT_KEEP
LEFT_DROP
RIGHT_KEEP
RIGHT_DROP
END
```

`DEF@i` and `CAND@j` may be represented as token type plus integer payload
rather than as separate vocabulary entries for every integer.

Fixed-order mask bits are the v1 mask representation. For a selected candidate:

```text
left terms:  t0, t1, ..., tn
right terms: u0, u1, ..., um
```

the decoder emits:

```text
LEFT_KEEP|LEFT_DROP for t0
LEFT_KEEP|LEFT_DROP for t1
...
RIGHT_KEEP|RIGHT_DROP for u0
RIGHT_KEEP|RIGHT_DROP for u1
...
END
```

If Rust requires a nonempty mask for a side with terms, the decoder enforces it
with the phase mask. When all previous bits on a side are `DROP`, the final bit
for that side is forced to `KEEP`.

## Dynamic Token Masks

V1 uses a small decoder state machine instead of a general context-free grammar
engine:

```text
choose_def_or_stop
choose_candidate
emit_left_mask
emit_right_mask
done
```

At each phase, the decoder asks the state machine for legal next tokens, sets
all illegal token logits to negative infinity, and samples from the masked,
renormalized distribution.

This makes invalid final decisions impossible by construction:

- no out-of-range definition index
- no out-of-range candidate index
- no malformed candidate sequence
- no duplicate mask term
- no missing mask bits
- no empty mask when a nonempty mask is required

A cheap-mask false positive may be sampled during stage 1, but it is not
returned as a rewrite decision. It is recorded as a rejected probe, the mask is
refined, and decoding continues.

The same state machine is used for scoring. A provided choice is replayed token
by token through the masks. If any token is illegal at its phase, scoring fails
with a concrete error.

## Log Probability

For a terminal path step:

```text
log_prob =
  sum rejected DEF probe log_probs
  + log p(STOP | state, current refined mask)
```

For a rewrite path step:

```text
log_prob =
  sum rejected DEF probe log_probs
  + log p(accepted DEF@i | state, current refined mask)
  + log p(CAND@j | action_space context)
  + sum_t log p(left_bit_t | prefix, action_space context)
  + sum_u log p(right_bit_u | prefix, action_space context)
  + log p(END | prefix)
```

Each rejected probe log probability is computed under the stage-1 mask active
for that attempt.

In v1, `END` is deterministic after all mask bits have been emitted, so its log
probability may be recorded as zero. Keeping `END` in the decision token stream
preserves room for future variable-length decision encodings.

## Model Architecture

V1 uses a small causal Transformer policy module. The model receives:

```text
context tokens + generated decision prefix
```

and returns next-token logits.

The two stages may be separate forward calls, but they should share one policy
model family and token schema:

```text
Call 1: STATE context -> DEF@i | STOP
Call 2: STATE + ACTION_SPACE context -> CAND@j, mask bits, END
```

The implementation may rebuild the context between calls rather than attempting
to cache activations. Caching is an optimization, not part of v1.

## Error Handling

- Empty definition mask: `STOP` is the only legal token.
- `action_space_for_def(i)` returns `None` after a sampled `DEF@i`: record a
  rejected stage-1 probe, keep Rust's refined mask, and restart stage 1.
- `action_space_for_def(i)` returns `None` while scoring a trace that marks the
  definition as accepted: raise `ValueError("invalid def_index ...")`.
- Invalid `def_index` during scoring: raise `ValueError("invalid def_index ...")`.
- Invalid `candidate_index` during scoring: raise
  `ValueError("invalid candidate_index ...")`.
- Invalid mask length during scoring: raise
  `ValueError("invalid left_mask length ...")` or the right-side equivalent.
- Invalid all-drop mask when a nonempty side is required: raise a side-specific
  `ValueError`.
- Token budget exceeded: raise a tokenizer error. V1 does not silently truncate
  state or action-space tokens.

## Testing

Tokenizer tests:

- `TensorDef` tokenization preserves snapshot field order and raw IDs.
- coefficients, `sum_indices`, factors, tensor IDs, and factor index lists are
  represented faithfully.
- state context wraps multiple definitions deterministically.
- action-space context wraps candidates and nested `TensorDef` values
  deterministically.
- schema version is present in tokenizer or checkpoint metadata.

Mask and decoder tests:

- `STOP` is available at the definition stage.
- `STOP` is the only legal token when the cheap definition mask is empty.
- invalid definition and candidate tokens are masked out.
- stage-1 sampling does not enumerate exact action spaces for every definition.
- cheap definition-mask false positives are recorded as rejected probes and
  refined lazily.
- scoring a sample trace includes rejected stage-1 probe log probabilities.
- fixed-order left and right mask bits are emitted for the selected candidate.
- nonempty side masks are forced or rejected according to Rust decision
  requirements.

Policy interface tests:

- sampled non-`STOP` decisions can be applied through
  `RewriteState.step_with_space`.
- scoring a sampled decision reproduces the same token path and a finite log
  probability.
- illegal scored choices fail with clear phase-specific errors.

## Alternatives Considered

Structured Transformer encoder with policy heads:

- Pros: easier batching, direct softmax/Bernoulli heads, straightforward
  legality handling.
- Cons: less natural as a sampleable next-token distribution over rewrite
  paths, and less aligned with REINFORCE path log-prob accounting.

Typed graph encoder:

- Pros: stronger symbolic inductive bias and explicit contraction/index-sharing
  structure.
- Cons: larger feature plumbing and graph batching step. It is better treated
  as a future representation version.

DeepSets-style set embedding:

- Pros: natural for permutation-invariant term or factor sets.
- Cons: weaker fit for the requested process-level tokenization of
  `state + action_space -> decision`, and less direct for constrained
  sequential sampling.

## Acceptance Criteria

- The design defines a v1 faithful `TensorDef` tokenizer.
- The design defines state and action-space context wrappers.
- The design includes `STOP` as a terminal path action.
- The design defines a two-stage masked autoregressive policy.
- The design exposes sampling and scoring operations suitable for REINFORCE.
- The design keeps Rust authoritative for legal rewrite state and action-space
  generation.
- The design explicitly excludes ID canonicalization, graph encoding, and RL
  algorithm changes from v1.

## References

- GitHub issue #4: https://github.com/Br0kenSmi1e/gristmill-rl/issues/4
- Deep Sets: https://arxiv.org/abs/1703.06114
- Graph representations for higher-order logic and theorem proving:
  https://arxiv.org/abs/1905.10006
- OpenAI Structured Outputs constrained decoding note:
  https://openai.com/index/introducing-structured-outputs-in-the-api/
