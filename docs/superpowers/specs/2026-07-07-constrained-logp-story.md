# Story Spec: Differentiable Grammar-Constrained LogP

## Product Owner Inputs

Outcome:

Build differentiable grammar-constrained log-probability helpers so model logits
can be scored against target token IDs while assigning probability only over
syntactically valid next tokens.

Core deliverable:

A grammar-constrained logp helper that is differentiable with respect to model
logits, so gradients can flow back to model parameters through:

```text
logp -> logits -> model params
```

Deliverables:
- Fixed-shape JAX helpers:
  - `constrained_token_log_probs(...) -> float[B, T]`
  - `constrained_sequence_log_prob(...) -> float[B]`
- Inputs:
  - `logits: float[B, T, V]`
  - `decoder_input_ids: int32[B, T]`
  - `labels: int32[B, T]`
  - `label_mask: bool[B, T]`
  - `grammar: FlatDefinitionGrammar`

Definitions:
- `decoder_input_ids`: prefix tokens fed to the decoder.
- `labels`: shifted next-token IDs to score, usually
  `decoder_input_ids[:, 1:]` plus EOS at the boundary.
- `label_mask`: true for real labels through EOS, false for padding.

Behavior:
- For each active position `t`, use
  `FlatDefinitionGrammar.valid_next_masks_for_decoder_input(decoder_input_ids)`
  to mask invalid next-token logits.
- Active grammar-valid labels get
  `log_softmax(grammar_masked_logits)[label]`.
- Active grammar-invalid labels get a large negative log-probability.
- Masked label positions return `0.0`.
- Sequence logp sums token logp across `T`.

JAX/autodiff constraints:
- Works on fixed-shape arrays.
- Works under `jax.jit`.
- Differentiable with respect to valid active logits.
- Grammar tables are fixed and non-trainable.
- No dynamic variable-length arrays.
- Does not coerce runtime input arrays; callers are responsible for passing
  floating logits, integer token IDs/labels, and boolean masks.

Non-goals:
- No loss helper.
- No example weights.
- No decoder-input builder.
- No sampler.
- No dataset/collator.
- No trainer integration.
- No model changes.
- No CLI/checkpoint work.

Acceptance criteria:
- Uses `FlatDefinitionGrammar.valid_next_masks_for_decoder_input`.
- Returns `[B, T]` token logp and `[B]` sequence logp.
- Invalid active labels receive large negative logp.
- Masked positions return `0.0`.
- Gradients through valid active logits are nonzero in tests.
- Invalid active labels do not create gradient signal for their invalid selected
  token.
- JIT test passes.
- Existing tests still pass.

## Role Boundary

The user is acting as Product Owner / Stakeholder.
Low-level implementation decisions are agent-owned.
User-provided implementation details are binding only when they express product,
compatibility, safety, performance, or operational constraints.

## Agent-Owned Notes

Assumptions:
- The helper belongs in a focused scoring module separate from grammar, model,
  sampler, and trainer code.
- The helper should not validate dynamic shapes or token ranges because it must
  stay JAX-friendly.
- Labels are explicit inputs even though they are normally shifted decoder
  inputs, because EOS/padding boundaries are caller-owned.

Risk level: medium

Story size: small

Escalation: none

Implementation hints:
- Build valid-next masks from the grammar.
- Mask invalid logits with `jnp.where(valid_next, logits, -jnp.inf)` before
  `jax.nn.log_softmax`.
- Use `jnp.take_along_axis` to select label log-probs.
- Use `jnp.where(label_mask, ..., 0.0)` for padding positions.

Follow-up stories:
- Decoder input/label/mask builder.
- Loss/objective helpers.
- Grammar-constrained autoregressive sampler.
- Dataset, trainer, CLI, and checkpoint integration.

## Definition of Done

- Acceptance criteria met
- Deliverables completed
- Non-goals avoided
- Verification run
- No unjustified scope, dependency, abstraction, public API, or broad refactor
  growth
