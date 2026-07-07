# Story Spec: Grammar-Constrained Flat Token Sampling

## Product Owner Inputs

Outcome:

Build a JAX-friendly sampler for the flat definition seq2seq path so source
token IDs can produce grammar-constrained generated target token streams and
the log probability of exactly what was sampled.

Deliverables:
- A fixed-shape autoregressive sampling helper.
- Inputs:
  - model callable producing logits from `source_ids` and `decoder_input_ids`
  - `source_ids: int32[B, source_len]`
  - `grammar: FlatDefinitionGrammar`
  - `rng: jax.Array`
  - `target_len: int`
- Output is a plain 3-tuple:
  - `generated_ids: int32[B, target_len]`
  - `token_log_probs: float[B, target_len]`
  - `sequence_log_prob: float[B]`
- Behavior:
  - Initialize each generated row as BOS followed by PAD.
  - At each active step, call the model with the current fixed-shape prefix.
  - Use the logits at the current decoder position to sample the next token.
  - Mask logits with the current grammar state before categorical sampling.
  - Carry grammar state through the sampling loop.
  - After EOS is sampled, fill later positions with PAD and contribute `0.0`
    token logp.
  - If EOS is not sampled before `target_len`, return the max-length stream
    unchanged; later decode or validation may reject it.

Autodiff/JAX constraints:
- Works on fixed-shape arrays.
- Works under `jax.jit` for fixed `target_len`.
- Uses `jax.lax.scan`.
- Carries grammar state directly; no dynamic prefix slicing.
- Sampling itself is discrete, but sampled token logp remains differentiable
  with respect to model logits and model parameters.
- Model calls during sampling use deterministic inference behavior.

Non-goals:
- No decode-to-definitions.
- No rewrite validation.
- No reward computation.
- No trainer integration.
- No CLI or checkpoint integration.
- No dataset/collator work.
- No beam search.
- No top-k or top-p sampling.
- No KV cache.
- No forced EOS at the final position.
- No full model protocol integration.

Acceptance criteria:
- Generated IDs have shape `[B, target_len]`.
- Token logp has shape `[B, target_len]`.
- Sequence logp has shape `[B]`.
- Every generated row starts with BOS.
- Tokens after EOS are PAD.
- Active sampled tokens are valid for the grammar state used at that step.
- `sequence_log_prob == sum(token_log_probs, axis=-1)`.
- Token logp after EOS is `0.0`.
- The helper is JIT-compatible for fixed batch and target shapes.
- Tests show gradients flow from returned sequence logp through model logits.
- Existing tokenizer, grammar, scoring, and flat seq2seq tests still pass.

## Role Boundary

The user is acting as Product Owner / Stakeholder.
Low-level implementation decisions are agent-owned.
User-provided implementation details are binding only when they express product,
compatibility, safety, performance, or operational constraints.

## Agent-Owned Notes

Assumptions:
- The sampler belongs in a focused module separate from tokenizer, grammar,
  scoring, model core, and trainer code.
- The sampler should accept a model-like callable rather than depend on a
  concrete model class.
- Returning logp with samples is part of the sampling contract because later
  policy-gradient training needs the probability of the sampled action.
- Max-length outputs without EOS are valid sampler outputs, but not necessarily
  valid generated definition streams.

Risk level: medium

Story size: small

Escalation: none

Implementation hints:
- Initialize `generated_ids` with BOS at position `0` and PAD elsewhere.
- Initialize grammar state, then consume BOS before the first sample.
- For step `t`, use `logits[:, t, :]` to sample token `t + 1`.
- Mask invalid logits with `jnp.where(valid_next, logits, -jnp.inf)`.
- Use `jax.nn.log_softmax` on masked logits and gather the sampled token logp.
- Keep inactive rows after EOS as PAD with zero logp.

Follow-up stories:
- Decode sampled IDs to definitions and validate/reject outputs.
- Reward and trainer integration.
- Dataset/collator integration.
- Higher-efficiency generation with cached decoder state if needed.

## Definition of Done

- Acceptance criteria met
- Deliverables completed
- Non-goals avoided
- Verification run
- No unjustified scope, dependency, abstraction, public API, or broad refactor
  growth
