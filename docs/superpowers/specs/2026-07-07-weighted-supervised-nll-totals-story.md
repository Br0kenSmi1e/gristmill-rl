# Story Spec: Weighted Supervised NLL Totals

## Product Owner Inputs

Outcome:

Add the supervised-training objective primitive for fully preprocessed
fixed-shape flat token batches. The helper returns additive weighted NLL totals
so future micro-batch gradient accumulation can combine micro-batch results
without recomputing denominators.

Deliverables:
- A supervised objective helper over prebuilt batch arrays.
- Input batch contract:
  - `source_ids: int32[B, source_len]`
  - `decoder_input_ids: int32[B, target_len]`
  - `target_ids: int32[B, target_len]`
  - `target_mask: bool[B, target_len]`
  - `example_weight: float32[B]`
- Helper inputs:
  - `logits: float[B, target_len, vocab_size]`
  - `decoder_input_ids`
  - `target_ids`
  - `target_mask`
  - `example_weight`
  - `grammar: FlatDefinitionGrammar`
- Helper output is only:
  - `weighted_nll_sum: float`
  - `weight_sum: float`

Non-goals:
- No weighted mean helper.
- No per-example NLL return.
- No token logp or sequence logp return.
- No decoder-input builder.
- No tokenizer or raw dataset preprocessing.
- No `.npz` loading or writing.
- No optimizer or train step.
- No model invocation wrapper.
- No CLI/checkpoint integration.
- No sampling/reward/REINFORCE.

Acceptance criteria:
- Tests verify weighted NLL sum math.
- Tests verify `weight_sum` equals `sum(example_weight)`.
- Tests verify masked target positions do not affect NLL.
- Tests verify zero total weight returns `weight_sum == 0`; weighted mean is
  caller-owned.
- JIT works.
- Gradients flow from `weighted_nll_sum` to logits.
- Existing scoring, sampling, grammar, and flat seq2seq tests remain green.

Constraints:
- JAX-friendly fixed-shape arrays.
- Caller owns dtype/shape contracts.
- The helper may ignore `source_ids`; it remains part of the batch contract for
  later train-step compatibility.

## Role Boundary

The user is acting as Product Owner / Stakeholder.
Low-level implementation decisions are agent-owned.
User-provided implementation details are binding only when they express product,
compatibility, safety, performance, or operational constraints.

## Agent-Owned Notes

Assumptions:
- The objective helper belongs above `scoring.py`, because `scoring.py` already
  owns constrained token/sequence log-probability.
- The helper should call `constrained_sequence_log_prob` internally and expose
  only accumulation-safe totals.

Risk level: low

Story size: small

Escalation: none

Implementation hints:
- Compute `example_nll = -constrained_sequence_log_prob(...)` internally.
- Compute `weighted_nll_sum = jnp.sum(example_weight * example_nll)`.
- Compute `weight_sum = jnp.sum(example_weight)`.
- Return a plain 2-tuple.

Follow-up stories:
- Offline preprocessing into this exact batch contract.
- NNX/Optax train step using these additive totals.

## Definition of Done

- Acceptance criteria met
- Deliverables completed
- Non-goals avoided
- Verification run
- No unjustified scope, dependency, abstraction, public API, or broad refactor
  growth
