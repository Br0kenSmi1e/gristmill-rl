# Story Spec: Shared Next-Token Scoring Path

## Product Owner Inputs

Outcome:

Make sequence scoring and sampling use the same grammar-constrained next-token
probability computation so their behavior cannot drift.

Deliverables:
- A basic helper in `scoring.py` for one-step constrained next-token log-probs.
- Sequence scoring uses that helper across all teacher-forced positions.
- Sampling uses that helper at each autoregressive step before sampling.
- Existing scoring and sampling behavior remains unchanged.

Non-goals:
- No trainer, model, CLI, checkpoint, dataset, reward, or decode integration.
- No new sampling algorithms.
- No new abstraction beyond the shared scoring primitive.
- No changes to grammar semantics.

Acceptance criteria:
- A focused test covers the new one-step scoring helper.
- Existing constrained sequence scoring tests pass.
- Existing flat token sampling tests pass.
- `sampling.py` no longer duplicates grammar-mask plus `log_softmax` logic.
- JIT and gradient behavior covered by existing tests remains green.
- No unrelated refactors or public API growth beyond the one helper.

Constraints:
- Keep the helper JAX-friendly and fixed-shape.
- Preserve caller-owned input contracts; do not add runtime validation.

## Role Boundary

The user is acting as Product Owner / Stakeholder.
Low-level implementation decisions are agent-owned.
User-provided implementation details are binding only when they express product,
compatibility, safety, performance, or operational constraints.

## Agent-Owned Notes

Assumptions:
- The shared primitive should accept raw logits and a valid-next mask, returning
  constrained log-probs over the vocabulary.
- Sampling can use constrained log-probs directly as categorical logits.

Risk level: low

Story size: small

Escalation: none

Implementation hints:
- Move `jnp.where(valid_next, logits, -jnp.inf)` plus safe `log_softmax` into
  the helper.
- Keep label gathering in `constrained_token_log_probs`.
- Keep RNG sampling and EOS/PAD handling in `sampling.py`.

Follow-up stories:
- None required for this refactor.

## Definition of Done

- Acceptance criteria met
- Deliverables completed
- Non-goals avoided
- Verification run
- No unjustified scope, dependency, abstraction, public API, or broad refactor
  growth
