# REINFORCE Implementation Suite Index Design

Status: planned
Feeds implementation plan: yes

## Summary

This index defines the implementation-facing REINFORCE spec suite. The suite is
split into three standalone phase specs that can be handed directly to plan
writers:

```text
phase 1: Rust/PyO3 row environment
phase 2: policy model sampling and scoring
phase 3: REINFORCE training
```

The earlier six `2026-06-09-reinforce-*` specs remain useful design background,
but implementation plans should use the three phase specs below as their primary
inputs. Each phase spec repeats the contracts, deliverables, tests, and exit
criteria needed for that phase so a plan writer does not need to chase the older
split by conceptual ownership.

## Canonical Implementation Specs

### Phase 1: Rust/PyO3 Row Environment

File: `2026-06-10-reinforce-row-env-implementation-design.md`

Refactors scalar rewrite boundaries without semantic changes, then builds thin
Rust/PyO3 row wrappers over those boundaries:

- scalar generate, public decision validation, and apply boundaries in
  `src/rewrite.rs`;
- `RewriteStateRow`;
- `ActionSpaceRow`;
- row action-space query over `RewriteState::action_space_for_def`;
- deterministic row action-space snapshots;
- row action validation over the scalar `validate_decision` boundary before
  mutation;
- row rewrite application over the scalar apply boundary;
- PyO3 conversion from padded row action input to exact scalar `Decision` values;
- Rayon-backed parallelism for row query, validation, and application;
- scalar-equivalence tests with injected choices.

This phase does not depend on the policy model or training loop.

### Phase 2: Policy Model Sampling And Scoring

File: `2026-06-10-reinforce-policy-model-implementation-design.md`

Builds the model-facing data contracts and probability code:

- state and action-space tokenization from snapshots;
- immutable target/action arrays;
- target sampling and scoring;
- action sampling and scoring;
- left/right bit-sequence decoding;
- padding and `jax.vmap` sampling/scoring;
- row-equivalent scalar/vectorized scoring.

This phase can use fixtures or row-environment snapshots. It does not depend on
the trainer.

### Phase 3: REINFORCE Training

File: `2026-06-10-reinforce-training-implementation-design.md`

Builds the runnable on-policy training path:

- row rollout orchestration;
- rectangular rollout table storage;
- terminal reward and advantage calculation;
- recomputed target/action logp;
- column-normalized REINFORCE loss;
- optimizer update;
- metrics and checkpoints;
- tiny end-to-end smoke tests.

This phase integrates the row environment and policy model.

## Background Specs

The following specs are background and source material for the implementation
suite:

- `2026-06-09-reinforce-system-overview-design.md`
- `2026-06-09-reinforce-parallel-row-wrapper-design.md`
- `2026-06-09-reinforce-policy-model-design.md`
- `2026-06-09-reinforce-row-table-overview-design.md`
- `2026-06-09-reinforce-scalar-step-design.md`
- `2026-06-09-reinforce-training-design.md`

If a phase spec and a background spec appear to conflict, use the phase spec for
implementation planning and update the docs before implementation if the conflict
changes behavior.

## Shared Invariants

All three phases must preserve these system invariants:

- Rust remains authoritative for rewrite legality, exact action-space
  generation, rewrite application, and cost evaluation.
- Target selection must not generate action spaces for unselected definitions.
- Sample position is the column identity throughout rollout and training.
- Row width is stable after a rollout begins.
- Stored model inputs and choices are immutable plain data, not live PyO3
  handles.
- Live `ActionSpaceRow` and validated-action handles are runtime-only.
- A sampled target or action can be scored later from stored arrays and choices.
- Row action validation must succeed for the row before rewrite application
  mutates any sample state.
- Masked entries must use safe padded values and contribute no logp, loss, or
  metric totals.
- Rewards, baselines, and advantages are stop-gradient constants.

## Plan Writer Handoff

For a single-phase plan, hand the plan writer only the relevant phase spec and
this index.

For a whole-project plan, hand the plan writer this index and all three phase
specs. The plan should preserve the phase order, but tasks inside a phase may be
parallelized when they do not share mutable code or test fixtures.

Each plan should end with the phase's exit criteria. Later phases should not
assume behavior that was not covered by earlier phase tests.
