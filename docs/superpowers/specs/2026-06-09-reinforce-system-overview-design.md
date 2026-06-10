# REINFORCE System Overview Design

Status: planned
Supersedes: the high-level architecture parts of the 2026-06-03 and 2026-06-05
REINFORCE prototype specs
Depends on: `2026-06-02-rewrite-state-api-design.md`
Feeds implementation plan: yes

## Summary

This spec defines the implementation roadmap for a runnable REINFORCE training
system over the symbolic rewrite kernel.

The system trains an attention-based policy that reads tokenized tensor
expressions, samples semantic rewrite choices, executes those choices through the
Rust kernel, stores immutable rollout data, recomputes differentiable log
probabilities, and applies an on-policy REINFORCE update.

The detailed design is split across focused contract specs:

```text
system overview       -> module boundaries and dependency order
policy model          -> tokenization, attention, logits, sampling, scoring
scalar step           -> authoritative one-sample rollout semantics
row table             -> rectangular rollout storage and score masks
parallel row wrapper  -> row execution and row scoring mechanics
training              -> rewards, advantages, loss, optimizer, metrics
```

Each detailed spec should be precise enough to generate implementation tasks and
tests without depending on undocumented design intent.

## Goals

- Build a runnable policy-gradient training path over the current symbolic
  kernel.
- Keep Rust authoritative for rewrite legality, exact action-space generation,
  rewrite application, and cost evaluation.
- Make the policy architecture explicit enough to implement and test.
- Keep scalar semantics, row storage, row parallelism, and training objective in
  separate specs with minimal overlap.
- Preserve stable sample-column alignment across parallel rollout.
- Store immutable rollout inputs and sampled choices, then recompute
  differentiable logp for training.
- Define enough invariants and acceptance criteria for later implementation
  plans and code reviews.

## Non-Goals

- Implementing AlphaZero, MCTS, replay, a value head, or off-policy learning.
- Differentiating through Rust rewrite application, action-space generation, or
  cost evaluation.
- Matching Gristmill's optimizer exactly in the first runnable version.
- Solving production-scale memory planning beyond explicit scoring chunks.
- Preserving the previous token-decoder prototype API as a public compatibility
  surface.

## Spec Suite

### Policy Model

File: `2026-06-09-reinforce-policy-model-design.md`

Owns:

- symbolic tensor tokenization;
- state and selected-action-space attention context;
- target logits over `STOP + def_index`;
- candidate logits;
- left/right bit-sequence mask decoding;
- sampling and differentiable rescoring contracts.

Does not own rollout control, reward computation, optimizer updates, or row
parallel execution.

### Scalar Step

File: `2026-06-09-reinforce-scalar-step-design.md`

Owns:

- one-sample step semantics;
- STOP, empty action space, valid action, and already-finished cases;
- target/action score-mask meaning for width 1;
- exact ordering of policy calls and Rust environment calls.

Does not define the neural architecture or training optimizer.

### Row Table

File: `2026-06-09-reinforce-row-table-overview-design.md`

Owns:

- sample, row, and column vocabulary;
- rectangular rollout storage;
- target/action tables, choices, masks, and padded values;
- row-to-column reward assignment.

Does not define private active-sample scheduling.

### Parallel Row Wrapper

File: `2026-06-09-reinforce-parallel-row-wrapper-design.md`

Owns:

- lifting scalar semantics to whole rows;
- active-sample filtering and result scatter as private mechanics;
- row scoring and scoring chunks;
- row-level invariants.

Does not change scalar behavior or model probability semantics.

### Training

File: `2026-06-09-reinforce-training-design.md`

Owns:

- reward and advantage calculation;
- STOP training modes;
- loss normalization;
- optimizer update unit;
- checkpoint and metric requirements.

Does not own model internals or Rust rewrite logic.

## Public Vocabulary

The public rollout vocabulary is intentionally small:

- A sample is one rollout instance.
- A row is all samples at the same rollout step.
- A column is one sample through time.

Additional implementation terms may appear inside a focused spec, but public
rollout APIs should not expose extra table concepts unless a later design updates
the vocabulary.

## Module Boundaries

The policy owns:

- constructing immutable target/action arrays;
- tokenizing those snapshots;
- sampling target/action choices;
- scoring stored target/action choices with differentiable logp;
- reporting policy diagnostics.

The rewrite environment owns:

- `RewriteState`;
- definition masks and lazy exact refinement;
- exact action-space generation for one selected definition;
- rewrite application;
- final or intermediate cost evaluation.

The trainer owns:

- rollout control over rows and columns;
- reward and advantage assignment per sample column;
- batching and scoring stored rows;
- REINFORCE loss assembly and normalization;
- optimizer updates, checkpoints, and training metrics.

The row wrapper owns:

- applying scalar semantics to all sample positions in one row;
- preserving row width and column alignment;
- hiding active-sample filtering, compaction, and scatter mechanics.

## End-To-End Data Flow

One optimizer update follows this flow:

```text
initial computations
  -> initialize one RewriteState per sample
  -> collect rows until all samples finish or max_steps is reached
  -> store immutable target/action row data and masks
  -> compute reward per sample column
  -> compute advantages over the logical batch
  -> score stored target/action choices in bounded chunks
  -> assemble REINFORCE loss
  -> apply one optimizer update
  -> discard rollout rows unless checkpoint/debug config says otherwise
```

Rollout may compute sampled logp for diagnostics, but sampled rollout logp is not
the gradient source. Training recomputes logp from stored arrays and choices.

## Implementation Order

The recommended implementation order is:

1. Policy data contracts and deterministic tokenization.
2. Scalar policy sampling and scoring for one sample.
3. Scalar rollout step using the real `RewriteState`.
4. Width-1 REINFORCE loss with deterministic fake advantages.
5. Rectangular row storage.
6. Multi-sample row stepping with scalar-equivalence tests.
7. Row scoring and score chunks.
8. Full training update with reward, baseline, optimizer, metrics, and
   checkpointing.

This order makes semantic bugs visible before adding parallel mechanics.

## Cross-Spec Invariants

- Target selection must not generate or inspect action spaces for unselected
  definitions.
- target/action arrays stored for training are immutable snapshots.
- Every scored choice has a score mask set to true.
- Every masked-out choice has safe padded values and contributes nothing to loss
  or metrics.
- Row width is stable from the first row to the last row.
- Sample position is the only column identity used for reward assignment.
- A sampled action can be scored later without holding live PyO3 `ActionSpace` or
  `RewriteState` handles.
- `STOP`, empty action space, valid action, and already-finished cases have the
  same mask semantics in scalar and row execution.
- Baseline and advantage values are treated as stop-gradient constants.

## Error Handling Strategy

Implementation should fail early for contract violations:

- illegal stored target choice during scoring;
- stored action choice whose candidate or bit sequence is invalid for the stored
  action arrays;
- missing immutable input data for a true score mask;
- non-safe padded values causing scorer indexing errors;
- row width changes across rollout;
- reward or advantage arrays whose length differs from row width.

These should be explicit exceptions or test failures, not silent dropped samples.

## Testing Requirements

The implementation plan should include tests at three levels:

- Contract tests for tokenization, masks, shapes, and scalar step cases.
- Equivalence tests showing row stepping matches independent scalar stepping.
- End-to-end smoke tests proving a tiny training update recomputes logp,
  produces finite loss, changes parameters for nonzero advantage, and writes a
  checkpoint.

Tests should include exact-empty action-space refinement, STOP, valid rewrite,
and already-finished samples.

## Acceptance Criteria

- The spec suite has one owner for model architecture, scalar semantics, row
  storage, row parallelism, and training objective.
- The implementation plan can be generated by reading the suite without relying
  on the deprecated prototype APIs.
- A width-1 rollout can train through recomputed target/action logp.
- A multi-sample row rollout preserves scalar semantics and sample alignment.
- The training loop can run a tiny end-to-end update over the real symbolic
  kernel.
