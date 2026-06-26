# REINFORCE Policy Refactor Suite Overview Design

Status: planned
Depends on:
- `2026-06-17-streamed-reinforce-policy-refactor-design.md`
- `2026-06-24-static-rollout-shapes-design.md`
- `2026-06-25-reusable-batched-policy-jit-api-design.md`
Feeds implementation plan: no, child specs feed implementation plans

## Summary

This overview defines the ownership boundaries for the next REINFORCE policy
refactor suite. The suite separates three concerns that are currently tangled in
the rollout and policy pipeline:

```text
tokenizer / padding / shape contract
expression proposal wrapper
training rollout
```

The overview is intentionally not an implementation design. It defines which
subsystem owns each problem, which dependencies may cross subsystem boundaries,
and which decisions must be left to the child specs.

## Motivation

The current profiling evidence shows that the bottleneck has moved away from
Rust/PyO3 symbolic rewrite logic and into the policy pipeline. The largest costs
are action-side model work, especially `sample_action`, plus
tokenization/padding/stacking overhead. An earlier batch-8 OOM exposed a large
attention-like intermediate, and current policy action dimensions accidentally
derive candidate and side-term widths from action token sequence length.

Those problems touch multiple modules, but they should not be fixed by one large
rewrite. The refactor needs a suite of smaller specs so each subsystem has a
clear owner and each performance change can be verified at the boundary it
actually changes.

## Primary Subsystems

### Tokenizer / Padding / Shape Contract

Owns representation preparation for policy calls.

This subsystem converts state and action-space snapshots into token arrays,
records observed token and structural lengths, applies configured padding, and
builds target/action policy batches. It also owns the explicit shape contract
for state tokens, action tokens, definition masks, candidate slots, and side-term
slots.

It must not own rollout control, target/action semantic decisions, exact-empty
replay, reward calculation, gradient accumulation, optimizer state, or Rust
rewrite validation/application.

It provides batch objects and metadata to the other subsystems. Those objects
must make real rows, dummy rows, configured padded lengths, and observed lengths
explicit.

### Expression Proposal Wrapper

Owns the expression-in/expression-out inference facade.

This subsystem presents the current policy as a proposal model:

```text
initial expression -> sampled rewrite trace -> validated symbolic apply -> final expression
```

It is a bridge toward a later proposal-model design, not a true seq2seq model.
Internally it may still use the current target/action policy rollout. Externally
it should expose proposal records that include final expression, cost, rewrite
trace, log probability, validity, and stop reason.

It must not own training loss, streamed gradient accumulation, optimizer state,
low-level token padding mechanics, or new symbolic rewrite semantics.

It uses tokenizer/padding batch objects when proposal generation is batched. It
uses Rust validation and application as the only authority for symbolic rewrite
legality and mutation.

### Training Rollout

Owns REINFORCE training semantics.

This subsystem controls active sample lifecycle, target sampling and scoring,
action-space querying, action sampling and scoring, exact-empty replay,
trajectory logp and gradient accumulation, and rollout metrics. It owns the
meaning of `stop_count`, `empty_action_space_count`, `target_score_count`,
`action_score_count`, and related training metrics.

It must not own token serialization details, static padding mechanics, public
expression proposal API design, model attention internals, or Rust rewrite
semantics.

It consumes tokenizer/padding batch builders and policy batched sample/score
APIs. It calls Rust row APIs for action-space generation, validation, and
rewrite application.

## Ownership Matrix

| Concern | Owner |
| --- | --- |
| Snapshot-to-token conversion | Tokenizer / Padding |
| Host or device padding mechanics | Tokenizer / Padding |
| Configured padded lengths | Tokenizer / Padding |
| Observed length metadata | Tokenizer / Padding |
| Dummy policy rows | Tokenizer / Padding |
| Candidate and side-term shape axes | Tokenizer / Padding with policy API contract |
| Policy target/action probability semantics | Policy API, consumed by wrapper and rollout |
| Expression-level proposal result schema | Expression Proposal Wrapper |
| Rewrite trace presentation | Expression Proposal Wrapper |
| Active/stopped sample lifecycle | Training Rollout |
| Exact-empty replay | Training Rollout |
| REINFORCE logp and gradient accumulation | Training Rollout |
| Reward, advantage, and optimizer integration | Training Rollout |
| Rust action-space generation | Rust rewrite environment |
| Rust decision validation and application | Rust rewrite environment |
| Attention implementation details | Policy model internals |

## Boundary Contracts

### Tokenizer/Padding To Policy Callers

The tokenizer/padding subsystem provides immutable policy batch inputs and shape
metadata. Callers may read masks and metadata, but must not infer semantic
rollout state from padding values alone. Real/dummy row status must be explicit.

### Policy API To Rollout And Wrapper

Policy sampling and scoring APIs own probability semantics for target and action
choices. Rollout and wrapper code may decide when to call those APIs, but should
not duplicate candidate, side-term, or log-probability calculations.

### Rollout To Rust

The training rollout may query action spaces and apply validated actions through
Rust row APIs. It must pass exact symbolic decisions derived from policy choices
and valid masks. Rust remains the authority for legality and mutation.

### Wrapper To Rust

The proposal wrapper may use scalar or row Rust APIs to apply sampled rewrite
traces. It must report invalid proposals rather than bypassing Rust validation.

### Model Internals To Callers

Model implementation details, including explicit or fused attention, must remain
behind the policy API. Neither rollout nor tokenizer/padding should branch on the
attention backend.

## Dependency Direction

The intended dependency direction is:

```text
Rust rewrite environment
        |
        | snapshots, action spaces, validation, application
        v
tokenizer / padding ----> policy API/model internals
        |                         |
        | policy batches           | sampling/scoring
        v                         v
training rollout          expression proposal wrapper
```

Tokenizer/padding is shared infrastructure. Training rollout and expression
proposal wrapper both depend on it, but tokenizer/padding must not depend on
either of them.

The expression proposal wrapper and training rollout may share small trace or
choice conversion helpers only if those helpers are policy-level data utilities,
not rollout-control utilities.

## Child Specs

### Tokenizer / Padding / Shape Contract Spec

This child spec defines the explicit shape contract, batch objects, batch builder
strategy boundaries, configured-vs-observed length metadata, dummy row
representation, and tokenizer/padding performance acceptance criteria.

It must not define rollout lifecycle behavior or expression proposal result
semantics.

### Expression Proposal Wrapper Spec

This child spec defines the expression-level proposal API, proposal result data
model, trace representation, validity and stop-reason vocabulary, and how the
wrapper uses current target/action policy semantics without becoming a true
seq2seq model.

It must not define training loss behavior or low-level padding mechanics.

### Training Rollout Refactor Spec

This child spec defines how rollout semantics consume target/action batch
objects, where scoring and gradient accumulation occur, how exact-empty replay
and dummy rows are handled, and which metrics must remain unchanged.

It must not define token serialization, expression-level public inference API, or
model attention backend behavior.

## Cross-Cutting Performance Discipline

Performance work belongs to the subsystem whose boundary it changes:

- tokenization, padding, stacking, and transfer overhead belong to
  tokenizer/padding;
- candidate and side-term width inflation belongs to the tokenizer/padding and
  policy API shape contract;
- repeated policy compile churn belongs to batch shape strategy and policy API
  call shapes;
- full-attention compute and memory belong to model internals;
- Rust row work is not the current primary bottleneck and should not be changed
  for performance without new evidence.

Every child spec that proposes a performance change must state the profiling
evidence that motivates it and the measurements that will verify it.

## Shared Invariants

- Rust remains authoritative for rewrite legality, action-space generation,
  rewrite application, and cost evaluation.
- Symbolic rewrite semantics do not change.
- Tokenizer changes preserve token semantic meaning.
- Padding and dummy rows are policy-call mechanics, not rollout semantics.
- Candidate and side-term dimensions are explicit shape contract dimensions, not
  accidental consequences of action token sequence length.
- Training metrics keep their current meanings unless a child spec explicitly
  names and justifies a breaking change.
- The expression proposal wrapper is not a true seq2seq model in this suite.
- Model backend changes must preserve policy API behavior within expected
  floating-point tolerance.
- Each implementation checkpoint must pass deterministic correctness checks
  before later performance work builds on it.

## Out Of Scope For This Overview

- Exact dataclass fields and function signatures.
- File-level move plans.
- CLI flag names and checkpoint schema changes.
- Test case code.
- Attention backend selection.
- A true seq2seq proposal model.
- Rust rewrite-kernel redesign.
- Implementation task ordering inside a child spec.

## Acceptance Criteria

This overview is accepted when it gives enough boundary information for the
three child specs to be written independently without assigning the same
responsibility to multiple subsystems.

The suite is ready for implementation planning only after the child specs are
written, reviewed, and checked against this overview's dependency direction and
shared invariants.
