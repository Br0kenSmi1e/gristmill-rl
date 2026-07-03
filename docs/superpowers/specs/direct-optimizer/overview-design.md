# Direct Optimizer Overview Design

## Purpose And Scope

This project adds a self-contained direct optimizer under
`python/gristmill_symbolics/direct_optimizer/`.

The new path trains a conditional proposal model:

```text
p_theta(y_definitions | x)
```

where `x` is an initial `TensorComputation`, `y` is an optimized equivalent
`TensorComputation`, and the cost signal is `y.log_total_flops()`.

Training examples come from verified `(x, y, outputs)` pairs. For each fixed
`x` group, lower-cost `y` candidates receive larger cost-softmax weights, so
weighted maximum likelihood pushes probability mass toward cheaper equivalent
computations.

The direct optimizer does not use the existing action-selector model, rewrite
trajectory rollout, or REINFORCE trainer. It reuses only stable symbolic pieces:
`TensorComputation` and `equivalent_computations`.

Non-goals for the first implementation:

- no rewrite-action trajectory modeling;
- no REINFORCE, GFlowNet, or beam search;
- no changes to existing action-selector checkpoints;
- no integration with `gristmill_symbolics.cli.checkpoint`;
- no dependency on `gristmill_symbolics.model.tokenizer`;
- no grammar-constrained decoder in the MVP, though the model and sampler APIs
  should leave room for one later.

## Package Boundary And Module Split

The direct optimizer lives under:

```text
python/gristmill_symbolics/direct_optimizer/
```

It is self-contained relative to the existing training stack. It may import:

```text
gristmill_symbolics.TensorComputation
gristmill_symbolics.equivalent_computations
jax / flax / optax / numpy / json
```

It should not import or depend on:

```text
gristmill_symbolics.model.tokenizer
gristmill_symbolics.model.transformer_action_selector
gristmill_symbolics.trainer.reinforce
gristmill_symbolics.cli.train_state
gristmill_symbolics.cli.checkpoint
```

Conceptual modules:

- Converter: symbolic computation to and from custom DSL text.
- Dataset utilities: raw candidates to verified weighted processed rows.
- Model: conditional encoder-decoder Transformer over DSL token ids.
- Trainer: weighted supervised maximum-likelihood loop and direct checkpointing.
- Sampler: candidate generation plus symbolic parse, verify, and cost gate.

Cross-module dependency rule:

```text
converter: may use TensorComputation and JSON
dataset: may use converter + verifier + flops
model: DSL text/token ids only
trainer: dataset rows + model only
sampler: converter + model + verifier + flops
```

Representation commitment:

```text
source DSL = full input computation x
target DSL = all definitions of y only
```

The DSL is owned by `direct_optimizer`, not by the existing action-selector
tokenizer.

## End-To-End Data Flow

Training data starts as raw candidates:

```text
(x, y, outputs, optional costs, metadata)
```

Processing flow:

```text
raw JSONL
  -> parse x/y as TensorComputation
  -> verify equivalent_computations(x, y, outputs)
  -> compute missing costs
  -> convert x to source DSL
  -> convert y definitions to target DSL
  -> group by canonical x + outputs
  -> dedupe repeated target DSL per group
  -> compute cost-softmax weights
  -> processed JSONL
```

Training flow:

```text
processed JSONL
  -> build/load DSL vocabulary
  -> encode source/target token ids
  -> weighted teacher-forced MLE
  -> direct optimizer checkpoint
```

Inference flow:

```text
x + outputs + checkpoint
  -> source DSL
  -> sample many target DSL candidates
  -> reconstruct y from x + target DSL
  -> reject parse/reconstruction failures
  -> reject non-equivalent candidates
  -> compute valid candidate costs
  -> return lowest-cost valid y
```

The correctness gate is always symbolic:

```text
TensorComputation reconstruction + equivalent_computations
```

The model never emits a trusted optimized computation directly. It emits
proposals.

## Core Invariants

The direct optimizer must preserve these invariants:

- Existing action-selector, REINFORCE, and CLI checkpoint code remain untouched
  unless a test failure reveals an unavoidable import/package issue.
- The model learns only over the direct optimizer DSL. It does not parse JSON,
  compute flops, or call the verifier.
- Dataset processing rejects examples that fail parsing, Rust validation, output
  validation, or equivalence verification.
- Processed examples are grouped by canonical `(x, outputs)`, not just raw JSON
  text.
- Duplicate target definitions within a group are collapsed before weighting.
- Cost weights are normalized per group and sum to `1.0` within numerical
  tolerance.
- Lower final log flops produce higher weight when `beta > 0`.
- Inference never returns an unverified candidate.
- Invalid generated DSL is counted as a parse/reconstruction failure, not a
  process crash.
- Checkpoints for this path are direct-optimizer checkpoints only and are not
  registered in `gristmill_symbolics.cli.checkpoint`.

Future extension invariant:

- The model sampling API should allow an optional grammar/logit mask hook later,
  but the MVP does not implement grammar-constrained decoding.

## DSL Strategy

The direct optimizer uses a custom deterministic DSL, not JSON and not the
existing action-selector tokenizer.

The DSL has two views:

```text
source DSL: full TensorComputation x
target DSL: definitions-only y
```

Design goals:

- deterministic output for the same snapshot;
- exact round-trip for valid DSL;
- whitespace-tokenizable for the MVP model;
- structured enough to support future grammar masking;
- readable enough for debugging processed datasets;
- strict parser errors via `ValueError`.

A representative MVP shape is line-oriented, with atomic whitespace tokens:

```text
range id=0 size=8
tensor id=0
def base=3
ext id=0 range=0
term
coeff numer=1 denom=1
sum id=2 range=0
factor tensor=0 indices=0,2
endterm
enddef
```

Target DSL contains only the `def ... enddef` blocks.

The detailed converter spec will define the exact grammar, ordering rules,
escaping rules if needed, parser behavior, and reconstruction semantics.

## Model And Future Grammar Masking

The model is an encoder-decoder Transformer over DSL token ids.

Training factorization:

```text
p_theta(y_definitions | x) = product_t p_theta(y_t | y_<t, x)
```

The encoder reads source DSL for `x`. The decoder autoregressively predicts
target DSL tokens for all definitions of `y`.

The model owns:

- vocabulary;
- source/target encoding;
- padding masks;
- teacher-forced next-token loss/log-prob;
- autoregressive sampling.

The model does not own:

- DSL parsing;
- `TensorComputation` reconstruction;
- symbolic equivalence;
- cost computation;
- dataset grouping or weights.

Future grammar masking should fit as a sampling hook:

```text
masked_logits = apply_valid_token_mask(logits, generation_state, x_context)
```

The MVP model should expose sampling in a way that can accept an optional mask
provider later, even when that provider is initially `None`.

## Test And Verification Strategy

Focused tests should cover module contracts rather than only end-to-end behavior.

Required coverage:

- Converter round-trips full source DSL and definitions-only target DSL.
- Reconstruction copies ranges/tensors from `x` and registers new generated
  definition bases with empty symmetry.
- Invalid DSL raises `ValueError`.
- Dataset processing filters non-equivalent candidates.
- Dataset grouping and deduplication are based on canonical DSL, not raw JSON
  spelling.
- Cost-softmax weights sum to one per group and prefer lower cost.
- Model loss accepts padded batches and ignores padding.
- A tiny training smoke test shows finite loss and at least one improving
  training metric.
- Sampler rejects invalid DSL and non-equivalent reconstructions.
- Sampler returns the lowest-cost verified candidate among valid proposals.
- Direct optimizer checkpoints round-trip without touching
  `gristmill_symbolics.cli.checkpoint`.

Focused verification before completion should include:

```bash
uv run pytest python/tests/test_model_trainer_cli_layout.py -q
uv run pytest python/tests/direct_optimizer -q
```

