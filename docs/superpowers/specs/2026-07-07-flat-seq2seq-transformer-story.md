# Story Spec: Flat Seq2Seq Transformer Wrapper

## Product Owner Inputs

Outcome:

Build a flat-token seq2seq Transformer wrapper so padded source token IDs and
decoder input token IDs can produce vocabulary logits using the stable reusable
Transformer core.

Deliverables:
- `FlatDefinitionSeq2SeqTransformer` in a new file separate from
  `nn/transformer.py`.
- Reuse existing `TransformerEncoder` and `TransformerDecoder`.
- Inputs:
  - `source_ids: int32[B, source_len]`
  - `decoder_input_ids: int32[B, target_len]`
- Output:
  - `logits: float[B, target_len, vocab_size]`
- Include:
  - shared token embedding
  - separate source and target positional embeddings
  - embedding dropout
  - output projection to `vocab_size`
  - padding masks derived from `pad_token_id`
  - zeroed vectors at pad-token positions before entering the core
- Export from `gristmill_symbolics.nn`.

Constraints:
- Keep `nn/transformer.py` stable; do not edit it unless a bug is found.
- No grammar masking inside the model.
- No loss function.
- No sampling loop.
- No trainer, CLI, checkpoint, dataset, or `ExpressionModel` protocol
  integration.
- No tokenizer object dependency in the model constructor; pass scalar values
  like `vocab_size` and `pad_token_id`.

Acceptance criteria:
- Model constructs with `nnx.Rngs`.
- Forward pass returns `[B, target_len, vocab_size]`.
- Source and target padding masks are derived from token IDs and passed to the
  core.
- Pad-token positions are zeroed after token plus position embedding.
- Deterministic mode is stable with dropout disabled.
- Tests verify the wrapper is separate from, and does not require changes to,
  the reusable Transformer core.

## Role Boundary

The user is acting as Product Owner / Stakeholder.
Low-level implementation decisions are agent-owned.
User-provided implementation details are binding only when they express product,
compatibility, safety, performance, or operational constraints.

## Agent-Owned Notes

Assumptions:
- The wrapper belongs under `gristmill_symbolics.nn` because it is an NNX neural
  module returning logits, not a full repo `ExpressionModel`.
- A small focused `flat_seq2seq.py` module is enough for this story.
- The wrapper should pass dtype, param dtype, dropout, and attention
  implementation through to the core to keep model construction consistent.

Risk level: medium

Story size: small

Escalation: none

Implementation hints:
- Use `nnx.Embed` for token and positional embeddings.
- Use `nnx.Linear` for the vocabulary output projection.
- Keep padding behavior local to the wrapper by constructing masks from token
  IDs and zeroing embedded pad positions.
- Avoid tokenizer imports.

Follow-up stories:
- Decoder-input/label/mask and token log-probability helpers.
- Grammar-masked supervised loss.
- Sampling loop carrying grammar state.
- Dataset, trainer, CLI, and checkpoint integration.

## Definition of Done

- Acceptance criteria met
- Deliverables completed
- Non-goals avoided
- Verification run
- No unjustified scope, dependency, abstraction, public API, or broad refactor
  growth
