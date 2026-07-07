# Story Spec: Tokenizer Generated Sequence Boundaries

## Product Owner Inputs

Outcome:

Make the flat definition tokenizer generation-ready by adding global BOS/EOS
tokens and replacing padded raw-sequence decode with a generated-sequence decode
contract.

Deliverables:
- Add `bos` and `eos` tokens to `FlatDefinitionTokenizer`.
- Expose `bos_token_id` and `eos_token_id`.
- Fix the first token IDs as `pad=0`, `bos=1`, `eos=2`, `def_start=3`,
  and `def_end=4`.
- Keep encode APIs content-only: definitions still encode as
  `def_start ... def_end` without BOS/EOS.
- Keep `encode_definitions_padded` for right-padding definition content.
- Remove `decode_definitions_padded`.
- Add `decode_definitions_generated(ids)` for generated streams shaped as
  `bos content eos pad...`.
- Update focused tokenizer tests and docs.

Non-goals:
- Decoder-input, label, or mask helpers.
- JAX or NumPy helpers.
- Transformer model, dataset collation, sampling loop, grammar mask, trainer,
  CLI, or checkpoint work.
- TensorComputation-level encode/decode.
- Preserving integer compatibility with already-generated tokenizer streams.

Acceptance criteria:
- `token_name(0..4)` returns `pad`, `bos`, `eos`, `def_start`, `def_end`.
- `bos_token_id == 1` and `eos_token_id == 2`.
- Encoded definitions begin with `def_start` and end with `def_end`; they do
  not include BOS/EOS.
- Raw `decode_definition` and `decode_definitions` reject BOS/EOS through the
  existing grammar.
- `decode_definitions_generated` requires a leading BOS.
- `decode_definitions_generated` reads content until the first EOS, allows only
  PAD after EOS, and rejects missing EOS, PAD before EOS, nested BOS in content,
  and non-PAD tokens after EOS.
- Empty generated content, `bos eos pad...`, decodes as `[]`.
- Padded encode still uses only PAD for right padding.
- `decode_definitions_padded` is no longer public.

Constraints:
- Tokenizer encode/decode remains plain Python sequence code; no JAX or NumPy in
  this layer.
- The vocabulary reset is intentional; old integer token streams from this
  branch must be regenerated.
- Keep the change scoped to tokenizer behavior and tests.

## Role Boundary

The user is acting as Product Owner / Stakeholder.
Low-level implementation decisions are agent-owned.
User-provided implementation details are binding only when they express product,
compatibility, safety, performance, or operational constraints.

## Agent-Owned Notes

Assumptions:
- `encode_definitions_padded` remains useful as a content-padding helper for
  fixed source/target content arrays.
- Generated decode returns normalized definition snapshot dicts by delegating to
  the existing raw sequence decoder after stripping BOS/EOS/PAD framing.
- A generated stream with empty content is valid because a model may choose EOS
  immediately.

Risk level: low

Story size: small

Escalation: none

Implementation hints:
- Insert BOS/EOS during vocabulary construction immediately after PAD.
- Add direct `bos_token_id` and `eos_token_id` properties parallel to
  `pad_token_id`.
- Implement generated decode as a framing validator that delegates valid content
  to `decode_definitions`.

Follow-up stories:
- Decoder-input, target-label, and mask helpers.
- Forward-only flat seq2seq Transformer.
- Dataset collation and generated-token sampling.

## Definition of Done

- Acceptance criteria met
- Deliverables completed
- Non-goals avoided
- Verification run
- No unjustified scope, dependency, abstraction, public API, or broad refactor
  growth
