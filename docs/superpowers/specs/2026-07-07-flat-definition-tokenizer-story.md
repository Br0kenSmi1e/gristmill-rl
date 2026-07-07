# Story Spec: Flat Definition Tokenizer

## Product Owner Inputs

Outcome:

Build the first Python ML rebuild piece: a definition-level tokenizer that
converts a TensorDef snapshot dict into a flat, fixed-vocabulary integer token
sequence and can decode valid raw token streams back into equivalent definition
snapshot dicts.

Deliverables:
- Public tokenizer object for definition-level encode/decode.
- Fixed integer vocabulary containing pad, def_start, def_end, rangeid*,
  tensorid*, indexid*, coeff_num*, and coeff_den* tokens.
- Debug token-name lookup for inspecting integer IDs.
- Raw round-trip: definition snapshot dict -> list[int] -> equivalent
  definition snapshot dict.
- Focused tests for valid round trips, vocabulary lookup, raw-only API, and
  tokenizer-owned rejection cases.

Non-goals:
- TensorComputation-level round-trip.
- Action-space or candidate-pair tokenization.
- State tokenization across all definitions.
- Padded encode/decode APIs, masks, or NumPy array conversion.
- Full TensorDef snapshot schema validation; CLI or data-loading code should
  own user-facing schema validation when that layer exists.
- JAX arrays, jax.jit, vmap, tracing, or model compilation.
- Model, trainer, CLI, checkpoint, batching, or training-loop work.
- Restoring the old columnar tokenizer or structure-heavy role tokens.

Acceptance criteria:
- A valid definition snapshot dict round-trips through raw integer tokens
  without changing semantic fields.
- Tokenization follows the flat positional grammar:
  - def_start begins a definition and def_end ends it.
  - first tensorid after def_start is the base tensor.
  - indexid/rangeid pairs before the first coefficient are external indices.
  - coeff_num/coeff_den starts a term.
  - indexid/rangeid pairs after a coefficient are sum indices.
  - tensorid inside a term starts a factor.
  - following indexid tokens are factor arguments until the next tensorid,
    coeff_num, or def_end.
- The reserved pad token remains token id 0 for future batching, but raw decode
  rejects pad tokens and the tokenizer exposes no padded encode/decode methods.
- Unsupported tensor/range/index IDs and unsupported coefficient
  numerator/denominator values are rejected with clear errors.
- Malformed raw token streams are rejected with clear errors.
- Encode accepts snapshot-like mappings and iterables when the fields it needs
  are present; it does not police extra keys, exact concrete container types,
  or the full TensorDef snapshot schema.
- Tests run on refactor/python-ml-rebuild without adding model/trainer
  dependencies.

Constraints:
- Primary ML-facing representation is integer IDs, not string tokens.
- Tokenizer raw encode/decode must use plain Python sequence APIs, with no JAX
  or NumPy dependency in this layer.
- Public API should be a tokenizer class because vocabulary bounds and
  coefficient sets are tokenizer configuration.
- The tokenizer is not a general input validator; schema-level checks belong in
  CLI or data-loading code.
- Input/output definition shape matches entries from
  TensorComputation.snapshot()["definitions"].

## Role Boundary

The user is acting as Product Owner / Stakeholder.
Low-level implementation decisions are agent-owned.
User-provided implementation details are binding only when they express product,
compatibility, safety, performance, or operational constraints.

## Agent-Owned Notes

Assumptions:
- The tokenizer constructor will own vocabulary limits such as max_range_id,
  max_tensor_id, max_index_id, supported coeff_nums, and supported coeff_dens.
- Error type choice and exact module/file layout are implementation decisions.
- Equivalent definition snapshot means the same normalized dict shape and values
  expected by current Python snapshot tests.

Risk level: medium

Story size: medium

Escalation: none

Implementation hints:
- Public shape likely resembles:
  - FlatDefinitionTokenizer(...)
  - encode_definition(defn) -> list[int]
  - decode_definition(ids) -> dict
  - pad_token_id -> int
  - token_name(token_id) -> str
- Keep the raw token-stream parser strict and explicit so invalid token order
  fails early.
- Keep encode direct and minimal; it should fail only when values cannot be
  mapped into the configured vocabulary or normal field access fails.
- Keep batching and padding outside this tokenizer until a later integration
  story needs them.

Follow-up stories:
- TensorComputation state tokenization by concatenating encoded definitions.
- Action-space candidate tokenization for flattened left/right definition pairs.
- Batch collation and model-facing integration.
- Trainer/model/CLI rebuild on top of the tokenizer.

## Definition of Done

- Acceptance criteria met.
- Deliverables completed.
- Non-goals avoided.
- Focused verification run.
- No unjustified scope, dependency, abstraction, public API, or broad refactor
  growth.
