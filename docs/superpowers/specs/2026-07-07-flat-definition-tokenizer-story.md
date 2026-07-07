# Story Spec: Flat Definition Tokenizer

## Product Owner Inputs

Outcome:

Build the first Python ML rebuild piece: a definition-level tokenizer that
converts a TensorDef snapshot dict into a flat, fixed-vocabulary integer token
sequence and can decode valid raw token streams back into equivalent definition
snapshot dicts.

Deliverables:
- Public tokenizer object for definition-level encode/decode.
- Fixed integer vocabulary containing pad, bos, eos, def_start, def_end,
  rangeid*, tensorid*, indexid*, coeff_num*, and coeff_den* tokens.
- Debug token-name lookup for inspecting integer IDs.
- Raw round-trip: definition snapshot dict -> list[int] -> equivalent
  definition snapshot dict.
- Raw sequence round-trip: sequence of definition snapshot dicts ->
  concatenated list[int] -> equivalent sequence of definition snapshot dicts.
- Thin padded sequence wrapper that right-pads concatenated definition tokens
  to a requested length.
- Generated sequence decode for `bos content eos pad...` model-output streams.
- Focused tests for valid round trips, vocabulary lookup,
  single-definition raw-only API, and tokenizer-owned rejection cases.

Non-goals:
- TensorComputation-level round-trip.
- Action-space or candidate-pair tokenization.
- TensorComputation state wrapper or model-facing batching around definition
  sequences.
- Padded single-definition APIs, masks, or NumPy array conversion.
- Full TensorDef snapshot schema validation; CLI or data-loading code should
  own user-facing schema validation when that layer exists.
- Constructor argument validation or normalization; configuration objects are
  stored as provided and may fail naturally while building or using the
  vocabulary if invalid.
- JAX arrays, jax.jit, vmap, tracing, or model compilation.
- Model, trainer, CLI, checkpoint, batching, or training-loop work.
- Restoring the old columnar tokenizer or structure-heavy role tokens.

Acceptance criteria:
- A valid definition snapshot dict round-trips through raw integer tokens
  without changing semantic fields.
- A valid sequence of definition snapshot dicts encodes as the concatenation of
  individual definition encodings and decodes by splitting on def_start/def_end.
- Empty definition sequences encode and decode as empty lists.
- A sequence of definition snapshot dicts can be encoded to an exact requested
  length by right-padding the concatenated raw token stream with pad token IDs.
- Padded sequence encode rejects requested lengths shorter than the raw
  concatenated token stream.
- Global sequence tokens have fixed IDs: `pad=0`, `bos=1`, `eos=2`,
  `def_start=3`, and `def_end=4`.
- Raw definition encode remains content-only and emits `def_start ... def_end`,
  not BOS/EOS.
- Generated sequence decode accepts `bos content eos pad...`, returns an empty
  list for `bos eos pad...`, and rejects missing BOS, missing EOS, PAD before
  EOS, nested BOS inside content, or non-PAD tokens after EOS.
- Tokenization follows the flat positional grammar:
  - def_start begins a definition and def_end ends it.
  - first tensorid after def_start is the base tensor.
  - indexid/rangeid pairs before the first coefficient are external indices.
  - coeff_num/coeff_den starts a term.
  - indexid/rangeid pairs after a coefficient are sum indices.
  - tensorid inside a term starts a factor.
  - following indexid tokens are factor arguments until the next tensorid,
    coeff_num, or def_end.
- The reserved pad token remains token id 0 for future batching. Raw decode
  rejects pad tokens; only generated sequence decode consumes padding after EOS.
- Unsupported tensor/range/index IDs and unsupported coefficient
  numerator/denominator values are rejected with clear errors.
- Malformed raw token streams are rejected with clear errors.
- Malformed concatenated definition streams are rejected when input is not a
  token sequence, a token appears before def_start, def_end is missing, pad
  appears, BOS/EOS appears, a token id is unknown, or a sliced definition
  violates the single-definition grammar.
- Encode accepts snapshot-like mappings and iterables when the fields it needs
  are present; it does not police extra keys, exact concrete container types,
  or the full TensorDef snapshot schema.
- The tokenizer constructor preserves its configuration inputs directly.
- Tests run on refactor/python-ml-rebuild without adding model/trainer
  dependencies.

Constraints:
- Primary ML-facing representation is integer IDs, not string tokens.
- Tokenizer raw and padded-wrapper encode/decode must use plain Python
  sequence APIs, with no JAX or NumPy dependency in this layer.
- Public API should be a tokenizer class because vocabulary bounds and
  coefficient sets are tokenizer configuration.
- The tokenizer is not a general input validator; schema-level checks belong in
  CLI or data-loading code.
- The tokenizer constructor should not coerce, deduplicate, or validate
  vocabulary configuration arguments.
- Input/output definition shape matches entries from
  TensorComputation.snapshot()["definitions"].

## Role Boundary

The user is acting as Product Owner / Stakeholder.
Low-level implementation decisions are agent-owned.
User-provided implementation details are binding only when they express product,
compatibility, safety, performance, or operational constraints.

## Agent-Owned Notes

Assumptions:
- The tokenizer constructor will store vocabulary limits such as max_range_id,
  max_tensor_id, max_index_id, supported coeff_nums, and supported coeff_dens
  exactly as passed.
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
  - encode_definitions(defns) -> list[int]
  - decode_definitions(ids) -> list[dict]
  - encode_definitions_padded(defns, length=...) -> list[int]
  - decode_definitions_generated(ids) -> list[dict]
  - pad_token_id -> int
  - bos_token_id -> int
  - eos_token_id -> int
  - token_name(token_id) -> str
- Keep the raw token-stream parser strict and explicit so invalid token order
  fails early.
- Keep encode direct and minimal; it should fail only when values cannot be
  mapped into the configured vocabulary or normal field access fails.
- Keep model-facing batching, masks, and array conversion outside this
  tokenizer until a later integration story needs them.

Follow-up stories:
- TensorComputation state tokenization that passes snapshot definitions through
  the raw sequence API.
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
