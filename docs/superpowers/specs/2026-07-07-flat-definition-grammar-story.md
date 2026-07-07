# Story Spec: Flat Definition Grammar Mask

## Product Owner Inputs

Outcome:

Build a reusable JAX-friendly grammar mask for `FlatDefinitionTokenizer` token
streams, so transformer logits can be constrained to syntactically valid flat
definition sequences without putting grammar logic inside the transformer.

Deliverables:
- Public tokenizer helpers:
  - `token_kind(token_id) -> str`
  - `token_ids_for_kind(kind) -> tuple[int, ...]`
- Standalone `FlatDefinitionGrammar`.
- Fixed-size grammar tables:
  - `category_by_id: int32[vocab_size]`
  - `allowed_by_state: bool[num_states, vocab_size]`
- Grammar APIs:
  - `initial_state(batch_shape) -> int32[...]`
  - `advance_state(state, token_id) -> int32[...]`
  - `valid_next_masks_for_decoder_input(decoder_input_ids) -> bool[B, T, V]`
  - `valid_next_mask_from_prefix(prefix_ids) -> bool[B, V]`
  - `apply_grammar_mask(logits, valid_next) -> logits`
- Focused tests.

Grammar:

```text
bos definition* eos pad*

definition := def_start tensorid ext* term* def_end
ext        := indexid rangeid
term       := coeff_num coeff_den sum_index* factor*
sum_index  := indexid rangeid
factor     := tensorid indexid*
```

After `eos`, only `pad` is valid. Invalid prefixes enter an error state with no
valid next tokens.

Teacher-forcing alignment:

For fixed padded training tensors:

```text
decoder_input_ids: bos       def_start tensorid0 def_end
target_ids:        def_start tensorid0  def_end   eos
```

`valid_next_masks_for_decoder_input(decoder_input_ids)[b, t]` means:

```text
after consuming decoder_input_ids[b, t], which token IDs are legal next?
```

So the mask at `[b, t]` should contain `target_ids[b, t]`. Usually that is also
`decoder_input_ids[b, t + 1]`, except near EOS/padding boundaries.

JAX constraints:
- APIs operate on fixed-shape padded arrays.
- No dynamic variable-length arrays inside grammar functions.
- No Python loops over batch or sequence in mask computation.
- Use `jax.lax.scan` for sequence progression.
- Future sampling should carry `grammar_state` and call `advance_state`, instead
  of dynamically slicing prefixes inside a JIT loop.
- Grammar object may be built in Python once, but runtime state/mask operations
  should be JAX-array friendly.

Non-goals:
- No seq2seq model wrapper.
- No loss function.
- No sampler.
- No trainer, CLI, checkpoint, or dataset changes.
- No grammar-aware tokenizer decoding.

Acceptance criteria:
- Tokenizer exposes public token-kind helpers.
- Grammar masks allow the expected legal next-token families at each FSM state.
- Teacher-forcing masks align with `target_ids[:, t]`.
- Prefix masks work for fixed-shape sampling-style prefixes.
- `advance_state` works on scalar and batched states/tokens.
- Invalid token streams produce all-false masks after the invalid transition.
- Existing tokenizer tests still pass.

## Role Boundary

The user is acting as Product Owner / Stakeholder.
Low-level implementation decisions are agent-owned.
User-provided implementation details are binding only when they express product,
compatibility, safety, performance, or operational constraints.

## Agent-Owned Notes

Assumptions:
- The FSM can live in a focused grammar module separate from the reusable
  transformer core.
- The tokenizer remains responsible only for token metadata and encode/decode;
  grammar-aware decoding is outside this story.
- `token_ids_for_kind(kind)` can return an empty tuple when no token has that
  kind.

Risk level: medium

Story size: small

Escalation: none

Implementation hints:
- Prefer a focused module under `gristmill_symbolics/`.
- Build grammar tables once from tokenizer public helpers.
- Keep runtime methods pure array operations using JAX arrays and
  `jax.lax.scan`.

Follow-up stories:
- Flat token seq2seq wrapper using tokenizer vocabulary IDs.
- Decoder-input/label/mask and sequence log-probability helpers.
- Sampling loop that carries grammar state.
- Dataset, trainer, CLI, and checkpoint integration.

## Definition of Done

- Acceptance criteria met
- Deliverables completed
- Non-goals avoided
- Verification run
- No unjustified scope, dependency, abstraction, public API, or broad refactor
  growth
