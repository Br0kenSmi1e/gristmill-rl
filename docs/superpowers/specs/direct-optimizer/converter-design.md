# Direct Optimizer Converter Design

## Role

The converter is the syntax and reconstruction boundary for the direct
optimizer. It owns the custom DSL and the structured token representation used by
the model.

It should feel like a small, strict compiler front-end for
`TensorComputation`:

1. Print symbolic objects into deterministic DSL text.
2. Parse DSL text back into structured snapshot pieces.
3. Reconstruct sampled candidate computations from a fixed input envelope and
   generated target definitions.

The converter is not a dataset module, verifier module, trainer, or model
tokenizer. It performs deterministic conversion, strict parsing, reconstruction,
and Rust `TensorComputation` validation.

## Public Contract

The public API should expose:

```python
computation_to_source_text(comp: TensorComputation) -> str
computation_to_target_text(comp: TensorComputation) -> str

source_text_to_snapshot(text: str) -> dict
target_text_to_definitions(text: str) -> list[dict]

target_text_to_computation(
    x: TensorComputation,
    target_text: str,
) -> TensorComputation
```

Meanings:

- `source_text` represents a full computation: ranges, tensors, and
  definitions.
- `target_text` represents all definitions of `y`, and only definitions.
- `target_text_to_computation(x, target_text)` reconstructs a full candidate
  `y` by copying `ranges` and existing `tensors` from `x`, parsing generated
  definitions, and registering missing definition-base tensors with empty
  symmetry.

The converter does not call `equivalent_computations`, compute flops, dedupe
examples, assign weights, or know about model parameters.

## Representation Overview

The converter owns both a readable DSL and a structured token representation.

Readable DSL is line-oriented and deterministic. It is intended for processed
datasets, debugging, checkpoints, and CLI visibility.

Example:

```text
range id range_id:0 size dim_size:8
tensor id tensor_id:0
endtensor
def base tensor_id:1
ext id index_id:0 range range_id:0
term
coeff numer coeff_num:1 denom coeff_den:1
factor tensor tensor_id:0
index index_id:0
endfactor
endterm
enddef
```

The same DSL parses into structured logical tokens:

```text
KEYWORD(range)
KEYWORD(id)
SCALAR(range_id, 0)
KEYWORD(size)
SCALAR(dim_size, 8)
...
```

Model arrays preserve that structure:

```python
{
    "kind": int32[..., length],
    "keyword": int32[..., length],
    "scalar_type": int32[..., length],
    "scalar_value": int32[..., length],
    "mask": bool[..., length],
}
```

This is still a sequence model input, but tokens are not collapsed into one flat
vocabulary. The model can distinguish `tensor_id:3`, `index_id:3`, and
`range_id:3` by scalar type.

## DSL Grammar

The DSL has structural keyword tokens and typed scalar tokens.

Typed scalar format:

```text
<scalar_type>:<integer_or_name>
```

Initial scalar types:

```text
range_id:<int>
tensor_id:<int>
index_id:<int>
dim_size:<positive_int>
coeff_num:<signed_int>
coeff_den:<positive_int>
sym_action:Identity
sym_action:Negate
axis:<int>
```

Keyword vocabulary:

```text
range id size
tensor symmetry action perm endtensor endsymmetry
def base ext term coeff numer denom sum factor index endfactor endterm enddef
```

Full source DSL supports:

```text
range id range_id:N size dim_size:N

tensor id tensor_id:N
symmetry action sym_action:Identity|Negate
perm axis:N
perm axis:N
endsymmetry
endtensor

def base tensor_id:N
ext id index_id:N range range_id:N
term
coeff numer coeff_num:Z denom coeff_den:N
sum id index_id:N range range_id:N
factor tensor tensor_id:N
index index_id:N
index index_id:N
endfactor
endterm
enddef
```

A tensor may have zero or more `symmetry ... endsymmetry` blocks. Permutations
are represented by repeated `perm axis:N` records in order. Symmetry blocks
preserve snapshot order.

Target DSL supports only the `def ... enddef` blocks.

Rules:

- one logical record per line;
- field order is fixed;
- block nesting is strict;
- repeated records preserve snapshot order;
- parser rejects unknown keywords, unknown scalar types, malformed scalar
  values, missing fields, extra fields, and invalid block nesting.

## Source And Target Semantics

Source DSL is a full computation snapshot:

```text
ranges + tensors + definitions
```

It must round-trip exactly:

```text
comp.snapshot() -> source DSL -> snapshot == comp.snapshot()
```

Target DSL is definitions only:

```text
definitions
```

It must round-trip exactly at the definitions level:

```text
comp.snapshot()["definitions"] -> target DSL -> definitions
```

The printer preserves snapshot order for:

- ranges;
- tensors;
- tensor symmetries;
- permutation entries;
- definitions;
- external indices;
- terms;
- sum indices;
- factors;
- factor indices.

The parser should not canonicalize or sort records. Canonicalization for
grouping comes from printing parsed and validated structures back through the
deterministic printer.

For target reconstruction:

```python
target_text_to_computation(x, target_text)
```

does:

1. Parse target text into `definitions`.
2. Copy `ranges` from `x.snapshot()`.
3. Copy `tensors` from `x.snapshot()`.
4. Add `{"id": base, "symmetry": []}` for every generated definition base not
   already present as a tensor.
5. Reject any factor tensor reference that is neither copied from `x` nor
   introduced as a generated definition base.
6. Build a full snapshot and validate it by calling
   `TensorComputation.from_json_string`.

## Structured Token Semantics

The converter exposes a structured token layer separate from readable DSL text.

Logical token kinds:

```text
PAD
BOS
EOS
KEYWORD
SCALAR
```

Keyword ids cover the fixed DSL keyword vocabulary:

```text
range id size tensor symmetry action perm endtensor endsymmetry
def base ext term coeff numer denom sum factor index endfactor endterm enddef
```

Scalar type ids cover:

```text
range_id tensor_id index_id dim_size coeff_num coeff_den sym_action axis
```

Each token has fields:

```python
{
    "kind": ...,
    "keyword": ...,
    "scalar_type": ...,
    "scalar_value": ...,
}
```

Rules:

- `KEYWORD` tokens set `keyword` and use sentinels for scalar fields.
- `SCALAR` tokens set `scalar_type` and `scalar_value`.
- `PAD`, `BOS`, and `EOS` use sentinels for keyword and scalar fields.
- `sym_action:Identity` and `sym_action:Negate` are scalar values encoded as
  small integers through the scalar-type table, not free strings in model
  arrays.
- The readable DSL text is the serialization format; structured tokens are the
  model-facing encoding of that DSL.
- Structured token encode/decode should round-trip through DSL text exactly for
  valid DSL.

## Error Handling And Boundaries

The converter raises `ValueError` for invalid DSL or impossible reconstruction.
Messages should include short context, for example:

```text
unknown keyword 'foo'
expected tensor_id scalar after base
unexpected endterm outside term
factor references unknown tensor_id:7
```

It should catch Rust validation errors during reconstruction and re-raise them as
`ValueError` with context. The sampler and dataset modules can count or report
these failures without knowing parser internals.

The converter should not:

- call `equivalent_computations`;
- compute `log_total_flops`;
- group, dedupe, or weight examples;
- own model parameters or vocabulary;
- import `gristmill_symbolics.model.tokenizer`;
- import the action-selector, REINFORCE trainer, or CLI checkpoint modules.

It may import only:

```text
json
gristmill_symbolics.TensorComputation
```

## Tests

Required focused tests:

- Full source round-trip:

  ```text
  TensorComputation -> source DSL -> snapshot
  ```

  matches the original snapshot.

- Target definitions round-trip:

  ```text
  TensorComputation -> target DSL -> definitions
  ```

  matches the original `snapshot()["definitions"]`.

- Reconstruction copies envelope: generated target definitions plus `x` copy
  `ranges` and existing `tensors`.
- Reconstruction registers new definition bases: if generated definitions
  include `base tensor_id:9` and `x` has no tensor `9`, reconstructed snapshot
  includes `{"id": 9, "symmetry": []}`.
- Reconstruction rejects unresolved factor tensors: a factor using
  `tensor_id:99` fails unless `99` is in copied `x.tensors` or generated
  definition bases.
- Symmetry source round-trip: source DSL preserves tensor symmetry action and
  permutation order.
- Structured token round-trip: valid DSL text to structured tokens to DSL text
  is exact.
- Parser rejects malformed DSL: unknown keyword, wrong scalar type, malformed
  scalar value, missing `enddef`, and invalid nesting.

Verification command:

```bash
uv run pytest python/tests/direct_optimizer/test_converter.py -q
```

