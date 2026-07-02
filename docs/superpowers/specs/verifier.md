# Task: Implement rewrite-independent TensorComputation equivalence verifier

Implement a verifier for `TensorComputation` equivalence in the Rust symbolic engine.

The verifier should not depend on the rewrite engine, bicliques, decisions, action spaces, or factorization logic. It should verify semantic equivalence by:

1. Inlining all non-output intermediate tensor definitions.
2. Aligning external indices of corresponding outputs.
3. Subtracting the two expanded output definitions.
4. Canonicalizing and merging terms.
5. Checking whether the normal-form difference is zero.

## Public API

Add a new module, probably:

```rust
pub mod verify;
```

Expose an API like:

```rust
pub fn equivalent_computations(
    lhs: &TensorComputation,
    rhs: &TensorComputation,
    outputs: &[TensorId],
) -> Result<bool, VerifyError>;
```

A diagnostic API can be added later.

## Scope / assumptions for MVP

Assume:

* each tensor has at most one definition;
* dependency graph is acyclic;
* tensor/range IDs have the same meaning in both computations;
* inputs pass `TensorComputation::validate()`;
* tensors not defined in a computation are leaf/input tensors;
* outputs are explicitly provided by the caller.

Later, add pre-checks for duplicate definitions and cycles.

## Core equivalence definition

For every output tensor `O`:

```text
NF(inline(lhs, O) - inline(rhs, O)) == 0
```

where `NF` means canonicalize terms, merge like terms, and remove zero coefficients.

## Inline algorithm

Use a non-recursive global elimination pass.

```text
INLINE_ALL_INTERMEDIATES(comp, outputs):

    working_defs = comp.definitions
    fresh_id = 1 + max IndexId used anywhere in comp

    while working_defs contains a non-output definition:

        source = last non-output TensorDef in working_defs

        for each target in working_defs except source:
            target = INLINE_SOURCE_INTO_TARGET(target, source, fresh_id)
            update fresh_id

        remove source from working_defs

    return remaining output definitions
```

## Inlining one source into one target

```text
INLINE_SOURCE_INTO_TARGET(target, source, fresh_id):

    new_terms = []

    for term in target.terms:
        new_terms += INLINE_SOURCE_INTO_TERM(term, source, fresh_id)
        update fresh_id

    return target with terms = new_terms
```

## Inlining inside one term

A term is:

```text
coeff * sum_{dummy indices} factor_1 * factor_2 * ... * factor_n
```

For each factor:

* if `factor.tensor != source.base`, keep it as a one-term expression;
* if `factor.tensor == source.base`, replace it by the instantiated RHS terms of `source`.

Then take the Cartesian product of all factor expansions.

```text
INLINE_SOURCE_INTO_TERM(term, source, fresh_id):

    expanded_factor_lists = []

    for factor in term.factors:

        if factor.tensor == source.base:
            expanded_factor_lists.append(
                INSTANTIATE_SOURCE_AT_FACTOR(source, factor, fresh_id)
            )
            update fresh_id
        else:
            expanded_factor_lists.append(
                [TERM_CONTAINING_ONLY_THIS_FACTOR(factor)]
            )

    products = CARTESIAN_PRODUCT(expanded_factor_lists)

    result_terms = []

    for product in products:
        new_term = MULTIPLY_TERMS(product)

        new_term.coeff *= term.coeff
        new_term.sum_indices += term.sum_indices

        result_terms.append(new_term)

    return result_terms
```

## Instantiating a source definition at a factor

Example:

```text
source:
    tau[x, y] = sum_k B[x, k] C[k, y]

factor:
    tau[i, j]

result:
    sum_k' B[i, k'] C[k', j]
```

Behavior:

```text
INSTANTIATE_SOURCE_AT_FACTOR(source, factor, fresh_id):

    1. Map source external indices to actual factor indices.
           x -> i
           y -> j

    2. For every RHS term of source:
           rename each source-term dummy index to a fresh IndexId
           preserve the original dummy's RangeId

    3. Apply both mappings to all factor indices in the RHS term.

    4. Return the instantiated RHS terms and updated fresh_id.
```

Important rule:

```text
same old dummy inside one source term -> same fresh dummy
different source instantiation occurrence -> different fresh dummy
```

This prevents accidental dummy capture.

## After inlining

For each output:

```text
lhs_output = expanded lhs output
rhs_output = expanded rhs output

rhs_output = ALIGN_EXTERNAL_INDICES(rhs_output, lhs_output)

diff = SUBTRACT(lhs_output, rhs_output)

normal = CANONICALIZE_AND_MERGE(diff)

equivalent iff normal has no terms
```

## External index alignment

Before subtraction, rename RHS external indices to match LHS external indices.

Example:

```text
R[i] = A[i]
R[j] = A[j]
```

These should compare equal.

Require:

* same output tensor base;
* same external arity;
* compatible external ranges.

## Canonicalize and merge

Use existing public canonicalization helpers where possible:

* `canon::build_tensor_symmetry_map`
* `canon::canon_term`

Process:

```text
CANONICALIZE_AND_MERGE(def):

    pool = build index pool from the whole diff definition
    symmetry = build tensor symmetry map

    table = map from term structure to Rational coefficient

    for term in def.terms:
        canon_term = canon_term(term, symmetry, pool)

        key = canon_term without coefficient

        table[key] += canon_term.coeff

    remove zero coefficients

    return normal form
```

A term key should include:

```text
sum_indices
factors
```

but not coefficient.

## Tests to add

### 1. Direct equality

```text
R[i] = A[i]
```

vs.

```text
R[i] = A[i]
```

should verify true.

### 2. External index alpha-renaming

```text
R[i] = A[i]
```

vs.

```text
R[j] = A[j]
```

should verify true.

### 3. Simple intermediate inline

```text
tau[i] = B[i] + C[i]
R[i] = A[i] tau[i]
```

vs.

```text
R[i] = A[i] B[i] + A[i] C[i]
```

should verify true.

### 4. Dummy hygiene

```text
tau[x] = sum_k B[x, k]
R[i] = sum_k A[i, k] tau[k]
```

should expand to:

```text
R[i] = sum_{k, l} A[i, k] B[k, l]
```

not:

```text
sum_k A[i, k] B[k, k]
```

### 5. Same intermediate used twice

```text
tau[x] = sum_k A[x, k]
R[i] = tau[i] tau[i]
```

should expand to two independent dummy indices:

```text
sum_{k, l} A[i, k] A[i, l]
```

### 6. Non-equivalent computations

```text
R[i] = A[i]
```

vs.

```text
R[i] = B[i]
```

should verify false.

## Implementation note

Do not use rewrite-specific structures such as `Decision`, `Factorization`, bicliques, or action spaces. This verifier should only depend on the semantic IR:

* `TensorComputation`
* `TensorDef`
* `Term`
* `Factor`
* `Index`
* `TensorId`
* `IndexId`
* `RangeId`
* `Rational`
* canonicalization helpers
