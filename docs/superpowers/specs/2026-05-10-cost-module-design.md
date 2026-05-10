# `cost` Module Design

## Goal

Add the v1 scalar objective needed by the sampled AlphaZero prototype.

The module should compute the natural logarithm of the simple FLOP cost for a
`TensorComputation` without materializing the raw integer FLOP count. This keeps
the RL-facing objective stable for large symbolic sizes and avoids integer
overflow in normal use.

## Scope

This module includes:

- simple FLOP objective evaluation
- log-space arithmetic
- zero range-size errors
- zero total-cost errors
- focused tests for the objective formula and numerical behavior

This module excludes:

- parenthesized contraction cost
- FLOPs plus IO or memory-weighted objectives
- rewrite profitability scoring
- Python bindings
- feature extraction
- structural validation of `TensorComputation`

## Public API

Expose the module from `src/lib.rs`:

```rust
pub mod cost;
```

Public API:

```rust
use crate::repr::{RangeId, TensorComputation};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CostError {
    ZeroRangeSize { range: RangeId },
    ZeroTotalFlops,
}

pub fn log_total_flops(comp: &TensorComputation) -> Result<f64, CostError>;
```

`log_total_flops` is the only public function in v1. Raw integer FLOP counts
are not part of the public API.

## Validity Contract

The cost module assumes the input representation is structurally valid.

That means:

- range IDs referenced by definitions and terms must exist in
  `comp.ranges()`
- tensor and index consistency must already have been checked by
  `TensorComputation::validate()` or trusted fixture construction

The module should not duplicate `repr::validate()` by returning unknown-range
or unknown-index errors. If callers pass structurally invalid computations,
behavior is outside this module's contract.

The cost module does perform objective-specific validation:

- range sizes used in cost products must be nonzero
- total FLOP cost must be nonzero before taking a logarithm

## Objective Formula

The v1 objective matches the older simple `rustymill::cost` model.

For one term in one definition:

```text
ext_size = product of active definition external index range sizes
sum_size = product of term sum-index range sizes

if sum_size == 1:
  term_flops = ext_size + ext_size
else:
  term_flops = 2 * ext_size * sum_size + ext_size
```

For the whole computation:

```text
total_flops = sum(term_flops for every term in every definition)
log_total_flops = ln(total_flops)
```

The implementation must preserve this formula in log space. It must not compute
the raw `total_flops` as a `u64`.

Empty products have size `1`:

- scalar outputs have `ext_size = 1`
- terms with no summed indices have `sum_size = 1`

Definitions with no terms contribute no cost.

## Log-Space Algorithm

Compute product sizes as sums of logarithms:

```text
log_ext_size = sum ln(range.size) over def.ext_indices
log_sum_size = sum ln(range.size) over term.sum_indices
```

If any referenced range has size `0`, return:

```rust
CostError::ZeroRangeSize { range }
```

Per-term log cost:

```text
if sum_product.is_one:
  log_term = ln(2) + log_ext_size
else:
  log_term = logaddexp(
      ln(2) + log_ext_size + log_sum_size,
      log_ext_size,
  )
```

The branch preserves the old raw formula's `sum_size == 1` rule. It must not
branch only on whether `term.sum_indices` is empty, because explicit size-one
summed ranges also have `sum_size == 1`.

Total log cost is a fold with `logaddexp`:

```rust
let mut total: Option<f64> = None;

for def in comp.definitions() {
    for term in &def.terms {
        let log_term = log_term_flops(term, def, comp.ranges())?;
        total = Some(match total {
            None => log_term,
            Some(acc) => logaddexp(acc, log_term),
        });
    }
}

total.ok_or(CostError::ZeroTotalFlops)
```

No separate `logsumexp` helper is needed in v1.

## Numerical Helper

Private helper:

```rust
fn logaddexp(a: f64, b: f64) -> f64;
```

It should use a stable implementation:

```text
max(a, b) + ln(exp(a - max) + exp(b - max))
```

The helper only needs to handle finite inputs produced by this module. It does
not need special public behavior for NaN or infinities.

Recommended private helper shape:

```rust
fn log_term_flops(
    term: &Term,
    def: &TensorDef,
    ranges: &[Range],
) -> Result<f64, CostError>;

fn log_size_product(indices: &[Index], ranges: &[Range]) -> Result<LogProduct, CostError>;
```

These helpers remain private. Unit tests inside `src/cost.rs` may exercise them
directly if useful.

## Size-One Summed Indices

The old raw formula branches on `sum_size == 1`, not on whether there are
summed indices. In log-space v1, explicit size-one summed indices should also
produce the no-summation cost:

```text
term_flops = ext_size + ext_size
```

To support that without raw product materialization, `log_size_product` should
return enough information to know whether the product is exactly one.

Recommended internal type:

```rust
struct LogProduct {
    value: f64,
    is_one: bool,
}
```

`is_one` is true when every index range in the product has size `1`, including
the empty product. Since zero sizes return an error, this is exact.

Then:

```text
if sum_product.is_one:
  log_term = ln(2) + log_ext_size
else:
  log_term = logaddexp(
      ln(2) + log_ext_size + log_sum_size,
      log_ext_size,
  )
```

## Error Semantics

`ZeroRangeSize` means a range referenced by the computation has size `0`.

`ZeroTotalFlops` means no term contributed any positive cost. This covers:

- an empty computation
- computations containing only zero-term definitions

The function returns `Ok(f64)` only for finite positive total cost.

## Integration

`src/lib.rs` should expose:

```rust
pub mod cost;
```

No existing rewrite or CLI code needs to call cost in this slice. The PyO3
environment spec will depend on `cost::log_total_flops` later.

## Testing

Most tests should live in `src/cost.rs` because the helper API is private.

Required coverage:

- simple contraction:

```text
T[a,b] = sum_c A[a,c] * B[c,b]
range size = 10
ext_size = 100
sum_size = 10
flops = 2100
log_total_flops = ln(2100)
```

- no summation:

```text
T[a,b] = A[a,b]
range size = 10
flops = 200
log_total_flops = ln(200)
```

- scalar output:

```text
E = sum_ab A[a,b] * B[a,b]
range size = 10
ext_size = 1
sum_size = 100
flops = 201
log_total_flops = ln(201)
```

- multiple definitions:

```text
two definitions each costing 2100
log_total_flops = ln(4200)
```

- explicit size-one summed index:

```text
sum_size = 1
flops = ext_size + ext_size
```

- zero-size range returns `CostError::ZeroRangeSize`
- empty computation returns `CostError::ZeroTotalFlops`
- computation with only zero-term definitions returns `CostError::ZeroTotalFlops`
- huge range sizes produce a finite log result rather than overflowing an
  integer product
- `logaddexp` handles strongly separated values without losing the larger value

Comparisons should use a small absolute tolerance for floating-point results.

## Acceptance Criteria

The cost module is complete when:

- `src/cost.rs` exists
- `src/lib.rs` exposes `pub mod cost`
- `log_total_flops` is the only public cost function
- raw integer total FLOP counts are not materialized by the public path
- zero range sizes return `ZeroRangeSize`
- empty or zero-term computations return `ZeroTotalFlops`
- simple small computations match `ln(expected_flops)`
- huge symbolic sizes return finite logs instead of overflow errors
- existing tests continue to pass
