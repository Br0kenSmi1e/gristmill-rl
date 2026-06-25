# Reusable Batched Policy JIT API Design

## Problem

Issue #20 profiles show that the remaining small-batch CCSD REINFORCE runtime
cost is dominated by repeated JAX transformation and compilation of stable-shape
policy internals. Static rollout shapes removed the worst dynamic policy-shape
churn, but grouped compile signatures still scale with update count:

```text
updates=1  compiles=2130
updates=2  compiles=3103
updates=3  compiles=4433
```

The repeated signatures are stable-shape action-policy internals, especially
`jit(scan)` calls associated with `_sample_side` and `_score_side`. The rollout
loop currently rebuilds `jax.vmap(...)` and `jax.value_and_grad(...)` wrappers
inside the loop. That makes the transformation boundary unstable even when the
array shapes are already static.

## Goal

Add reusable, module-level batched policy APIs that create the pure-JAX policy
transforms once and let rollout reuse them on every step:

```text
batched_sample_target
batched_score_target_grad
batched_sample_action
batched_score_action_grad
```

The implementation should reduce repeated compile signatures without changing
default rollout behavior, scalar policy semantics, metrics, or Rust row
environment responsibilities.

## Non-Goals

- Do not add `jit(train_update)`.
- Do not move Python/Rust rollout state, action-space queries, validation, or
  action application into JAX.
- Do not redesign the scalar policy model or action representation.
- Do not address the separate full-attention memory wall, including larger
  batch shapes such as `float32[2,4096,4096]`.
- Do not make `batch_size=8` memory-safe in this change.

## Architecture

Add `python/gristmill_symbolics/policy/batched.py`. This module owns the stable
JAX transformation boundaries and defines four import-time callables:

```python
batched_sample_target = jax.jit(
    jax.vmap(sample_target, in_axes=(None, 0, 0, 0, 0))
)

batched_score_target_grad = jax.jit(
    jax.vmap(
        jax.value_and_grad(score_target, argnums=0),
        in_axes=(None, 0, 0, 0, 0),
    )
)

batched_sample_action = jax.jit(
    jax.vmap(sample_action, in_axes=(None, 0, 0, 0, 0, 0, 0))
)

batched_score_action_grad = jax.jit(
    jax.vmap(
        jax.value_and_grad(score_action, argnums=0),
        in_axes=(None, 0, 0, 0, 0, 0, 0),
    )
)
```

Export these callables from `python/gristmill_symbolics/policy/__init__.py` so
they are public, reusable policy APIs.

Update `python/gristmill_symbolics/reinforce/rollout.py` to import and call the
batched APIs instead of constructing inline `vmap` and `value_and_grad`
transforms in the rollout loop. Rollout remains the owner of active sample
selection, static padding, dummy rows, masking, metrics, validation, and
gradient accumulation.

## Components And Data Flow

`batched_sample_target` accepts:

- `params`
- batched state token tree
- batched state token mask
- batched definition mask
- per-row RNG keys

It returns a `[batch]` target-choice array.

`batched_score_target_grad` accepts the same inputs plus `[batch]` target
choices. It returns `(target_logps, target_grads)`, where `target_logps` has
shape `[batch]` and every gradient leaf has shape
`[batch, *param_leaf_shape]`.

`batched_sample_action` accepts:

- `params`
- batched state token tree
- batched state token mask
- `[batch]` selected definition indices
- batched action-space token tree
- batched action-space token mask
- per-row RNG keys

It returns an action-choice pytree with one batch axis on every leaf.

`batched_score_action_grad` accepts the same action inputs plus the batched
action-choice pytree. It returns `(action_logps, action_grads)`, with the same
gradient leaf convention as the target scorer.

In rollout, existing static-mode row masking still happens after these calls and
before gradient accumulation. Dummy and inactive rows therefore continue to
contribute exactly zero.

## Error Handling And Compatibility

The batched APIs do not add new validation rules. They inherit scalar validation
from `score_target`, `sample_target`, `score_action`, and `sample_action`.
Invalid choices and shape mismatches should fail through the same scalar policy
or JAX paths as today.

Rollout `TrainingError` behavior remains unchanged for invalid configs,
too-small static pads, unexpected action-space kinds, and failed Rust action
application.

Existing scalar policy APIs remain exported and untouched. Rollout should import
the batched callables at module scope so tests can monkeypatch
`rollout_module.batched_sample_target`, `rollout_module.batched_score_target_grad`,
`rollout_module.batched_sample_action`, and
`rollout_module.batched_score_action_grad` directly.

## Testing Strategy

Add policy API tests that compare each batched callable against the current
inline `jax.vmap(...)` behavior:

- `batched_sample_target` matches row-wise `sample_target`.
- `batched_score_target_grad` matches row-wise `value_and_grad(score_target)`.
- `batched_sample_action` matches row-wise `sample_action`.
- `batched_score_action_grad` matches row-wise `value_and_grad(score_action)`.

Add package/export tests to ensure the four batched APIs are available from
`gristmill_symbolics.policy`.

Update rollout tests so behavioral assertions remain focused on fixed-seed
metrics and gradient semantics rather than on the old inline transform
construction. Tests that monkeypatch policy calls should patch the rollout
module's batched callable imports.

## Profiling Acceptance Evidence

After implementation, compare the profiling run from issue #20 before and after
the change using the same input and seed. Acceptance evidence:

1. Default and static-shape behavior remain unchanged for existing tests.
2. Static-shape profile keeps the same final rollout/training metrics for a
   fixed input and seed.
3. `grep -c '^Compiling' stderr.log` drops materially for
   `batch_size=2, updates=1/2/3`.
4. Grouped compile signatures no longer show `128 * updates` repeated
   `jit(scan)` compiles for the action-side sampling/scoring internals.
5. Wall time improves without increasing peak GPU memory.

The profiling evidence should be reported separately from the unit test run
because the RTX 4060 Ti CCSD workload is an environment-specific benchmark, not
a routine CI test.
