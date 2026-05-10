# Sampled AlphaZero RL Prototype Design

## Goal

Build the first end-to-end reinforcement-learning prototype for the rewrite
kernel in this repository.

The prototype should be close to the intended sampled AlphaZero architecture:

- Rust owns symbolic correctness, legal action generation, validation,
  transition, terminal detection, and cost.
- Python/JAX owns feature extraction, action sampling, sampled PUCT, model
  training, replay records, and experiment runners.
- The first success criterion is structural: run a short training loop on a
  given JSON computation, update model parameters, and report cost/loss metrics.

This is not a benchmark-quality optimizer yet. It is the smallest useful system
that exercises the real Rust rewrite environment from Python/JAX search and
learning code.

## Context

The current crate already contains the deterministic rewrite kernel:

```text
repr -> split -> canon -> graph -> biclique -> rewrite -> io
```

The `rewrite` module exposes:

- `next_action_space`
- `validate_decision`
- `build_rewrite`
- `apply_rewrite`

The `random-rewrite` CLI proves that callers can choose decisions outside the
kernel and apply them through the existing APIs.

The older `rustymill` RL specs remain conceptually useful, but this repository
has different module names and a cleaner action-facing rewrite boundary. The
prototype should build on the current crate shape rather than porting the old
standalone design literally.

## Architecture

The prototype has three layers.

### Rust Core

The Rust core stays the deterministic symbolic kernel.

It adds a small `cost` module with simple FLOP evaluation and `log_total_flops`.
It continues to use the existing `rewrite` APIs for action-space generation and
state transition.

The Rust core does not contain:

- RL training
- policy scoring
- PUCT
- feature extraction for neural models
- Python-specific model code

### PyO3 Extension

A separate Rust extension crate lives under `python/` and depends on
`gristmill-symbolics` by path.

This crate exposes the environment boundary to Python. It owns the PyO3 classes
and conversion helpers, but it does not own learning algorithms.

The extension exposes opaque handles for Rust values that need their internal
sidecars preserved, especially `ActionSpace`.

### Python Package

Python package code lives under `python/gristmill_rl/`.

Python owns:

- symbolic-to-array feature extraction
- autoregressive action sampling
- sampled PUCT
- replay records
- the JAX policy/value model
- training and experiment runners

The first runner accepts a JSON input path. It can use tiny controlled inputs or
existing fixture-style computations. Success does not require outperforming any
baseline.

## Cost Objective

The v1 objective is simple FLOP count, matching the older `rustymill::cost`
model.

For one definition:

```text
ext_size = product of active definition external index range sizes
sum_size = product of term sum-index range sizes

if sum_size == 1:
  term_flops = ext_size + ext_size
else:
  term_flops = 2 * ext_size * sum_size + ext_size

def_flops = sum(term_flops)
```

For a computation:

```text
total_flops = sum(def_flops)
log_total_flops = ln(total_flops)
```

The raw FLOP path should be checked or fallible, so overflow becomes an error
instead of silently wrapping. RL-facing APIs should use `log_total_flops` because
search values and value targets need a stable scale.

Parenthesized contraction cost is out of scope for v1. The module boundary
should leave room for later objective variants, but v1 only implements this
simple objective.

## Environment API

The PyO3 environment owns a `TensorComputation`.

Target Python-facing shape:

```python
env = RlEnv.load_json(path)
clone = env.clone()

log_cost = env.log_total_flops()
space = env.next_action_space(start_from=0)
terminal = env.is_terminal(start_from=0)
result = env.apply_decision_with_space(space, decision)
```

`start_from` is explicit. This keeps the environment behavior aligned with the
existing rewrite loop and avoids hiding cursor state inside the environment.

`ApplyResult` reports:

- `applied`
- `def_index`
- `next_start_from`
- `log_total_flops`
- `terminal`
- `error`

Invalid actions are not terminal. Terminal means
`next_action_space(next_start_from)` returns no action space after a successful
apply. `next_start_from` is the rewritten definition index from the stored
action space, matching the existing rewrite loop.

`log_total_flops` is fallible at the Rust boundary. It returns an error when
raw FLOP computation overflows or when the total cost is zero and therefore has
no finite natural logarithm.

## Action Representation

Rust keeps the current action semantics:

```text
Decision {
  candidate_index,
  left_mask,
  right_mask
}
```

`candidate_index` indexes one maximal-biclique factorization template from the
returned `ActionSpace`.

`left_mask` and `right_mask` index the left and right side terms in that
candidate template. `true` means keep the term. Both masks must keep at least
one term.

Python treats a sampled action as one complete macro-action:

```python
SampledAction(
    candidate_index: int,
    left_mask: list[bool],
    right_mask: list[bool],
    prior: float,
)
```

The policy samples a complete action autoregressively:

```text
state/action-space features
  -> sample candidate_index
  -> condition on selected candidate
  -> sample left_mask
  -> condition on selected candidate and left_mask
  -> sample right_mask
  -> validate through Rust
```

Invalid sampled actions are discarded and resampled up to a configurable cap. If
too few valid unique actions are found, search proceeds with the valid set it
has. Deterministic fallbacks should exist for debugging, such as full masks for
the first candidate or uniform random valid actions.

PUCT sees only complete sampled actions. It does not search over candidate and
mask substeps separately.

## Stored Action Spaces

Each search-tree node represents one Rust environment state plus the
`start_from` cursor for that state.

A node starts unexpanded. On first expansion only, Python calls:

```python
action_space = env.next_action_space(start_from)
```

If there is no action space, the node is terminal.

Otherwise, the node stores:

```python
SearchNode(
    env_snapshot,
    start_from,
    action_space_handle,
    action_space_features,
    children,
)
```

This is a correctness requirement.

`Decision.candidate_index` and masks are scoped to the exact `ActionSpace` that
produced them. The current Rust action space also contains private graph and
biclique sidecars that are required to build the rewrite. A later recomputation
could produce different ordering or a different opaque sidecar value.

When traversing a child, Python clones the node's `env_snapshot` and applies:

```python
child_env.apply_decision_with_space(action_space_handle, child.decision)
```

This uses the stored opaque action-space handle, not a recomputed action space.
After applying, the returned `next_start_from` is the child state's next cursor.
It is the rewritten `def_index`, matching the existing rewrite loop.

Subsequent visits to an expanded node reuse the stored children and priors and
update only visit statistics. If progressive widening is added later, it must
extend the stored child set using the same stored action-space handle.

## Sampled PUCT

At each expanded node, the sampled actions become the node's children. Each
child receives a prior from the autoregressive policy probability.

Selection uses the standard PUCT shape:

```text
score = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
```

A simulation clones the stored environment state, applies a selected child
decision through the stored action-space handle, and either expands the new
state or evaluates it with the value head.

The root action is chosen by visit count after search, not by raw prior.

The sampled action set is fixed for a node once that node is expanded. This
makes the policy target well-defined for replay.

## Replay And Training Targets

Replay records are scoped to the sampled action set available at a search root:

```python
ReplayItem(
    state_snapshot,
    action_space_snapshot,
    sampled_actions,
    visit_distribution,
    state_log_flops,
)
```

At episode end, compute:

```text
final_log_flops
```

For each recorded state, the value target is:

```text
z_t = state_log_flops - final_log_flops
```

This trains the value head to predict expected remaining log-cost improvement
from the current state. There is no dense per-step reward in v1.

The policy target is the MCTS root visit distribution over the sampled full
actions for that root. The policy loss trains the model to match that
distribution; the value loss regresses to `z_t`.

## Feature Extraction And Model

Feature extraction is Python-owned in v1. Rust exports faithful symbolic values
and opaque action-space handles; Python builds padded arrays from exported
snapshots.

The initial model can be small but structured:

```text
state encoder:
  global scalar/count features
  active-definition summary

candidate encoder:
  per-candidate template summary
  left-side term summaries
  right-side term summaries

heads:
  candidate logits
  left term logits conditioned on selected candidate
  right term logits conditioned on selected candidate and left-mask summary
  scalar value head
```

Initial feature categories:

- number of definitions, tensors, and ranges
- active definition term count
- active definition factor-count and sum-index summaries
- current `log_total_flops`
- candidate left and right term counts
- candidate rewritten-definition term count
- per-term factor count
- per-term sum-index count
- coefficient sign and magnitude summaries

The first implementation does not require a graph neural network or
transformer. It needs consistent padded tensors and masks, with a clear path to
richer encoders later.

Feature extraction for a search node must use the stored action-space snapshot,
not a recomputed action space.

## Training Runner

The first runner is intentionally small.

Target command shape:

```text
python -m gristmill_rl.train --input <path.json> --episodes 2 --simulations 8
```

The runner should:

1. load the JSON computation through the PyO3 environment
2. run short sampled-PUCT-guided episodes
3. collect replay records
4. compute final log FLOP targets
5. apply at least one JAX optimizer update
6. print concise metrics

Metrics should include:

- episode length
- initial log FLOPs
- final log FLOPs
- number of valid sampled actions per root
- policy loss
- value loss

## Scope

In scope for v1:

- Rust `cost` module with checked simple FLOP cost and `log_total_flops`
- separate PyO3 extension crate under `python/`
- Python package under `python/gristmill_rl/`
- environment object with JSON load, clone, action-space handle, apply with
  stored space, terminal checks, and log FLOPs
- Python feature extraction from exported symbolic/action-space snapshots
- autoregressive JAX policy/value model
- sampled valid action generation
- sampled PUCT with one-time node expansion and stored action spaces
- short training runner over a given JSON input

Out of scope for v1:

- beating greedy or random baselines
- parenthesized term cost
- learned definition selection
- Rust-side neural inference
- full action enumeration
- stable candidate IDs across independently generated action spaces
- production packaging polish

## Testing

Rust tests:

- simple FLOP cost for small hand-built computations
- checked overflow behavior for raw cost
- `log_total_flops` returns finite values for valid nonzero-cost computations

PyO3 tests:

- load JSON into an environment
- clone an environment
- compute log cost
- get an action-space handle for an actionable input
- apply a decision with its stored action-space handle
- reject invalid decisions through Rust validation
- document or guard against mismatched action-space lifecycle misuse

Python tests:

- feature extraction produces stable padded shapes on a tiny input
- autoregressive sampler produces valid decisions when using deterministic
  fallback policy
- PUCT computes a node action space once and reuses the stored handle
- training runner completes at least one short update and records metrics

## Acceptance Criteria

The prototype design is complete when:

- the Rust core exposes the simple log FLOP objective
- Python can load a given JSON computation through PyO3
- sampled PUCT can expand nodes with stored action-space handles
- decisions are applied through the stored action-space handle that produced
  them
- replay records train policy targets over sampled full actions
- value targets use `state_log_flops - final_log_flops`
- a short training command runs end to end and updates JAX parameters

The prototype is not expected to produce better final costs than existing
baselines in v1.
