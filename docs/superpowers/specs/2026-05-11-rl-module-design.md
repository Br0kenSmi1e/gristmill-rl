# RL Module Design

## Goal

Build the pure Python reinforcement-learning layer for the sampled AlphaZero
prototype.

The RL module should run a short end-to-end loop over the existing rewrite
kernel:

1. load a JSON computation through the implemented PyO3 bindings
2. run sampled PUCT at each rewrite state
3. collect replay examples from search roots
4. train a small Flax NNX policy/value model with Optax
5. report cost, search, and loss metrics

This slice treats the Rust core and PyO3 extension as fixed. It does not add
new Rust APIs, new PyO3 classes, or an `RlEnv` wrapper.

## Existing Boundary

The implemented PyO3 module exposes `gristmill_symbolics.TensorComputation`
and `gristmill_symbolics.ActionSpace` directly.

The RL module uses this API:

```python
comp = TensorComputation.load_json(path)
child = comp.clone()
state_snapshot = comp.snapshot()
log_cost = comp.log_total_flops()
space = comp.next_action_space(start_from)
space_snapshot = space.snapshot()
comp.apply_decision_with_space(space, decision)
```

`apply_decision_with_space` mutates the receiving `TensorComputation` and
returns `None`. The next rewrite cursor is `space.def_index`, matching the
existing Rust `random-rewrite` loop.

`ActionSpace` handles are valid only for the computation state that produced
them and clones of that same pre-rewrite state. The RL module must preserve
that lifecycle rule.

## Package Shape

Python package code lives under `python/gristmill_rl/`.

Required v1 modules:

```text
gristmill_rl/
  __init__.py
  features.py
  actions.py
  model.py
  search.py
  replay.py
  train.py
```

Responsibilities:

- `features.py`: convert computation and action-space snapshots into padded
  arrays and masks
- `actions.py`: define `SampledAction`, autoregressive sampling, deterministic
  fallbacks, validation, and deduplication
- `model.py`: define the Flax NNX policy/value model and loss helpers
- `search.py`: implement sampled PUCT with stored `TensorComputation` clones
  and stored `ActionSpace` handles
- `replay.py`: define temporary episode traces and completed replay items
- `train.py`: provide the CLI runner and training loop

No module should recompute an old action space to interpret old action indices.

## Search Node Invariant

Each search-tree node represents one local state:

```text
TensorComputation clone + start_from cursor
```

On first expansion, the node calls:

```python
space = comp.next_action_space(start_from)
```

This happens at most once per node. The node then stores:

```text
comp: TensorComputation
start_from: int
action_space: ActionSpace | None
action_space_snapshot: dict | None
sampled_actions: list[SampledAction]
children: list[SearchChild]
expanded: bool
```

If `action_space` is `None`, the node is terminal.

If an action space exists, all child transitions from that node must apply
decisions through the exact stored `ActionSpace` handle:

```python
child_comp = node.comp.clone()
child_comp.apply_decision_with_space(node.action_space, child.decision)
child_start_from = node.action_space.def_index
```

The child node owns the rewritten `TensorComputation` clone and computes its
own action space later, on first expansion.

## Action Identity And Ordering

`candidate_index` is local to one stored `ActionSpace`.

The Rust candidate order may change when the same symbolic state is recomputed,
because candidate generation can depend on hash-map iteration order. The RL
module accepts this by never treating candidate indices as stable global IDs.

For each search node and replay item, these values are scoped together:

```text
action_space_snapshot
sampled_actions
visit_distribution
```

Training uses the stored snapshot and stored sampled actions. It must not
reload a state, recompute an action space, and apply old candidate indices to
the recomputed order.

Across episodes, action spaces are recomputed and search trees are rebuilt.
That is acceptable because each new episode uses the newly produced
`ActionSpace` handle and local candidate ordering. Cross-episode action-space
caching is out of scope for v1.

The model should use shared candidate and term encoders. It should not assign
semantic meaning to raw candidate row position.

## Episode Data Flow

One episode mirrors the existing Rust `random-rewrite` control flow, with MCTS
replacing random decision choice:

```text
current_comp = TensorComputation.load_json(input).clone()
start_from = 0

for step in range(max_steps):
    root = SearchNode(current_comp.clone(), start_from)
    result = run_sampled_puct(root, params, simulations, actions_per_node)

    if root is terminal:
        stop

    save root trace record:
        state_snapshot
        action_space_snapshot
        sampled_actions
        visit_distribution
        state_log_flops

    chosen_action = sample_from_visit_counts(result.visit_counts, temperature)
    current_comp.apply_decision_with_space(root.action_space, chosen_action.decision)
    start_from = root.action_space.def_index

final_log_flops = current_comp.log_total_flops()
complete value targets for the episode trace
append completed items to replay
run train_steps optimizer updates
```

The episode stops when either `next_action_space(start_from)` returns `None` at
root expansion or `max_steps` is reached.

## Replay

Replay means stored supervised training examples generated by previous search
roots. It does not mean re-running the environment.

There are two related containers:

```text
EpisodeTrace
  temporary records from the current episode
  lacks value targets until final_log_flops is known

ReplayBuffer
  completed records from finished episodes
  stores policy targets and value targets
  sampled for optimizer minibatches
```

One root search produces one trace record:

```text
input:
  state_snapshot
  action_space_snapshot
  sampled_actions

policy target:
  MCTS visit distribution over sampled_actions

state scalar:
  state_log_flops
```

After the episode finishes:

```text
value_target = state_log_flops - final_log_flops
```

Completed records are appended to an in-memory replay buffer with fixed
capacity. When full, the buffer evicts the oldest records.

For v1, replay is in-memory only. Persistence, prioritized replay,
multi-process self-play, and large-scale sampling are out of scope.

## Features

Feature extraction is Python-owned and snapshot-based.

Initial state features:

- number of definitions, tensors, and ranges
- `start_from`
- current `log_total_flops`
- active definition term count
- active definition factor-count summaries
- active definition sum-index-count summaries
- coefficient sign and magnitude summaries

Initial candidate and term features:

- candidate left-term count
- candidate right-term count
- rewritten-definition term count
- per-term factor count
- per-term sum-index count
- per-term coefficient sign and magnitude summaries

Feature extraction pads candidates and terms to configured maxima and returns
explicit masks. Padded rows are ignored by model heads and losses.

Shape overflow is handled by truncating to configured maxima and masking. The
runner should report truncation counts in debug metrics so small defaults do not
silently hide too much action-space structure.

## Model

Use Flax NNX and Optax.

The v1 model is a small structured MLP policy/value network:

```text
StateEncoder:
  MLP over global state features

CandidateEncoder:
  shared MLP over candidate feature rows

TermEncoder:
  shared MLP over left/right term feature rows

Heads:
  candidate logits over candidate rows
  left mask bit logits conditioned on selected candidate
  right mask bit logits conditioned on selected candidate and left-mask summary
  scalar value head
```

The model predicts:

- priors for complete sampled actions, by multiplying autoregressive
  probabilities
- scalar value for the current state

The value head predicts expected remaining log-cost improvement from the current
state, on the same scale as:

```text
state_log_flops - final_log_flops
```

## Autoregressive Action Sampling

Python treats one decision as one complete macro-action:

```python
SampledAction(
    decision={
        "candidate_index": int,
        "left_mask": list[bool],
        "right_mask": list[bool],
    },
    prior=float,
)
```

Sampling uses the model autoregressively:

```text
sample candidate_index
sample nonempty left_mask conditioned on selected candidate
sample nonempty right_mask conditioned on selected candidate and left_mask summary
validate through the PyO3/Rust apply path on a clone
deduplicate complete decisions
repeat until K valid unique actions or sample_attempts is reached
```

The default node action budget is fixed:

```text
actions_per_node = K
sample_attempts = cap
```

If too few valid unique actions are found, search proceeds with the valid set.
If zero valid actions are found, the node is treated as terminal for search and
action selection.

Deterministic fallbacks should exist for tests and debugging:

- first candidate with full masks
- uniform random candidate with nonempty random masks

## Sampled PUCT

PUCT stores complete sampled actions as children. It does not search over
candidate/mask substeps separately.

Selection score:

```text
score = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
```

Each simulation:

1. starts at the root
2. selects child actions by PUCT
3. clones the parent node's `TensorComputation`
4. applies the selected decision through the parent node's stored
   `ActionSpace`
5. expands/evaluates the reached leaf
6. backs up the value through selected edges

Nonterminal leaves use the NNX value head. Terminal leaves use the exact
observed improvement from the state where the simulation started to the
terminal leaf:

```text
simulation_value = simulation_start_log_flops - terminal_leaf_log_flops
```

That keeps search backups on the same scale as replay value targets:

```text
state_log_flops - final_log_flops
```

At root action selection time, the real rewrite action is sampled from visit
counts with temperature. A temperature of `0` means argmax and is available for
debugging and evaluation.

## Training

After each completed episode:

```text
final_log_flops = current_comp.log_total_flops()
for each trace item:
    value_target = item.state_log_flops - final_log_flops
    replay.append(completed_item)

for train_step in range(train_steps):
    batch = replay.sample(batch_size)
    loss = policy_loss + value_loss
    apply optax update to NNX model parameters
```

Policy loss is cross entropy over the sampled actions stored in the replay
item. The log probability of a complete action is:

```text
log p(candidate)
+ log p(left_mask | candidate)
+ log p(right_mask | candidate, left_mask)
```

Value loss is mean squared error against the stored value target.

The first implementation should prove that at least one trainable parameter
changes during a training step.

## CLI

Target command:

```text
python -m gristmill_rl.train --input <path.json> --episodes 2 --simulations 8
```

Useful defaults:

```text
--episodes 2
--max-steps 4
--simulations 8
--actions-per-node 8
--sample-attempts 64
--train-steps 1
--batch-size 4
--replay-capacity 256
--temperature 1.0
--c-puct 1.5
--seed 0
```

Metrics printed by the runner:

- episode index
- episode length
- initial log FLOPs
- final log FLOPs
- per-root valid sampled action counts
- replay size
- policy loss
- value loss
- total loss
- whether a parameter update changed model parameters

## Error Handling

V1 error handling is direct and fail-fast:

- malformed input JSON or invalid symbolic representation raises through
  `TensorComputation.load_json`
- cost errors fail the runner with a clear message
- invalid sampled decisions are discarded during action sampling
- too few valid sampled actions are allowed
- zero valid sampled actions makes the node terminal for search
- feature truncation is allowed but counted and reported

The RL module should not catch broad exceptions around training. Unexpected
shape or model errors should fail tests and the CLI.

## Testing

Required Python tests:

- feature extraction produces stable padded shapes from an actionable fixture
- masks correctly mark candidate and term padding
- deterministic action fallback produces a valid decision
- action sampler deduplicates decisions
- sampled decisions can be applied through the stored `ActionSpace`
- search node expansion calls `next_action_space` once
- child transitions use the parent node's stored `ActionSpace`
- root visit distribution sums to `1`
- episode trace converts to value targets after `final_log_flops`
- replay capacity evicts oldest items
- NNX forward pass returns candidate/mask logits and value
- one training step changes at least one parameter
- the CLI completes a tiny run and prints cost/search/loss metrics

Rust and PyO3 tests already cover the fixed boundary and should not need to
change for this slice.

## Acceptance Criteria

The RL module v1 is complete when:

- `python/gristmill_rl/` exists with the package shape above
- the package uses `TensorComputation` and `ActionSpace` directly
- each search node computes and stores its action space at most once
- decisions are applied through the exact stored `ActionSpace` that produced
  them
- replay records store action-space snapshots and sampled actions together
- the model uses Flax NNX and Optax
- sampled PUCT can run with fixed `actions_per_node`
- the runner completes a short command on a JSON input
- at least one optimizer update changes model parameters
- metrics include cost, valid sampled action counts, replay size, and losses

The prototype is not expected to beat random or greedy rewrite baselines in
v1.

## Out Of Scope

- Rust or PyO3 API changes
- an `RlEnv` wrapper class
- cross-episode action-space caching
- stable global candidate IDs
- canonical state/action fingerprints
- full action enumeration
- graph neural networks or transformers
- replay persistence
- prioritized replay
- multiprocessing or distributed self-play
- benchmark-quality optimization results
