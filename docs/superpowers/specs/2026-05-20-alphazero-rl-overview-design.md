# AlphaZero RL Overview Design

## Goal

Redesign the Python RL layer as an AlphaZero-style optimizer for symbolic tensor
rewrites.

The system keeps the existing PyO3 rewrite API as the environment boundary and
organizes the Python RL layer around three conceptual modules:

```text
Network -> MCTS -> Converter -> Network
```

The objective is state-level and replaceable:

```text
objective(TensorComputation) -> float
```

Lower objective values are better. The current objective is log FLOPs, but the
overview design must not depend on FLOP-specific decomposition.

## Existing Boundary

The redesigned RL layer continues to use the exposed PyO3 operations:

```text
TensorComputation.clone()
TensorComputation.snapshot()
TensorComputation.next_action_space(start_from)
TensorComputation.apply_decision_with_space(space, decision)
TensorComputation.log_total_flops()
ActionSpace.snapshot()
```

Future objective functions may replace `log_total_flops()` behind the objective
interface. The overview design assumes only that a complete state can be scored.

## Top-Level Shape

The RL layer has three roots:

```text
RL Layer
├── Network
├── MCTS
└── Converter
```

All other components are submodules of one of these roots.

## Network

The Network owns representation and prediction.

It consumes symbolic state and action-space information and returns guidance for
search:

```text
Network(state, action_space)
  -> proposed or scored actions
  -> value(state)
```

The Network may include encoders, action proposal heads, action scoring heads,
and value heads. Those internals are intentionally left to a Network-specific
spec.

The Network does not own tree search, replay semantics, episode control flow, or
objective bookkeeping.

## MCTS

MCTS owns search.

It uses the PyO3 environment operations plus Network guidance to build temporary
search trees. For a root state, bounded MCTS returns:

```text
root action visit distribution
root actions considered by search
best state and objective found during search
search statistics
```

MCTS may clone states, request action spaces, apply decisions, score states, and
query the Network. It does not train the Network and does not own replay storage.

## Converter

The Converter owns learning data.

It consumes MCTS results and episode traces, then builds Network training
examples:

```text
state snapshot
action-space snapshot
actions considered at the root
policy target from MCTS visits
value target from final or best-found objective
optional sample weight or priority
```

The Converter owns target construction, replay storage, batch preparation, and
metrics derived from training examples. It does not perform MCTS and does not
define the Network architecture.

## Runtime Flow

One training run repeatedly creates episodes. An episode is one actual rewrite
trajectory:

```text
initial state
  -> chosen rewrite
  -> chosen rewrite
  -> ...
  -> terminal state or episode budget
```

At each episode step:

```text
1. MCTS searches from the current state using the current Network.
2. Converter records the root state and MCTS result.
3. The episode runner chooses one real action from the root visit distribution.
4. The chosen action is applied to produce the next state.
```

After an episode ends:

```text
1. Converter finalizes policy and value targets.
2. Converter appends examples to replay.
3. Network trains from Converter-produced batches.
```

The conceptual training loop is:

```text
Network guides MCTS.
MCTS produces stronger search targets.
Converter turns search results into training data.
Training improves the Network.
```

## Design Principles

- Keep objective evaluation state-level and replaceable.
- Keep Network, MCTS, and Converter responsibilities separate.
- Treat MCTS trees as temporary search structures, not replay storage.
- Treat replay as training data extracted from search and episode traces.
- Track best-found objective values separately from the latest sampled episode
  endpoint.
- Keep overview-level interfaces stable while allowing module-specific redesigns.

## Out Of Scope

This overview does not define:

- exact hierarchical encoder structure
- exact action proposal or action scoring architecture
- exact PUCT formula
- exact tree reuse strategy
- exact replay priority formula
- exact value target formula
- exact loss functions
- checkpoint format
- monitor dashboard changes
- implementation task ordering

Those details belong in module-specific specs for Network, MCTS, and Converter.

## Acceptance Criteria

The overview design is complete when:

- the RL layer is described as three roots: Network, MCTS, and Converter
- each root has a clear ownership boundary
- the data flow between roots is explicit
- the design preserves the current PyO3 environment boundary
- the objective is defined as a replaceable state-level function
- module internals are intentionally deferred to later specs
