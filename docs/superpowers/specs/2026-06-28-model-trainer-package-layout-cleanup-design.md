# Model And Trainer Package Layout Cleanup Design

## Goal

Restructure the Python package so the model and trainer abstractions are visible
in the filesystem and independent in code.

The target architecture is:

- `gristmill_symbolics.model` defines the expression-model protocol and owns
  concrete model implementations.
- `gristmill_symbolics.trainer` defines the trainer protocol and owns concrete
  trainer implementations.
- `gristmill_symbolics.cli` composes a concrete model and concrete trainer for
  training, checkpointing, and command-line execution.

The current `policy` package becomes internals of the concrete transformer action
selector model. The current `reinforce` package is split into trainer logic and
CLI/orchestration logic.

## Motivation

The previous refactor introduced protocols, but the package layout still makes
the boundaries hard to see:

- `reinforce/model.py` contains the concrete expression model.
- `reinforce/trainer.py` contains the concrete trainer.
- `policy/` contains the neural policy internals used only by that model.
- `train_state.py` uses an adapter to pass model configuration through a trainer
  call that should not know model-specific configuration exists.

The cleaned layout should make these facts obvious:

- A model chooses from an input action space and provides log-probability
  gradients.
- A trainer updates parameters by calling only the model protocol.
- CLI/checkpoint code wires concrete model and trainer objects together.

## Target Layout

```text
python/gristmill_symbolics/
  model/
    __init__.py
    protocols.py

    transformer_action_selector/
      __init__.py
      model.py
      rollout.py
      api.py
      batched.py
      constants.py
      tokenize.py
      tree.py
      types.py

  trainer/
    __init__.py
    protocols.py

    reinforce/
      __init__.py
      trainer.py
      objective.py

  cli/
    __init__.py
    train.py
    checkpoint.py
    train_state.py
```

The old `python/gristmill_symbolics/policy/` package should be removed after its
files move into `model/transformer_action_selector/`.

The old `python/gristmill_symbolics/reinforce/` package should be removed or
reduced to a temporary compatibility shim only if needed during migration. The
final checkpoint should not keep two supported training APIs.

## Protocols

`model/protocols.py` defines the model protocol:

```python
from collections.abc import Mapping
from typing import Protocol


class ExpressionModel(Protocol):
    @property
    def batch_size(self) -> int:
        ...

    def init_params(self, rng) -> object:
        ...

    def sample_with_logp_grad(
        self,
        params,
        rng,
        row,
    ) -> tuple[object, object, object, Mapping[str, object]]:
        ...
```

`trainer/protocols.py` defines the trainer protocol:

```python
from collections.abc import Mapping, Sequence
from typing import Protocol

from gristmill_symbolics.model.protocols import ExpressionModel


class Trainer(Protocol):
    @property
    def batch_size(self) -> int:
        ...

    def init_opt_state(self, params) -> object:
        ...

    def update(
        self,
        params,
        opt_state,
        batch: Sequence[object],
        model: ExpressionModel,
        rng,
    ) -> tuple[object, object, Mapping[str, object]]:
        ...
```

The protocol calls do not accept model or trainer config objects. Concrete
instances own their own settings, parameter initialization, and optimizer-state
initialization.

## Concrete Model

The current policy implementation becomes:

```text
gristmill_symbolics.model.transformer_action_selector
```

It exports:

```python
TransformerActionSelectorModel
```

The concrete model follows `ExpressionModel`:

```python
class TransformerActionSelectorModel(ExpressionModel):
    def __init__(
        self,
        *,
        batch_size: int,
        max_steps: int,
        state_token_pad_to: int,
        action_token_pad_to: int,
        definition_pad_to: int,
        d_model: int = 32,
        num_attention_layers: int = 1,
        id_vocab_size: int = 128,
        init_scale: float = 0.02,
        stop_bias_init: float = -20.0,
    ):
        ...

    @property
    def batch_size(self) -> int:
        ...

    def init_params(self, rng):
        ...

    def sample_with_logp_grad(self, params, rng, row):
        ...
```

Constructor settings replace the current `CurrentTransformerModelConfig`
and `PolicyConfig` public dataclasses. If the implementation wants an internal
private settings object, it may use one, but the public construction API is the
model constructor.

Model responsibilities:

- tokenize rewrite states and action spaces;
- run transformer-backed target/action sampling;
- compute trajectory log probabilities and per-sample log-probability gradients;
- own static rollout shape settings such as batch size and pad widths;
- return only model-level diagnostics in `metrics`.

Model non-responsibilities:

- reward computation;
- advantage computation;
- optimizer updates;
- checkpoint orchestration;
- CLI argument parsing.

## Concrete Trainer

The REINFORCE implementation becomes:

```text
gristmill_symbolics.trainer.reinforce
```

It exports:

```python
ReinforceTrainer
```

The concrete trainer follows `Trainer`:

```python
class ReinforceTrainer(Trainer):
    def __init__(
        self,
        *,
        batch_size: int,
        learning_rate: float = 1.0e-3,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1.0e-8,
        reward_kind: str = "log_flops_improvement",
        standardize_baseline: bool = False,
        baseline_epsilon: float = 1.0e-8,
    ):
        ...

    @property
    def batch_size(self) -> int:
        ...

    def init_opt_state(self, params):
        ...

    def update(self, params, opt_state, batch, model, rng):
        ...
```

Constructor settings replace the current `ReinforceTrainerConfig`,
`OptimizerConfig`, `RewardConfig`, and `BaselineConfig` public dataclasses. If the
implementation wants internal private settings objects, it may use them, but the
public construction API is the trainer constructor.

Trainer responsibilities:

- validate batch length;
- convert batch states to a row-like object before calling the model protocol;
- call only `model.sample_with_logp_grad(params, rng, row)`;
- validate returned `logp` and `grad_logp` shapes;
- compute reward and advantage;
- compute REINFORCE gradients and surrogate loss diagnostics;
- apply optimizer updates;
- return compact trainer metrics.

Trainer non-responsibilities:

- model-specific config;
- tokenization;
- action-space sampling internals;
- checkpoint schema;
- CLI argument parsing.

## CLI And Checkpoint Orchestration

The CLI package owns composition:

```text
gristmill_symbolics.cli
```

`cli/train.py` constructs:

```python
model = TransformerActionSelectorModel(...)
trainer = ReinforceTrainer(...)
```

Then it calls:

```python
train_state, metrics = advance_train_state(
    train_state,
    initial_states,
    model=model,
    trainer=trainer,
)
```

`cli/train_state.py` owns:

```python
TrainState(params, opt_state, root_key, update_index)
UpdateMetrics(...)
init_train_state(...)
advance_train_state(...)
```

`advance_train_state` should not contain a model config adapter. Its call should
be direct:

```python
new_params, new_opt_state, trainer_metrics = trainer.update(
    state.params,
    state.opt_state,
    list(initial_states),
    model,
    rng,
)
```

Initial state creation asks the objects to initialize their owned state:

```python
params = model.init_params(params_key)
opt_state = trainer.init_opt_state(params)
```

`cli/checkpoint.py` owns serialization. Since model and trainer settings are
stored on concrete instances rather than in public config dataclasses,
checkpoints store constructor dictionaries:

```python
{
    "schema_version": 3,
    "model": {
        "kind": "transformer_action_selector",
        "kwargs": {
            "d_model": 8,
            "num_attention_layers": 1,
            "id_vocab_size": 128,
            "init_scale": 0.02,
            "stop_bias_init": -20.0,
            "batch_size": 2,
            "max_steps": 64,
            "state_token_pad_to": 3072,
            "action_token_pad_to": 4096,
            "definition_pad_to": 128,
        },
    },
    "trainer": {
        "kind": "reinforce",
        "kwargs": {
            "batch_size": 2,
            "learning_rate": 0.001,
            "b1": 0.9,
            "b2": 0.999,
            "eps": 1.0e-8,
            "reward_kind": "log_flops_improvement",
            "standardize_baseline": False,
            "baseline_epsilon": 1.0e-8,
        },
    },
    "policy_params": ...,
    "optimizer_state": ...,
    "root_key": ...,
    "update_index": ...,
    "recent_metrics": ...,
}
```

Checkpoint loading reconstructs the concrete objects from `kind` and `kwargs`.
The initial cleanup only needs to support the two concrete kinds above.

## Dependency Rules

Allowed:

```text
trainer.reinforce -> trainer.protocols
trainer.reinforce -> model.protocols
trainer.reinforce -> core RewriteStateRow binding

model.transformer_action_selector -> model.protocols
model.transformer_action_selector -> core RewriteStateRow binding

cli -> model concrete implementation
cli -> trainer concrete implementation
cli -> checkpoint/train state utilities
```

Forbidden:

```text
model.* -> trainer.*
trainer.reinforce -> model.transformer_action_selector
trainer.reinforce -> cli.*
model.transformer_action_selector -> cli.*
```

The model and trainer concrete classes must be independently importable.

## Public API

The final public imports should prefer:

```python
from gristmill_symbolics.model import ExpressionModel
from gristmill_symbolics.model.transformer_action_selector import (
    TransformerActionSelectorModel,
)
from gristmill_symbolics.trainer import Trainer
from gristmill_symbolics.trainer.reinforce import (
    ReinforceTrainer,
)
```

CLI/checkpoint APIs should be under:

```python
from gristmill_symbolics.cli.train_state import init_train_state, advance_train_state
from gristmill_symbolics.cli.checkpoint import save_checkpoint, load_checkpoint
```

The old `gristmill_symbolics.policy` and `gristmill_symbolics.reinforce` import
paths should not be retained as supported public APIs after the cleanup.

## Testing Strategy

Tests should move with ownership:

```text
tests/model/transformer_action_selector/
  test_api.py
  test_batched.py
  test_model.py
  test_rollout.py
  test_tokenize.py
  test_tree.py

tests/trainer/reinforce/
  test_trainer.py
  test_objective.py

tests/cli/
  test_train.py
  test_checkpoint.py
```

The migration should preserve existing behavior through focused tests:

- policy/tokenization tests pass after moving under the model package;
- scalar-oracle rollout coverage remains;
- trainer protocol tests verify trainer works with a fake model that satisfies
  `ExpressionModel`;
- model tests verify model works without importing trainer;
- CLI tests verify composition, checkpoint round-trip, and compact metrics;
- import tests verify old public paths are gone or explicitly unsupported.

## Non-Goals

- No symbolic rewrite semantic changes.
- No tokenizer or padding performance refactor.
- No model architecture change.
- No trainer algorithm change.
- No new registry/plugin system beyond the simple checkpoint `kind` dispatch.
- No compatibility migration for checkpoint schema versions older than the
  current development checkpoint unless explicitly requested later.

## Acceptance Criteria

- The filesystem exposes `model`, `trainer`, and `cli` as top-level package
  boundaries.
- `TransformerActionSelectorModel` follows `ExpressionModel`.
- `ReinforceTrainer` follows `Trainer`.
- Model and trainer concrete classes have no concrete cross-imports.
- Model/trainer settings are constructor-owned; no public
  `TransformerActionSelectorConfig`, `PolicyConfig`, `ReinforceTrainerConfig`,
  `OptimizerConfig`, `RewardConfig`, or `BaselineConfig` dataclasses remain.
- `_ConfiguredModel` is removed.
- `gristmill_symbolics.policy` is removed.
- `gristmill_symbolics.reinforce` is removed or reduced to a non-supported
  temporary shim, with tests enforcing the final supported imports.
- Full Python and Rust test suites pass.
