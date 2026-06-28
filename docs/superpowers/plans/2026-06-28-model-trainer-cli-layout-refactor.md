# Model Trainer CLI Layout Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the current policy and reinforce code into explicit `model`, `trainer`, and `cli` package boundaries, with constructor-owned concrete model/trainer instances and no supported legacy training path.

**Architecture:** Use a staged package migration. First add failing tests for the new import surface and config-free protocols, then move policy internals under `model/transformer_action_selector`, move REINFORCE update logic under `trainer/reinforce`, move orchestration/checkpointing under `cli`, prove deterministic equivalence with the current protocol path, and finally remove `gristmill_symbolics.policy` and `gristmill_symbolics.reinforce` as supported APIs.

**Tech Stack:** Python 3.11, JAX, NumPy, Optax, PyO3 `RewriteState` / `RewriteStateRow`, pytest, uv, Rust cargo tests.

---

## Worktree And Baseline

Use the existing isolated worktree:

```bash
cd /Users/longli/rcode/gristmill-symbolics/.worktrees/refactor/model-trainer-protocols
git branch --show-current
git log --oneline -1
git status --short
```

Expected:

```text
refactor/model-trainer-protocols
8b4017d docs: design model trainer package cleanup
```

`git status --short` should be empty before implementation starts.

Important PR state:

- Draft PR: `https://github.com/Br0kenSmi1e/gristmill-rl/pull/25`
- Commit `8b4017d docs: design model trainer package cleanup` is local and should be pushed to PR #25 before implementation work is published.

Push command, if network access is available:

```bash
git -c http.version=HTTP/1.1 push -u https://github.com/Br0kenSmi1e/gristmill-rl.git refactor/model-trainer-protocols
```

Baseline already verified in this worktree before writing this plan:

```bash
cd /Users/longli/rcode/gristmill-symbolics/.worktrees/refactor/model-trainer-protocols/python
uv run pytest -q
```

Result:

```text
186 passed in 77.85s
```

Rust baseline:

```bash
cd /Users/longli/rcode/gristmill-symbolics/.worktrees/refactor/model-trainer-protocols
cargo test
```

Result: all Rust unit, integration, and doc tests passed.

## Accepted Spec

Read before implementation:

```bash
sed -n '1,520p' docs/superpowers/specs/2026-06-28-model-trainer-package-layout-cleanup-design.md
```

Critical accepted decisions:

- `gristmill_symbolics.model.protocols.ExpressionModel` is the model protocol.
- `gristmill_symbolics.trainer.protocols.Trainer` is the trainer protocol.
- Concrete model package: `gristmill_symbolics.model.transformer_action_selector`.
- Concrete model class: `TransformerActionSelectorModel`.
- Concrete trainer package: `gristmill_symbolics.trainer.reinforce`.
- Concrete trainer class: `ReinforceTrainer`.
- `cli` composes concrete model and trainer instances.
- Protocol calls are config-free:

```python
params = model.init_params(rng)
out_row, logp, grad_logp, metrics = model.sample_with_logp_grad(params, rng, row)

opt_state = trainer.init_opt_state(params)
new_params, new_opt_state, metrics = trainer.update(
    params,
    opt_state,
    batch,
    model,
    rng,
)
```

- No public `PolicyConfig`, `CurrentTransformerModelConfig`, `ReinforceTrainerConfig`, `OptimizerConfig`, `RewardConfig`, or `BaselineConfig` dataclasses remain.
- `_ConfiguredModel` is removed.
- Final supported imports do not include `gristmill_symbolics.policy` or `gristmill_symbolics.reinforce`.
- The final supported training workflow is `trainer.update(...) -> model.sample_with_logp_grad(...)`.

## Files

Create:

- `python/gristmill_symbolics/model/__init__.py` - public model protocol export.
- `python/gristmill_symbolics/model/protocols.py` - config-free `ExpressionModel` protocol.
- `python/gristmill_symbolics/model/transformer_action_selector/__init__.py` - public concrete model export.
- `python/gristmill_symbolics/model/transformer_action_selector/api.py` - moved policy target/action scoring API.
- `python/gristmill_symbolics/model/transformer_action_selector/batched.py` - moved batched JAX wrappers.
- `python/gristmill_symbolics/model/transformer_action_selector/constants.py` - moved token constants.
- `python/gristmill_symbolics/model/transformer_action_selector/model.py` - moved transformer parameter/encoding helpers plus `TransformerActionSelectorModel`.
- `python/gristmill_symbolics/model/transformer_action_selector/rollout.py` - moved static rollout helper from current `reinforce/rollout.py`.
- `python/gristmill_symbolics/model/transformer_action_selector/tokenize.py` - moved tokenizer.
- `python/gristmill_symbolics/model/transformer_action_selector/tree.py` - moved token-tree helpers.
- `python/gristmill_symbolics/model/transformer_action_selector/types.py` - moved action/token type aliases and private model settings.
- `python/gristmill_symbolics/trainer/__init__.py` - public trainer protocol export.
- `python/gristmill_symbolics/trainer/protocols.py` - config-free `Trainer` protocol.
- `python/gristmill_symbolics/trainer/reinforce/__init__.py` - public concrete trainer export.
- `python/gristmill_symbolics/trainer/reinforce/objective.py` - moved reward/advantage helpers with constructor-owned trainer settings.
- `python/gristmill_symbolics/trainer/reinforce/trainer.py` - moved REINFORCE update implementation and private optimizer helper.
- `python/gristmill_symbolics/cli/__init__.py` - CLI package marker.
- `python/gristmill_symbolics/cli/checkpoint.py` - schema version 3 checkpoint save/load.
- `python/gristmill_symbolics/cli/train.py` - command-line composition of concrete model and trainer.
- `python/gristmill_symbolics/cli/train_state.py` - `TrainState`, `UpdateMetrics`, `init_train_state`, `advance_train_state`.
- `python/gristmill_symbolics/_training.py` - private shared `TrainingError` exception only, to avoid model/trainer cross-imports for errors.
- `python/tests/model/transformer_action_selector/` - migrated model tests.
- `python/tests/trainer/reinforce/` - migrated trainer tests.
- `python/tests/cli/` - migrated CLI/checkpoint/train-state tests.
- `python/tests/test_model_trainer_cli_layout.py` - package-boundary and unsupported-old-import tests.

Modify:

- `python/gristmill_symbolics/__init__.py` - keep Rust binding exports unchanged; do not re-export training APIs here.
- Existing `python/tests/test_policy_*.py`, `python/tests/test_current_transformer_model.py`, `python/tests/test_reinforce_*.py` - migrate or replace with ownership-aligned tests, then remove old files.
- `python/tests/policy_fixtures.py` - keep shared fixtures; update imports only if needed.

Remove in the final cleanup checkpoint:

- `python/gristmill_symbolics/policy/`
- `python/gristmill_symbolics/reinforce/`
- old flat tests that assert `policy` or `reinforce` public APIs.

## Dependency Rules To Enforce

Allowed:

```text
trainer.reinforce -> trainer.protocols
trainer.reinforce -> model.protocols
trainer.reinforce -> gristmill_symbolics._training
trainer.reinforce -> gristmill_symbolics RewriteStateRow binding

model.transformer_action_selector -> model.protocols
model.transformer_action_selector -> gristmill_symbolics._training
model.transformer_action_selector -> gristmill_symbolics RewriteStateRow binding

cli -> concrete model
cli -> concrete trainer
cli -> checkpoint/train_state utilities
```

Forbidden:

```text
model.* -> trainer.*
trainer.reinforce -> model.transformer_action_selector
trainer.reinforce -> cli.*
model.transformer_action_selector -> cli.*
```

## Task 1: Add Failing Boundary And Protocol Tests

**Files:**
- Create: `python/tests/test_model_trainer_cli_layout.py`
- Create: `python/tests/model/transformer_action_selector/test_model_protocol.py`
- Create: `python/tests/trainer/reinforce/test_trainer_protocol.py`
- Create: `python/tests/cli/test_train_state_protocol_composition.py`

- [ ] **Step 1: Create package boundary tests**

Create `python/tests/test_model_trainer_cli_layout.py`:

```python
import importlib

import pytest


def test_new_public_import_surface_exists():
    model_protocols = importlib.import_module("gristmill_symbolics.model.protocols")
    trainer_protocols = importlib.import_module("gristmill_symbolics.trainer.protocols")
    model_pkg = importlib.import_module(
        "gristmill_symbolics.model.transformer_action_selector"
    )
    trainer_pkg = importlib.import_module("gristmill_symbolics.trainer.reinforce")
    train_state = importlib.import_module("gristmill_symbolics.cli.train_state")
    checkpoint = importlib.import_module("gristmill_symbolics.cli.checkpoint")

    assert hasattr(model_protocols, "ExpressionModel")
    assert hasattr(trainer_protocols, "Trainer")
    assert hasattr(model_pkg, "TransformerActionSelectorModel")
    assert hasattr(trainer_pkg, "ReinforceTrainer")
    assert hasattr(train_state, "init_train_state")
    assert hasattr(train_state, "advance_train_state")
    assert hasattr(checkpoint, "save_checkpoint")
    assert hasattr(checkpoint, "load_checkpoint")


@pytest.mark.parametrize(
    "module_name",
    [
        "gristmill_symbolics.policy",
        "gristmill_symbolics.reinforce",
    ],
)
def test_old_public_training_packages_are_not_supported(module_name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


def test_no_public_config_dataclasses_are_exported():
    model_pkg = importlib.import_module(
        "gristmill_symbolics.model.transformer_action_selector"
    )
    trainer_pkg = importlib.import_module("gristmill_symbolics.trainer.reinforce")

    assert model_pkg.__all__ == ("TransformerActionSelectorModel",)
    assert trainer_pkg.__all__ == ("ReinforceTrainer",)

    forbidden = {
        "PolicyConfig",
        "CurrentTransformerModelConfig",
        "TransformerActionSelectorConfig",
        "OptimizerConfig",
        "RewardConfig",
        "BaselineConfig",
        "ReinforceTrainerConfig",
    }
    assert forbidden.isdisjoint(set(getattr(model_pkg, "__all__", ())))
    assert forbidden.isdisjoint(set(getattr(trainer_pkg, "__all__", ())))
    for name in forbidden:
        assert not hasattr(model_pkg, name)
        assert not hasattr(trainer_pkg, name)
```

- [ ] **Step 2: Create model protocol shape tests**

Create `python/tests/model/transformer_action_selector/test_model_protocol.py`:

```python
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gristmill_symbolics import RewriteState, RewriteStateRow, TensorComputation
from gristmill_symbolics.model.transformer_action_selector import (
    TransformerActionSelectorModel,
)
from gristmill_symbolics._training import TrainingError
from tests.policy_fixtures import actionable_json
from tests.test_bindings import exact_empty_json


def _state_from_json(text):
    return RewriteState.from_computation(TensorComputation.from_json_string(text))


def _floating_leaves(tree):
    return [
        leaf
        for leaf in jax.tree_util.tree_leaves(tree)
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating)
    ]


def _model(**overrides):
    values = {
        "batch_size": 2,
        "max_steps": 2,
        "state_token_pad_to": 512,
        "action_token_pad_to": 512,
        "definition_pad_to": 8,
        "d_model": 8,
        "stop_bias_init": -20.0,
    }
    values.update(overrides)
    return TransformerActionSelectorModel(**values)


def test_transformer_action_selector_model_protocol_is_config_free():
    model = _model()

    assert model.batch_size == 2
    assert list(inspect.signature(model.init_params).parameters) == ["rng"]
    assert list(inspect.signature(model.sample_with_logp_grad).parameters) == [
        "params",
        "rng",
        "row",
    ]


def test_transformer_action_selector_model_initializes_params_and_samples():
    model = _model()
    params = model.init_params(jax.random.PRNGKey(0))
    row = RewriteStateRow.from_states(
        [_state_from_json(actionable_json()), _state_from_json(exact_empty_json())]
    )

    out_row, logp, grad_logp, metrics = model.sample_with_logp_grad(
        params,
        jax.random.PRNGKey(23),
        row,
    )

    assert out_row is row
    assert logp.shape == (2,)
    assert set(metrics) == {"stopped"}
    assert metrics["stopped"].shape == (2,)
    assert metrics["stopped"].dtype == bool
    assert np.isfinite(np.asarray(logp)).all()
    for leaf in _floating_leaves(grad_logp):
        assert leaf.shape[0] == 2
        assert bool(jnp.all(jnp.isfinite(leaf)))


def test_transformer_action_selector_model_rejects_batch_size_mismatch():
    model = _model(batch_size=2)
    params = model.init_params(jax.random.PRNGKey(0))
    row = RewriteStateRow.from_states([_state_from_json(actionable_json())])

    with pytest.raises(TrainingError, match="row batch size|batch_size"):
        model.sample_with_logp_grad(params, jax.random.PRNGKey(0), row)
```

- [ ] **Step 3: Create trainer protocol tests**

Create `python/tests/trainer/reinforce/test_trainer_protocol.py`:

```python
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gristmill_symbolics import RewriteState, TensorComputation
from gristmill_symbolics.trainer.reinforce import ReinforceTrainer
from gristmill_symbolics._training import TrainingError
from tests.policy_fixtures import actionable_json
from tests.test_bindings import exact_empty_json


def _state_from_json(text):
    return RewriteState.from_computation(TensorComputation.from_json_string(text))


def _batch():
    return [_state_from_json(actionable_json()), _state_from_json(exact_empty_json())]


class FakeOutRow:
    def __init__(self, final_log_flops):
        self._final_log_flops = np.asarray(final_log_flops, dtype=np.float64)

    def log_total_flops(self):
        return self._final_log_flops


class FakeModel:
    batch_size = 2

    def __init__(self, *, final_log_flops, logp, grad_logp):
        self.final_log_flops = final_log_flops
        self.logp = logp
        self.grad_logp = grad_logp
        self.calls = []

    def sample_with_logp_grad(self, params, rng, row):
        self.calls.append((params, rng, row))
        return (
            FakeOutRow(self.final_log_flops),
            self.logp,
            self.grad_logp,
            {"stopped": np.asarray([False, True], dtype=bool)},
        )


def _simple_params():
    return {"w": jnp.asarray([1.0, -2.0], dtype=jnp.float32)}


def _zero_grad_logp(params, batch_size):
    return jax.tree_util.tree_map(
        lambda leaf: jnp.zeros((batch_size, *leaf.shape), dtype=leaf.dtype),
        params,
    )


def test_reinforce_trainer_protocol_is_config_free():
    trainer = ReinforceTrainer(batch_size=2, learning_rate=1.0e-2)

    assert trainer.batch_size == 2
    assert list(inspect.signature(trainer.init_opt_state).parameters) == ["params"]
    assert list(inspect.signature(trainer.update).parameters) == [
        "params",
        "opt_state",
        "batch",
        "model",
        "rng",
    ]


def test_reinforce_trainer_calls_model_without_config_and_updates():
    params = _simple_params()
    trainer = ReinforceTrainer(batch_size=2, learning_rate=1.0e-2)
    opt_state = trainer.init_opt_state(params)
    batch = _batch()
    initial = np.asarray([state.log_total_flops() for state in batch], dtype=np.float64)
    final = initial - np.asarray([1.0, -1.0], dtype=np.float64)
    model = FakeModel(
        final_log_flops=final,
        logp=jnp.asarray([-0.5, -1.5], dtype=jnp.float32),
        grad_logp={"w": jnp.asarray([[2.0, 0.0], [0.0, 4.0]], dtype=jnp.float32)},
    )

    new_params, new_opt_state, metrics = trainer.update(
        params,
        opt_state,
        batch,
        model,
        jax.random.PRNGKey(0),
    )

    assert len(model.calls) == 1
    assert len(model.calls[0]) == 3
    assert new_opt_state is not opt_state
    assert metrics["reward_mean"] == pytest.approx(0.0)
    assert metrics["reward_std"] == pytest.approx(1.0)
    assert metrics["objective_loss_mean"] == pytest.approx(-metrics["reward_mean"])
    assert np.isfinite(metrics["surrogate_loss"])
    assert metrics["final_flops_best"] == pytest.approx(float(np.min(final)))
    assert metrics["params_changed"] is True
    assert not jnp.array_equal(new_params["w"], params["w"])


def test_reinforce_trainer_validates_batch_length_before_model_call():
    params = _simple_params()
    trainer = ReinforceTrainer(batch_size=2, learning_rate=1.0e-2)
    model = FakeModel(
        final_log_flops=np.asarray([0.0]),
        logp=jnp.asarray([0.0], dtype=jnp.float32),
        grad_logp=_zero_grad_logp(params, 1),
    )

    with pytest.raises(TrainingError, match="batch length"):
        trainer.update(
            params,
            trainer.init_opt_state(params),
            [_state_from_json(actionable_json())],
            model,
            jax.random.PRNGKey(0),
        )
    assert model.calls == []
```

- [ ] **Step 4: Create train-state composition tests**

Create `python/tests/cli/test_train_state_protocol_composition.py`:

```python
from dataclasses import fields

import jax
import jax.numpy as jnp
import pytest

from gristmill_symbolics import RewriteState, TensorComputation
from gristmill_symbolics.cli.train_state import (
    advance_train_state,
    init_train_state,
)
from tests.policy_fixtures import actionable_json
from tests.test_bindings import exact_empty_json


def _state_from_json(text):
    return RewriteState.from_computation(TensorComputation.from_json_string(text))


def _batch():
    return [_state_from_json(actionable_json()), _state_from_json(exact_empty_json())]


class RecordingModel:
    batch_size = 2

    def __init__(self):
        self.init_rng = None

    def init_params(self, rng):
        self.init_rng = rng
        return {"w": jnp.asarray([1.0], dtype=jnp.float32)}


class RecordingTrainer:
    batch_size = 2

    def __init__(self):
        self.init_params = None
        self.calls = []

    def init_opt_state(self, params):
        self.init_params = params
        return {"step": 0}

    def update(self, params, opt_state, batch, model, rng):
        self.calls.append((params, opt_state, batch, model, rng))
        return params, opt_state, {
            "reward_mean": 1.0,
            "reward_std": 0.0,
            "objective_loss_mean": -1.0,
            "surrogate_loss": 0.25,
            "final_flops_best": 3.0,
            "params_changed": False,
        }


def test_init_train_state_asks_model_and_trainer_to_initialize_owned_state():
    model = RecordingModel()
    trainer = RecordingTrainer()

    state = init_train_state(model, trainer, seed=11)

    assert [field.name for field in fields(type(state))] == [
        "params",
        "opt_state",
        "root_key",
        "update_index",
    ]
    assert state.update_index == 0
    assert model.init_rng is not None
    assert trainer.init_params is state.params


def test_advance_train_state_calls_trainer_directly_without_adapter_or_config():
    model = RecordingModel()
    trainer = RecordingTrainer()
    state = init_train_state(model, trainer, seed=31, update_index=7)

    new_state, metrics = advance_train_state(
        state,
        _batch(),
        model=model,
        trainer=trainer,
    )

    assert len(trainer.calls) == 1
    params, opt_state, batch, called_model, rng = trainer.calls[0]
    assert params is state.params
    assert opt_state is state.opt_state
    assert len(batch) == 2
    assert called_model is model
    assert jnp.array_equal(rng, jax.random.fold_in(state.root_key, 7))
    assert new_state.root_key is state.root_key
    assert new_state.update_index == 8
    assert metrics.update_index == 7
    assert metrics.params_changed is False
```

- [ ] **Step 5: Run new tests and verify they fail**

Run:

```bash
cd /Users/longli/rcode/gristmill-symbolics/.worktrees/refactor/model-trainer-protocols/python
uv run pytest \
  tests/test_model_trainer_cli_layout.py \
  tests/model/transformer_action_selector/test_model_protocol.py \
  tests/trainer/reinforce/test_trainer_protocol.py \
  tests/cli/test_train_state_protocol_composition.py \
  -q
```

Expected: FAIL with missing `gristmill_symbolics.model`, `gristmill_symbolics.trainer`, and `gristmill_symbolics.cli` modules.

## Task 2: Create New Protocol Packages

**Files:**
- Create: `python/gristmill_symbolics/model/__init__.py`
- Create: `python/gristmill_symbolics/model/protocols.py`
- Create: `python/gristmill_symbolics/trainer/__init__.py`
- Create: `python/gristmill_symbolics/trainer/protocols.py`
- Create: `python/gristmill_symbolics/_training.py`

- [ ] **Step 1: Add private shared training exception**

Create `python/gristmill_symbolics/_training.py`:

```python
from __future__ import annotations


class TrainingError(RuntimeError):
    """Raised when model/trainer/CLI training contracts are violated."""
```

- [ ] **Step 2: Add model protocol**

Create `python/gristmill_symbolics/model/protocols.py`:

```python
from __future__ import annotations

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

- [ ] **Step 3: Add model package export**

Create `python/gristmill_symbolics/model/__init__.py`:

```python
"""Expression model protocols and implementations."""

from .protocols import ExpressionModel

__all__ = ("ExpressionModel",)
```

- [ ] **Step 4: Add trainer protocol**

Create `python/gristmill_symbolics/trainer/protocols.py`:

```python
from __future__ import annotations

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

- [ ] **Step 5: Add trainer package export**

Create `python/gristmill_symbolics/trainer/__init__.py`:

```python
"""Trainer protocols and implementations."""

from .protocols import Trainer

__all__ = ("Trainer",)
```

- [ ] **Step 6: Run protocol smoke tests**

Run:

```bash
cd /Users/longli/rcode/gristmill-symbolics/.worktrees/refactor/model-trainer-protocols/python
uv run pytest tests/test_model_trainer_cli_layout.py::test_new_public_import_surface_exists -q
```

Expected: still FAIL because concrete model, trainer, and CLI modules do not exist yet. The protocol imports should be resolved.

- [ ] **Step 7: Commit protocol package surface**

```bash
git add python/gristmill_symbolics/_training.py \
  python/gristmill_symbolics/model \
  python/gristmill_symbolics/trainer \
  python/tests/test_model_trainer_cli_layout.py \
  python/tests/model/transformer_action_selector/test_model_protocol.py \
  python/tests/trainer/reinforce/test_trainer_protocol.py \
  python/tests/cli/test_train_state_protocol_composition.py
git commit -m "test: define model trainer cli layout expectations"
```

## Task 3: Move Policy Internals Into Transformer Action Selector Model

**Files:**
- Create: `python/gristmill_symbolics/model/transformer_action_selector/`
- Move: `python/gristmill_symbolics/policy/api.py` to `python/gristmill_symbolics/model/transformer_action_selector/api.py`
- Move: `python/gristmill_symbolics/policy/batched.py` to `python/gristmill_symbolics/model/transformer_action_selector/batched.py`
- Move: `python/gristmill_symbolics/policy/constants.py` to `python/gristmill_symbolics/model/transformer_action_selector/constants.py`
- Move: `python/gristmill_symbolics/policy/model.py` to `python/gristmill_symbolics/model/transformer_action_selector/model.py`
- Move: `python/gristmill_symbolics/policy/tokenize.py` to `python/gristmill_symbolics/model/transformer_action_selector/tokenize.py`
- Move: `python/gristmill_symbolics/policy/tree.py` to `python/gristmill_symbolics/model/transformer_action_selector/tree.py`
- Move: `python/gristmill_symbolics/policy/types.py` to `python/gristmill_symbolics/model/transformer_action_selector/types.py`
- Move: `python/gristmill_symbolics/reinforce/rollout.py` to `python/gristmill_symbolics/model/transformer_action_selector/rollout.py`
- Modify: `python/gristmill_symbolics/model/transformer_action_selector/model.py`
- Modify: `python/gristmill_symbolics/model/transformer_action_selector/rollout.py`
- Create: `python/gristmill_symbolics/model/transformer_action_selector/__init__.py`

- [ ] **Step 1: Move files with history**

Run:

```bash
mkdir -p python/gristmill_symbolics/model/transformer_action_selector
git mv python/gristmill_symbolics/policy/api.py python/gristmill_symbolics/model/transformer_action_selector/api.py
git mv python/gristmill_symbolics/policy/batched.py python/gristmill_symbolics/model/transformer_action_selector/batched.py
git mv python/gristmill_symbolics/policy/constants.py python/gristmill_symbolics/model/transformer_action_selector/constants.py
git mv python/gristmill_symbolics/policy/model.py python/gristmill_symbolics/model/transformer_action_selector/model.py
git mv python/gristmill_symbolics/policy/tokenize.py python/gristmill_symbolics/model/transformer_action_selector/tokenize.py
git mv python/gristmill_symbolics/policy/tree.py python/gristmill_symbolics/model/transformer_action_selector/tree.py
git mv python/gristmill_symbolics/policy/types.py python/gristmill_symbolics/model/transformer_action_selector/types.py
git mv python/gristmill_symbolics/reinforce/rollout.py python/gristmill_symbolics/model/transformer_action_selector/rollout.py
```

- [ ] **Step 2: Privatize model settings**

In `python/gristmill_symbolics/model/transformer_action_selector/types.py`, replace public `PolicyConfig` with private `_PolicySettings`:

```python
@dataclass(frozen=True)
class _PolicySettings:
    d_model: int = 32
    num_attention_layers: int = 1
    id_vocab_size: int = 128
    init_scale: float = 0.02
    stop_bias_init: float = -20.0
```

Keep `TokenTree`, `ActionChoiceTree`, `make_action_choice`, and `action_choice_to_python` unchanged.

In `python/gristmill_symbolics/model/transformer_action_selector/model.py`, update the import and signature:

```python
from .types import _PolicySettings, TokenTree


def init_policy_params(config: _PolicySettings, rng) -> dict[str, object]:
    ...
```

- [ ] **Step 3: Add concrete model class**

Append this class to `python/gristmill_symbolics/model/transformer_action_selector/model.py` after the existing helper functions:

```python
class TransformerActionSelectorModel:
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
        self._batch_size = _positive_int("batch_size", batch_size)
        self._max_steps = _positive_int("max_steps", max_steps)
        self._state_token_pad_to = _positive_int(
            "state_token_pad_to", state_token_pad_to
        )
        self._action_token_pad_to = _positive_int(
            "action_token_pad_to", action_token_pad_to
        )
        self._definition_pad_to = _positive_int(
            "definition_pad_to", definition_pad_to
        )
        self._settings = _PolicySettings(
            d_model=_positive_int("d_model", d_model),
            num_attention_layers=_positive_int(
                "num_attention_layers", num_attention_layers
            ),
            id_vocab_size=_positive_int("id_vocab_size", id_vocab_size),
            init_scale=_finite_float("init_scale", init_scale),
            stop_bias_init=_finite_float("stop_bias_init", stop_bias_init),
        )

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @property
    def state_token_pad_to(self) -> int:
        return self._state_token_pad_to

    @property
    def action_token_pad_to(self) -> int:
        return self._action_token_pad_to

    @property
    def definition_pad_to(self) -> int:
        return self._definition_pad_to

    def constructor_kwargs(self) -> dict[str, object]:
        return {
            "batch_size": self._batch_size,
            "max_steps": self._max_steps,
            "state_token_pad_to": self._state_token_pad_to,
            "action_token_pad_to": self._action_token_pad_to,
            "definition_pad_to": self._definition_pad_to,
            "d_model": self._settings.d_model,
            "num_attention_layers": self._settings.num_attention_layers,
            "id_vocab_size": self._settings.id_vocab_size,
            "init_scale": self._settings.init_scale,
            "stop_bias_init": self._settings.stop_bias_init,
        }

    def init_params(self, rng):
        return init_policy_params(self._settings, rng)

    def sample_with_logp_grad(self, params, rng, row):
        from .rollout import _sample_static_model_rollout

        result = _sample_static_model_rollout(params, rng, row, self)
        return result.out_row, result.logp, result.grad_logp, {
            "stopped": result.stopped,
        }
```

Add these helper functions near the top of the same file:

```python
def _positive_int(name: str, value: int) -> int:
    if type(value) is not int:
        from gristmill_symbolics._training import TrainingError

        raise TrainingError(f"{name} must be an int")
    if value <= 0:
        from gristmill_symbolics._training import TrainingError

        raise TrainingError(f"{name} must be positive")
    return value


def _finite_float(name: str, value: float) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        from gristmill_symbolics._training import TrainingError

        raise TrainingError(f"{name} must be finite")
    return parsed
```

Ensure `math` is imported in this file:

```python
import math
```

- [ ] **Step 4: Update rollout imports and settings access**

In `python/gristmill_symbolics/model/transformer_action_selector/rollout.py`, replace policy imports with package-local internal imports:

```python
from gristmill_symbolics._training import TrainingError

from .batched import (
    batched_sample_action,
    batched_sample_target,
    batched_score_action_grad,
    batched_score_target_grad,
)
from .tokenize import (
    tokenize_action_space_snapshot,
    tokenize_state_snapshot,
)
from .tree import stack_token_trees
from .types import TokenTree, action_choice_to_python
```

Remove imports from `gristmill_symbolics.reinforce.types`.

Define decision constants locally in `rollout.py`:

```python
DECISION_TARGET = 0
DECISION_ACTION = 1
```

Change `_sample_static_model_rollout` to accept the model instance:

```python
def _sample_static_model_rollout(
    params,
    rng,
    row: RewriteStateRow,
    model,
) -> _StaticModelRolloutResult:
    _validate_streamed_gradient_param_dtypes(params)

    if int(row.len()) != model.batch_size:
        raise TrainingError(
            f"row batch size {row.len()} differs from batch_size {model.batch_size}"
        )

    rng_grid = _make_decision_rng_grid(
        rng,
        max_steps=model.max_steps,
        batch_size=model.batch_size,
    )
```

Replace remaining `config.` reads with `model.` reads:

```text
config.batch_size -> model.batch_size
config.max_steps -> model.max_steps
config.state_token_pad_to -> model.state_token_pad_to
config.action_token_pad_to -> model.action_token_pad_to
config.definition_pad_to -> model.definition_pad_to
```

- [ ] **Step 5: Add concrete model package export**

Create `python/gristmill_symbolics/model/transformer_action_selector/__init__.py`:

```python
"""Transformer action selector expression model."""

from .model import TransformerActionSelectorModel

__all__ = ("TransformerActionSelectorModel",)
```

This `__all__` intentionally excludes every former policy helper and every config dataclass name. Tests that need internals should import from explicit submodules such as `gristmill_symbolics.model.transformer_action_selector.api`, `...batched`, `...tokenize`, `...tree`, or `...types`.

- [ ] **Step 6: Run focused model tests**

Run:

```bash
cd /Users/longli/rcode/gristmill-symbolics/.worktrees/refactor/model-trainer-protocols/python
uv run pytest tests/model/transformer_action_selector/test_model_protocol.py -q
```

Expected: PASS for the new model protocol tests, or fail only on old imports not yet migrated outside this focused file.

- [ ] **Step 7: Commit model package migration**

```bash
git add python/gristmill_symbolics/model/transformer_action_selector \
  python/gristmill_symbolics/policy \
  python/gristmill_symbolics/reinforce/rollout.py \
  python/tests/model/transformer_action_selector/test_model_protocol.py
git commit -m "feat: add transformer action selector model package"
```

## Task 4: Move REINFORCE Trainer Into Trainer Package

**Files:**
- Move: `python/gristmill_symbolics/reinforce/objective.py` to `python/gristmill_symbolics/trainer/reinforce/objective.py`
- Move/adapt: `python/gristmill_symbolics/reinforce/trainer.py` to `python/gristmill_symbolics/trainer/reinforce/trainer.py`
- Create: `python/gristmill_symbolics/trainer/reinforce/__init__.py`
- Modify: `python/gristmill_symbolics/trainer/reinforce/objective.py`
- Modify: `python/gristmill_symbolics/trainer/reinforce/trainer.py`

- [ ] **Step 1: Move trainer files with history**

Run:

```bash
mkdir -p python/gristmill_symbolics/trainer/reinforce
git mv python/gristmill_symbolics/reinforce/objective.py python/gristmill_symbolics/trainer/reinforce/objective.py
git mv python/gristmill_symbolics/reinforce/trainer.py python/gristmill_symbolics/trainer/reinforce/trainer.py
```

- [ ] **Step 2: Replace config dataclass usage with constructor-owned settings**

In `python/gristmill_symbolics/trainer/reinforce/trainer.py`, keep existing validation helpers and update imports to:

```python
from collections.abc import Sequence
import math

import jax
import jax.numpy as jnp
import numpy as np
import optax

from gristmill_symbolics import RewriteState, RewriteStateRow
from gristmill_symbolics._training import TrainingError

from .objective import compute_advantages
```

Add private optimizer helper:

```python
def _make_optimizer(
    *,
    learning_rate: float,
    b1: float,
    b2: float,
    eps: float,
) -> optax.GradientTransformation:
    if not (math.isfinite(learning_rate) and learning_rate > 0.0):
        raise TrainingError("learning_rate must be finite and positive")
    if not (math.isfinite(b1) and 0.0 <= b1 < 1.0):
        raise TrainingError("b1 must be finite and satisfy 0.0 <= b1 < 1.0")
    if not (math.isfinite(b2) and 0.0 <= b2 < 1.0):
        raise TrainingError("b2 must be finite and satisfy 0.0 <= b2 < 1.0")
    if not (math.isfinite(eps) and eps > 0.0):
        raise TrainingError("eps must be finite and positive")
    return optax.adam(learning_rate=learning_rate, b1=b1, b2=b2, eps=eps)
```

Replace the existing `ReinforceTrainer` with:

```python
class ReinforceTrainer:
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
        self._batch_size = _positive_int("batch_size", batch_size)
        self._learning_rate = float(learning_rate)
        self._b1 = float(b1)
        self._b2 = float(b2)
        self._eps = float(eps)
        self._reward_kind = str(reward_kind)
        self._standardize_baseline = bool(standardize_baseline)
        self._baseline_epsilon = float(baseline_epsilon)
        self._optimizer()

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def constructor_kwargs(self) -> dict[str, object]:
        return {
            "batch_size": self._batch_size,
            "learning_rate": self._learning_rate,
            "b1": self._b1,
            "b2": self._b2,
            "eps": self._eps,
            "reward_kind": self._reward_kind,
            "standardize_baseline": self._standardize_baseline,
            "baseline_epsilon": self._baseline_epsilon,
        }

    def _optimizer(self) -> optax.GradientTransformation:
        return _make_optimizer(
            learning_rate=self._learning_rate,
            b1=self._b1,
            b2=self._b2,
            eps=self._eps,
        )

    def init_opt_state(self, params):
        return self._optimizer().init(params)

    def update(self, params, opt_state, batch: Sequence[RewriteState], model, rng):
        initial_states = list(batch)
        if len(initial_states) != self._batch_size:
            raise TrainingError(
                f"batch length {len(initial_states)} differs from "
                f"batch_size {self._batch_size}"
            )
        if getattr(model, "batch_size", self._batch_size) != self._batch_size:
            raise TrainingError("model batch_size must match trainer batch_size")

        initial_log_flops = [state.log_total_flops() for state in initial_states]
        row = RewriteStateRow.from_states(initial_states)
        out_row, logp, grad_logp, _model_metrics = model.sample_with_logp_grad(
            params,
            rng,
            row,
        )
        logp = _validate_logp(logp, self._batch_size)
        grad_logp = _validate_grad_logp(params, grad_logp, self._batch_size)

        raw_final_log_flops = out_row.log_total_flops()
        reward = _compute_reward(
            initial_log_flops,
            raw_final_log_flops,
            self._reward_kind,
        )
        final_log_flops = _as_float64_array("final_log_flops", raw_final_log_flops)
        advantage = compute_advantages(
            reward,
            standardize=self._standardize_baseline,
            epsilon=self._baseline_epsilon,
        )

        grads = _reinforce_grad_loss(grad_logp, advantage)
        surrogate_loss = _surrogate_loss(logp, advantage)
        if not bool(np.isfinite(np.asarray(surrogate_loss))):
            raise TrainingError("surrogate_loss is non-finite")

        updates, new_opt_state = self._optimizer().update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        _validate_finite_params(new_params)

        return new_params, new_opt_state, {
            "reward_mean": float(np.mean(reward)),
            "reward_std": float(np.std(reward)),
            "objective_loss_mean": float(-np.mean(reward)),
            "surrogate_loss": float(np.asarray(surrogate_loss)),
            "final_flops_best": float(np.min(final_log_flops)),
            "params_changed": _params_changed(params, new_params),
        }
```

Add `_positive_int` to this file:

```python
def _positive_int(name: str, value: int) -> int:
    if type(value) is not int:
        raise TrainingError(f"{name} must be an int")
    if value <= 0:
        raise TrainingError(f"{name} must be positive")
    return value
```

Change `_compute_reward` signature from config object to string:

```python
def _compute_reward(initial_log_flops, final_log_flops, reward_kind: str) -> np.ndarray:
    if reward_kind != "log_flops_improvement":
        raise TrainingError(f"unsupported reward kind {reward_kind!r}")
    ...
```

- [ ] **Step 3: Move advantage config into function parameters**

In `python/gristmill_symbolics/trainer/reinforce/objective.py`, replace config imports and signatures with:

```python
from __future__ import annotations

import numpy as np

from gristmill_symbolics._training import TrainingError


def compute_rewards(final_metrics: object, *, reward_kind: str) -> np.ndarray:
    if reward_kind != "log_flops_improvement":
        raise TrainingError(f"unsupported reward kind {reward_kind!r}")
    ...


def compute_advantages(
    reward: np.ndarray,
    *,
    standardize: bool = False,
    epsilon: float = 1.0e-8,
) -> np.ndarray:
    values = np.asarray(reward, dtype=np.float64)
    ...
    if standardize:
        std = np.std(advantage, dtype=np.float64)
        advantage = (advantage - np.mean(advantage, dtype=np.float64)) / (
            std + epsilon
        )
    ...
```

Keep the existing body validations for reward shape, stopped/max_steps shape, empty reward, and finite values.

- [ ] **Step 4: Add trainer package export**

Create `python/gristmill_symbolics/trainer/reinforce/__init__.py`:

```python
"""REINFORCE trainer implementation."""

from .trainer import ReinforceTrainer

__all__ = ("ReinforceTrainer",)
```

Tests and internal callers that need objective helpers should import them from `gristmill_symbolics.trainer.reinforce.objective`.

- [ ] **Step 5: Run focused trainer tests**

Run:

```bash
cd /Users/longli/rcode/gristmill-symbolics/.worktrees/refactor/model-trainer-protocols/python
uv run pytest tests/trainer/reinforce/test_trainer_protocol.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit trainer package migration**

```bash
git add python/gristmill_symbolics/trainer/reinforce \
  python/gristmill_symbolics/reinforce/objective.py \
  python/gristmill_symbolics/reinforce/trainer.py \
  python/tests/trainer/reinforce/test_trainer_protocol.py
git commit -m "feat: add reinforce trainer package"
```

## Task 5: Move Train State, Checkpoint, And CLI Composition

**Files:**
- Move/adapt: `python/gristmill_symbolics/reinforce/train_state.py` to `python/gristmill_symbolics/cli/train_state.py`
- Move/adapt: `python/gristmill_symbolics/reinforce/checkpoint.py` to `python/gristmill_symbolics/cli/checkpoint.py`
- Move/adapt: `python/gristmill_symbolics/reinforce/train.py` to `python/gristmill_symbolics/cli/train.py`
- Create: `python/gristmill_symbolics/cli/__init__.py`
- Create: `python/tests/cli/test_checkpoint.py`
- Create: `python/tests/cli/test_train.py`

- [ ] **Step 1: Move CLI files with history**

Run:

```bash
mkdir -p python/gristmill_symbolics/cli
git mv python/gristmill_symbolics/reinforce/train_state.py python/gristmill_symbolics/cli/train_state.py
git mv python/gristmill_symbolics/reinforce/checkpoint.py python/gristmill_symbolics/cli/checkpoint.py
git mv python/gristmill_symbolics/reinforce/train.py python/gristmill_symbolics/cli/train.py
```

Create `python/gristmill_symbolics/cli/__init__.py`:

```python
"""Command-line training orchestration."""
```

- [ ] **Step 2: Replace train state config path with object path**

In `python/gristmill_symbolics/cli/train_state.py`, keep `TrainState`, `UpdateMetrics`, `_params_changed`, `_validate_finite_params`, `_reinforce_grad_loss`, and `_surrogate_loss`. Remove `make_optimizer`, `_ConfiguredModel`, `train_update`, config imports, and default concrete construction.

Use this public init function:

```python
def init_train_state(
    model,
    trainer,
    *,
    seed: int,
    update_index: int = 0,
) -> TrainState:
    _validate_matching_batch_sizes(model, trainer)
    root_key = jax.random.PRNGKey(int(seed))
    params_key = jax.random.fold_in(root_key, np.uint32(0xFFFFFFFF))
    params = model.init_params(params_key)
    return TrainState(
        params=params,
        opt_state=trainer.init_opt_state(params),
        root_key=root_key,
        update_index=int(update_index),
    )
```

Use this public advance function:

```python
def advance_train_state(
    state: TrainState,
    initial_states: Sequence[RewriteState],
    *,
    model,
    trainer,
):
    _validate_matching_batch_sizes(model, trainer)
    rng = jax.random.fold_in(state.root_key, int(state.update_index))
    new_params, new_opt_state, trainer_metrics = trainer.update(
        state.params,
        state.opt_state,
        list(initial_states),
        model,
        rng,
    )
    metrics = UpdateMetrics(
        update_index=state.update_index,
        batch_size=trainer.batch_size,
        reward_mean=float(trainer_metrics["reward_mean"]),
        reward_std=float(trainer_metrics["reward_std"]),
        objective_loss_mean=float(trainer_metrics["objective_loss_mean"]),
        surrogate_loss=float(trainer_metrics["surrogate_loss"]),
        final_flops_best=float(trainer_metrics["final_flops_best"]),
        params_changed=bool(trainer_metrics["params_changed"]),
    )
    return (
        TrainState(
            params=new_params,
            opt_state=new_opt_state,
            root_key=state.root_key,
            update_index=state.update_index + 1,
        ),
        metrics,
    )
```

Add the batch-size helper:

```python
def _validate_matching_batch_sizes(model, trainer) -> None:
    if model.batch_size != trainer.batch_size:
        raise TrainingError("model batch_size must match trainer batch_size")
```

- [ ] **Step 3: Add schema version 3 checkpoint tests**

Create `python/tests/cli/test_checkpoint.py`:

```python
import pickle

import jax
import jax.numpy as jnp
import pytest

from gristmill_symbolics.cli.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointData,
    load_checkpoint,
    save_checkpoint,
)
from gristmill_symbolics.cli.train_state import UpdateMetrics, init_train_state
from gristmill_symbolics.model.transformer_action_selector import (
    TransformerActionSelectorModel,
)
from gristmill_symbolics.trainer.reinforce import ReinforceTrainer
from gristmill_symbolics._training import TrainingError


def _assert_pytrees_equal(left, right):
    assert jax.tree_util.tree_structure(left) == jax.tree_util.tree_structure(right)
    for left_leaf, right_leaf in zip(
        jax.tree_util.tree_leaves(left),
        jax.tree_util.tree_leaves(right),
        strict=True,
    ):
        if hasattr(left_leaf, "dtype") or hasattr(right_leaf, "dtype"):
            assert bool(jnp.array_equal(left_leaf, right_leaf))
        else:
            assert left_leaf == right_leaf


def _model():
    return TransformerActionSelectorModel(
        batch_size=2,
        max_steps=3,
        state_token_pad_to=256,
        action_token_pad_to=256,
        definition_pad_to=8,
        d_model=8,
        num_attention_layers=2,
        id_vocab_size=64,
        init_scale=0.03,
        stop_bias_init=-7.0,
    )


def _trainer():
    return ReinforceTrainer(
        batch_size=2,
        learning_rate=1.0e-2,
        b1=0.8,
        b2=0.95,
        eps=1.0e-5,
        standardize_baseline=True,
        baseline_epsilon=1.0e-6,
    )


def test_checkpoint_round_trips_objects_state_and_metrics(tmp_path):
    model = _model()
    trainer = _trainer()
    state = init_train_state(model, trainer, seed=13, update_index=5)
    recent_metrics = (
        UpdateMetrics(
            update_index=5,
            batch_size=2,
            reward_mean=1.5,
            reward_std=0.25,
            objective_loss_mean=-1.5,
            surrogate_loss=-0.125,
            final_flops_best=7.25,
            params_changed=True,
        ),
    )
    path = tmp_path / "checkpoint.pkl"

    save_checkpoint(
        path,
        state,
        model=model,
        trainer=trainer,
        recent_metrics=recent_metrics,
    )
    loaded = load_checkpoint(path)

    assert isinstance(loaded, CheckpointData)
    assert isinstance(loaded.model, TransformerActionSelectorModel)
    assert isinstance(loaded.trainer, ReinforceTrainer)
    assert loaded.model.constructor_kwargs() == model.constructor_kwargs()
    assert loaded.trainer.constructor_kwargs() == trainer.constructor_kwargs()
    assert loaded.train_state.update_index == state.update_index
    _assert_pytrees_equal(loaded.train_state.params, state.params)
    _assert_pytrees_equal(loaded.train_state.opt_state, state.opt_state)
    assert loaded.recent_metrics == recent_metrics

    with path.open("rb") as handle:
        payload = pickle.load(handle)
    assert payload["schema_version"] == 3
    assert payload["model"] == {
        "kind": "transformer_action_selector",
        "kwargs": model.constructor_kwargs(),
    }
    assert payload["trainer"] == {
        "kind": "reinforce",
        "kwargs": trainer.constructor_kwargs(),
    }
    assert "policy_config" not in payload
    assert "optimizer_config" not in payload
    assert "model_config" not in payload
    assert "trainer_config" not in payload


def test_checkpoint_rejects_unknown_model_kind(tmp_path):
    path = tmp_path / "bad.pkl"
    with path.open("wb") as handle:
        pickle.dump(
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "model": {"kind": "unknown", "kwargs": {}},
                "trainer": {"kind": "reinforce", "kwargs": {"batch_size": 1}},
                "policy_params": {},
                "optimizer_state": {},
                "root_key": [0, 0],
                "update_index": 0,
                "recent_metrics": (),
            },
            handle,
        )

    with pytest.raises(TrainingError, match="unknown model kind"):
        load_checkpoint(path)
```

- [ ] **Step 4: Implement checkpoint schema 3**

In `python/gristmill_symbolics/cli/checkpoint.py`, define:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
import pickle

import jax.numpy as jnp
import numpy as np

from gristmill_symbolics._training import TrainingError
from gristmill_symbolics.cli.train_state import TrainState, UpdateMetrics
from gristmill_symbolics.model.transformer_action_selector import (
    TransformerActionSelectorModel,
)
from gristmill_symbolics.trainer.reinforce import ReinforceTrainer

CHECKPOINT_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class CheckpointData:
    train_state: TrainState
    model: TransformerActionSelectorModel
    trainer: ReinforceTrainer
    recent_metrics: tuple[UpdateMetrics, ...]
```

Use these helpers:

```python
def _model_payload(model) -> dict[str, object]:
    if isinstance(model, TransformerActionSelectorModel):
        return {
            "kind": "transformer_action_selector",
            "kwargs": model.constructor_kwargs(),
        }
    raise TrainingError(f"unsupported model type {type(model).__name__}")


def _trainer_payload(trainer) -> dict[str, object]:
    if isinstance(trainer, ReinforceTrainer):
        return {"kind": "reinforce", "kwargs": trainer.constructor_kwargs()}
    raise TrainingError(f"unsupported trainer type {type(trainer).__name__}")


def _load_model(payload: dict[str, object]):
    kind = payload["kind"]
    kwargs = payload["kwargs"]
    if kind == "transformer_action_selector":
        return TransformerActionSelectorModel(**kwargs)
    raise TrainingError(f"unknown model kind {kind!r}")


def _load_trainer(payload: dict[str, object]):
    kind = payload["kind"]
    kwargs = payload["kwargs"]
    if kind == "reinforce":
        return ReinforceTrainer(**kwargs)
    raise TrainingError(f"unknown trainer kind {kind!r}")
```

Use this save function:

```python
def save_checkpoint(
    path,
    train_state: TrainState,
    *,
    model,
    trainer,
    recent_metrics: tuple[UpdateMetrics, ...],
) -> None:
    if model.batch_size != trainer.batch_size:
        raise TrainingError("model batch_size must match trainer batch_size")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model": _model_payload(model),
        "trainer": _trainer_payload(trainer),
        "policy_params": train_state.params,
        "optimizer_state": train_state.opt_state,
        "update_index": int(train_state.update_index),
        "root_key": np.asarray(train_state.root_key, dtype=np.uint32),
        "recent_metrics": tuple(asdict(metrics) for metrics in recent_metrics),
    }
    with open(path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

Use this load function:

```python
def load_checkpoint(path) -> CheckpointData:
    with open(path, "rb") as handle:
        payload = pickle.load(handle)

    if not isinstance(payload, dict):
        raise TrainingError("checkpoint payload must be a dict")
    schema_version = payload.get("schema_version")
    if schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise TrainingError(
            "unsupported checkpoint schema "
            f"{schema_version}; expected {CHECKPOINT_SCHEMA_VERSION}"
        )

    try:
        model = _load_model(payload["model"])
        trainer = _load_trainer(payload["trainer"])
        if model.batch_size != trainer.batch_size:
            raise TrainingError("model batch_size must match trainer batch_size")
        train_state = TrainState(
            params=payload["policy_params"],
            opt_state=payload["optimizer_state"],
            root_key=jnp.asarray(payload["root_key"], dtype=jnp.uint32),
            update_index=int(payload["update_index"]),
        )
        return CheckpointData(
            train_state=train_state,
            model=model,
            trainer=trainer,
            recent_metrics=tuple(
                UpdateMetrics(**metrics) for metrics in payload["recent_metrics"]
            ),
        )
    except TrainingError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise TrainingError(f"invalid checkpoint payload: {exc}") from exc
```

- [ ] **Step 5: Update CLI composition**

In `python/gristmill_symbolics/cli/train.py`, use concrete objects directly:

```python
from gristmill_symbolics.model.transformer_action_selector import (
    TransformerActionSelectorModel,
)
from gristmill_symbolics.trainer.reinforce import ReinforceTrainer

from .checkpoint import load_checkpoint, save_checkpoint
from .train_state import advance_train_state, init_train_state
```

For fresh training:

```python
model = TransformerActionSelectorModel(
    batch_size=args.batch_size,
    max_steps=args.max_steps,
    state_token_pad_to=args.state_token_pad_to,
    action_token_pad_to=args.action_token_pad_to,
    definition_pad_to=args.definition_pad_to,
    d_model=8,
)
trainer = ReinforceTrainer(
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
)
train_state = init_train_state(model, trainer, seed=args.seed)
recent_metrics = []
```

For checkpoint resume:

```python
checkpoint = load_checkpoint(args.checkpoint_in)
train_state = checkpoint.train_state
model = checkpoint.model
trainer = checkpoint.trainer
recent_metrics = list(checkpoint.recent_metrics)
```

For each update:

```python
initial_states = [
    RewriteState.from_computation(comp)
    for _ in range(trainer.batch_size)
]
train_state, metrics = advance_train_state(
    train_state,
    initial_states,
    model=model,
    trainer=trainer,
)
```

For checkpoint output:

```python
save_checkpoint(
    args.checkpoint_out,
    train_state,
    model=model,
    trainer=trainer,
    recent_metrics=tuple(recent_metrics[-10:]),
)
```

- [ ] **Step 6: Run focused CLI tests**

Run:

```bash
cd /Users/longli/rcode/gristmill-symbolics/.worktrees/refactor/model-trainer-protocols/python
uv run pytest tests/cli/test_train_state_protocol_composition.py tests/cli/test_checkpoint.py tests/cli/test_train.py -q
```

Expected: PASS after `tests/cli/test_train.py` is migrated from the current `test_reinforce_cli.py` imports and assertions to `gristmill_symbolics.cli.train` and checkpoint object assertions.

- [ ] **Step 7: Commit CLI/checkpoint migration**

```bash
git add python/gristmill_symbolics/cli \
  python/gristmill_symbolics/reinforce/train_state.py \
  python/gristmill_symbolics/reinforce/checkpoint.py \
  python/gristmill_symbolics/reinforce/train.py \
  python/tests/cli
git commit -m "feat: compose model and trainer from cli package"
```

## Task 6: Prove Deterministic Equivalence Before Removing Legacy Imports

**Files:**
- Create: `python/tests/test_model_trainer_layout_equivalence.py`

This task compares the new object-composed path against the previous current code semantics. Because the source files have moved by this point, the reference is reconstructed with the same seeds, constructor kwargs, and update path shape, not by keeping a second production workflow.

- [ ] **Step 1: Add deterministic one-update equivalence test**

Create `python/tests/test_model_trainer_layout_equivalence.py`:

```python
import jax
import jax.numpy as jnp

from gristmill_symbolics import RewriteState, TensorComputation
from gristmill_symbolics.cli.train_state import advance_train_state, init_train_state
from gristmill_symbolics.model.transformer_action_selector import (
    TransformerActionSelectorModel,
)
from gristmill_symbolics.trainer.reinforce import ReinforceTrainer
from tests.policy_fixtures import actionable_json
from tests.test_bindings import exact_empty_json


def _state_from_json(text):
    return RewriteState.from_computation(TensorComputation.from_json_string(text))


def _batch():
    return [_state_from_json(actionable_json()), _state_from_json(exact_empty_json())]


def _assert_pytrees_equal(left, right):
    assert jax.tree_util.tree_structure(left) == jax.tree_util.tree_structure(right)
    for left_leaf, right_leaf in zip(
        jax.tree_util.tree_leaves(left),
        jax.tree_util.tree_leaves(right),
        strict=True,
    ):
        if hasattr(left_leaf, "dtype") or hasattr(right_leaf, "dtype"):
            assert bool(jnp.array_equal(left_leaf, right_leaf))
        else:
            assert left_leaf == right_leaf


def test_object_composed_training_is_deterministic_for_same_seed_and_kwargs():
    model_kwargs = {
        "batch_size": 2,
        "max_steps": 2,
        "state_token_pad_to": 512,
        "action_token_pad_to": 512,
        "definition_pad_to": 8,
        "d_model": 8,
        "stop_bias_init": -20.0,
    }
    trainer_kwargs = {"batch_size": 2, "learning_rate": 1.0e-2}

    left_model = TransformerActionSelectorModel(**model_kwargs)
    left_trainer = ReinforceTrainer(**trainer_kwargs)
    right_model = TransformerActionSelectorModel(**model_kwargs)
    right_trainer = ReinforceTrainer(**trainer_kwargs)

    left_state = init_train_state(left_model, left_trainer, seed=29)
    right_state = init_train_state(right_model, right_trainer, seed=29)

    left_next, left_metrics = advance_train_state(
        left_state,
        _batch(),
        model=left_model,
        trainer=left_trainer,
    )
    right_next, right_metrics = advance_train_state(
        right_state,
        _batch(),
        model=right_model,
        trainer=right_trainer,
    )

    _assert_pytrees_equal(left_next.params, right_next.params)
    _assert_pytrees_equal(left_next.opt_state, right_next.opt_state)
    assert left_next.update_index == right_next.update_index == 1
    assert left_metrics == right_metrics
```

- [ ] **Step 2: Add scalar-oracle rollout equivalence migration**

Migrate the existing scalar-oracle assertions from `python/tests/test_reinforce_streaming.py` into `python/tests/model/transformer_action_selector/test_rollout.py`.

Change imports to:

```python
from gristmill_symbolics.model.transformer_action_selector import (
    TransformerActionSelectorModel,
)
from gristmill_symbolics.model.transformer_action_selector.api import (
    sample_action,
    sample_target,
    score_action,
    score_target,
)
from gristmill_symbolics.model.transformer_action_selector.rollout import (
    DECISION_ACTION,
    DECISION_TARGET,
    _dummy_action_policy_item,
    _dummy_state_policy_item,
    _make_decision_rng_grid,
    _mask_tree_rows,
    _sample_static_model_rollout,
    _stack_bool_masks,
)
from gristmill_symbolics.model.transformer_action_selector.tokenize import (
    tokenize_action_space_snapshot,
    tokenize_state_snapshot,
)
from gristmill_symbolics.model.transformer_action_selector.types import (
    action_choice_to_python,
)
from gristmill_symbolics.cli.train_state import (
    _reinforce_grad_loss,
    _surrogate_loss,
)
from gristmill_symbolics._training import TrainingError
```

Replace `_model_config()` with:

```python
def _model(**overrides):
    values = {
        "batch_size": 1,
        "max_steps": 1,
        "state_token_pad_to": 512,
        "action_token_pad_to": 512,
        "definition_pad_to": 8,
        "d_model": 8,
        "stop_bias_init": -20.0,
    }
    values.update(overrides)
    return TransformerActionSelectorModel(**values)


def _params(model):
    return model.init_params(jax.random.PRNGKey(0))
```

Update direct rollout calls from:

```python
result = _sample_static_model_rollout(params, rng, row, config)
```

to:

```python
result = _sample_static_model_rollout(params, rng, row, model)
```

Update `config.batch_size`, `config.max_steps`, and pad-width reads in tests to the equivalent model properties.

- [ ] **Step 3: Run equivalence and rollout tests**

Run:

```bash
cd /Users/longli/rcode/gristmill-symbolics/.worktrees/refactor/model-trainer-protocols/python
uv run pytest \
  tests/test_model_trainer_layout_equivalence.py \
  tests/model/transformer_action_selector/test_rollout.py \
  -q
```

Expected: PASS.

- [ ] **Step 4: Commit equivalence coverage**

```bash
git add python/tests/test_model_trainer_layout_equivalence.py \
  python/tests/model/transformer_action_selector/test_rollout.py \
  python/tests/test_reinforce_streaming.py
git commit -m "test: prove model trainer layout equivalence"
```

## Task 7: Migrate Remaining Tests To New Ownership

**Files:**
- Move/update policy tests into `python/tests/model/transformer_action_selector/`
- Move/update reinforce objective tests into `python/tests/trainer/reinforce/`
- Move/update reinforce train/checkpoint/CLI tests into `python/tests/cli/`
- Remove old flat test files after migrated equivalents pass.

- [ ] **Step 1: Move model-owned tests**

Run:

```bash
mkdir -p python/tests/model/transformer_action_selector
git mv python/tests/test_policy_action.py python/tests/model/transformer_action_selector/test_action.py
git mv python/tests/test_policy_batched.py python/tests/model/transformer_action_selector/test_batched.py
git mv python/tests/test_policy_jit_grad.py python/tests/model/transformer_action_selector/test_jit_grad.py
git mv python/tests/test_policy_model.py python/tests/model/transformer_action_selector/test_transformer_params.py
git mv python/tests/test_policy_package.py python/tests/model/transformer_action_selector/test_package.py
git mv python/tests/test_policy_target.py python/tests/model/transformer_action_selector/test_target.py
git mv python/tests/test_policy_tokenize_action.py python/tests/model/transformer_action_selector/test_tokenize_action.py
git mv python/tests/test_policy_tokenize_state.py python/tests/model/transformer_action_selector/test_tokenize_state.py
git mv python/tests/test_policy_tree.py python/tests/model/transformer_action_selector/test_tree.py
git mv python/tests/test_policy_vmap.py python/tests/model/transformer_action_selector/test_vmap.py
git mv python/tests/test_current_transformer_model.py python/tests/model/transformer_action_selector/test_model.py
git mv python/tests/test_reinforce_rollout.py python/tests/model/transformer_action_selector/test_rng_grid.py
```

Update imports in these files:

```text
gristmill_symbolics.policy import TransformerActionSelectorModel -> gristmill_symbolics.model.transformer_action_selector
gristmill_symbolics.policy import low-level helpers -> explicit gristmill_symbolics.model.transformer_action_selector.<submodule>
gristmill_symbolics.policy.constants -> gristmill_symbolics.model.transformer_action_selector.constants
gristmill_symbolics.policy.model -> gristmill_symbolics.model.transformer_action_selector.model
gristmill_symbolics.policy.tokenize -> gristmill_symbolics.model.transformer_action_selector.tokenize
gristmill_symbolics.policy.tree -> gristmill_symbolics.model.transformer_action_selector.tree
CurrentTransformerModel -> TransformerActionSelectorModel
```

Replace `PolicyConfig(...)` plus `init_policy_params(...)` pairs with:

```python
model = TransformerActionSelectorModel(
    batch_size=1,
    max_steps=1,
    state_token_pad_to=512,
    action_token_pad_to=512,
    definition_pad_to=8,
    d_model=16,
)
params = model.init_params(jax.random.PRNGKey(0))
```

For low-level tests that only need parameter shapes, use any valid static shape values. The static rollout shapes do not affect `init_params`.

- [ ] **Step 2: Move trainer-owned tests**

Run:

```bash
mkdir -p python/tests/trainer/reinforce
git mv python/tests/test_reinforce_objective.py python/tests/trainer/reinforce/test_objective.py
git mv python/tests/test_reinforce_trainer_protocol.py python/tests/trainer/reinforce/test_trainer.py
git mv python/tests/test_reinforce_protocols.py python/tests/trainer/reinforce/test_protocol_exports.py
```

Update imports:

```text
gristmill_symbolics.reinforce -> gristmill_symbolics.trainer.reinforce
gristmill_symbolics.reinforce.types.TrainingError -> gristmill_symbolics._training.TrainingError
OptimizerConfig/ReinforceTrainerConfig/RewardConfig/BaselineConfig -> constructor kwargs
make_optimizer(config) -> ReinforceTrainer(...).init_opt_state(params)
```

For standardization tests, replace:

```python
ReinforceTrainerConfig(
    batch_size=2,
    optimizer_config=OptimizerConfig(learning_rate=1.0e-2),
    baseline_config=BaselineConfig(standardize=True, epsilon=1.0e-12),
)
```

with:

```python
ReinforceTrainer(
    batch_size=2,
    learning_rate=1.0e-2,
    standardize_baseline=True,
    baseline_epsilon=1.0e-12,
)
```

- [ ] **Step 3: Move CLI-owned tests**

Run:

```bash
mkdir -p python/tests/cli
git mv python/tests/test_reinforce_checkpoint.py python/tests/cli/test_checkpoint_legacy_migrated.py
git mv python/tests/test_reinforce_cli.py python/tests/cli/test_train.py
git mv python/tests/test_reinforce_train.py python/tests/cli/test_train_state.py
git mv python/tests/test_reinforce_package.py python/tests/cli/test_package.py
```

Update imports:

```text
gristmill_symbolics.reinforce.checkpoint -> gristmill_symbolics.cli.checkpoint
gristmill_symbolics.reinforce.train -> gristmill_symbolics.cli.train
gristmill_symbolics.reinforce.train_state -> gristmill_symbolics.cli.train_state
gristmill_symbolics.policy.PolicyConfig -> concrete model constructor kwargs
CurrentTransformerModelConfig -> TransformerActionSelectorModel(...)
ReinforceTrainerConfig -> ReinforceTrainer(...)
init_train_state(policy_config, optimizer_config, seed=...) -> init_train_state(model, trainer, seed=...)
advance_train_state(..., model_config=..., trainer_config=...) -> advance_train_state(..., model=model, trainer=trainer)
```

Replace checkpoint assertions:

```python
assert checkpoint.model_config.batch_size == 2
assert checkpoint.trainer_config.batch_size == 2
```

with:

```python
assert checkpoint.model.batch_size == 2
assert checkpoint.trainer.batch_size == 2
```

- [ ] **Step 4: Run migrated test directories**

Run:

```bash
cd /Users/longli/rcode/gristmill-symbolics/.worktrees/refactor/model-trainer-protocols/python
uv run pytest tests/model/transformer_action_selector tests/trainer/reinforce tests/cli tests/test_model_trainer_cli_layout.py tests/test_model_trainer_layout_equivalence.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit test migration**

```bash
git add python/tests
git commit -m "test: migrate training tests to model trainer cli layout"
```

## Task 8: Remove Legacy Policy And Reinforce Packages

**Files:**
- Delete: `python/gristmill_symbolics/policy/`
- Delete: `python/gristmill_symbolics/reinforce/`
- Modify: tests that still reference old public paths.

- [ ] **Step 1: Verify no production imports still depend on old packages**

Run:

```bash
cd /Users/longli/rcode/gristmill-symbolics/.worktrees/refactor/model-trainer-protocols
rg "gristmill_symbolics\\.(policy|reinforce)|from gristmill_symbolics import .*policy|from gristmill_symbolics import .*reinforce" python/gristmill_symbolics -n
```

Expected: no matches.

- [ ] **Step 2: Remove old packages**

Run:

```bash
git rm -r python/gristmill_symbolics/policy python/gristmill_symbolics/reinforce
```

- [ ] **Step 3: Enforce unsupported old imports and no legacy config classes**

Run:

```bash
cd /Users/longli/rcode/gristmill-symbolics/.worktrees/refactor/model-trainer-protocols
rg "gristmill_symbolics\\.(policy|reinforce)|PolicyConfig|CurrentTransformerModelConfig|CurrentTransformerModel|ReinforceTrainerConfig|OptimizerConfig|RewardConfig|BaselineConfig|_ConfiguredModel" python/gristmill_symbolics python/tests -n
```

Expected: no matches.

If test names mention old packages in historical comments, rename them. Do not leave old package imports in tests.

- [ ] **Step 4: Enforce model/trainer dependency boundaries**

Run:

```bash
cd /Users/longli/rcode/gristmill-symbolics/.worktrees/refactor/model-trainer-protocols
rg "gristmill_symbolics\\.trainer|from gristmill_symbolics\\.trainer|\\.trainer" python/gristmill_symbolics/model -n
rg "gristmill_symbolics\\.model\\.transformer_action_selector" python/gristmill_symbolics/trainer -n
rg "gristmill_symbolics\\.cli|from gristmill_symbolics\\.cli|\\.cli" python/gristmill_symbolics/model python/gristmill_symbolics/trainer -n
```

Expected: no matches.

- [ ] **Step 5: Enforce no direct trainer-to-rollout bypass names remain**

Run:

```bash
cd /Users/longli/rcode/gristmill-symbolics/.worktrees/refactor/model-trainer-protocols
rg "_collect_streamed_rollout_gradients|_StreamedRolloutResult|RolloutConfig|LossConfig|PolicyState|FinalColumnMetrics|static_policy_batch|target_score_count|action_score_count|valid_action_count|empty_action_space_count|make_rng_grid|_ConfiguredModel" python/gristmill_symbolics python/tests -n
```

Expected: no matches.

- [ ] **Step 6: Run full Python suite**

Run:

```bash
cd /Users/longli/rcode/gristmill-symbolics/.worktrees/refactor/model-trainer-protocols/python
uv run pytest -q
```

Expected: PASS.

- [ ] **Step 7: Commit legacy package removal**

```bash
git add python/gristmill_symbolics python/tests
git commit -m "refactor: remove legacy policy reinforce packages"
```

## Task 9: Final Verification And PR Update

**Files:**
- No source edits unless verification exposes a real issue.

- [ ] **Step 1: Run full Python tests**

```bash
cd /Users/longli/rcode/gristmill-symbolics/.worktrees/refactor/model-trainer-protocols/python
uv run pytest -q
```

Expected: all Python tests pass.

- [ ] **Step 2: Run Rust tests**

```bash
cd /Users/longli/rcode/gristmill-symbolics/.worktrees/refactor/model-trainer-protocols
cargo test
```

Expected: all Rust tests pass.

- [ ] **Step 3: Run final import and bypass scans**

```bash
cd /Users/longli/rcode/gristmill-symbolics/.worktrees/refactor/model-trainer-protocols
rg "gristmill_symbolics\\.(policy|reinforce)|PolicyConfig|CurrentTransformerModelConfig|CurrentTransformerModel|ReinforceTrainerConfig|OptimizerConfig|RewardConfig|BaselineConfig|_ConfiguredModel" python/gristmill_symbolics python/tests -n
rg "_collect_streamed_rollout_gradients|_StreamedRolloutResult|RolloutConfig|LossConfig|PolicyState|FinalColumnMetrics|static_policy_batch|target_score_count|action_score_count|valid_action_count|empty_action_space_count|make_rng_grid" python/gristmill_symbolics python/tests -n
rg "gristmill_symbolics\\.trainer|from gristmill_symbolics\\.trainer|\\.trainer" python/gristmill_symbolics/model -n
rg "gristmill_symbolics\\.model\\.transformer_action_selector" python/gristmill_symbolics/trainer -n
rg "gristmill_symbolics\\.cli|from gristmill_symbolics\\.cli|\\.cli" python/gristmill_symbolics/model python/gristmill_symbolics/trainer -n
```

Expected: all scans produce no matches.

- [ ] **Step 4: Review final diff**

```bash
git status --short
git log --oneline --decorate -8
git diff --stat main...HEAD
```

Expected:

- working tree is clean after final commit;
- latest commits are small and reviewable;
- diff removes old `policy/` and `reinforce/` supported packages;
- diff adds `model/`, `trainer/`, and `cli/` packages.

- [ ] **Step 5: Push branch**

If `8b4017d` and implementation commits have not been pushed yet:

```bash
git -c http.version=HTTP/1.1 push -u https://github.com/Br0kenSmi1e/gristmill-rl.git refactor/model-trainer-protocols
```

- [ ] **Step 6: Final commit if verification required small fixes**

Only if verification exposed fixes:

```bash
git add <changed-files>
git commit -m "fix: verify model trainer cli package cleanup"
```

## Acceptance Checklist

- [ ] `gristmill_symbolics.model` exists and exports `ExpressionModel`.
- [ ] `gristmill_symbolics.model.transformer_action_selector` exists and exports `TransformerActionSelectorModel`.
- [ ] `TransformerActionSelectorModel.__init__` owns all former policy/model config settings.
- [ ] `TransformerActionSelectorModel.init_params(rng)` initializes params.
- [ ] `TransformerActionSelectorModel.sample_with_logp_grad(params, rng, row)` is config-free.
- [ ] `gristmill_symbolics.trainer` exists and exports `Trainer`.
- [ ] `gristmill_symbolics.trainer.reinforce` exists and exports `ReinforceTrainer`.
- [ ] `ReinforceTrainer.__init__` owns all former optimizer/reward/baseline config settings.
- [ ] `ReinforceTrainer.init_opt_state(params)` initializes optimizer state.
- [ ] `ReinforceTrainer.update(params, opt_state, batch, model, rng)` is config-free.
- [ ] `trainer.update` calls only `model.sample_with_logp_grad`, not rollout helpers.
- [ ] `cli.train_state.advance_train_state` calls trainer directly and does not use `_ConfiguredModel`.
- [ ] checkpoint schema is version 3 and stores model/trainer constructor kwargs.
- [ ] `gristmill_symbolics.policy` is removed.
- [ ] `gristmill_symbolics.reinforce` is removed or absent as a supported API.
- [ ] Model package does not import trainer or cli packages.
- [ ] Trainer package does not import concrete model or cli packages.
- [ ] Full Python suite passes.
- [ ] Rust `cargo test` passes.
