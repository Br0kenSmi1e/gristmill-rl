# Model Trainer Protocols Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor current REINFORCE training so the supported training workflow is `trainer.update(...) -> model.sample_with_logp_grad(...)` with unchanged current transformer rollout behavior.

**Architecture:** Use a strangler refactor. First add protocol/config surfaces and deterministic equivalence tests while the old streamed rollout remains private reference code. Then extract the static current-transformer rollout behind `CurrentTransformerModel.sample_with_logp_grad`, add a protocol-shaped `ReinforceTrainer.update`, switch CLI/checkpoint/public training orchestration to the new path, and finally remove or privatize direct trainer-to-rollout bypasses.

**Tech Stack:** Python 3.11, JAX, NumPy, Optax, PyO3 `RewriteState` / `RewriteStateRow`, pytest, uv.

---

## Baseline

Worktree created from local `main`:

```bash
/Users/longli/rcode/gristmill-symbolics/.worktrees/refactor/model-trainer-protocols
```

Branch:

```bash
refactor/model-trainer-protocols
```

Focused baseline command already run:

```bash
cd /Users/longli/rcode/gristmill-symbolics/.worktrees/refactor/model-trainer-protocols/python
uv run pytest tests/test_reinforce_streaming.py tests/test_reinforce_train.py tests/test_reinforce_checkpoint.py tests/test_reinforce_cli.py tests/test_reinforce_package.py tests/test_reinforce_rollout.py tests/test_reinforce_objective.py -q
```

Baseline result:

```text
64 passed in 63.50s
```

## Files

Create:

- `python/gristmill_symbolics/reinforce/protocols.py` - structural `ExpressionModel` and `Trainer` protocols.
- `python/gristmill_symbolics/reinforce/model.py` - current target/action transformer model adapter.
- `python/gristmill_symbolics/reinforce/trainer.py` - REINFORCE trainer protocol implementation.
- `python/tests/test_reinforce_protocols.py` - protocol/config/export tests.
- `python/tests/test_current_transformer_model.py` - model adapter and rollout-equivalence tests.
- `python/tests/test_reinforce_trainer_protocol.py` - trainer tests with fake model outputs.
- `python/tests/test_reinforce_protocol_equivalence.py` - temporary old-vs-new deterministic equivalence tests.

Modify:

- `python/gristmill_symbolics/reinforce/rollout.py` - extract static rollout body into model-private helper; keep old streamed path only until equivalence checkpoint, then remove direct legacy collector.
- `python/gristmill_symbolics/reinforce/types.py` - split model/trainer configs, simplify train state and update metrics, remove legacy direct-rollout config requirements after migration.
- `python/gristmill_symbolics/reinforce/train_state.py` - keep optimizer/init helpers, add state runner that folds update index into RNG and calls `trainer.update`; remove direct rollout call.
- `python/gristmill_symbolics/reinforce/train.py` - switch CLI to model/trainer configs and protocol runner; make static pads mandatory for current adapter.
- `python/gristmill_symbolics/reinforce/checkpoint.py` - store `schema_version`, `train_state`, `model_config`, `trainer_config`, and compact recent metrics.
- `python/gristmill_symbolics/reinforce/__init__.py` - export new supported workflow classes and remove direct streamed rollout-era exports from the public contract.
- Existing tests in `python/tests/test_reinforce_streaming.py`, `python/tests/test_reinforce_train.py`, `python/tests/test_reinforce_checkpoint.py`, `python/tests/test_reinforce_cli.py`, `python/tests/test_reinforce_package.py`, and `python/tests/test_reinforce_rollout.py` - move assertions from legacy rollout/training API to model/trainer/runner API.

No production changes should touch:

- `python/gristmill_symbolics/policy/api.py`
- `python/gristmill_symbolics/policy/batched.py`
- `python/gristmill_symbolics/policy/model.py`
- `python/gristmill_symbolics/policy/tokenize.py`

Those files are current policy semantics and tokenizer internals. They are read-only for this refactor unless a test exposes a direct incompatibility.

## Invariants

- Symbolic rewrite legality, action-space generation, validation, application, and cost remain Rust/PyO3-owned.
- The model adapter consumes a `RewriteStateRow` and returns the same final row object under normal execution.
- Static shapes are mandatory for `CurrentTransformerModel`; no dynamic rollout mode is added to the new boundary.
- `model.sample_with_logp_grad(params, rng, row, config)` returns `(out_row, logp, grad_logp, metrics)`.
- Model metrics expose only `{"stopped": np.ndarray[bool]}` at the protocol boundary.
- `trainer.update(params, opt_state, batch, model, rng, config)` never inspects tokens, masks, action choices, target choices, action spaces, or exact-empty replay internals.
- Final supported path is not allowed to call `_collect_streamed_rollout_gradients` or any replacement direct rollout collector from trainer/CLI/checkpoint code.

---

### Task 1: Add Protocol And Config Surface

**Files:**
- Create: `python/gristmill_symbolics/reinforce/protocols.py`
- Modify: `python/gristmill_symbolics/reinforce/types.py`
- Modify: `python/gristmill_symbolics/reinforce/__init__.py`
- Test: `python/tests/test_reinforce_protocols.py`

- [ ] **Step 1: Write failing protocol/config tests**

Create `python/tests/test_reinforce_protocols.py`:

```python
import pytest

import gristmill_symbolics.reinforce as reinforce
from gristmill_symbolics.policy import PolicyConfig
from gristmill_symbolics.reinforce import (
    CurrentTransformerModelConfig,
    OptimizerConfig,
    ReinforceTrainerConfig,
    TrainingError,
    validate_model_config,
    validate_trainer_config,
)


def test_current_transformer_model_config_requires_static_positive_shapes():
    config = CurrentTransformerModelConfig(
        policy_config=PolicyConfig(d_model=8),
        batch_size=2,
        max_steps=3,
        state_token_pad_to=128,
        action_token_pad_to=256,
        definition_pad_to=8,
    )

    validate_model_config(config)
    assert config.batch_size == 2
    assert config.max_steps == 3


@pytest.mark.parametrize(
    ("kwargs", "field_name"),
    [
        ({"batch_size": 0}, "batch_size"),
        ({"max_steps": 0}, "max_steps"),
        ({"state_token_pad_to": None}, "state_token_pad_to"),
        ({"action_token_pad_to": 0}, "action_token_pad_to"),
        ({"definition_pad_to": True}, "definition_pad_to"),
    ],
)
def test_current_transformer_model_config_rejects_invalid_static_shapes(
    kwargs, field_name
):
    values = {
        "policy_config": PolicyConfig(d_model=8),
        "batch_size": 1,
        "max_steps": 1,
        "state_token_pad_to": 128,
        "action_token_pad_to": 128,
        "definition_pad_to": 4,
    }
    values.update(kwargs)

    with pytest.raises(TrainingError, match=field_name):
        validate_model_config(CurrentTransformerModelConfig(**values))


def test_reinforce_trainer_config_owns_batch_reward_baseline_and_optimizer():
    config = ReinforceTrainerConfig(
        batch_size=2,
        optimizer_config=OptimizerConfig(learning_rate=1.0e-2),
    )

    validate_trainer_config(config)
    assert config.batch_size == 2
    assert config.reward_config.kind == "log_flops_improvement"
    assert config.baseline_config.standardize is False
    assert config.optimizer_config.learning_rate == pytest.approx(1.0e-2)


def test_reinforce_package_exports_protocol_boundary_names():
    expected = {
        "BaselineConfig",
        "CheckpointData",
        "CurrentTransformerModel",
        "CurrentTransformerModelConfig",
        "ExpressionModel",
        "ReinforceTrainer",
        "ReinforceTrainerConfig",
        "RewardConfig",
        "TrainState",
        "Trainer",
        "TrainingError",
        "UpdateMetrics",
        "advance_train_state",
        "compute_advantages",
        "compute_rewards",
        "init_train_state",
        "load_checkpoint",
        "make_optimizer",
        "save_checkpoint",
        "validate_model_config",
        "validate_trainer_config",
    }

    assert expected.issubset(set(reinforce.__all__))
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd python
uv run pytest tests/test_reinforce_protocols.py -q
```

Expected: FAIL with missing `CurrentTransformerModelConfig`, `ReinforceTrainerConfig`, `ExpressionModel`, `Trainer`, and `advance_train_state`.

- [ ] **Step 3: Add protocol definitions**

Create `python/gristmill_symbolics/reinforce/protocols.py`:

```python
from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol


class ExpressionModel(Protocol):
    def sample_with_logp_grad(
        self,
        params,
        rng,
        row,
        config,
    ) -> tuple[object, object, object, Mapping[str, object]]:
        ...


class Trainer(Protocol):
    def update(
        self,
        params,
        opt_state,
        batch,
        model: ExpressionModel,
        rng,
        config,
    ) -> tuple[object, object, Mapping[str, object]]:
        ...
```

- [ ] **Step 4: Add model/trainer config dataclasses in `types.py`**

In `python/gristmill_symbolics/reinforce/types.py`, keep `TrainingError`, `RewardConfig`, `BaselineConfig`, `OptimizerConfig`, `TrainState`, `UpdateMetrics`, and `CheckpointData` unchanged in this task. Add these new dataclasses after `OptimizerConfig`:

```python
@dataclass(frozen=True)
class CurrentTransformerModelConfig:
    policy_config: PolicyConfig
    batch_size: int
    max_steps: int
    state_token_pad_to: int
    action_token_pad_to: int
    definition_pad_to: int


@dataclass(frozen=True)
class ReinforceTrainerConfig:
    batch_size: int
    optimizer_config: OptimizerConfig
    reward_config: RewardConfig = RewardConfig()
    baseline_config: BaselineConfig = BaselineConfig()
```

Add validators:

```python
def _validate_positive_int(name: str, value: int) -> None:
    if type(value) is not int:
        raise TrainingError(f"{name} must be an int")
    if value <= 0:
        raise TrainingError(f"{name} must be positive")


def validate_model_config(config: CurrentTransformerModelConfig) -> None:
    if not isinstance(config.policy_config, PolicyConfig):
        raise TrainingError("policy_config must be a PolicyConfig")
    _validate_positive_int("batch_size", config.batch_size)
    _validate_positive_int("max_steps", config.max_steps)
    _validate_positive_int("state_token_pad_to", config.state_token_pad_to)
    _validate_positive_int("action_token_pad_to", config.action_token_pad_to)
    _validate_positive_int("definition_pad_to", config.definition_pad_to)


def validate_trainer_config(config: ReinforceTrainerConfig) -> None:
    _validate_positive_int("batch_size", config.batch_size)
    if not isinstance(config.optimizer_config, OptimizerConfig):
        raise TrainingError("optimizer_config must be an OptimizerConfig")
    if not isinstance(config.reward_config, RewardConfig):
        raise TrainingError("reward_config must be a RewardConfig")
    if not isinstance(config.baseline_config, BaselineConfig):
        raise TrainingError("baseline_config must be a BaselineConfig")
```

Keep `RolloutConfig`, `LossConfig`, `PolicyState`, `TrainState`, `UpdateMetrics`, `CheckpointData`, `FinalColumnMetrics`, `validate_rollout_config`, and `validate_policy_state` unchanged so existing tests and the old reference workflow still run. The protocol state/metric/checkpoint replacement happens in Task 5 after deterministic equivalence is proven.


- [ ] **Step 5: Export the new names**

Update `python/gristmill_symbolics/reinforce/__init__.py` to import and export `ExpressionModel`, `Trainer`, `CurrentTransformerModelConfig`, `ReinforceTrainerConfig`, `validate_model_config`, and `validate_trainer_config`. Add lazy wrappers for `CurrentTransformerModel`, `ReinforceTrainer`, and `advance_train_state`:

```python
def advance_train_state(*args, **kwargs):
    from .train_state import advance_train_state as _advance_train_state

    return _advance_train_state(*args, **kwargs)


def CurrentTransformerModel(*args, **kwargs):
    from .model import CurrentTransformerModel as _CurrentTransformerModel

    return _CurrentTransformerModel(*args, **kwargs)


def ReinforceTrainer(*args, **kwargs):
    from .trainer import ReinforceTrainer as _ReinforceTrainer

    return _ReinforceTrainer(*args, **kwargs)
```

- [ ] **Step 6: Run protocol tests**

Run:

```bash
cd python
uv run pytest tests/test_reinforce_protocols.py -q
```

Expected: PASS.

- [ ] **Step 7: Run existing focused baseline**

Run:

```bash
cd python
uv run pytest tests/test_reinforce_streaming.py tests/test_reinforce_train.py tests/test_reinforce_checkpoint.py tests/test_reinforce_cli.py tests/test_reinforce_package.py tests/test_reinforce_rollout.py tests/test_reinforce_objective.py tests/test_reinforce_protocols.py -q
```

Expected: PASS. Existing tests still use legacy types, and new tests cover the protocol surface.

- [ ] **Step 8: Commit**

```bash
git add python/gristmill_symbolics/reinforce/protocols.py python/gristmill_symbolics/reinforce/types.py python/gristmill_symbolics/reinforce/__init__.py python/tests/test_reinforce_protocols.py
git commit -m "feat: add reinforce model trainer protocol surface"
```

---

### Task 2: Extract Static Current Transformer Model Adapter

**Files:**
- Create: `python/gristmill_symbolics/reinforce/model.py`
- Modify: `python/gristmill_symbolics/reinforce/rollout.py`
- Modify: `python/gristmill_symbolics/reinforce/__init__.py`
- Test: `python/tests/test_current_transformer_model.py`

- [ ] **Step 1: Write failing model adapter tests**

Create `python/tests/test_current_transformer_model.py`:

```python
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gristmill_symbolics import RewriteState, RewriteStateRow, TensorComputation
from gristmill_symbolics.policy import PolicyConfig, init_policy_params
from gristmill_symbolics.reinforce import (
    CurrentTransformerModel,
    CurrentTransformerModelConfig,
    TrainingError,
)
from gristmill_symbolics.reinforce.rollout import _collect_streamed_rollout_gradients
from gristmill_symbolics.reinforce.types import PolicyState, RolloutConfig
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


def _tree_allclose(left, right, *, atol=1.0e-5):
    assert jax.tree_util.tree_structure(left) == jax.tree_util.tree_structure(right)
    for left_leaf, right_leaf in zip(
        _floating_leaves(left), _floating_leaves(right), strict=True
    ):
        assert jnp.allclose(left_leaf, right_leaf, atol=atol, rtol=atol)


def _model_config(**overrides):
    values = {
        "policy_config": PolicyConfig(d_model=8, stop_bias_init=-20.0),
        "batch_size": 2,
        "max_steps": 2,
        "state_token_pad_to": 512,
        "action_token_pad_to": 512,
        "definition_pad_to": 8,
    }
    values.update(overrides)
    return CurrentTransformerModelConfig(**values)


def test_current_transformer_model_matches_legacy_static_rollout():
    config = _model_config()
    params = init_policy_params(config.policy_config, jax.random.PRNGKey(0))
    initial_json = [actionable_json(), exact_empty_json()]
    legacy_policy = PolicyState(config=config.policy_config, params=params)
    root_key = jax.random.PRNGKey(23)
    update_index = 4
    legacy = _collect_streamed_rollout_gradients(
        legacy_policy,
        [_state_from_json(text) for text in initial_json],
        RolloutConfig(
            batch_size=config.batch_size,
            max_steps=config.max_steps,
            seed=23,
            static_policy_batch=True,
            state_token_pad_to=config.state_token_pad_to,
            action_token_pad_to=config.action_token_pad_to,
            definition_pad_to=config.definition_pad_to,
        ),
        update_index=update_index,
        root_key=root_key,
    )
    row = RewriteStateRow.from_states([_state_from_json(text) for text in initial_json])
    model = CurrentTransformerModel()

    out_row, logp, grad_logp, metrics = model.sample_with_logp_grad(
        params,
        jax.random.fold_in(root_key, update_index),
        row,
        config,
    )

    assert out_row is row
    assert np.allclose(out_row.log_total_flops(), legacy.final.final_log_flops)
    assert jnp.allclose(logp, legacy.trajectory_logp, atol=1.0e-5)
    _tree_allclose(grad_logp, legacy.trajectory_grad_logp)
    assert set(metrics) == {"stopped"}
    assert metrics["stopped"].tolist() == legacy.final.stopped.tolist()


def test_current_transformer_model_rejects_batch_size_mismatch():
    config = _model_config(batch_size=2)
    params = init_policy_params(config.policy_config, jax.random.PRNGKey(0))
    row = RewriteStateRow.from_states([_state_from_json(actionable_json())])

    with pytest.raises(TrainingError, match="row batch size|batch_size"):
        CurrentTransformerModel().sample_with_logp_grad(
            params,
            jax.random.PRNGKey(0),
            row,
            config,
        )


def test_current_transformer_model_static_pad_errors_name_dimension():
    config = _model_config(batch_size=1, state_token_pad_to=1)
    params = init_policy_params(config.policy_config, jax.random.PRNGKey(0))
    row = RewriteStateRow.from_states([_state_from_json(actionable_json())])

    with pytest.raises(
        TrainingError,
        match="state token length .* exceeds state_token_pad_to 1",
    ):
        CurrentTransformerModel().sample_with_logp_grad(
            params,
            jax.random.PRNGKey(0),
            row,
            config,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd python
uv run pytest tests/test_current_transformer_model.py -q
```

Expected: FAIL with missing `reinforce.model.CurrentTransformerModel`.

- [ ] **Step 3: Add RNG helper for protocol RNG**

In `python/gristmill_symbolics/reinforce/rollout.py`, split the existing RNG helper:

```python
def _make_decision_rng_grid(rng, max_steps: int, batch_size: int):
    flat_keys = jax.random.split(rng, max_steps * batch_size * 2)
    return flat_keys.reshape((max_steps, batch_size, 2, *flat_keys.shape[1:]))


def make_rng_grid(root_key, update_index: int, max_steps: int, batch_size: int):
    update_key = jax.random.fold_in(root_key, int(update_index))
    return _make_decision_rng_grid(update_key, max_steps, batch_size)
```

This preserves old `make_rng_grid` behavior and gives the adapter a direct-`rng` primitive.

- [ ] **Step 4: Extract static row-native helper**

In `python/gristmill_symbolics/reinforce/rollout.py`, add a private result dataclass:

```python
@dataclass(frozen=True)
class _StaticModelRolloutResult:
    out_row: RewriteStateRow
    logp: jax.Array
    grad_logp: object
    stopped: np.ndarray
```

Add `_sample_static_model_rollout(params, rng, row, config)` below `_dummy_action_policy_item`. Move the body of `_collect_streamed_rollout_gradients` lines 132-443 into this helper with these exact ownership changes:

```python
def _sample_static_model_rollout(
    params,
    rng,
    row: RewriteStateRow,
    config: CurrentTransformerModelConfig,
) -> _StaticModelRolloutResult:
    validate_model_config(config)
    _validate_streamed_gradient_param_dtypes(params)

    if int(row.len()) != config.batch_size:
        raise TrainingError(
            f"row batch size {row.len()} differs from batch_size {config.batch_size}"
        )

    rng_grid = _make_decision_rng_grid(
        rng,
        max_steps=config.max_steps,
        batch_size=config.batch_size,
    )
    active = [True] * config.batch_size
    stopped = [False] * config.batch_size
    exact_empty_def_masks: list[jax.Array | None] = [None] * config.batch_size
    trajectory_logp = jnp.zeros((config.batch_size,), dtype=jnp.float32)
    trajectory_grad_logp = _zero_trajectory_grad(params, config.batch_size)
    static_state_pad_to = config.state_token_pad_to
    static_definition_pad_to = config.definition_pad_to
    static_action_pad_to = config.action_token_pad_to
```

Inside the moved loop:

- Replace every `policy.params` with `params`.
- Replace `if not active_indices and not static_policy_batch:` with `if not active_indices: continue`.
- Replace `target_policy_samples = list(range(config.batch_size)) if static_policy_batch else active_indices` with `target_policy_samples = list(range(config.batch_size))`.
- Keep the dummy state/action rows and zero masking exactly as in the current static branch.
- Remove legacy counter increments from the helper return path except local values needed by the old reference wrapper during Task 2.
- Return:

```python
return _StaticModelRolloutResult(
    out_row=row,
    logp=trajectory_logp,
    grad_logp=trajectory_grad_logp,
    stopped=np.asarray(stopped, dtype=bool),
)
```

During this task, leave `_collect_streamed_rollout_gradients` in place. For its static branch, it may call `_sample_static_model_rollout(...)` and then wrap the result back into `_StreamedRolloutResult`; for its dynamic branch, keep the current code untouched until Task 6 cleanup.

- [ ] **Step 5: Implement model adapter**

Create `python/gristmill_symbolics/reinforce/model.py`:

```python
from __future__ import annotations

from gristmill_symbolics import RewriteStateRow

from .rollout import _sample_static_model_rollout
from .types import CurrentTransformerModelConfig, validate_model_config


class CurrentTransformerModel:
    def sample_with_logp_grad(
        self,
        params,
        rng,
        row: RewriteStateRow,
        config: CurrentTransformerModelConfig,
    ):
        validate_model_config(config)
        result = _sample_static_model_rollout(params, rng, row, config)
        return result.out_row, result.logp, result.grad_logp, {
            "stopped": result.stopped,
        }
```

- [ ] **Step 6: Run model adapter tests**

Run:

```bash
cd python
uv run pytest tests/test_current_transformer_model.py -q
```

Expected: PASS.

- [ ] **Step 7: Run existing rollout tests**

Run:

```bash
cd python
uv run pytest tests/test_reinforce_streaming.py tests/test_reinforce_rollout.py tests/test_current_transformer_model.py -q
```

Expected: PASS. The old direct streamed tests still pass while the new adapter equivalence test proves the static model boundary.

- [ ] **Step 8: Commit**

```bash
git add python/gristmill_symbolics/reinforce/model.py python/gristmill_symbolics/reinforce/rollout.py python/gristmill_symbolics/reinforce/__init__.py python/tests/test_current_transformer_model.py
git commit -m "feat: add current transformer model adapter"
```

---

### Task 3: Implement Protocol-Shaped REINFORCE Trainer

**Files:**
- Create: `python/gristmill_symbolics/reinforce/trainer.py`
- Modify: `python/gristmill_symbolics/reinforce/train_state.py`
- Modify: `python/gristmill_symbolics/reinforce/__init__.py`
- Test: `python/tests/test_reinforce_trainer_protocol.py`

- [ ] **Step 1: Write failing trainer tests**

Create `python/tests/test_reinforce_trainer_protocol.py`:

```python
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gristmill_symbolics import RewriteState, TensorComputation
from gristmill_symbolics.policy import PolicyConfig, init_policy_params
from gristmill_symbolics.reinforce import (
    BaselineConfig,
    OptimizerConfig,
    ReinforceTrainer,
    ReinforceTrainerConfig,
    TrainingError,
    make_optimizer,
)
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
    def __init__(self, *, final_log_flops, logp, grad_logp, metrics=None):
        self.final_log_flops = final_log_flops
        self.logp = logp
        self.grad_logp = grad_logp
        self.metrics = metrics or {"stopped": np.asarray([False, True], dtype=bool)}
        self.calls = []

    def sample_with_logp_grad(self, params, rng, row, config):
        self.calls.append((params, rng, row, config))
        return FakeOutRow(self.final_log_flops), self.logp, self.grad_logp, self.metrics


def _simple_params():
    return {"w": jnp.asarray([1.0, -2.0], dtype=jnp.float32)}


def _zero_grad_logp(params, batch_size):
    return jax.tree_util.tree_map(
        lambda leaf: jnp.zeros((batch_size, *leaf.shape), dtype=leaf.dtype),
        params,
    )


def test_reinforce_trainer_calls_model_and_updates_from_model_outputs():
    params = _simple_params()
    optimizer = make_optimizer(OptimizerConfig(learning_rate=1.0e-2))
    opt_state = optimizer.init(params)
    batch = _batch()
    initial = np.asarray([state.log_total_flops() for state in batch], dtype=np.float64)
    final = initial - np.asarray([1.0, -1.0], dtype=np.float64)
    grad_logp = {"w": jnp.asarray([[2.0, 0.0], [0.0, 4.0]], dtype=jnp.float32)}
    model = FakeModel(
        final_log_flops=final,
        logp=jnp.asarray([-0.5, -1.5], dtype=jnp.float32),
        grad_logp=grad_logp,
    )
    config = ReinforceTrainerConfig(
        batch_size=2,
        optimizer_config=OptimizerConfig(learning_rate=1.0e-2),
    )

    new_params, new_opt_state, metrics = ReinforceTrainer().update(
        params,
        opt_state,
        batch,
        model,
        jax.random.PRNGKey(0),
        config,
    )

    assert model.calls
    assert new_opt_state is not opt_state
    assert metrics["reward_mean"] == pytest.approx(0.0)
    assert metrics["reward_std"] == pytest.approx(1.0)
    assert metrics["objective_loss_mean"] == pytest.approx(-metrics["reward_mean"])
    assert np.isfinite(metrics["surrogate_loss"])
    assert metrics["final_flops_best"] == pytest.approx(float(np.min(final)))
    assert metrics["params_changed"] is True
    assert not jnp.array_equal(new_params["w"], params["w"])


def test_reinforce_trainer_standardizes_advantage_when_configured():
    params = _simple_params()
    optimizer = make_optimizer(OptimizerConfig(learning_rate=1.0e-2))
    batch = _batch()
    initial = np.asarray([state.log_total_flops() for state in batch], dtype=np.float64)
    final = initial - np.asarray([2.0, 4.0], dtype=np.float64)
    model = FakeModel(
        final_log_flops=final,
        logp=jnp.asarray([-1.0, -1.0], dtype=jnp.float32),
        grad_logp=_zero_grad_logp(params, 2),
    )
    config = ReinforceTrainerConfig(
        batch_size=2,
        optimizer_config=OptimizerConfig(learning_rate=1.0e-2),
        baseline_config=BaselineConfig(standardize=True, epsilon=1.0e-12),
    )

    _new_params, _new_opt_state, metrics = ReinforceTrainer().update(
        params,
        optimizer.init(params),
        batch,
        model,
        jax.random.PRNGKey(0),
        config,
    )

    assert metrics["reward_mean"] == pytest.approx(3.0)
    assert np.isfinite(metrics["surrogate_loss"])


def test_reinforce_trainer_validates_batch_length_before_model_call():
    params = _simple_params()
    optimizer = make_optimizer(OptimizerConfig(learning_rate=1.0e-2))
    model = FakeModel(
        final_log_flops=np.asarray([0.0]),
        logp=jnp.asarray([0.0], dtype=jnp.float32),
        grad_logp=_zero_grad_logp(params, 1),
    )

    with pytest.raises(TrainingError, match="batch length"):
        ReinforceTrainer().update(
            params,
            optimizer.init(params),
            [_state_from_json(actionable_json())],
            model,
            jax.random.PRNGKey(0),
            ReinforceTrainerConfig(
                batch_size=2,
                optimizer_config=OptimizerConfig(learning_rate=1.0e-2),
            ),
        )
    assert model.calls == []


@pytest.mark.parametrize(
    ("logp", "grad_logp", "message"),
    [
        (jnp.asarray([[0.0]], dtype=jnp.float32), _zero_grad_logp(_simple_params(), 2), "logp"),
        (jnp.asarray([0.0, jnp.nan], dtype=jnp.float32), _zero_grad_logp(_simple_params(), 2), "logp"),
        (jnp.asarray([0.0, 0.0], dtype=jnp.float32), {"w": jnp.zeros((1, 2), dtype=jnp.float32)}, "leading dimension"),
        (jnp.asarray([0.0, 0.0], dtype=jnp.float32), {"w": jnp.asarray([[0.0, 0.0], [jnp.inf, 0.0]], dtype=jnp.float32)}, "grad_logp"),
    ],
)
def test_reinforce_trainer_validates_model_output_protocol(logp, grad_logp, message):
    params = _simple_params()
    optimizer = make_optimizer(OptimizerConfig(learning_rate=1.0e-2))
    batch = _batch()
    final = np.asarray([state.log_total_flops() for state in batch], dtype=np.float64)
    model = FakeModel(final_log_flops=final, logp=logp, grad_logp=grad_logp)

    with pytest.raises(TrainingError, match=message):
        ReinforceTrainer().update(
            params,
            optimizer.init(params),
            batch,
            model,
            jax.random.PRNGKey(0),
            ReinforceTrainerConfig(
                batch_size=2,
                optimizer_config=OptimizerConfig(learning_rate=1.0e-2),
            ),
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd python
uv run pytest tests/test_reinforce_trainer_protocol.py -q
```

Expected: FAIL with missing `ReinforceTrainer`.

- [ ] **Step 3: Implement trainer helpers**

Create `python/gristmill_symbolics/reinforce/trainer.py` with:

```python
from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

from gristmill_symbolics import RewriteState, RewriteStateRow

from .objective import compute_advantages
from .train_state import (
    _params_changed,
    _reinforce_grad_loss,
    _surrogate_loss,
    _validate_finite_params,
    make_optimizer,
)
from .types import (
    ReinforceTrainerConfig,
    TrainingError,
    validate_trainer_config,
)
```

Add output validation:

```python
def _validate_logp(logp, batch_size: int):
    values = jnp.asarray(logp)
    if values.shape != (batch_size,):
        raise TrainingError(f"logp must have shape {(batch_size,)}, got {values.shape}")
    if not bool(jnp.all(jnp.isfinite(values))):
        raise TrainingError("logp contains non-finite values")
    return values


def _validate_grad_logp(params, grad_logp, batch_size: int):
    if jax.tree_util.tree_structure(params) != jax.tree_util.tree_structure(grad_logp):
        raise TrainingError("grad_logp pytree must match params pytree")
    for param_leaf, grad_leaf in zip(
        jax.tree_util.tree_leaves(params),
        jax.tree_util.tree_leaves(grad_logp),
        strict=True,
    ):
        param_leaf = jnp.asarray(param_leaf)
        grad_leaf = jnp.asarray(grad_leaf)
        if grad_leaf.shape[0] != batch_size:
            raise TrainingError(
                "grad_logp floating leaves must have leading dimension "
                f"{batch_size}, got {grad_leaf.shape}"
            )
        if grad_leaf.shape[1:] != param_leaf.shape:
            raise TrainingError(
                "grad_logp leaf shape after the sample axis must match params leaf "
                f"shape {param_leaf.shape}, got {grad_leaf.shape[1:]}"
            )
        if jnp.issubdtype(grad_leaf.dtype, jnp.floating):
            if not bool(jnp.all(jnp.isfinite(grad_leaf))):
                raise TrainingError("grad_logp contains non-finite values")
    return grad_logp
```

Add trainer:

```python
class ReinforceTrainer:
    def update(
        self,
        params,
        opt_state,
        batch: Sequence[RewriteState],
        model,
        rng,
        config: ReinforceTrainerConfig,
    ):
        validate_trainer_config(config)
        initial_states = list(batch)
        if len(initial_states) != config.batch_size:
            raise TrainingError(
                f"batch length {len(initial_states)} differs from batch_size {config.batch_size}"
            )

        initial_log_flops = np.asarray(
            [state.log_total_flops() for state in initial_states],
            dtype=np.float64,
        )
        row = RewriteStateRow.from_states(initial_states)
        out_row, logp, grad_logp, _model_metrics = model.sample_with_logp_grad(
            params,
            rng,
            row,
            config,
        )
        logp = _validate_logp(logp, config.batch_size)
        grad_logp = _validate_grad_logp(params, grad_logp, config.batch_size)

        final_log_flops = np.asarray(out_row.log_total_flops(), dtype=np.float64)
        if final_log_flops.shape != initial_log_flops.shape:
            raise TrainingError(
                "final_log_flops shape does not match initial_log_flops shape: "
                f"{final_log_flops.shape} != {initial_log_flops.shape}"
            )
        reward = initial_log_flops - final_log_flops
        if not bool(np.all(np.isfinite(reward))):
            raise TrainingError("reward contains non-finite values")
        advantage = compute_advantages(reward, config.baseline_config)

        grads = _reinforce_grad_loss(grad_logp, advantage)
        surrogate_loss = _surrogate_loss(logp, advantage)
        if not bool(np.isfinite(np.asarray(surrogate_loss))):
            raise TrainingError("surrogate_loss is non-finite")

        optimizer = make_optimizer(config.optimizer_config)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
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

The fake-model tests pass `config` into the model. In Task 4, `_ConfiguredModel` binds the model config before the trainer calls `model.sample_with_logp_grad`, so the trainer remains unaware of model-private configuration while the protocol signature stays canonical.

- [ ] **Step 4: Run trainer protocol tests**

Run:

```bash
cd python
uv run pytest tests/test_reinforce_trainer_protocol.py -q
```

Expected: PASS.

- [ ] **Step 5: Run objective/train helper tests**

Run:

```bash
cd python
uv run pytest tests/test_reinforce_trainer_protocol.py tests/test_reinforce_objective.py tests/test_reinforce_train.py -q
```

Expected: PASS. Existing `train_update` still uses the legacy direct path in this task.

- [ ] **Step 6: Commit**

```bash
git add python/gristmill_symbolics/reinforce/trainer.py python/gristmill_symbolics/reinforce/__init__.py python/tests/test_reinforce_trainer_protocol.py
git commit -m "feat: add reinforce trainer protocol implementation"
```

---

### Task 4: Prove Old-Vs-New Deterministic Equivalence

**Files:**
- Modify: `python/gristmill_symbolics/reinforce/train_state.py`
- Test: `python/tests/test_reinforce_protocol_equivalence.py`

- [ ] **Step 1: Write failing equivalence tests**

Create `python/tests/test_reinforce_protocol_equivalence.py`:

```python
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gristmill_symbolics import RewriteState, TensorComputation
from gristmill_symbolics.policy import PolicyConfig
from gristmill_symbolics.reinforce import (
    CurrentTransformerModel,
    CurrentTransformerModelConfig,
    OptimizerConfig,
    ReinforceTrainer,
    ReinforceTrainerConfig,
    init_train_state,
    train_update,
)
from gristmill_symbolics.reinforce.train_state import _ConfiguredModel
from gristmill_symbolics.reinforce.types import RolloutConfig
from tests.policy_fixtures import actionable_json
from tests.test_bindings import exact_empty_json


def _state_from_json(text):
    return RewriteState.from_computation(TensorComputation.from_json_string(text))


def _batch():
    return [_state_from_json(actionable_json()), _state_from_json(exact_empty_json())]


def _tree_allclose(left, right, *, atol=1.0e-5):
    assert jax.tree_util.tree_structure(left) == jax.tree_util.tree_structure(right)
    for left_leaf, right_leaf in zip(
        jax.tree_util.tree_leaves(left),
        jax.tree_util.tree_leaves(right),
        strict=True,
    ):
        if hasattr(left_leaf, "dtype") and jnp.issubdtype(left_leaf.dtype, jnp.floating):
            assert jnp.allclose(left_leaf, right_leaf, atol=atol, rtol=atol)


def test_new_trainer_model_path_matches_legacy_static_train_update():
    policy_config = PolicyConfig(d_model=8, stop_bias_init=-20.0)
    optimizer_config = OptimizerConfig(learning_rate=1.0e-2)
    state = init_train_state(policy_config, optimizer_config, seed=29)
    legacy_config = RolloutConfig(
        batch_size=2,
        max_steps=2,
        seed=29,
        static_policy_batch=True,
        state_token_pad_to=512,
        action_token_pad_to=512,
        definition_pad_to=8,
    )
    legacy_state, legacy_metrics = train_update(state, _batch(), legacy_config)

    model_config = CurrentTransformerModelConfig(
        policy_config=policy_config,
        batch_size=2,
        max_steps=2,
        state_token_pad_to=512,
        action_token_pad_to=512,
        definition_pad_to=8,
    )
    trainer_config = ReinforceTrainerConfig(
        batch_size=2,
        optimizer_config=optimizer_config,
    )
    rng = jax.random.fold_in(state.root_key, state.update_index)
    new_params, new_opt_state, new_metrics = ReinforceTrainer().update(
        state.policy.params,
        state.opt_state,
        _batch(),
        _ConfiguredModel(CurrentTransformerModel(), model_config),
        rng,
        trainer_config,
    )

    _tree_allclose(new_params, legacy_state.policy.params)
    _tree_allclose(new_opt_state, legacy_state.opt_state)
    assert new_metrics["reward_mean"] == pytest.approx(legacy_metrics.reward_mean)
    assert new_metrics["reward_std"] == pytest.approx(legacy_metrics.reward_std)
    assert new_metrics["objective_loss_mean"] == pytest.approx(
        legacy_metrics.objective_loss_mean
    )
    assert new_metrics["surrogate_loss"] == pytest.approx(
        legacy_metrics.surrogate_loss,
        abs=1.0e-5,
    )
    assert new_metrics["final_flops_best"] == pytest.approx(
        legacy_metrics.final_log_flops_best
    )
    assert new_metrics["params_changed"] is legacy_metrics.params_changed
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd python
uv run pytest tests/test_reinforce_protocol_equivalence.py -q
```

Expected: FAIL because `_ConfiguredModel` does not exist yet.

- [ ] **Step 3: Add model-config binding helper**

In `python/gristmill_symbolics/reinforce/train_state.py`, add:

```python
class _ConfiguredModel:
    def __init__(self, model, model_config):
        self._model = model
        self._model_config = model_config

    def sample_with_logp_grad(self, params, rng, row, _trainer_config):
        return self._model.sample_with_logp_grad(
            params,
            rng,
            row,
            self._model_config,
        )
```

Use this wrapper in state orchestration only. Do not make `ReinforceTrainer` inspect model config.

For this task only, keep old `train_update` unchanged for equivalence reference. The public state runner is added in Task 5 after `TrainState` becomes protocol-shaped.

- [ ] **Step 4: Run equivalence tests**

Run:

```bash
cd python
uv run pytest tests/test_reinforce_protocol_equivalence.py -q
```

Expected: PASS.

- [ ] **Step 5: Run full focused protocol slice**

Run:

```bash
cd python
uv run pytest tests/test_current_transformer_model.py tests/test_reinforce_trainer_protocol.py tests/test_reinforce_protocol_equivalence.py tests/test_reinforce_train.py tests/test_reinforce_streaming.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add python/gristmill_symbolics/reinforce/train_state.py python/tests/test_reinforce_protocol_equivalence.py
git commit -m "test: prove protocol path matches legacy static training"
```

---

### Task 5: Switch Public Training, CLI, And Checkpoint To Protocol Path

**Files:**
- Modify: `python/gristmill_symbolics/reinforce/train_state.py`
- Modify: `python/gristmill_symbolics/reinforce/train.py`
- Modify: `python/gristmill_symbolics/reinforce/checkpoint.py`
- Modify: `python/gristmill_symbolics/reinforce/types.py`
- Modify: `python/tests/test_reinforce_train.py`
- Modify: `python/tests/test_reinforce_cli.py`
- Modify: `python/tests/test_reinforce_checkpoint.py`
- Modify: `python/tests/test_reinforce_package.py`

- [ ] **Step 1: Write failing public runner tests**

Update `python/tests/test_reinforce_train.py` so the primary update tests use `advance_train_state`, `CurrentTransformerModel`, and `ReinforceTrainer`:

```python
from gristmill_symbolics.reinforce import (
    CurrentTransformerModel,
    CurrentTransformerModelConfig,
    ReinforceTrainer,
    ReinforceTrainerConfig,
    TrainState,
    advance_train_state,
)
```

Replace one old `train_update` test with:

```python
def test_advance_train_state_uses_protocol_path_and_increments_update_index():
    policy_config = PolicyConfig(d_model=8, stop_bias_init=-20.0)
    optimizer_config = OptimizerConfig(learning_rate=1.0e-2)
    state = init_train_state(policy_config, optimizer_config, seed=29)
    model_config = CurrentTransformerModelConfig(
        policy_config=policy_config,
        batch_size=2,
        max_steps=2,
        state_token_pad_to=512,
        action_token_pad_to=512,
        definition_pad_to=8,
    )
    trainer_config = ReinforceTrainerConfig(
        batch_size=2,
        optimizer_config=optimizer_config,
    )

    new_state, metrics = advance_train_state(
        state,
        _mixed_initial_states(),
        model=CurrentTransformerModel(),
        trainer=ReinforceTrainer(),
        model_config=model_config,
        trainer_config=trainer_config,
    )

    assert new_state.update_index == 1
    assert metrics.update_index == 0
    assert metrics.batch_size == 2
    assert np.isfinite(metrics.objective_loss_mean)
    assert np.isfinite(metrics.surrogate_loss)
    assert np.isfinite(metrics.final_flops_best)
    assert metrics.params_changed is True
```

Add the protocol-state shape assertion:

```python
def test_train_state_is_protocol_state_only():
    state = TrainState(
        params={"w": jnp.asarray([1.0])},
        opt_state=("opt",),
        root_key=jax.random.PRNGKey(0),
        update_index=7,
    )

    assert state.params["w"].shape == (1,)
    assert state.opt_state == ("opt",)
    assert state.update_index == 7
    assert not hasattr(state, "policy")
    assert not hasattr(state, "optimizer_config")
```

Add a runner RNG test:

```python
class RecordingTrainer:
    def __init__(self):
        self.rng = None

    def update(self, params, opt_state, batch, model, rng, config):
        self.rng = rng
        return params, opt_state, {
            "reward_mean": 0.0,
            "reward_std": 0.0,
            "objective_loss_mean": -0.0,
            "surrogate_loss": -0.0,
            "final_flops_best": 0.0,
            "params_changed": False,
        }


class UnusedModel:
    pass


def test_advance_train_state_folds_update_index_into_root_key_before_update():
    policy_config = PolicyConfig(d_model=8)
    optimizer_config = OptimizerConfig(learning_rate=1.0e-2)
    state = init_train_state(policy_config, optimizer_config, seed=3, update_index=5)
    trainer = RecordingTrainer()

    advance_train_state(
        state,
        _mixed_initial_states(),
        model=UnusedModel(),
        trainer=trainer,
        model_config=CurrentTransformerModelConfig(
            policy_config=policy_config,
            batch_size=2,
            max_steps=1,
            state_token_pad_to=512,
            action_token_pad_to=512,
            definition_pad_to=8,
        ),
        trainer_config=ReinforceTrainerConfig(
            batch_size=2,
            optimizer_config=optimizer_config,
        ),
    )

    assert jnp.array_equal(trainer.rng, jax.random.fold_in(state.root_key, 5))
```

- [ ] **Step 2: Run train tests to verify failure**

Run:

```bash
cd python
uv run pytest tests/test_reinforce_train.py -q
```

Expected: FAIL until `init_train_state` returns protocol `TrainState` and public tests stop expecting legacy rollout metrics.

- [ ] **Step 3: Switch state, metrics, and checkpoint dataclasses to protocol shape**

In `python/gristmill_symbolics/reinforce/types.py`, replace the old `TrainState`, `UpdateMetrics`, and `CheckpointData` definitions with:

```python
@dataclass(frozen=True)
class TrainState:
    params: object
    opt_state: object
    root_key: jax.Array
    update_index: int


@dataclass(frozen=True)
class UpdateMetrics:
    update_index: int
    batch_size: int
    reward_mean: float
    reward_std: float
    objective_loss_mean: float
    surrogate_loss: float
    final_flops_best: float
    params_changed: bool


@dataclass(frozen=True)
class CheckpointData:
    train_state: TrainState
    model_config: CurrentTransformerModelConfig
    trainer_config: ReinforceTrainerConfig
    recent_metrics: tuple[UpdateMetrics, ...]
```

Keep `RolloutConfig`, `LossConfig`, `PolicyState`, and `FinalColumnMetrics` in this task because `test_reinforce_protocol_equivalence.py` and old rollout tests still use them as the temporary reference path. They are removed in Task 6.

- [ ] **Step 4: Switch `init_train_state` to protocol state**

In `python/gristmill_symbolics/reinforce/train_state.py`, change `init_train_state` to return `TrainState(params=..., opt_state=..., root_key=..., update_index=...)`:

```python
def init_train_state(
    policy_config: PolicyConfig,
    optimizer_config: OptimizerConfig,
    *,
    seed: int,
    update_index: int = 0,
) -> TrainState:
    root_key = jax.random.PRNGKey(int(seed))
    params_key = jax.random.fold_in(root_key, np.uint32(0xFFFFFFFF))
    params = init_policy_params(policy_config, params_key)
    optimizer = make_optimizer(optimizer_config)
    return TrainState(
        params=params,
        opt_state=optimizer.init(params),
        root_key=root_key,
        update_index=int(update_index),
    )
```

Update all local references from `state.policy.params` to `state.params`.

- [ ] **Step 5: Add `advance_train_state` for protocol state**

In `python/gristmill_symbolics/reinforce/train_state.py`, add the public state runner. The runner body uses only protocol `TrainState`:

```python
def advance_train_state(
    state: TrainState,
    batch: Sequence[RewriteState],
    *,
    model,
    trainer,
    model_config: CurrentTransformerModelConfig,
    trainer_config: ReinforceTrainerConfig,
):
    rng = jax.random.fold_in(state.root_key, int(state.update_index))
    new_params, new_opt_state, trainer_metrics = trainer.update(
        state.params,
        state.opt_state,
        batch,
        _ConfiguredModel(model, model_config),
        rng,
        trainer_config,
    )
    metrics = UpdateMetrics(
        update_index=state.update_index,
        batch_size=trainer_config.batch_size,
        reward_mean=trainer_metrics["reward_mean"],
        reward_std=trainer_metrics["reward_std"],
        objective_loss_mean=trainer_metrics["objective_loss_mean"],
        surrogate_loss=trainer_metrics["surrogate_loss"],
        final_flops_best=trainer_metrics["final_flops_best"],
        params_changed=trainer_metrics["params_changed"],
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

Update `python/tests/test_reinforce_protocol_equivalence.py` only if it still references the public runner; the preferred equivalence test from Task 4 compares direct `trainer.update` mapping outputs and does not require a state-shape edit.

- [ ] **Step 6: Replace public `train_update` with protocol wrapper or remove it from supported tests**

Do not leave `train_update` calling `_collect_streamed_rollout_gradients`. Either remove it from `__all__` or keep this compatibility wrapper only:

```python
def train_update(
    state: TrainState,
    initial_states: Sequence[RewriteState],
    *,
    model_config: CurrentTransformerModelConfig,
    trainer_config: ReinforceTrainerConfig,
    model=None,
    trainer=None,
):
    from .model import CurrentTransformerModel
    from .trainer import ReinforceTrainer

    return advance_train_state(
        state,
        initial_states,
        model=CurrentTransformerModel() if model is None else model,
        trainer=ReinforceTrainer() if trainer is None else trainer,
        model_config=model_config,
        trainer_config=trainer_config,
    )
```

This wrapper is allowed because it uses the supported workflow and is not a direct trainer-to-rollout path.

- [ ] **Step 7: Update CLI tests for compact metrics and static config**

In `python/tests/test_reinforce_cli.py`, update expectations:

```python
assert "reward_mean" in line
assert "reward_std" in line
assert "objective_loss_mean" in line
assert "surrogate_loss" in line
assert "final_flops_best" in line
assert "params_changed" in line
assert "target_score_count" not in line
assert "action_score_count" not in line
assert "stop_count" not in line
```

Replace `checkpoint.rollout_config` assertions with:

```python
assert checkpoint.model_config.state_token_pad_to == 256
assert checkpoint.model_config.action_token_pad_to == 256
assert checkpoint.model_config.definition_pad_to == 4
assert checkpoint.trainer_config.batch_size == 1
```

- [ ] **Step 8: Switch CLI implementation**

In `python/gristmill_symbolics/reinforce/train.py`:

- Remove `--static-policy-batch`; static is mandatory.
- Keep `--state-token-pad-to`, `--action-token-pad-to`, and `--definition-pad-to`, but require them when no checkpoint is loaded.
- Build configs:

```python
model_config = CurrentTransformerModelConfig(
    policy_config=PolicyConfig(d_model=8),
    batch_size=args.batch_size,
    max_steps=args.max_steps,
    state_token_pad_to=args.state_token_pad_to,
    action_token_pad_to=args.action_token_pad_to,
    definition_pad_to=args.definition_pad_to,
)
trainer_config = ReinforceTrainerConfig(
    batch_size=args.batch_size,
    optimizer_config=OptimizerConfig(learning_rate=args.learning_rate),
)
train_state = init_train_state(
    model_config.policy_config,
    trainer_config.optimizer_config,
    seed=args.seed,
)
```

Inside the update loop:

```python
train_state, metrics = advance_train_state(
    train_state,
    initial_states,
    model=CurrentTransformerModel(),
    trainer=ReinforceTrainer(),
    model_config=model_config,
    trainer_config=trainer_config,
)
print(json.dumps(asdict(metrics), sort_keys=True))
```

- [ ] **Step 9: Switch checkpoint schema**

In `python/gristmill_symbolics/reinforce/checkpoint.py`, store:

```python
payload = {
    "schema_version": CHECKPOINT_SCHEMA_VERSION,
    "policy_config": asdict(model_config.policy_config),
    "policy_params": train_state.params,
    "optimizer_config": asdict(trainer_config.optimizer_config),
    "optimizer_state": train_state.opt_state,
    "model_config": {
        "batch_size": model_config.batch_size,
        "max_steps": model_config.max_steps,
        "state_token_pad_to": model_config.state_token_pad_to,
        "action_token_pad_to": model_config.action_token_pad_to,
        "definition_pad_to": model_config.definition_pad_to,
    },
    "trainer_config": {
        "batch_size": trainer_config.batch_size,
        "reward_config": asdict(trainer_config.reward_config),
        "baseline_config": asdict(trainer_config.baseline_config),
    },
    "update_index": int(train_state.update_index),
    "root_key": np.asarray(train_state.root_key, dtype=np.uint32),
    "recent_metrics": tuple(asdict(metrics) for metrics in recent_metrics),
}
```

Load into:

```python
policy_config = PolicyConfig(**payload["policy_config"])
optimizer_config = OptimizerConfig(**payload["optimizer_config"])
model_config = CurrentTransformerModelConfig(
    policy_config=policy_config,
    **payload["model_config"],
)
trainer_payload = payload["trainer_config"]
trainer_config = ReinforceTrainerConfig(
    batch_size=int(trainer_payload["batch_size"]),
    optimizer_config=optimizer_config,
    reward_config=RewardConfig(**trainer_payload["reward_config"]),
    baseline_config=BaselineConfig(**trainer_payload["baseline_config"]),
)
train_state = TrainState(
    params=payload["policy_params"],
    opt_state=payload["optimizer_state"],
    root_key=jnp.asarray(payload["root_key"], dtype=jnp.uint32),
    update_index=int(payload["update_index"]),
)
```

Remove `tokenizer_schema_version`, `rollout_config`, and `loss_config` from newly written checkpoint payloads. Do not add migration from schema version 1; the accepted spec defers old checkpoint migration.

- [ ] **Step 10: Update checkpoint tests**

In `python/tests/test_reinforce_checkpoint.py`, construct and assert `model_config` and `trainer_config` instead of `rollout_config`, `reward_config`, `baseline_config`, and `loss_config` as separate checkpoint top-level fields. Recent metrics should use the compact `UpdateMetrics` dataclass:

```python
UpdateMetrics(
    update_index=5,
    batch_size=2,
    reward_mean=1.5,
    reward_std=0.25,
    objective_loss_mean=-1.5,
    surrogate_loss=-0.125,
    final_flops_best=7.25,
    params_changed=True,
)
```

Delete the tokenizer-schema-version rejection test and replace it with:

```python
def test_checkpoint_rejects_missing_model_config(tmp_path):
    path = tmp_path / "bad.pkl"
    with path.open("wb") as handle:
        pickle.dump({"schema_version": CHECKPOINT_SCHEMA_VERSION}, handle)

    with pytest.raises(TrainingError, match="model_config|policy_config"):
        load_checkpoint(path)
```

- [ ] **Step 11: Run public training/checkpoint/CLI tests**

Run:

```bash
cd python
uv run pytest tests/test_reinforce_train.py tests/test_reinforce_cli.py tests/test_reinforce_checkpoint.py tests/test_reinforce_package.py -q
```

Expected: PASS.

- [ ] **Step 12: Run protocol/model/trainer tests**

Run:

```bash
cd python
uv run pytest tests/test_reinforce_protocols.py tests/test_current_transformer_model.py tests/test_reinforce_trainer_protocol.py tests/test_reinforce_protocol_equivalence.py -q
```

Expected: PASS.

- [ ] **Step 13: Commit**

```bash
git add python/gristmill_symbolics/reinforce/train_state.py python/gristmill_symbolics/reinforce/train.py python/gristmill_symbolics/reinforce/checkpoint.py python/gristmill_symbolics/reinforce/types.py python/tests/test_reinforce_train.py python/tests/test_reinforce_cli.py python/tests/test_reinforce_checkpoint.py python/tests/test_reinforce_package.py
git commit -m "refactor: switch reinforce public training to protocols"
```

---

### Task 6: Remove Legacy Direct Rollout Workflow

**Files:**
- Modify: `python/gristmill_symbolics/reinforce/rollout.py`
- Modify: `python/gristmill_symbolics/reinforce/types.py`
- Modify: `python/gristmill_symbolics/reinforce/__init__.py`
- Modify: `python/tests/test_reinforce_streaming.py`
- Modify: `python/tests/test_reinforce_rollout.py`
- Modify: `python/tests/test_current_transformer_model.py`
- Modify: `python/tests/test_reinforce_protocol_equivalence.py`

- [ ] **Step 1: Write cleanup assertions that fail while legacy direct path remains**

Add to `python/tests/test_reinforce_package.py`:

```python
def test_public_reinforce_package_does_not_export_legacy_rollout_workflow():
    assert "RolloutConfig" not in reinforce.__all__
    assert "LossConfig" not in reinforce.__all__
    assert "FinalColumnMetrics" not in reinforce.__all__
    assert "make_rng_grid" not in reinforce.__all__
```

Add to `python/tests/test_current_transformer_model.py`:

```python
def test_current_transformer_model_metrics_do_not_expose_legacy_rollout_counters():
    config = _model_config(batch_size=1, max_steps=1)
    params = init_policy_params(config.policy_config, jax.random.PRNGKey(0))
    row = RewriteStateRow.from_states([_state_from_json(actionable_json())])

    _out_row, _logp, _grad_logp, metrics = CurrentTransformerModel().sample_with_logp_grad(
        params,
        jax.random.PRNGKey(0),
        row,
        config,
    )

    assert set(metrics) == {"stopped"}
```

- [ ] **Step 2: Run cleanup tests to verify they fail**

Run:

```bash
cd python
uv run pytest tests/test_reinforce_package.py tests/test_current_transformer_model.py -q
```

Expected: FAIL because legacy exports and/or legacy references still exist.

- [ ] **Step 3: Remove old direct collector**

In `python/gristmill_symbolics/reinforce/rollout.py`:

- Delete `_StreamedRolloutResult`.
- Delete `_collect_streamed_rollout_gradients`.
- Keep `_sample_static_model_rollout` as the only rollout execution helper.
- Keep dummy helpers, pad validation helpers, tree masking helpers, and `_make_decision_rng_grid` as private model infrastructure.
- Rename `_validate_streamed_gradient_param_dtypes` to `_validate_gradient_param_dtypes` and update calls.
- Keep `make_rng_grid` only if `tests/test_reinforce_rollout.py` still needs a public helper. Prefer replacing its test with a private `_make_decision_rng_grid` test and removing `make_rng_grid` from public exports.

The resulting call graph must be:

```text
advance_train_state
  -> trainer.update
       -> _ConfiguredModel.sample_with_logp_grad
            -> CurrentTransformerModel.sample_with_logp_grad
                 -> _sample_static_model_rollout
```

There must be no path:

```text
train_state/train.py/checkpoint.py -> rollout collector
```

- [ ] **Step 4: Remove legacy config/types from public contract**

In `python/gristmill_symbolics/reinforce/types.py`:

- Delete `LossConfig`.
- Delete `FinalColumnMetrics` if no non-legacy tests need it. If `objective.py` still uses it for focused reward tests, make it `_FinalColumnMetrics` and keep it module-private.
- Delete `RolloutConfig`, `PolicyState`, `validate_rollout_config`, and `validate_policy_state` after all tests have moved to model/trainer configs.
- Keep `DECISION_TARGET` and `DECISION_ACTION` because `_make_decision_rng_grid` and model rollout still use them internally.
- Keep `CHECKPOINT_SCHEMA_VERSION`, but bump it to `2` because newly written checkpoints no longer include tokenizer/rollout/loss fields.
- Delete `TOKENIZER_SCHEMA_VERSION`.

- [ ] **Step 5: Rewrite streaming tests as model-private tests**

In `python/tests/test_reinforce_streaming.py`:

- Rename the file to `python/tests/test_current_transformer_model_rollout.py` if desired.
- Replace imports of `_collect_streamed_rollout_gradients` with `CurrentTransformerModel`.
- Keep helper tests for `_stack_bool_masks`, `_mask_tree_rows`, `_dummy_action_policy_item`, and static pad failures.
- Replace dynamic-vs-static equivalence tests with current model vs scalar oracle tests. Use the already-present `_scalar_rollout_oracle` helper and call:

```python
row = RewriteStateRow.from_states([_state_from_json(_two_actionable_json())])
out_row, logp, grad_logp, metrics = CurrentTransformerModel().sample_with_logp_grad(
    policy.params,
    jax.random.fold_in(root, 0),
    row,
    CurrentTransformerModelConfig(
        policy_config=policy.config,
        batch_size=1,
        max_steps=2,
        state_token_pad_to=512,
        action_token_pad_to=512,
        definition_pad_to=8,
    ),
)
```

Assert `logp[0]`, `grad_logp` row `0`, final row flops, and `metrics["stopped"]`.

- [ ] **Step 6: Remove temporary equivalence test file**

Delete `python/tests/test_reinforce_protocol_equivalence.py` after Task 5 has passed. Its purpose was to prove the strangler switch while old direct code existed. The final suite should not import the removed legacy collector.

- [ ] **Step 7: Scan for forbidden legacy bypass references**

Run:

```bash
rg "_collect_streamed_rollout_gradients|RolloutConfig|LossConfig|PolicyState|FinalColumnMetrics|static_policy_batch|target_score_count|action_score_count|valid_action_count|empty_action_space_count|make_rng_grid" python/gristmill_symbolics python/tests
```

Expected: no hits except:

- `DECISION_TARGET` / `DECISION_ACTION` tests or private model rollout tests if they still verify RNG axes;
- any private helper names that do not expose direct trainer-to-rollout workflow;
- changelog/spec/plan documents are not part of this scan.

- [ ] **Step 8: Run cleanup tests**

Run:

```bash
cd python
uv run pytest tests/test_current_transformer_model.py tests/test_current_transformer_model_rollout.py tests/test_reinforce_trainer_protocol.py tests/test_reinforce_train.py tests/test_reinforce_checkpoint.py tests/test_reinforce_cli.py tests/test_reinforce_package.py tests/test_reinforce_objective.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add python/gristmill_symbolics/reinforce/rollout.py python/gristmill_symbolics/reinforce/types.py python/gristmill_symbolics/reinforce/__init__.py python/tests
git commit -m "refactor: remove legacy direct reinforce rollout workflow"
```

---

### Task 7: Final Verification

**Files:**
- No production edits expected.

- [ ] **Step 1: Run focused Python suite**

Run:

```bash
cd python
uv run pytest tests/test_policy_batched.py tests/test_policy_jit_grad.py tests/test_policy_model.py tests/test_policy_package.py tests/test_policy_target.py tests/test_policy_tokenize_action.py tests/test_policy_tokenize_state.py tests/test_policy_tree.py tests/test_policy_vmap.py tests/test_current_transformer_model.py tests/test_current_transformer_model_rollout.py tests/test_reinforce_trainer_protocol.py tests/test_reinforce_train.py tests/test_reinforce_checkpoint.py tests/test_reinforce_cli.py tests/test_reinforce_package.py tests/test_reinforce_rollout.py tests/test_reinforce_objective.py -q
```

Expected: PASS.

- [ ] **Step 2: Run full Python suite**

Run:

```bash
cd python
uv run pytest -q
```

Expected: PASS.

- [ ] **Step 3: Run Rust suite**

Run:

```bash
cargo test
```

Expected: PASS.

- [ ] **Step 4: Confirm no unsupported legacy training path remains**

Run:

```bash
rg "_collect_streamed_rollout_gradients|static_policy_batch|target_score_count|action_score_count|valid_action_count|empty_action_space_count|LossConfig|RolloutConfig" python/gristmill_symbolics python/tests
```

Expected: no hits.

Run:

```bash
rg "sample_with_logp_grad|trainer.update|advance_train_state" python/gristmill_symbolics/reinforce
```

Expected: hits show the supported workflow:

```text
train_state.py: trainer.update(...)
trainer.py: model.sample_with_logp_grad(...)
model.py: def sample_with_logp_grad(...)
```

- [ ] **Step 5: Inspect git status and final diff**

Run:

```bash
git status --short
git diff --stat main...HEAD
```

Expected: only intended Python code, tests, and this plan changed.

- [ ] **Step 6: Final commit if verification-only edits were needed**

If Task 7 required any test cleanup or small fixes:

```bash
git add python/gristmill_symbolics python/tests
git commit -m "test: verify reinforce protocol refactor"
```

If no files changed, do not create an empty commit.

## Self-Review

Spec coverage:

- Model protocol: Task 2 implements `CurrentTransformerModel.sample_with_logp_grad(params, rng, row, config)`.
- Trainer protocol: Task 3 implements `ReinforceTrainer.update(params, opt_state, batch, model, rng, config)`.
- Current transformer adapter static-only behavior: Task 2 extracts the current static branch and preserves dummy rows, exact-empty replay, pad failures, logp, and per-sample gradients.
- Trainer-owned reward/advantage/update behavior: Task 3 covers reward, advantage, gradient reduction, surrogate loss, optimizer update, finite param validation, and compact metrics.
- Deterministic equivalence: Task 4 compares old static training against the new protocol path before removing the old path.
- CLI/checkpoint/public path switch: Task 5 switches train runner, CLI, checkpoint schema, and package exports.
- Cleanup of parallel workflow: Task 6 removes the legacy direct collector from supported code and scans for forbidden direct-path references.

Placeholder scan:

- No placeholder or incomplete-task markers remain.
- Every task has exact files, commands, expected results, and a commit point.

Type consistency:

- Protocol names are `ExpressionModel` and `Trainer`.
- Concrete classes are `CurrentTransformerModel` and `ReinforceTrainer`.
- Config names are `CurrentTransformerModelConfig` and `ReinforceTrainerConfig`.
- State runner is `advance_train_state`.
- Compact metrics are `UpdateMetrics` with `update_index`, `batch_size`, `reward_mean`, `reward_std`, `objective_loss_mean`, `surrogate_loss`, `final_flops_best`, and `params_changed`.
