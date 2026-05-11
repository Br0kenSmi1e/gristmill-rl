# RL Checkpointing And Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add PyO3 JSON output, param-only Flax NNX/Orbax checkpoints, training resume from checkpoints, and checkpoint-driven rewrite sampling.

**Architecture:** Expose existing Rust JSON serialization to Python, then build a small Python checkpoint module that stores metadata plus NNX state in an Orbax directory. Refactor the current training rollout into a shared `rollout.py` function so both `train.py` and the new `sample.py` use the same model-guided sampled PUCT path and the same one-action-space-per-node invariant.

**Tech Stack:** Rust/PyO3, Python 3.11+, uv, NumPy, JAX, Flax NNX, Orbax Checkpoint, Optax, pytest.

---

## File Structure

- Modify `python/src/lib.rs`: expose `TensorComputation.to_json_string()` and `TensorComputation.write_json(path)`.
- Modify `python/tests/test_bindings.py`: add Python serialization round-trip tests.
- Modify `python/pyproject.toml`: add explicit `orbax-checkpoint` runtime dependency.
- Generate `python/uv.lock`: lock the explicit Orbax dependency.
- Create `python/gristmill_rl/checkpoint.py`: metadata dataclasses and checkpoint save/load helpers.
- Create `python/tests/test_rl_checkpoint.py`: checkpoint round-trip and validation tests.
- Create `python/gristmill_rl/rollout.py`: shared model-guided rewrite episode runner.
- Create `python/tests/test_rl_rollout.py`: shared rollout behavior tests.
- Modify `python/gristmill_rl/train.py`: use shared rollout and add `--checkpoint-in`, `--checkpoint-out`, `--checkpoint-overwrite`, `--hidden-dim`.
- Modify `python/tests/test_rl_train.py`: add checkpoint-out and checkpoint-in CLI tests.
- Create `python/gristmill_rl/sample.py`: checkpoint-driven sampling CLI.
- Create `python/tests/test_rl_sample.py`: sampling CLI output tests.

The implementation must not create an `RlEnv` class and must not change Rust rewrite semantics or `ActionSpace` behavior.

---

### Task 1: Expose TensorComputation JSON Writing In PyO3

**Files:**
- Modify: `python/src/lib.rs`
- Modify: `python/tests/test_bindings.py`

- [ ] **Step 1: Add failing Python serialization tests**

Append these tests to `python/tests/test_bindings.py`:

```python
def test_to_json_string_round_trips_basic_fixture():
    comp = TensorComputation.load_json(BASIC_FIXTURE)

    text = comp.to_json_string()
    loaded = TensorComputation.from_json_string(text)

    assert loaded.snapshot() == comp.snapshot()


def test_write_json_round_trips_basic_fixture(tmp_path):
    comp = TensorComputation.load_json(BASIC_FIXTURE)
    output = tmp_path / "written.json"

    comp.write_json(output)
    loaded = TensorComputation.load_json(output)

    assert loaded.snapshot() == comp.snapshot()


def test_write_json_round_trips_rewritten_computation(tmp_path):
    comp = TensorComputation.from_json_string(actionable_json())
    space = comp.next_action_space(0)
    assert space is not None
    template = space.snapshot()["candidate_templates"][0]
    comp.apply_decision_with_space(
        space,
        {
            "candidate_index": 0,
            "left_mask": [True] * len(template["left_definition"]["terms"]),
            "right_mask": [True] * len(template["right_definition"]["terms"]),
        },
    )
    output = tmp_path / "rewritten.json"

    comp.write_json(output)
    loaded = TensorComputation.load_json(output)

    assert loaded.snapshot() == comp.snapshot()
```

- [ ] **Step 2: Run the new binding tests to verify they fail**

Run:

```bash
cd python
uv run pytest tests/test_bindings.py::test_to_json_string_round_trips_basic_fixture tests/test_bindings.py::test_write_json_round_trips_basic_fixture tests/test_bindings.py::test_write_json_round_trips_rewritten_computation -q
```

Expected: FAIL with `AttributeError` for missing `to_json_string` and `write_json`.

- [ ] **Step 3: Add PyO3 methods**

In `python/src/lib.rs`, add these methods inside the existing `#[pymethods] impl PyTensorComputation` block after `from_json_string`:

```rust
    fn to_json_string(&self) -> PyResult<String> {
        io::to_json(&self.inner).map_err(py_gristmill_display_error)
    }

    fn write_json(&self, path: PathBuf) -> PyResult<()> {
        io::write_json(path, &self.inner).map_err(py_gristmill_display_error)
    }
```

Do not add any new Rust serialization path. These methods must call the existing `io` helpers.

- [ ] **Step 4: Rebuild the Python extension and run the binding tests**

Run:

```bash
cd python
uv run maturin develop
uv run pytest tests/test_bindings.py -q
```

Expected: extension rebuild succeeds and all binding tests pass.

- [ ] **Step 5: Run Rust tests**

Run:

```bash
cargo test
```

Expected: all Rust tests pass.

- [ ] **Step 6: Commit PyO3 serialization**

Run:

```bash
git add python/src/lib.rs python/tests/test_bindings.py
git commit -m "feat: expose tensor computation json writing"
```

Expected: commit succeeds.

---

### Task 2: Add Orbax Checkpoint Save/Load Helpers

**Files:**
- Modify: `python/pyproject.toml`
- Generate: `python/uv.lock`
- Create: `python/gristmill_rl/checkpoint.py`
- Create: `python/tests/test_rl_checkpoint.py`

- [ ] **Step 1: Add failing checkpoint tests**

Create `python/tests/test_rl_checkpoint.py`:

```python
import json

import numpy as np
import pytest

from gristmill_rl.checkpoint import load_checkpoint, save_checkpoint
from gristmill_rl.features import FeatureConfig, extract_features
from gristmill_rl.model import PolicyValueModel
from gristmill_rl.model import train_step, TrainConfig

from .rl_fixtures import actionable_space


def checkpoint_features(config=FeatureConfig(max_candidates=4, max_left_terms=3, max_right_terms=3)):
    comp, space = actionable_space()
    return extract_features(
        comp_snapshot=comp.snapshot(),
        action_space_snapshot=space.snapshot(),
        start_from=0,
        log_total_flops=comp.log_total_flops(),
        config=config,
    )


def model_value(model, features):
    return float(model(features).value)


def test_save_and_load_checkpoint_restores_model_outputs(tmp_path):
    feature_config = FeatureConfig(max_candidates=4, max_left_terms=3, max_right_terms=3)
    features = checkpoint_features(feature_config)
    model = PolicyValueModel(hidden_dim=16, rng_seed=0)
    expected_value = model_value(model, features)
    checkpoint = tmp_path / "checkpoint"

    save_checkpoint(
        checkpoint,
        model=model,
        feature_config=feature_config,
        hidden_dim=16,
        metadata={"seed": 0, "episodes": 3},
    )
    loaded = load_checkpoint(checkpoint)

    assert loaded.metadata.schema_version == 1
    assert loaded.metadata.hidden_dim == 16
    assert loaded.feature_config == feature_config
    assert loaded.metadata.metadata["episodes"] == 3
    np.testing.assert_allclose(model_value(loaded.model, features), expected_value)


def test_checkpoint_loads_trained_parameters(tmp_path):
    feature_config = FeatureConfig(max_candidates=4, max_left_terms=3, max_right_terms=3)
    features = checkpoint_features(feature_config)
    action = {
        "features": features,
        "actions": [],
        "policy_target": np.asarray([], dtype=np.float32),
        "value_target": 0.5,
    }
    model = PolicyValueModel(hidden_dim=16, rng_seed=0)
    before = model_value(model, features)
    train_step(model, batch=[action], config=TrainConfig(learning_rate=1e-2))
    after = model_value(model, features)
    checkpoint = tmp_path / "trained"

    save_checkpoint(
        checkpoint,
        model=model,
        feature_config=feature_config,
        hidden_dim=16,
        overwrite=False,
    )
    loaded = load_checkpoint(checkpoint)

    assert after != before
    np.testing.assert_allclose(model_value(loaded.model, features), after)


def test_save_checkpoint_refuses_existing_directory_without_overwrite(tmp_path):
    model = PolicyValueModel(hidden_dim=16, rng_seed=0)
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(
        checkpoint,
        model=model,
        feature_config=FeatureConfig(),
        hidden_dim=16,
    )

    with pytest.raises(FileExistsError):
        save_checkpoint(
            checkpoint,
            model=model,
            feature_config=FeatureConfig(),
            hidden_dim=16,
        )


def test_save_checkpoint_overwrite_replaces_metadata(tmp_path):
    model = PolicyValueModel(hidden_dim=16, rng_seed=0)
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(
        checkpoint,
        model=model,
        feature_config=FeatureConfig(),
        hidden_dim=16,
        metadata={"episodes": 1},
    )

    save_checkpoint(
        checkpoint,
        model=model,
        feature_config=FeatureConfig(max_candidates=4),
        hidden_dim=16,
        metadata={"episodes": 2},
        overwrite=True,
    )

    metadata = json.loads((checkpoint / "metadata.json").read_text())
    assert metadata["features"]["max_candidates"] == 4
    assert metadata["metadata"]["episodes"] == 2


def test_load_checkpoint_rejects_unknown_schema(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 999,
                "model": {"class": "PolicyValueModel", "hidden_dim": 16},
                "features": {
                    "max_candidates": 16,
                    "max_left_terms": 8,
                    "max_right_terms": 8,
                },
                "metadata": {},
            }
        )
    )

    with pytest.raises(ValueError, match="schema"):
        load_checkpoint(checkpoint)
```

- [ ] **Step 2: Run checkpoint tests to verify they fail**

Run:

```bash
cd python
uv run pytest tests/test_rl_checkpoint.py -q
```

Expected: FAIL because `gristmill_rl.checkpoint` does not exist.

- [ ] **Step 3: Add explicit Orbax dependency**

Modify `python/pyproject.toml` so `[project].dependencies` includes:

```toml
    "orbax-checkpoint>=0.11",
```

Then run:

```bash
cd python
uv lock
uv sync
```

Expected: `python/uv.lock` updates or remains consistent with the explicit dependency.

- [ ] **Step 4: Implement checkpoint module**

Create `python/gristmill_rl/checkpoint.py`:

```python
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import orbax.checkpoint as ocp
from flax import nnx

from gristmill_rl.features import FeatureConfig
from gristmill_rl.model import PolicyValueModel


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CheckpointMetadata:
    schema_version: int
    hidden_dim: int
    feature_config: FeatureConfig
    metadata: dict[str, Any]


@dataclass(frozen=True)
class LoadedCheckpoint:
    model: PolicyValueModel
    feature_config: FeatureConfig
    metadata: CheckpointMetadata


def _metadata_path(path: Path) -> Path:
    return path / "metadata.json"


def _state_path(path: Path) -> Path:
    return path / "state"


def _metadata_payload(
    *,
    hidden_dim: int,
    feature_config: FeatureConfig,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model": {
            "class": "PolicyValueModel",
            "hidden_dim": int(hidden_dim),
        },
        "features": asdict(feature_config),
        "metadata": dict(metadata or {}),
    }


def _parse_metadata(payload: dict[str, Any]) -> CheckpointMetadata:
    schema_version = int(payload["schema_version"])
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint schema_version {schema_version}")
    model = payload["model"]
    if model["class"] != "PolicyValueModel":
        raise ValueError(f"unsupported checkpoint model class {model['class']!r}")
    features = payload["features"]
    feature_config = FeatureConfig(
        max_candidates=int(features["max_candidates"]),
        max_left_terms=int(features["max_left_terms"]),
        max_right_terms=int(features["max_right_terms"]),
    )
    return CheckpointMetadata(
        schema_version=schema_version,
        hidden_dim=int(model["hidden_dim"]),
        feature_config=feature_config,
        metadata=dict(payload.get("metadata", {})),
    )


def save_checkpoint(
    path: Path,
    *,
    model: PolicyValueModel,
    feature_config: FeatureConfig,
    hidden_dim: int,
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> None:
    path = Path(path)
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"checkpoint already exists: {path}")
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()

    payload = _metadata_payload(
        hidden_dim=hidden_dim,
        feature_config=feature_config,
        metadata=metadata,
    )
    _metadata_path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))

    _, state = nnx.split(model.module)
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(_state_path(path), state, force=True)


def load_checkpoint(path: Path) -> LoadedCheckpoint:
    path = Path(path)
    payload = json.loads(_metadata_path(path).read_text())
    metadata = _parse_metadata(payload)

    model = PolicyValueModel(hidden_dim=metadata.hidden_dim, rng_seed=0)
    _, abstract_state = nnx.split(model.module)
    checkpointer = ocp.PyTreeCheckpointer()
    restored_state = checkpointer.restore(_state_path(path), item=abstract_state)
    nnx.update(model.module, restored_state)
    return LoadedCheckpoint(
        model=model,
        feature_config=metadata.feature_config,
        metadata=metadata,
    )
```

- [ ] **Step 5: Run checkpoint tests**

Run:

```bash
cd python
uv run pytest tests/test_rl_checkpoint.py -q
```

Expected: all checkpoint tests pass.

- [ ] **Step 6: Run all Python tests**

Run:

```bash
cd python
uv run pytest tests -q
```

Expected: all Python tests pass.

- [ ] **Step 7: Commit checkpoint helpers**

Run:

```bash
git add python/pyproject.toml python/uv.lock python/gristmill_rl/checkpoint.py python/tests/test_rl_checkpoint.py
git commit -m "feat: add rl checkpoint helpers"
```

Expected: commit succeeds.

---

### Task 3: Extract Shared Policy Rollout Logic

**Files:**
- Create: `python/gristmill_rl/rollout.py`
- Create: `python/tests/test_rl_rollout.py`
- Modify: `python/gristmill_rl/train.py`
- Modify: `python/tests/test_rl_train.py`

- [ ] **Step 1: Add failing rollout tests**

Create `python/tests/test_rl_rollout.py`:

```python
import numpy as np

from gristmill_rl.features import FeatureConfig
from gristmill_rl.model import PolicyValueModel
from gristmill_rl.rollout import RolloutConfig, run_policy_rollout

from .rl_fixtures import actionable_comp


def test_policy_rollout_returns_trace_and_rewritten_comp():
    comp = actionable_comp()
    model = PolicyValueModel(hidden_dim=16, rng_seed=0)

    result = run_policy_rollout(
        comp,
        model=model,
        feature_config=FeatureConfig(max_candidates=4, max_left_terms=3, max_right_terms=3),
        config=RolloutConfig(
            max_steps=1,
            simulations=2,
            actions_per_node=1,
            sample_attempts=4,
            temperature=0.0,
            c_puct=1.5,
        ),
        rng=np.random.default_rng(0),
    )

    assert result.steps == 1
    assert len(result.trace.records) == 1
    assert result.final_log_flops == result.comp.log_total_flops()
    assert result.comp.snapshot() != comp.snapshot()
    assert result.valid_action_counts == [1]


def test_policy_rollout_on_terminal_comp_returns_zero_step_trace():
    from gristmill_symbolics import TensorComputation

    terminal = TensorComputation.from_json_string(
        '{"ranges":[],"tensors":[],"definitions":[]}'
    )
    model = PolicyValueModel(hidden_dim=16, rng_seed=0)

    result = run_policy_rollout(
        terminal,
        model=model,
        feature_config=FeatureConfig(max_candidates=4, max_left_terms=3, max_right_terms=3),
        config=RolloutConfig(max_steps=2, simulations=1, actions_per_node=1, sample_attempts=2),
        rng=np.random.default_rng(0),
    )

    assert result.steps == 0
    assert len(result.trace.records) == 0
    assert result.terminal
    assert result.comp.snapshot() == terminal.snapshot()
```

- [ ] **Step 2: Run rollout tests to verify they fail**

Run:

```bash
cd python
uv run pytest tests/test_rl_rollout.py -q
```

Expected: FAIL because `gristmill_rl.rollout` does not exist.

- [ ] **Step 3: Implement shared rollout module**

Create `python/gristmill_rl/rollout.py` by moving the rollout-specific logic out of `train.py`. The module must expose:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from gristmill_rl.actions import SampledAction, make_model_proposal_fn, sample_valid_actions
from gristmill_rl.features import FeatureConfig, extract_features
from gristmill_rl.model import PolicyValueModel
from gristmill_rl.replay import EpisodeTrace, RootTraceRecord
from gristmill_rl.search import SearchConfig, SearchNode, run_sampled_puct


@dataclass(frozen=True)
class RolloutConfig:
    max_steps: int = 4
    simulations: int = 8
    actions_per_node: int = 8
    sample_attempts: int = 64
    temperature: float = 1.0
    c_puct: float = 1.5


@dataclass(frozen=True)
class RolloutResult:
    comp: Any
    trace: EpisodeTrace
    initial_log_flops: float
    final_log_flops: float
    steps: int
    terminal: bool
    valid_action_counts: list[int]
```

Move `_sample_from_visit_counts` and `_proposal_for_node` from `train.py` into `rollout.py`. Keep `_sample_from_visit_counts` private and keep its existing validation behavior.

Add:

```python
def run_policy_rollout(
    comp: Any,
    *,
    model: PolicyValueModel,
    feature_config: FeatureConfig,
    config: RolloutConfig,
    rng: np.random.Generator,
) -> RolloutResult:
    current_comp = comp.clone()
    trace = EpisodeTrace()
    start_from = 0
    initial_log_flops = float(current_comp.log_total_flops())
    valid_action_counts: list[int] = []
    terminal = False

    for _ in range(config.max_steps):
        state_log_flops = float(current_comp.log_total_flops())
        root = SearchNode(comp=current_comp.clone(), start_from=start_from)

        def value_fn(node: SearchNode) -> float:
            child_log_flops = float(node.comp.log_total_flops())
            node.expand(
                proposal_fn=_proposal_for_node(
                    node,
                    model=model,
                    feature_config=feature_config,
                    rng=rng,
                    actions_per_node=config.actions_per_node,
                    sample_attempts=config.sample_attempts,
                )
            )
            if node.terminal or node.action_space_snapshot is None:
                return state_log_flops - child_log_flops
            features = extract_features(
                comp_snapshot=node.comp.snapshot(),
                action_space_snapshot=node.action_space_snapshot,
                start_from=node.start_from,
                log_total_flops=child_log_flops,
                config=feature_config,
            )
            return float(model(features).value)

        result = run_sampled_puct(
            root,
            config=SearchConfig(
                simulations=config.simulations,
                actions_per_node=config.actions_per_node,
                c_puct=config.c_puct,
            ),
            proposal_fn=_proposal_for_node(
                root,
                model=model,
                feature_config=feature_config,
                rng=rng,
                actions_per_node=config.actions_per_node,
                sample_attempts=config.sample_attempts,
            ),
            value_fn=value_fn,
        )

        if (
            result.selected_action is None
            or root.action_space is None
            or root.action_space_snapshot is None
            or not root.sampled_actions
        ):
            terminal = True
            break

        trace.append(
            RootTraceRecord(
                state_snapshot=current_comp.snapshot(),
                action_space_snapshot=root.action_space_snapshot,
                sampled_actions=root.sampled_actions,
                visit_distribution=result.visit_distribution,
                state_log_flops=state_log_flops,
                start_from=start_from,
            )
        )
        chosen = _sample_from_visit_counts(
            root.sampled_actions,
            result.visit_distribution,
            temperature=config.temperature,
            rng=rng,
        )
        current_comp.apply_decision_with_space(root.action_space, chosen.decision)
        start_from = int(root.action_space.def_index)
        valid_action_counts.append(result.valid_action_count)

    final_log_flops = float(current_comp.log_total_flops())
    return RolloutResult(
        comp=current_comp,
        trace=trace,
        initial_log_flops=initial_log_flops,
        final_log_flops=final_log_flops,
        steps=len(valid_action_counts),
        terminal=terminal,
        valid_action_counts=valid_action_counts,
    )
```

Keep `_proposal_for_node` using `node.action_space`; it must not call `node.comp.next_action_space(...)`.

- [ ] **Step 4: Refactor train.py to use rollout**

In `python/gristmill_rl/train.py`:

- Remove local `_sample_from_visit_counts` and `_proposal_for_node`.
- Import `RolloutConfig` and `run_policy_rollout`.
- In the episode loop, replace the inner max-step search block with:

```python
        rollout = run_policy_rollout(
            _load_comp(config.input),
            model=model,
            feature_config=feature_config,
            config=RolloutConfig(
                max_steps=config.max_steps,
                simulations=config.simulations,
                actions_per_node=config.actions_per_node,
                sample_attempts=config.sample_attempts,
                temperature=config.temperature,
                c_puct=config.c_puct,
            ),
            rng=rng,
        )

        completed_items = rollout.trace.complete(final_log_flops=rollout.final_log_flops)
        replay.extend(completed_items)
        last_initial_log_flops = rollout.initial_log_flops
        last_final_log_flops = rollout.final_log_flops
        last_episode_steps = rollout.steps
        last_episode_records = len(completed_items)
```

Keep existing per-episode metrics keys stable.

- [ ] **Step 5: Run rollout and train tests**

Run:

```bash
cd python
uv run pytest tests/test_rl_rollout.py tests/test_rl_train.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Run all Python tests**

Run:

```bash
cd python
uv run pytest tests -q
```

Expected: all Python tests pass.

- [ ] **Step 7: Commit shared rollout**

Run:

```bash
git add python/gristmill_rl/rollout.py python/gristmill_rl/train.py python/tests/test_rl_rollout.py python/tests/test_rl_train.py
git commit -m "refactor: share rl policy rollout"
```

Expected: commit succeeds.

---

### Task 4: Add Training Checkpoint In/Out Flags

**Files:**
- Modify: `python/gristmill_rl/train.py`
- Modify: `python/tests/test_rl_train.py`

- [ ] **Step 1: Add failing training checkpoint CLI tests**

Append these tests to `python/tests/test_rl_train.py`:

```python
def test_train_cli_writes_checkpoint(tmp_path):
    input_path = tmp_path / "input.json"
    checkpoint_path = tmp_path / "checkpoint"
    input_path.write_text(actionable_json())

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gristmill_rl.train",
            "--input",
            str(input_path),
            "--episodes",
            "1",
            "--max-steps",
            "1",
            "--simulations",
            "2",
            "--actions-per-node",
            "1",
            "--sample-attempts",
            "4",
            "--train-steps",
            "1",
            "--batch-size",
            "1",
            "--hidden-dim",
            "16",
            "--checkpoint-out",
            str(checkpoint_path),
            "--seed",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    metrics = json.loads(result.stdout.strip().splitlines()[-1])
    assert metrics["checkpoint_out"] == str(checkpoint_path)
    assert (checkpoint_path / "metadata.json").exists()
    assert (checkpoint_path / "state").exists()


def test_train_cli_loads_checkpoint_and_continues_training(tmp_path):
    input_path = tmp_path / "input.json"
    first_checkpoint = tmp_path / "first"
    second_checkpoint = tmp_path / "second"
    input_path.write_text(actionable_json())

    subprocess.run(
        [
            sys.executable,
            "-m",
            "gristmill_rl.train",
            "--input",
            str(input_path),
            "--episodes",
            "1",
            "--max-steps",
            "1",
            "--simulations",
            "2",
            "--actions-per-node",
            "1",
            "--sample-attempts",
            "4",
            "--train-steps",
            "1",
            "--batch-size",
            "1",
            "--hidden-dim",
            "16",
            "--checkpoint-out",
            str(first_checkpoint),
            "--seed",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gristmill_rl.train",
            "--input",
            str(input_path),
            "--episodes",
            "1",
            "--max-steps",
            "1",
            "--simulations",
            "2",
            "--actions-per-node",
            "1",
            "--sample-attempts",
            "4",
            "--train-steps",
            "1",
            "--batch-size",
            "1",
            "--checkpoint-in",
            str(first_checkpoint),
            "--checkpoint-out",
            str(second_checkpoint),
            "--seed",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    metrics = json.loads(result.stdout.strip().splitlines()[-1])
    assert metrics["checkpoint_in"] == str(first_checkpoint)
    assert metrics["checkpoint_out"] == str(second_checkpoint)
    assert metrics["replay_size"] >= 1
    assert metrics["params_changed"]
    assert (second_checkpoint / "metadata.json").exists()
```

- [ ] **Step 2: Run new train checkpoint tests to verify they fail**

Run:

```bash
cd python
uv run pytest tests/test_rl_train.py::test_train_cli_writes_checkpoint tests/test_rl_train.py::test_train_cli_loads_checkpoint_and_continues_training -q
```

Expected: FAIL because train CLI does not support checkpoint flags.

- [ ] **Step 3: Extend RunnerConfig and argument parsing**

In `python/gristmill_rl/train.py`, add imports:

```python
from gristmill_rl.checkpoint import load_checkpoint, save_checkpoint
```

Add fields to `RunnerConfig`:

```python
    hidden_dim: int | None = None
    checkpoint_in: Path | None = None
    checkpoint_out: Path | None = None
    checkpoint_overwrite: bool = False
```

Add parser arguments:

```python
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--checkpoint-in", type=Path, default=None)
    parser.add_argument("--checkpoint-out", type=Path, default=None)
    parser.add_argument("--checkpoint-overwrite", action="store_true")
```

Pass them into `RunnerConfig(...)`.

- [ ] **Step 4: Load model and feature config from checkpoint when provided**

Replace fresh model initialization in `run(config)` with:

```python
    if config.checkpoint_in is None:
        hidden_dim = config.hidden_dim if config.hidden_dim is not None else 32
        model = PolicyValueModel(hidden_dim=hidden_dim, rng_seed=config.seed)
        feature_config = FeatureConfig()
        checkpoint_in = None
    else:
        loaded = load_checkpoint(config.checkpoint_in)
        if (
            config.hidden_dim is not None
            and config.hidden_dim != loaded.metadata.hidden_dim
        ):
            raise ValueError(
                f"--hidden-dim {config.hidden_dim} does not match checkpoint "
                f"hidden_dim {loaded.metadata.hidden_dim}"
            )
        model = loaded.model
        feature_config = loaded.feature_config
        hidden_dim = loaded.metadata.hidden_dim
        checkpoint_in = str(config.checkpoint_in)
```

Keep replay, RNG, and optimizer fresh.

- [ ] **Step 5: Save final checkpoint and report metrics**

After all episodes and train steps, before returning from `run(config)`, add:

```python
    checkpoint_out = None
    if config.checkpoint_out is not None:
        save_checkpoint(
            config.checkpoint_out,
            model=model,
            feature_config=feature_config,
            hidden_dim=hidden_dim,
            metadata={"seed": config.seed, "episodes": config.episodes},
            overwrite=config.checkpoint_overwrite,
        )
        checkpoint_out = str(config.checkpoint_out)
```

Add these fields to the returned final metrics dict:

```python
        "checkpoint_in": checkpoint_in,
        "checkpoint_out": checkpoint_out,
```

Per-episode metrics may also include these fields, but the final JSON line must include them.

- [ ] **Step 6: Run training checkpoint tests**

Run:

```bash
cd python
uv run pytest tests/test_rl_train.py -q
```

Expected: all train tests pass.

- [ ] **Step 7: Run all Python tests**

Run:

```bash
cd python
uv run pytest tests -q
```

Expected: all Python tests pass.

- [ ] **Step 8: Commit training checkpoint flags**

Run:

```bash
git add python/gristmill_rl/train.py python/tests/test_rl_train.py
git commit -m "feat: add rl training checkpoints"
```

Expected: commit succeeds.

---

### Task 5: Add Checkpoint-Driven Sampling CLI

**Files:**
- Create: `python/gristmill_rl/sample.py`
- Create: `python/tests/test_rl_sample.py`

- [ ] **Step 1: Add failing sampling CLI tests**

Create `python/tests/test_rl_sample.py`:

```python
import json
import subprocess
import sys

from gristmill_symbolics import TensorComputation

from .rl_fixtures import actionable_json


def make_checkpoint(tmp_path):
    input_path = tmp_path / "input.json"
    checkpoint_path = tmp_path / "checkpoint"
    input_path.write_text(actionable_json())
    subprocess.run(
        [
            sys.executable,
            "-m",
            "gristmill_rl.train",
            "--input",
            str(input_path),
            "--episodes",
            "1",
            "--max-steps",
            "1",
            "--simulations",
            "2",
            "--actions-per-node",
            "1",
            "--sample-attempts",
            "4",
            "--train-steps",
            "1",
            "--batch-size",
            "1",
            "--hidden-dim",
            "16",
            "--checkpoint-out",
            str(checkpoint_path),
            "--seed",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return input_path, checkpoint_path


def test_sample_cli_writes_rewritten_outputs(tmp_path):
    input_path, checkpoint_path = make_checkpoint(tmp_path)
    output_dir = tmp_path / "samples"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gristmill_rl.sample",
            "--checkpoint",
            str(checkpoint_path),
            "--input",
            str(input_path),
            "--samples",
            "2",
            "--max-steps",
            "1",
            "--simulations",
            "2",
            "--actions-per-node",
            "1",
            "--sample-attempts",
            "4",
            "--temperature",
            "0.0",
            "--output-dir",
            str(output_dir),
            "--seed",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(result.stdout.strip().splitlines()[-1])
    assert summary["samples"] == 2
    assert summary["output_dir"] == str(output_dir)
    for index in range(2):
        sample_dir = output_dir / f"sample-{index:03d}"
        final_path = sample_dir / "final.json"
        metrics_path = sample_dir / "metrics.json"
        assert final_path.exists()
        assert metrics_path.exists()
        TensorComputation.load_json(final_path)
        metrics = json.loads(metrics_path.read_text())
        assert metrics["sample"] == index
        assert metrics["steps"] >= 0
        assert metrics["checkpoint"] == str(checkpoint_path)


def test_sample_cli_refuses_existing_sample_directory_without_overwrite(tmp_path):
    input_path, checkpoint_path = make_checkpoint(tmp_path)
    output_dir = tmp_path / "samples"
    sample_dir = output_dir / "sample-000"
    sample_dir.mkdir(parents=True)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gristmill_rl.sample",
            "--checkpoint",
            str(checkpoint_path),
            "--input",
            str(input_path),
            "--samples",
            "1",
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "sample-000" in result.stderr
```

- [ ] **Step 2: Run sampling tests to verify they fail**

Run:

```bash
cd python
uv run pytest tests/test_rl_sample.py -q
```

Expected: FAIL because `gristmill_rl.sample` does not exist.

- [ ] **Step 3: Implement sampling CLI**

Create `python/gristmill_rl/sample.py`:

```python
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from gristmill_symbolics import TensorComputation

from gristmill_rl.checkpoint import load_checkpoint
from gristmill_rl.rollout import RolloutConfig, run_policy_rollout


@dataclass(frozen=True)
class SampleConfig:
    checkpoint: Path
    input: Path
    output_dir: Path
    samples: int = 1
    max_steps: int = 4
    simulations: int = 8
    actions_per_node: int = 8
    sample_attempts: int = 64
    temperature: float = 1.0
    c_puct: float = 1.5
    seed: int = 0
    overwrite_output: bool = False


def parse_args(argv: Sequence[str] | None = None) -> SampleConfig:
    parser = argparse.ArgumentParser(
        description="Sample rewritten TensorComputation outputs from an RL checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=SampleConfig.samples)
    parser.add_argument("--max-steps", type=int, default=SampleConfig.max_steps)
    parser.add_argument("--simulations", type=int, default=SampleConfig.simulations)
    parser.add_argument("--actions-per-node", type=int, default=SampleConfig.actions_per_node)
    parser.add_argument("--sample-attempts", type=int, default=SampleConfig.sample_attempts)
    parser.add_argument("--temperature", type=float, default=SampleConfig.temperature)
    parser.add_argument("--c-puct", type=float, default=SampleConfig.c_puct)
    parser.add_argument("--seed", type=int, default=SampleConfig.seed)
    parser.add_argument("--overwrite-output", action="store_true")
    return SampleConfig(**vars(parser.parse_args(argv)))


def _prepare_sample_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"sample output already exists: {path}")
    path.mkdir(parents=True, exist_ok=True)


def run(config: SampleConfig) -> dict[str, int | str | float]:
    loaded = load_checkpoint(config.checkpoint)
    rng = np.random.default_rng(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rollout_config = RolloutConfig(
        max_steps=config.max_steps,
        simulations=config.simulations,
        actions_per_node=config.actions_per_node,
        sample_attempts=config.sample_attempts,
        temperature=config.temperature,
        c_puct=config.c_puct,
    )
    total_steps = 0
    best_final_log_flops: float | None = None

    for sample_index in range(config.samples):
        sample_dir = config.output_dir / f"sample-{sample_index:03d}"
        _prepare_sample_dir(sample_dir, overwrite=config.overwrite_output)
        comp = TensorComputation.load_json(config.input)
        rollout = run_policy_rollout(
            comp,
            model=loaded.model,
            feature_config=loaded.feature_config,
            config=rollout_config,
            rng=rng,
        )
        final_path = sample_dir / "final.json"
        metrics_path = sample_dir / "metrics.json"
        rollout.comp.write_json(final_path)
        metrics = {
            "sample": sample_index,
            "steps": rollout.steps,
            "terminal": rollout.terminal,
            "initial_log_flops": rollout.initial_log_flops,
            "final_log_flops": rollout.final_log_flops,
            "valid_action_counts": rollout.valid_action_counts,
            "checkpoint": str(config.checkpoint),
            "input": str(config.input),
        }
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
        print(json.dumps(metrics, sort_keys=True))
        total_steps += rollout.steps
        if best_final_log_flops is None:
            best_final_log_flops = rollout.final_log_flops
        else:
            best_final_log_flops = min(best_final_log_flops, rollout.final_log_flops)

    summary = {
        "samples": config.samples,
        "output_dir": str(config.output_dir),
        "total_steps": total_steps,
        "best_final_log_flops": float(best_final_log_flops or 0.0),
    }
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    metrics = run(parse_args(argv))
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run sampling tests**

Run:

```bash
cd python
uv run pytest tests/test_rl_sample.py -q
```

Expected: all sampling tests pass.

- [ ] **Step 5: Run all Python tests**

Run:

```bash
cd python
uv run pytest tests -q
```

Expected: all Python tests pass.

- [ ] **Step 6: Commit sampling CLI**

Run:

```bash
git add python/gristmill_rl/sample.py python/tests/test_rl_sample.py
git commit -m "feat: add rl checkpoint sampling cli"
```

Expected: commit succeeds.

---

### Task 6: Final Verification

**Files:**
- No source changes expected.

- [ ] **Step 1: Run all Python tests**

Run:

```bash
cd python
uv run pytest -q
```

Expected: all Python tests pass.

- [ ] **Step 2: Run Rust tests**

Run:

```bash
cargo test
```

Expected: all Rust tests pass.

- [ ] **Step 3: Run a manual checkpoint and sampling smoke path**

Run:

```bash
cd python
uv run python -c 'from pathlib import Path; from tests.rl_fixtures import actionable_json; Path("/tmp/gristmill-actionable.json").write_text(actionable_json())'
uv run python -m gristmill_rl.train \
  --input /tmp/gristmill-actionable.json \
  --episodes 1 \
  --max-steps 1 \
  --simulations 2 \
  --actions-per-node 1 \
  --sample-attempts 4 \
  --train-steps 1 \
  --batch-size 1 \
  --hidden-dim 16 \
  --checkpoint-out /tmp/gristmill-rl-checkpoint \
  --checkpoint-overwrite \
  --seed 0
uv run python -m gristmill_rl.sample \
  --checkpoint /tmp/gristmill-rl-checkpoint \
  --input /tmp/gristmill-actionable.json \
  --samples 2 \
  --max-steps 1 \
  --simulations 2 \
  --actions-per-node 1 \
  --sample-attempts 4 \
  --temperature 0.0 \
  --output-dir /tmp/gristmill-rl-samples \
  --overwrite-output \
  --seed 0
uv run python -c 'from gristmill_symbolics import TensorComputation; TensorComputation.load_json("/tmp/gristmill-rl-samples/sample-000/final.json"); TensorComputation.load_json("/tmp/gristmill-rl-samples/sample-001/final.json")'
```

Expected: all commands exit successfully and both sampled final JSON files load.

- [ ] **Step 4: Check git status**

Run:

```bash
git status --short
```

Expected: clean working tree.

---

## Self-Review Checklist

- Spec coverage: PyO3 JSON output is Task 1; explicit Orbax dependency and checkpoint API are Task 2; shared rollout is Task 3; training resume/save flags are Task 4; sampling CLI and output layout are Task 5; final Python/Rust/manual verification is Task 6.
- Red-flag scan: plan contains no unresolved implementation choices.
- Type consistency: `FeatureConfig`, `PolicyValueModel`, `RolloutConfig`, `RolloutResult`, `CheckpointMetadata`, and `LoadedCheckpoint` are introduced before subsequent tasks use them.
- Scope check: the plan intentionally implements param-only checkpointing only. It does not add replay, optimizer, RNG, or exact episode-counter resume.
