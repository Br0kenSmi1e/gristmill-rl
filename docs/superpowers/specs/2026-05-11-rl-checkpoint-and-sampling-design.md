# RL Checkpointing And Sampling Design

## Summary

Add three persistence features to the RL prototype:

1. Python bindings can write `TensorComputation` values back to canonical JSON.
2. Training can save a param-only Flax NNX/Orbax checkpoint and load one to continue training.
3. A sampling command can load a checkpoint and write learned rewrite outputs as JSON computations.

This design keeps the Rust/PyO3 rewrite API fixed in behavior. The only PyO3 change is exposing existing Rust JSON serialization functions to Python.

## Goals

- Save the current neural network after training in an Orbax-style checkpoint directory.
- Load a checkpoint to reconstruct the same `PolicyValueModel` architecture and restore its parameters.
- Continue training from restored model parameters without restoring replay, optimizer state, RNG state, or episode counters.
- Load a checkpoint for inference-only sampling and write final rewritten `TensorComputation` JSON files.
- Write sampled outputs through Rust serialization, not by dumping Python snapshots.
- Share the model-guided rewrite episode logic between training and sampling.

## Non-Goals

- Full training resume with optimizer state, replay buffer, RNG state, or exact episode counter.
- A model registry or checkpoint retention policy.
- Step-by-step output snapshots during sampling.
- Changes to Rust rewrite semantics, `ActionSpace`, or action ordering.
- A new `RlEnv` abstraction.

## PyO3 JSON Serialization

Expose two methods on Python `TensorComputation`:

```python
comp.to_json_string() -> str
comp.write_json(path) -> None
```

Both methods delegate to the existing Rust `io::to_json` and `io::write_json` functions. This keeps Python sampling output identical to the Rust CLI JSON path.

`write_json` accepts a filesystem path and creates or replaces that file. Parent directory creation remains the responsibility of the Python caller, matching normal CLI behavior.

Tests cover:

- `to_json_string()` returns JSON that `TensorComputation.from_json_string()` can load.
- `write_json(path)` writes JSON that `TensorComputation.load_json(path)` can load.
- A rewritten computation written from Python round-trips through the Rust validation path.

## Checkpoint Format

Checkpoints are directories, following the usual Orbax/Flax style:

```text
checkpoint/
  metadata.json
  state/
```

`metadata.json` stores the data needed to reconstruct the model around the saved parameters:

```json
{
  "schema_version": 1,
  "model": {
    "class": "PolicyValueModel",
    "hidden_dim": 32
  },
  "features": {
    "max_candidates": 16,
    "max_left_terms": 8,
    "max_right_terms": 8
  },
  "metadata": {
    "seed": 0,
    "episodes": 100
  }
}
```

`state/` stores the Flax NNX model state through `orbax.checkpoint`. The Python package should add an explicit `orbax-checkpoint` runtime dependency because `gristmill_rl` imports it directly, even if Flax already brings Orbax transitively.

The checkpoint is param-only. Reconstructing a model requires:

- the installed `gristmill_rl.model.PolicyValueModel` code,
- the architecture config from `metadata.json`,
- the `FeatureConfig` shape from `metadata.json`,
- the saved NNX state from `state/`.

It does not include optimizer state, replay records, RNG state, or counters.

## Checkpoint API

Add `python/gristmill_rl/checkpoint.py` with:

```python
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


def save_checkpoint(
    path: Path,
    *,
    model: PolicyValueModel,
    feature_config: FeatureConfig,
    hidden_dim: int,
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> None: ...


def load_checkpoint(path: Path) -> LoadedCheckpoint: ...
```

`save_checkpoint` refuses to overwrite an existing checkpoint directory unless `overwrite=True`. It creates parent directories as needed.

`load_checkpoint` validates `schema_version`, reconstructs `PolicyValueModel(hidden_dim=...)`, restores the NNX state, and returns the model plus feature config. Unknown schema versions fail with a clear `ValueError`.

## Shared Rollout Logic

Training and sampling should share a single model-guided episode runner instead of duplicating search/proposal code.

Add `python/gristmill_rl/rollout.py` with:

```python
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
    comp: TensorComputation
    trace: EpisodeTrace
    initial_log_flops: float
    final_log_flops: float
    steps: int
    terminal: bool
    valid_action_counts: list[int]
```

`run_policy_rollout(comp, *, model, feature_config, config, rng) -> RolloutResult` performs one rewrite episode:

- uses `SearchNode(comp=current_comp.clone(), start_from=start_from)`,
- proposes actions with `make_model_proposal_fn` and `sample_valid_actions`,
- validates actions through the `ActionSpace` stored by `SearchNode.expand`,
- evaluates children with the model or immediate terminal flop improvement,
- appends replay `RootTraceRecord` entries using the root node's stored action-space snapshot and sampled actions,
- applies the chosen executable action to the episode's mutable computation,
- never recomputes an action space for a node that has already expanded.

`train.py` uses `RolloutResult.trace` to complete replay items. `sample.py` uses `RolloutResult.comp.write_json(...)` to write outputs.

## Training CLI Changes

Extend `python -m gristmill_rl.train` with:

```text
--checkpoint-in PATH
--checkpoint-out PATH
--checkpoint-overwrite
--hidden-dim N
```

Behavior:

- Without `--checkpoint-in`, training starts a fresh `PolicyValueModel(hidden_dim=--hidden-dim, rng_seed=--seed)`.
- With `--checkpoint-in`, training loads model parameters, model architecture, and feature config from the checkpoint.
- If `--checkpoint-in` is set and `--hidden-dim` is explicitly provided with a value that differs from the checkpoint metadata, training fails before starting.
- Replay buffer, optimizer state, and RNG are fresh even when loading a checkpoint.
- With `--checkpoint-out`, training saves the final model after all requested episodes.
- The final JSON metrics include `checkpoint_in` and `checkpoint_out` string fields when present.

Example:

```bash
uv run python -m gristmill_rl.train \
  --input input.json \
  --episodes 100 \
  --checkpoint-out checkpoints/run-001
```

Continue param-only training:

```bash
uv run python -m gristmill_rl.train \
  --input input.json \
  --episodes 50 \
  --checkpoint-in checkpoints/run-001 \
  --checkpoint-out checkpoints/run-002
```

## Sampling CLI

Add `python/gristmill_rl/sample.py`.

Example:

```bash
uv run python -m gristmill_rl.sample \
  --checkpoint checkpoints/run-001 \
  --input input.json \
  --samples 8 \
  --max-steps 20 \
  --simulations 8 \
  --actions-per-node 4 \
  --sample-attempts 32 \
  --temperature 0.2 \
  --output-dir outputs/run-001 \
  --seed 0
```

For each sample, the command loads a fresh input computation, runs one model-guided rollout without training, and writes:

```text
outputs/run-001/
  sample-000/
    final.json
    metrics.json
  sample-001/
    final.json
    metrics.json
```

`final.json` is written with `TensorComputation.write_json`. `metrics.json` contains:

- sample index,
- steps,
- terminal flag,
- initial and final log flops,
- valid action counts per step,
- checkpoint path,
- input path.

The command also prints one JSON metrics line per sample and a final summary JSON line.

## Error Handling

- Loading a missing checkpoint fails before training or sampling starts.
- Checkpoint metadata schema mismatch fails with `ValueError`.
- Existing checkpoint output directories are refused unless `--checkpoint-overwrite` is set.
- Sampling creates `--output-dir` if needed. Existing per-sample directories are refused unless `--overwrite-output` is set.
- If a sampled input has no action space, sampling still writes `final.json` identical to the input plus metrics with `steps: 0`.
- JSON output parent directories are created by the Python CLI before calling `TensorComputation.write_json`.

## Tests

PyO3 serialization tests:

- `to_json_string` round-trips a fixture.
- `write_json` round-trips a fixture through `load_json`.
- Rewritten computation output from Python round-trips.

Checkpoint tests:

- Save a model, mutate a fresh model, load checkpoint, and verify predictions match the saved model on fixed features.
- Loading missing or unsupported schema checkpoint fails clearly.
- Existing checkpoint directory is refused unless overwrite is enabled.

Training tests:

- Existing tiny training smoke test still passes.
- `--checkpoint-out` creates a checkpoint with metadata and state.
- `--checkpoint-in` can load that checkpoint and complete another tiny training run.

Sampling tests:

- Save a tiny checkpoint, run `python -m gristmill_rl.sample`, and verify `sample-000/final.json` and `metrics.json` exist.
- `final.json` loads through `TensorComputation.load_json`.
- Sampling a no-action-space fixture writes output with zero steps.

Regression tests:

- Shared rollout does not recompute action space during proposal validation.
- Sampling and training both use the same stored action-space snapshot/action list semantics for replay or output decisions.

## Implementation Notes

- Keep the direct `TensorComputation`/`ActionSpace` style. Do not add an environment wrapper.
- Keep model persistence separate from replay persistence.
- Keep `SampledAction.decision` as the PyO3-executable decision. `score_decision` remains model-only scoring metadata.
- Avoid storing Python snapshots as final computation output; snapshots are for features and tests, while final outputs use Rust serialization.
