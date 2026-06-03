from __future__ import annotations

import json
import math
import shutil
import uuid
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path
from typing import Any

from flax import nnx
import jax
import numpy as np
import orbax.checkpoint as ocp

from reinforce_training.objective import TrainConfig, create_optimizer
from reinforce_training.rollout import PolicyConfig, RolloutConfig

SCHEMA_VERSION = 1

_PACKAGE = "reinforce_training"
_MODEL_CLASS = "CausalTransformerScorer"
_OPTIMIZER = "adam"
_SEED_SCHEME = "seed + update_index * batch_size + episode_index"


@dataclass(frozen=True)
class LoadedCheckpoint:
    scorer: Any
    optimizer: Any
    policy_config: PolicyConfig
    train_config: TrainConfig
    rollout_config: RolloutConfig
    update_count: int
    seed: int
    metadata: dict[str, Any]


def _metadata_path(path: Path) -> Path:
    return path / "metadata.json"


def _state_path(path: Path) -> Path:
    return path / "state"


def _temporary_checkpoint_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")


def _backup_checkpoint_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.bak-{uuid.uuid4().hex}")


def _checkpoint_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _nonnegative_integer(value: Any, field_name: str) -> int:
    value = _integer(value, field_name)
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _learning_rate(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be finite and positive")
    learning_rate = float(value)
    if learning_rate <= 0.0 or not math.isfinite(learning_rate):
        raise ValueError(f"{field_name} must be finite and positive")
    return learning_rate


def _train_config_payload(train_config: TrainConfig, field_name: str) -> dict[str, float]:
    return {"learning_rate": _learning_rate(train_config.learning_rate, field_name)}


def _json_value_path(path: str, key: str) -> str:
    if key.isidentifier():
        return f"{path}.{key}"
    return f"{path}[{key!r}]"


def _validate_json_value(value: Any, path: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} keys must be strings")
            _validate_json_value(child, _json_value_path(path, key))
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _validate_json_value(child, f"{path}[{index}]")
        return
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite")
        return
    raise ValueError(f"{path} must be JSON-compatible")


def _validate_user_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata.metadata must be an object")
    _validate_json_value(metadata, "checkpoint metadata.metadata")
    return metadata


def _state_shape_signature(
    state: Any,
) -> tuple[tuple[tuple[str, ...], tuple[int, ...], str], ...]:
    leaves_with_path, _ = jax.tree_util.tree_flatten_with_path(state)
    return tuple(
        (
            tuple(str(part) for part in path),
            tuple(np.asarray(leaf).shape),
            str(np.asarray(leaf).dtype),
        )
        for path, leaf in leaves_with_path
    )


def _validate_state_shape_matches(
    actual_state: Any,
    expected_state: Any,
    message: str,
) -> None:
    if _state_shape_signature(actual_state) != _state_shape_signature(expected_state):
        raise ValueError(message)


def _iter_nested_optimizer_functions(value: Any, *, depth: int = 0, seen=None):
    if seen is None:
        seen = set()
    if depth > 8 or id(value) in seen:
        return
    seen.add(id(value))

    if callable(value):
        yield value
        closure = getattr(value, "__closure__", None)
        if closure is None:
            return
        for cell in closure:
            try:
                child = cell.cell_contents
            except ValueError:
                continue
            yield from _iter_nested_optimizer_functions(
                child,
                depth=depth + 1,
                seen=seen,
            )
        return

    if isinstance(value, (tuple, list)):
        for child in value:
            yield from _iter_nested_optimizer_functions(
                child,
                depth=depth + 1,
                seen=seen,
            )
        return

    for attr_name in ("init", "update"):
        child = getattr(value, attr_name, None)
        if callable(child):
            yield from _iter_nested_optimizer_functions(
                child,
                depth=depth + 1,
                seen=seen,
            )


def _extract_create_optimizer_learning_rate(optimizer) -> float:
    """Extract LR from the Optax Adam transform produced by create_optimizer."""
    tx = getattr(optimizer, "tx", None)
    update_fn = getattr(tx, "update", None)
    if not callable(update_fn):
        raise ValueError(
            "optimizer learning_rate could not be verified; use create_optimizer"
        )

    learning_rates = []
    for function in _iter_nested_optimizer_functions(update_fn):
        if getattr(function, "__qualname__", None) != "scale.<locals>.update_fn":
            continue
        closure = getattr(function, "__closure__", None) or ()
        values = []
        for cell in closure:
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            if isinstance(value, Real) and not isinstance(value, bool):
                values.append(float(value))
        if len(values) == 1 and values[0] < 0.0 and math.isfinite(values[0]):
            learning_rates.append(-values[0])

    if len(learning_rates) != 1:
        raise ValueError(
            "optimizer learning_rate could not be verified; use create_optimizer"
        )
    return learning_rates[0]


def _validate_optimizer_learning_rate(optimizer, train_config: TrainConfig) -> None:
    optimizer_learning_rate = _extract_create_optimizer_learning_rate(optimizer)
    if not math.isclose(
        optimizer_learning_rate,
        train_config.learning_rate,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError(
            "optimizer learning_rate does not match train_config.learning_rate"
        )


def _validate_checkpoint_state_shapes(
    *,
    scorer,
    optimizer,
    policy_config: PolicyConfig,
    train_config: TrainConfig,
) -> None:
    expected_scorer = policy_config.create_scorer(seed=0)
    _validate_state_shape_matches(
        nnx.state(scorer, nnx.Param),
        nnx.state(expected_scorer, nnx.Param),
        "scorer state shape does not match policy_config",
    )
    expected_optimizer = create_optimizer(expected_scorer, train_config)
    # Optax/NNX state does not expose the original optimizer hyperparameters in a
    # stable way here, so this validates topology compatibility by state shape.
    _validate_state_shape_matches(
        nnx.state(optimizer),
        nnx.state(expected_optimizer),
        "optimizer state shape does not match policy_config and train_config",
    )
    _validate_optimizer_learning_rate(optimizer, train_config)


def _serialized_metadata(
    *,
    policy_config: PolicyConfig,
    train_config: TrainConfig,
    rollout_config: RolloutConfig,
    update_count: int,
    seed: int,
    metadata: dict[str, Any],
) -> str:
    train_config_payload = _train_config_payload(
        train_config,
        "checkpoint metadata.train_config.learning_rate",
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "package": _PACKAGE,
        "model_class": _MODEL_CLASS,
        "policy_config": asdict(policy_config),
        "train_config": train_config_payload,
        "rollout_config": asdict(rollout_config),
        "optimizer": _OPTIMIZER,
        "learning_rate": train_config_payload["learning_rate"],
        "update_count": update_count,
        "seed": seed,
        "seed_scheme": _SEED_SCHEME,
        "metadata": metadata,
    }
    try:
        return json.dumps(payload, allow_nan=False, indent=2, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint metadata must be JSON serializable") from exc


def _write_metadata(path: Path, metadata_json: str) -> None:
    _metadata_path(path).write_text(metadata_json)


def _reject_json_constant(value: str):
    raise ValueError(
        f"checkpoint metadata must be strict JSON; invalid constant {value}"
    )


def _read_json_metadata(path: Path) -> dict[str, Any]:
    metadata_file = _metadata_path(path)
    if not metadata_file.exists():
        raise FileNotFoundError(f"checkpoint metadata not found: {metadata_file}")
    try:
        payload = json.loads(
            metadata_file.read_text(),
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError("checkpoint metadata must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("checkpoint metadata must be an object")
    return payload


def _metadata_object(payload: dict[str, Any], field_name: str) -> dict[str, Any]:
    value = payload.get(field_name)
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint metadata.{field_name} must be an object")
    return value


def _config_from_metadata(config_type, payload: dict[str, Any], field_name: str):
    try:
        return config_type(**_metadata_object(payload, field_name))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"checkpoint metadata.{field_name} is invalid") from exc


def _read_metadata(
    path: Path,
) -> tuple[PolicyConfig, TrainConfig, RolloutConfig, int, int, dict[str, Any]]:
    payload = _read_json_metadata(path)

    schema_version = payload.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported checkpoint schema_version {schema_version}")

    package = payload.get("package")
    if package != _PACKAGE:
        raise ValueError(f"Unsupported checkpoint package {package!r}")

    model_class = payload.get("model_class")
    if model_class != _MODEL_CLASS:
        raise ValueError(f"Unsupported checkpoint model_class {model_class!r}")

    optimizer = payload.get("optimizer")
    if optimizer != _OPTIMIZER:
        raise ValueError(f"Unsupported checkpoint optimizer {optimizer!r}")

    seed_scheme = payload.get("seed_scheme")
    if seed_scheme != _SEED_SCHEME:
        raise ValueError(f"Unsupported checkpoint seed_scheme {seed_scheme!r}")

    policy_config = _config_from_metadata(PolicyConfig, payload, "policy_config")
    train_config = _config_from_metadata(TrainConfig, payload, "train_config")
    train_learning_rate = _learning_rate(
        train_config.learning_rate,
        "checkpoint metadata.train_config.learning_rate",
    )
    metadata_learning_rate = _learning_rate(
        payload.get("learning_rate"),
        "checkpoint metadata.learning_rate",
    )
    if metadata_learning_rate != train_learning_rate:
        raise ValueError(
            "checkpoint metadata.learning_rate must match train_config.learning_rate"
        )
    train_config = TrainConfig(learning_rate=train_learning_rate)
    rollout_config = _config_from_metadata(RolloutConfig, payload, "rollout_config")
    update_count = _nonnegative_integer(payload.get("update_count"), "update_count")
    seed = _integer(payload.get("seed"), "seed")

    user_metadata = payload.get("metadata", {})
    if not isinstance(user_metadata, dict):
        raise ValueError("checkpoint metadata.metadata must be an object")
    _validate_json_value(user_metadata, "checkpoint metadata.metadata")

    return (
        policy_config,
        train_config,
        rollout_config,
        update_count,
        seed,
        user_metadata,
    )


def _state_payload(scorer, optimizer) -> dict[str, Any]:
    return {
        "model": nnx.state(scorer, nnx.Param),
        "optimizer": nnx.state(optimizer),
    }


def _publish_checkpoint(temp_path: Path, checkpoint_path: Path, *, overwrite: bool) -> None:
    backup_path: Path | None = None
    try:
        if checkpoint_path.exists():
            if not overwrite:
                raise FileExistsError(f"checkpoint path already exists: {checkpoint_path}")
            backup_path = _backup_checkpoint_path(checkpoint_path)
            checkpoint_path.rename(backup_path)

        temp_path.rename(checkpoint_path)
    except Exception:
        if (
            backup_path is not None
            and backup_path.exists()
            and not checkpoint_path.exists()
        ):
            backup_path.rename(checkpoint_path)
        raise
    else:
        if backup_path is not None:
            _remove_path(backup_path)


def save_checkpoint(
    path: str | Path,
    *,
    scorer,
    optimizer,
    policy_config: PolicyConfig,
    train_config: TrainConfig,
    rollout_config: RolloutConfig,
    update_count: int,
    seed: int,
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> None:
    update_count = _nonnegative_integer(update_count, "update_count")
    seed = _integer(seed, "seed")
    train_config = TrainConfig(
        learning_rate=_learning_rate(
            train_config.learning_rate,
            "checkpoint metadata.train_config.learning_rate",
        )
    )
    metadata_payload = _validate_user_metadata(metadata)
    metadata_json = _serialized_metadata(
        policy_config=policy_config,
        train_config=train_config,
        rollout_config=rollout_config,
        update_count=update_count,
        seed=seed,
        metadata=metadata_payload,
    )
    _validate_checkpoint_state_shapes(
        scorer=scorer,
        optimizer=optimizer,
        policy_config=policy_config,
        train_config=train_config,
    )

    checkpoint_path = _checkpoint_path(path)
    if checkpoint_path.exists() and not overwrite:
        raise FileExistsError(f"checkpoint path already exists: {checkpoint_path}")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temporary_checkpoint_path(checkpoint_path)
    try:
        temp_path.mkdir()
        ocp.PyTreeCheckpointer().save(
            _state_path(temp_path),
            _state_payload(scorer, optimizer),
            force=True,
        )
        _write_metadata(temp_path, metadata_json)
        _publish_checkpoint(temp_path, checkpoint_path, overwrite=overwrite)
    except Exception:
        _remove_path(temp_path)
        raise


def load_checkpoint(path: str | Path) -> LoadedCheckpoint:
    checkpoint_path = _checkpoint_path(path)
    (
        policy_config,
        train_config,
        rollout_config,
        update_count,
        seed,
        metadata,
    ) = _read_metadata(checkpoint_path)

    scorer = policy_config.create_scorer(seed=0)
    optimizer = create_optimizer(scorer, train_config)
    abstract_state = _state_payload(scorer, optimizer)
    restore_args = ocp.checkpoint_utils.construct_restore_args(abstract_state)
    restored_state = ocp.PyTreeCheckpointer().restore(
        _state_path(checkpoint_path),
        item=abstract_state,
        restore_args=restore_args,
    )
    nnx.update(scorer, restored_state["model"])
    nnx.update(optimizer, restored_state["optimizer"])

    return LoadedCheckpoint(
        scorer=scorer,
        optimizer=optimizer,
        policy_config=policy_config,
        train_config=train_config,
        rollout_config=rollout_config,
        update_count=update_count,
        seed=seed,
        metadata=metadata,
    )
