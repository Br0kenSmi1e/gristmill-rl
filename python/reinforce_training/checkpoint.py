from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from flax import nnx
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


def _validate_user_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata.metadata must be an object")
    return metadata


def _serialized_metadata(
    *,
    policy_config: PolicyConfig,
    train_config: TrainConfig,
    rollout_config: RolloutConfig,
    update_count: int,
    seed: int,
    metadata: dict[str, Any],
) -> str:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "package": _PACKAGE,
        "model_class": _MODEL_CLASS,
        "policy_config": asdict(policy_config),
        "train_config": asdict(train_config),
        "rollout_config": asdict(rollout_config),
        "optimizer": _OPTIMIZER,
        "learning_rate": train_config.learning_rate,
        "update_count": update_count,
        "seed": seed,
        "seed_scheme": _SEED_SCHEME,
        "metadata": metadata,
    }
    try:
        return json.dumps(payload, indent=2, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint metadata must be JSON serializable") from exc


def _write_metadata(path: Path, metadata_json: str) -> None:
    _metadata_path(path).write_text(metadata_json)


def _read_json_metadata(path: Path) -> dict[str, Any]:
    metadata_file = _metadata_path(path)
    if not metadata_file.exists():
        raise FileNotFoundError(f"checkpoint metadata not found: {metadata_file}")
    try:
        payload = json.loads(metadata_file.read_text())
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
    rollout_config = _config_from_metadata(RolloutConfig, payload, "rollout_config")
    update_count = _nonnegative_integer(payload.get("update_count"), "update_count")
    seed = _integer(payload.get("seed"), "seed")

    user_metadata = payload.get("metadata", {})
    if not isinstance(user_metadata, dict):
        raise ValueError("checkpoint metadata.metadata must be an object")

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
    metadata_payload = _validate_user_metadata(metadata)
    metadata_json = _serialized_metadata(
        policy_config=policy_config,
        train_config=train_config,
        rollout_config=rollout_config,
        update_count=update_count,
        seed=seed,
        metadata=metadata_payload,
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
