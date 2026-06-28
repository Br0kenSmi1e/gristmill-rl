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


def save_checkpoint(
    path,
    train_state: TrainState,
    *,
    model,
    trainer,
    recent_metrics: tuple[UpdateMetrics, ...],
) -> None:
    model_payload = _model_payload(model)
    trainer_payload = _trainer_payload(trainer)
    if model.batch_size != trainer.batch_size:
        raise TrainingError("model batch_size must match trainer batch_size")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model": model_payload,
        "trainer": trainer_payload,
        "policy_params": train_state.params,
        "optimizer_state": train_state.opt_state,
        "update_index": int(train_state.update_index),
        "root_key": np.asarray(train_state.root_key, dtype=np.uint32),
        "recent_metrics": tuple(asdict(metrics) for metrics in recent_metrics),
    }
    with open(path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


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
        root_key = jnp.asarray(payload["root_key"], dtype=jnp.uint32)
        if root_key.shape != (2,):
            raise TrainingError("root_key must have shape (2,)")
        train_state = TrainState(
            params=payload["policy_params"],
            opt_state=payload["optimizer_state"],
            root_key=root_key,
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
