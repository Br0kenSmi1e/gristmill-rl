import pickle

import jax.numpy as jnp
import pytest

from gristmill_symbolics._training import TrainingError
from gristmill_symbolics.cli.checkpoint import load_checkpoint
from gristmill_symbolics.cli.checkpoint import save_checkpoint
from gristmill_symbolics.cli.train_state import init_train_state
from gristmill_symbolics.model.transformer_action_selector import (
    TransformerActionSelectorModel,
)
from gristmill_symbolics.trainer.reinforce import ReinforceTrainer


def _model():
    return TransformerActionSelectorModel(
        state_token_pad_to=128,
        action_token_pad_to=128,
        definition_pad_to=4,
        candidate_pad_to=8,
        side_term_pad_to=8,
        d_model=8,
    )


def _trainer(*, batch_size=2, max_steps=1):
    return ReinforceTrainer(
        batch_size=batch_size,
        max_steps=max_steps,
        learning_rate=1.0e-2,
    )


def test_checkpoint_restores_root_key_as_jax_uint32_array(tmp_path):
    model = _model()
    trainer = _trainer()
    state = init_train_state(model, trainer, seed=13)
    path = tmp_path / "checkpoint.pkl"

    save_checkpoint(path, state, model=model, trainer=trainer, recent_metrics=())

    loaded = load_checkpoint(path)

    assert isinstance(loaded.train_state.root_key, jnp.ndarray)
    assert loaded.train_state.root_key.dtype == jnp.uint32
    assert jnp.array_equal(loaded.train_state.root_key, state.root_key)


def test_checkpoint_allows_trainer_batch_size_to_be_checkpoint_local(tmp_path):
    model = _model()
    trainer = _trainer(batch_size=2)
    state = init_train_state(model, trainer, seed=13)
    path = tmp_path / "checkpoint.pkl"
    save_checkpoint(path, state, model=model, trainer=trainer, recent_metrics=())
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    payload["trainer"]["kwargs"]["batch_size"] = 3
    with path.open("wb") as handle:
        pickle.dump(payload, handle)

    loaded = load_checkpoint(path)

    assert loaded.trainer.batch_size == 3


def test_checkpoint_rejects_old_schema_version(tmp_path):
    path = tmp_path / "bad.pkl"
    with path.open("wb") as handle:
        pickle.dump({"schema_version": 1}, handle)

    with pytest.raises(TrainingError, match="checkpoint schema"):
        load_checkpoint(path)


def test_checkpoint_rejects_unknown_schema_version(tmp_path):
    path = tmp_path / "bad.pkl"
    with path.open("wb") as handle:
        pickle.dump({"schema_version": 999}, handle)

    with pytest.raises(TrainingError, match="checkpoint schema"):
        load_checkpoint(path)
