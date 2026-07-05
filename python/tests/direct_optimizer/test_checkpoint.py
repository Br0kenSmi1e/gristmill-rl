from dataclasses import FrozenInstanceError
import json

import jax
import pytest
from flax import nnx

from gristmill_symbolics.direct_optimizer import checkpoint as checkpoint_module
from gristmill_symbolics.direct_optimizer.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    load_checkpoint,
    load_model_for_inference,
    save_checkpoint,
)
from gristmill_symbolics.direct_optimizer.converter import CONVERTER_SCHEMA_VERSION
from gristmill_symbolics.direct_optimizer.model import DirectOptimizerTransformer
from gristmill_symbolics.direct_optimizer.trainer import DirectOptimizerTrainer


def _model(**overrides):
    kwargs = {
        "source_len": 32,
        "target_len": 32,
        "scalar_value_min": -16,
        "scalar_value_max": 16,
        "d_model": 8,
        "num_layers": 1,
        "num_heads": 1,
        "rngs": nnx.Rngs(0),
    }
    kwargs.update(overrides)
    return DirectOptimizerTransformer(**kwargs)


def test_checkpoint_round_trips_model_optimizer_and_metadata(tmp_path):
    model = _model()
    trainer = DirectOptimizerTrainer(batch_size=2, learning_rate=1.0e-3)
    optimizer = trainer.init_optimizer(model)

    save_checkpoint(
        tmp_path,
        model=model,
        optimizer=optimizer,
        trainer=trainer,
        epoch=2,
        updates=5,
        last_train_loss=1.25,
        last_valid_loss=1.5,
    )

    loaded = load_checkpoint(tmp_path)

    assert loaded.metadata["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert loaded.metadata["converter_schema_version"] == CONVERTER_SCHEMA_VERSION
    assert loaded.metadata["epoch"] == 2
    assert loaded.metadata["updates"] == 5
    assert loaded.metadata["last_train_loss"] == pytest.approx(1.25)
    assert loaded.metadata["last_valid_loss"] == pytest.approx(1.5)
    assert loaded.metadata["model_kwargs"] == model.model_kwargs()
    assert loaded.metadata["trainer_kwargs"]["batch_size"] == 2
    assert jax.tree_util.tree_structure(nnx.state(loaded.model)) == (
        jax.tree_util.tree_structure(nnx.state(model))
    )
    assert loaded.optimizer is not None
    assert jax.tree_util.tree_structure(nnx.state(loaded.optimizer)) == (
        jax.tree_util.tree_structure(nnx.state(optimizer))
    )
    with pytest.raises(FrozenInstanceError):
        loaded.metadata = {}


def test_checkpoint_rejects_incompatible_static_model_kwargs(tmp_path):
    model = _model()
    trainer = DirectOptimizerTrainer(batch_size=2, learning_rate=1.0e-3)
    optimizer = trainer.init_optimizer(model)
    save_checkpoint(
        tmp_path,
        model=model,
        optimizer=optimizer,
        trainer=trainer,
        epoch=0,
        updates=0,
        last_train_loss=1.0,
    )

    with pytest.raises(ValueError, match="source_len"):
        load_checkpoint(
            tmp_path,
            expected_model_kwargs={**model.model_kwargs(), "source_len": 64},
        )


def test_failed_checkpoint_overwrite_leaves_existing_checkpoint_loadable(
    tmp_path,
    monkeypatch,
):
    model = _model()
    trainer = DirectOptimizerTrainer(batch_size=2, learning_rate=1.0e-3)
    optimizer = trainer.init_optimizer(model)
    save_checkpoint(
        tmp_path,
        model=model,
        optimizer=optimizer,
        trainer=trainer,
        epoch=1,
        updates=2,
        last_train_loss=3.0,
    )

    original_checkpointer = checkpoint_module.ocp.StandardCheckpointer
    save_calls = 0

    class FailingCheckpointer:
        def __init__(self):
            self._inner = original_checkpointer()

        def save(self, *args, **kwargs):
            nonlocal save_calls
            save_calls += 1
            if save_calls == 2:
                raise RuntimeError("injected checkpoint save failure")
            return self._inner.save(*args, **kwargs)

        def wait_until_finished(self):
            return self._inner.wait_until_finished()

        def restore(self, *args, **kwargs):
            return self._inner.restore(*args, **kwargs)

    monkeypatch.setattr(
        checkpoint_module.ocp,
        "StandardCheckpointer",
        FailingCheckpointer,
    )

    with pytest.raises(RuntimeError, match="injected checkpoint save failure"):
        save_checkpoint(
            tmp_path,
            model=model,
            optimizer=optimizer,
            trainer=trainer,
            epoch=9,
            updates=10,
            last_train_loss=11.0,
        )

    loaded = load_checkpoint(tmp_path)

    assert loaded.metadata["epoch"] == 1
    assert loaded.metadata["updates"] == 2
    assert loaded.metadata["last_train_loss"] == pytest.approx(3.0)


def test_load_model_for_inference_ignores_optimizer_state(tmp_path):
    model = _model()
    trainer = DirectOptimizerTrainer(batch_size=2, learning_rate=1.0e-3)
    optimizer = trainer.init_optimizer(model)
    save_checkpoint(
        tmp_path,
        model=model,
        optimizer=optimizer,
        trainer=trainer,
        epoch=1,
        updates=3,
        last_train_loss=2.0,
    )

    loaded_model, metadata = load_model_for_inference(tmp_path)

    assert isinstance(loaded_model, DirectOptimizerTransformer)
    assert metadata["model_kwargs"] == model.model_kwargs()


def test_checkpoint_rejects_unsupported_schema_version(tmp_path):
    model = _model()
    trainer = DirectOptimizerTrainer(batch_size=2, learning_rate=1.0e-3)
    optimizer = trainer.init_optimizer(model)
    save_checkpoint(
        tmp_path,
        model=model,
        optimizer=optimizer,
        trainer=trainer,
        epoch=0,
        updates=0,
        last_train_loss=1.0,
    )
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["schema_version"] = CHECKPOINT_SCHEMA_VERSION + 1
    metadata_path.write_text(json.dumps(metadata, sort_keys=True))

    with pytest.raises(ValueError, match="schema_version"):
        load_checkpoint(tmp_path)


def test_checkpoint_rejects_converter_schema_mismatch(tmp_path):
    model = _model()
    trainer = DirectOptimizerTrainer(batch_size=2, learning_rate=1.0e-3)
    optimizer = trainer.init_optimizer(model)
    save_checkpoint(
        tmp_path,
        model=model,
        optimizer=optimizer,
        trainer=trainer,
        epoch=0,
        updates=0,
        last_train_loss=1.0,
    )
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["converter_schema_version"] = CONVERTER_SCHEMA_VERSION + 1
    metadata_path.write_text(json.dumps(metadata, sort_keys=True))

    with pytest.raises(ValueError, match="converter_schema_version"):
        load_checkpoint(tmp_path)
