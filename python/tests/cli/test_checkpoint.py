import pickle

import jax
import jax.numpy as jnp
import pytest

from gristmill_symbolics._training import TrainingError
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


def test_checkpoint_rejects_root_key_with_invalid_shape(tmp_path):
    model = _model()
    trainer = _trainer()
    path = tmp_path / "bad-root-key.pkl"
    with path.open("wb") as handle:
        pickle.dump(
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "model": {
                    "kind": "transformer_action_selector",
                    "kwargs": model.constructor_kwargs(),
                },
                "trainer": {
                    "kind": "reinforce",
                    "kwargs": trainer.constructor_kwargs(),
                },
                "policy_params": {},
                "optimizer_state": {},
                "root_key": [[0, 0]],
                "update_index": 0,
                "recent_metrics": (),
            },
            handle,
        )

    with pytest.raises(TrainingError, match=r"root_key must have shape \(2,\)"):
        load_checkpoint(path)


def test_save_checkpoint_rejects_unsupported_model_without_batch_size(tmp_path):
    state = init_train_state(_model(), _trainer(), seed=17)

    with pytest.raises(TrainingError, match="unsupported model type"):
        save_checkpoint(
            tmp_path / "checkpoint.pkl",
            state,
            model=object(),
            trainer=_trainer(),
            recent_metrics=(),
        )
