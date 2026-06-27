import pickle

import jax
import jax.numpy as jnp
import pytest

from gristmill_symbolics.policy import PolicyConfig
from gristmill_symbolics.reinforce import (
    BaselineConfig,
    CheckpointData,
    CurrentTransformerModelConfig,
    OptimizerConfig,
    ReinforceTrainerConfig,
    RewardConfig,
    TrainingError,
    UpdateMetrics,
    init_train_state,
    load_checkpoint,
    save_checkpoint,
)


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


def test_checkpoint_round_trips_protocol_train_state_configs_and_metrics(tmp_path):
    policy_config = PolicyConfig(
        d_model=8,
        num_attention_layers=2,
        id_vocab_size=64,
        init_scale=0.03,
        stop_bias_init=-7.0,
    )
    optimizer_config = OptimizerConfig(
        learning_rate=1.0e-2,
        b1=0.8,
        b2=0.95,
        eps=1.0e-5,
    )
    model_config = CurrentTransformerModelConfig(
        policy_config=policy_config,
        batch_size=2,
        max_steps=3,
        state_token_pad_to=256,
        action_token_pad_to=256,
        definition_pad_to=8,
    )
    trainer_config = ReinforceTrainerConfig(
        batch_size=2,
        optimizer_config=optimizer_config,
        reward_config=RewardConfig(),
        baseline_config=BaselineConfig(standardize=True, epsilon=1.0e-6),
    )
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
    state = init_train_state(
        policy_config,
        optimizer_config,
        seed=13,
        update_index=5,
    )
    path = tmp_path / "checkpoint.pkl"

    save_checkpoint(
        path,
        state,
        model_config=model_config,
        trainer_config=trainer_config,
        recent_metrics=recent_metrics,
    )
    loaded = load_checkpoint(path)

    assert isinstance(loaded, CheckpointData)
    assert loaded.train_state.update_index == state.update_index
    assert not hasattr(loaded.train_state, "policy")
    assert not hasattr(loaded.train_state, "optimizer_config")
    _assert_pytrees_equal(loaded.train_state.params, state.params)
    _assert_pytrees_equal(loaded.train_state.opt_state, state.opt_state)
    assert loaded.model_config == model_config
    assert loaded.trainer_config == trainer_config
    assert loaded.recent_metrics == recent_metrics
    assert isinstance(loaded.recent_metrics[0], UpdateMetrics)

    with path.open("rb") as handle:
        payload = pickle.load(handle)
    assert payload["schema_version"] == 2
    assert "tokenizer_schema_version" not in payload
    assert "rollout_config" not in payload
    assert "loss_config" not in payload
    assert "policy_config" in payload
    assert "policy_params" in payload
    assert "optimizer_config" in payload
    assert "optimizer_state" in payload
    assert set(payload["model_config"]) == {
        "batch_size",
        "max_steps",
        "state_token_pad_to",
        "action_token_pad_to",
        "definition_pad_to",
    }
    assert set(payload["trainer_config"]) == {
        "batch_size",
        "reward_config",
        "baseline_config",
    }


def test_checkpoint_restores_root_key_as_jax_uint32_array(tmp_path):
    policy_config = PolicyConfig(d_model=8)
    optimizer_config = OptimizerConfig(learning_rate=1.0e-2)
    state = init_train_state(
        policy_config,
        optimizer_config,
        seed=13,
    )
    path = tmp_path / "checkpoint.pkl"

    save_checkpoint(
        path,
        state,
        model_config=CurrentTransformerModelConfig(
            policy_config=policy_config,
            batch_size=2,
            max_steps=1,
            state_token_pad_to=128,
            action_token_pad_to=128,
            definition_pad_to=4,
        ),
        trainer_config=ReinforceTrainerConfig(
            batch_size=2,
            optimizer_config=optimizer_config,
        ),
        recent_metrics=(),
    )

    loaded = load_checkpoint(path)

    assert isinstance(loaded.train_state.root_key, jnp.ndarray)
    assert loaded.train_state.root_key.dtype == jnp.uint32
    assert jnp.array_equal(loaded.train_state.root_key, state.root_key)


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
