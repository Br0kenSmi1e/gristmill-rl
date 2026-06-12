import pickle

import jax
import jax.numpy as jnp
import pytest

from gristmill_symbolics.policy import PolicyConfig
from gristmill_symbolics.reinforce import (
    BaselineConfig,
    CheckpointData,
    LossConfig,
    OptimizerConfig,
    RewardConfig,
    RolloutConfig,
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


def test_checkpoint_round_trips_train_state_configs_and_metrics(tmp_path):
    policy_config = PolicyConfig(
        d_model=8,
        num_attention_layers=2,
        id_vocab_size=64,
        max_candidates=8,
        max_side_terms=4,
        init_scale=0.03,
        stop_bias_init=-7.0,
    )
    optimizer_config = OptimizerConfig(
        learning_rate=1.0e-2,
        b1=0.8,
        b2=0.95,
        eps=1.0e-5,
    )
    rollout_config = RolloutConfig(batch_size=2, max_steps=3, seed=17)
    baseline_config = BaselineConfig(standardize=True, epsilon=1.0e-6)
    loss_config = LossConfig(require_scored_terms=False)
    recent_metrics = (
        UpdateMetrics(
            update_index=5,
            batch_size=2,
            max_steps=3,
            initial_log_flops_mean=10.0,
            final_log_flops_mean=8.5,
            final_log_flops_best=7.25,
            reward_mean=1.5,
            reward_std=0.25,
            advantage_mean=0.0,
            advantage_std=1.0,
            valid_action_count=4,
            stop_count=1,
            empty_action_space_count=0,
            finished_count=1,
            max_steps_count=1,
            target_score_count=6,
            action_score_count=4,
            loss=-0.125,
            target_logp_mean=-0.75,
            action_logp_mean=-1.25,
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
        rollout_config=rollout_config,
        reward_config=RewardConfig(),
        baseline_config=baseline_config,
        loss_config=loss_config,
        recent_metrics=recent_metrics,
    )
    loaded = load_checkpoint(path)

    assert isinstance(loaded, CheckpointData)
    assert loaded.train_state.update_index == state.update_index
    assert loaded.train_state.policy.config == state.policy.config
    assert loaded.train_state.optimizer_config == state.optimizer_config
    _assert_pytrees_equal(loaded.train_state.policy.params, state.policy.params)
    _assert_pytrees_equal(loaded.train_state.opt_state, state.opt_state)
    assert loaded.rollout_config.batch_size == 2
    assert loaded.rollout_config == rollout_config
    assert loaded.reward_config == RewardConfig()
    assert loaded.baseline_config == baseline_config
    assert loaded.loss_config == loss_config
    assert loaded.recent_metrics == recent_metrics
    assert isinstance(loaded.recent_metrics[0], UpdateMetrics)


def test_checkpoint_restores_root_key_as_jax_uint32_array(tmp_path):
    state = init_train_state(
        PolicyConfig(d_model=8, max_candidates=8, max_side_terms=4),
        OptimizerConfig(learning_rate=1.0e-2),
        seed=13,
    )
    path = tmp_path / "checkpoint.pkl"

    save_checkpoint(
        path,
        state,
        rollout_config=RolloutConfig(batch_size=2, max_steps=1, seed=13),
        reward_config=RewardConfig(),
        baseline_config=BaselineConfig(),
        loss_config=LossConfig(),
        recent_metrics=(),
    )

    loaded = load_checkpoint(path)

    assert isinstance(loaded.train_state.root_key, jnp.ndarray)
    assert loaded.train_state.root_key.dtype == jnp.uint32
    assert jnp.array_equal(loaded.train_state.root_key, state.root_key)


def test_checkpoint_rejects_unknown_schema_version(tmp_path):
    path = tmp_path / "bad.pkl"
    with path.open("wb") as handle:
        pickle.dump({"schema_version": 999}, handle)

    with pytest.raises(TrainingError, match="checkpoint schema"):
        load_checkpoint(path)


def test_checkpoint_rejects_unknown_tokenizer_schema_version(tmp_path):
    state = init_train_state(
        PolicyConfig(d_model=8, max_candidates=8, max_side_terms=4),
        OptimizerConfig(learning_rate=1.0e-2),
        seed=13,
    )
    path = tmp_path / "bad.pkl"

    save_checkpoint(
        path,
        state,
        rollout_config=RolloutConfig(batch_size=2, max_steps=1, seed=13),
        reward_config=RewardConfig(),
        baseline_config=BaselineConfig(),
        loss_config=LossConfig(),
        recent_metrics=(),
    )
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    payload["tokenizer_schema_version"] = 999
    with path.open("wb") as handle:
        pickle.dump(payload, handle)

    with pytest.raises(TrainingError, match="tokenizer schema"):
        load_checkpoint(path)
