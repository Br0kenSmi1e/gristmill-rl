import pickle

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
    init_train_state,
    load_checkpoint,
    save_checkpoint,
)


def test_checkpoint_round_trips_train_state_configs_and_metrics(tmp_path):
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

    assert isinstance(loaded, CheckpointData)
    assert loaded.train_state.update_index == state.update_index
    assert loaded.train_state.policy.config == state.policy.config
    assert loaded.rollout_config.batch_size == 2
    assert loaded.recent_metrics == ()


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
