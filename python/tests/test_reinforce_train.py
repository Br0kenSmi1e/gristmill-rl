from dataclasses import fields

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gristmill_symbolics.reinforce.train_state as train_state_module
from gristmill_symbolics import RewriteState, TensorComputation
from gristmill_symbolics.policy import PolicyConfig
from gristmill_symbolics.reinforce import (
    CurrentTransformerModel,
    CurrentTransformerModelConfig,
    OptimizerConfig,
    ReinforceTrainer,
    ReinforceTrainerConfig,
    TrainingError,
    advance_train_state,
    init_train_state,
    make_optimizer,
    train_update,
)
from tests.policy_fixtures import actionable_state
from tests.test_bindings import exact_empty_json


def _mixed_initial_states():
    return [
        actionable_state(),
        RewriteState.from_computation(TensorComputation.from_json_string(exact_empty_json())),
    ]


def _floating_leaves(tree):
    return [
        leaf
        for leaf in jax.tree_util.tree_leaves(tree)
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating)
    ]


def test_make_optimizer_returns_optax_gradient_transformation():
    optimizer = make_optimizer(OptimizerConfig(learning_rate=1.0e-2))

    assert hasattr(optimizer, "init")
    assert hasattr(optimizer, "update")


@pytest.mark.parametrize("learning_rate", [np.nan, 0.0, -1.0e-3])
def test_make_optimizer_rejects_non_finite_or_non_positive_learning_rate(
    learning_rate,
):
    with pytest.raises(TrainingError, match="learning_rate"):
        make_optimizer(OptimizerConfig(learning_rate=learning_rate))


@pytest.mark.parametrize(
    ("config", "field_name"),
    [
        (OptimizerConfig(b1=1.0), "b1"),
        (OptimizerConfig(b2=float("nan")), "b2"),
        (OptimizerConfig(eps=0.0), "eps"),
    ],
)
def test_make_optimizer_rejects_invalid_adam_hyperparameters(config, field_name):
    with pytest.raises(TrainingError, match=field_name):
        make_optimizer(config)


def test_init_train_state_creates_policy_params_and_opt_state():
    state = init_train_state(
        PolicyConfig(d_model=8),
        OptimizerConfig(learning_rate=1.0e-2),
        seed=11,
    )

    assert state.update_index == 0
    assert [field.name for field in fields(type(state))] == [
        "params",
        "opt_state",
        "root_key",
        "update_index",
    ]
    assert not hasattr(state, "policy")
    assert not hasattr(state, "optimizer_config")
    assert _floating_leaves(state.params)
    assert state.opt_state is not None


def test_advance_train_state_uses_protocol_path_and_increments_update_index():
    policy_config = PolicyConfig(d_model=8, stop_bias_init=-20.0)
    optimizer_config = OptimizerConfig(learning_rate=1.0e-2)
    state = init_train_state(policy_config, optimizer_config, seed=29)
    model_config = CurrentTransformerModelConfig(
        policy_config=policy_config,
        batch_size=2,
        max_steps=2,
        state_token_pad_to=512,
        action_token_pad_to=512,
        definition_pad_to=8,
    )
    trainer_config = ReinforceTrainerConfig(
        batch_size=2,
        optimizer_config=optimizer_config,
    )

    new_state, metrics = advance_train_state(
        state,
        _mixed_initial_states(),
        model=CurrentTransformerModel(),
        trainer=ReinforceTrainer(),
        model_config=model_config,
        trainer_config=trainer_config,
    )

    assert new_state.update_index == 1
    assert metrics.update_index == 0
    assert metrics.batch_size == 2
    assert np.isfinite(metrics.objective_loss_mean)
    assert np.isfinite(metrics.surrogate_loss)
    assert np.isfinite(metrics.final_flops_best)
    assert metrics.params_changed is True


def test_advance_train_state_folds_update_index_into_trainer_rng():
    policy_config = PolicyConfig(d_model=8)
    optimizer_config = OptimizerConfig(learning_rate=1.0e-2)
    state = init_train_state(
        policy_config,
        optimizer_config,
        seed=31,
        update_index=7,
    )
    model_config = CurrentTransformerModelConfig(
        policy_config=policy_config,
        batch_size=2,
        max_steps=1,
        state_token_pad_to=128,
        action_token_pad_to=128,
        definition_pad_to=4,
    )
    trainer_config = ReinforceTrainerConfig(
        batch_size=2,
        optimizer_config=optimizer_config,
    )

    class RecordingTrainer:
        def __init__(self):
            self.rng = None
            self.model = None
            self.config = None

        def update(self, params, opt_state, batch, model, rng, config):
            self.rng = rng
            self.model = model
            self.config = config
            return params, opt_state, {
                "reward_mean": 1.0,
                "reward_std": 0.0,
                "objective_loss_mean": -1.0,
                "surrogate_loss": 0.25,
                "final_flops_best": 3.0,
                "params_changed": False,
            }

    trainer = RecordingTrainer()

    new_state, metrics = advance_train_state(
        state,
        _mixed_initial_states(),
        model=object(),
        trainer=trainer,
        model_config=model_config,
        trainer_config=trainer_config,
    )

    assert jnp.array_equal(trainer.rng, jax.random.fold_in(state.root_key, 7))
    assert trainer.config is trainer_config
    assert new_state.root_key is state.root_key
    assert new_state.update_index == 8
    assert metrics.update_index == 7
    assert metrics.params_changed is False


def test_train_update_is_protocol_wrapper_not_streamed_rollout(monkeypatch):
    state = init_train_state(
        PolicyConfig(d_model=8),
        OptimizerConfig(learning_rate=1.0e-2),
        seed=12,
    )
    policy_config = PolicyConfig(d_model=8)
    optimizer_config = OptimizerConfig(learning_rate=1.0e-2)
    model_config = CurrentTransformerModelConfig(
        policy_config=policy_config,
        batch_size=2,
        max_steps=1,
        state_token_pad_to=128,
        action_token_pad_to=128,
        definition_pad_to=4,
    )
    trainer_config = ReinforceTrainerConfig(
        batch_size=2,
        optimizer_config=optimizer_config,
    )

    def fail_legacy_direct_rollout(*_args, **_kwargs):
        raise AssertionError("legacy direct rollout path was called")

    monkeypatch.setattr(
        train_state_module,
        "_collect_streamed_rollout_gradients",
        fail_legacy_direct_rollout,
        raising=False,
    )

    class RecordingTrainer:
        def __init__(self):
            self.called = False

        def update(self, params, opt_state, batch, model, rng, config):
            self.called = True
            return params, opt_state, {
                "reward_mean": 1.0,
                "reward_std": 0.0,
                "objective_loss_mean": -1.0,
                "surrogate_loss": 0.25,
                "final_flops_best": 3.0,
                "params_changed": False,
            }

    trainer = RecordingTrainer()

    new_state, metrics = train_update(
        state,
        _mixed_initial_states(),
        model=object(),
        trainer=trainer,
        model_config=model_config,
        trainer_config=trainer_config,
    )

    assert trainer.called is True
    assert new_state.update_index == 1
    assert metrics.reward_mean == pytest.approx(1.0)
    assert metrics.params_changed is False


def test_validate_finite_params_rejects_non_finite_float_leaf():
    with pytest.raises(TrainingError, match="updated policy parameters"):
        train_state_module._validate_finite_params({"w": jnp.asarray([1.0, jnp.nan])})
