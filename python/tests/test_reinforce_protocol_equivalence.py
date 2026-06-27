import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from gristmill_symbolics import RewriteState, TensorComputation
from gristmill_symbolics.policy import PolicyConfig
from gristmill_symbolics.reinforce import (
    CurrentTransformerModel,
    CurrentTransformerModelConfig,
    OptimizerConfig,
    ReinforceTrainer,
    ReinforceTrainerConfig,
    advance_train_state,
    init_train_state,
)
from gristmill_symbolics.reinforce.objective import compute_advantages, compute_rewards
from gristmill_symbolics.reinforce.rollout import _collect_streamed_rollout_gradients
from gristmill_symbolics.reinforce.train_state import (
    _params_changed,
    _reinforce_grad_loss,
    _surrogate_loss,
    _validate_finite_params,
    make_optimizer,
)
from gristmill_symbolics.reinforce.types import (
    BaselineConfig,
    LossConfig,
    PolicyState,
    RewardConfig,
    RolloutConfig,
    TrainingError,
)
from tests.policy_fixtures import actionable_json
from tests.test_bindings import exact_empty_json


def _state_from_json(text):
    return RewriteState.from_computation(TensorComputation.from_json_string(text))


def _batch():
    return [_state_from_json(actionable_json()), _state_from_json(exact_empty_json())]


def _tree_allclose(left, right, *, atol=1.0e-5):
    assert jax.tree_util.tree_structure(left) == jax.tree_util.tree_structure(right)
    for left_leaf, right_leaf in zip(
        jax.tree_util.tree_leaves(left),
        jax.tree_util.tree_leaves(right),
        strict=True,
    ):
        if hasattr(left_leaf, "dtype") and hasattr(right_leaf, "dtype"):
            assert left_leaf.dtype == right_leaf.dtype
            if jnp.issubdtype(left_leaf.dtype, jnp.inexact) or jnp.issubdtype(
                right_leaf.dtype,
                jnp.inexact,
            ):
                assert jnp.allclose(left_leaf, right_leaf, atol=atol, rtol=atol)
            else:
                assert jnp.array_equal(left_leaf, right_leaf)
        else:
            assert left_leaf == right_leaf


def _legacy_static_train_update(
    state,
    policy_config,
    optimizer_config,
    batch,
    rollout_config,
    reward_config=RewardConfig(),
    baseline_config=BaselineConfig(),
    loss_config=LossConfig(),
):
    policy = PolicyState(config=policy_config, params=state.params)
    streamed = _collect_streamed_rollout_gradients(
        policy,
        batch,
        rollout_config,
        update_index=state.update_index,
        root_key=state.root_key,
    )
    reward = compute_rewards(streamed.final, reward_config)
    advantage = compute_advantages(reward, baseline_config)
    if (
        loss_config.require_scored_terms
        and streamed.target_score_count + streamed.action_score_count == 0
    ):
        raise TrainingError("no scored policy terms in rollout batch")

    grads = _reinforce_grad_loss(streamed.trajectory_grad_logp, advantage)
    surrogate_loss = _surrogate_loss(streamed.trajectory_logp, advantage)
    optimizer = make_optimizer(optimizer_config)
    updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)
    _validate_finite_params(new_params)

    return new_params, opt_state, {
        "reward_mean": float(np.mean(reward)),
        "reward_std": float(np.std(reward)),
        "objective_loss_mean": float(-np.mean(reward)),
        "surrogate_loss": float(np.asarray(surrogate_loss)),
        "final_flops_best": float(np.min(streamed.final.final_log_flops)),
        "params_changed": _params_changed(state.params, new_params),
    }


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (
            {"count": jnp.asarray(1, dtype=jnp.int32)},
            {"count": jnp.asarray(2, dtype=jnp.int32)},
        ),
        (
            {"count": jnp.asarray(1, dtype=jnp.int16)},
            {"count": jnp.asarray(1, dtype=jnp.int32)},
        ),
        ({"tag": "new"}, {"tag": "legacy"}),
    ],
)
def test_tree_allclose_rejects_exact_leaf_mismatch(left, right):
    with pytest.raises(AssertionError):
        _tree_allclose(left, right)


def test_protocol_train_state_path_matches_private_legacy_static_update():
    policy_config = PolicyConfig(d_model=8, stop_bias_init=-20.0)
    optimizer_config = OptimizerConfig(learning_rate=1.0e-2)
    state = init_train_state(policy_config, optimizer_config, seed=29)
    legacy_config = RolloutConfig(
        batch_size=2,
        max_steps=2,
        seed=29,
        static_policy_batch=True,
        state_token_pad_to=512,
        action_token_pad_to=512,
        definition_pad_to=8,
    )
    legacy_params, legacy_opt_state, legacy_metrics = _legacy_static_train_update(
        state,
        policy_config,
        optimizer_config,
        _batch(),
        legacy_config,
    )

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
    new_state, new_metrics = advance_train_state(
        state,
        _batch(),
        model=CurrentTransformerModel(),
        trainer=ReinforceTrainer(),
        model_config=model_config,
        trainer_config=trainer_config,
    )

    _tree_allclose(new_state.params, legacy_params)
    _tree_allclose(new_state.opt_state, legacy_opt_state)
    assert new_metrics.reward_mean == pytest.approx(legacy_metrics["reward_mean"])
    assert new_metrics.reward_std == pytest.approx(legacy_metrics["reward_std"])
    assert new_metrics.objective_loss_mean == pytest.approx(
        legacy_metrics["objective_loss_mean"]
    )
    assert new_metrics.surrogate_loss == pytest.approx(
        legacy_metrics["surrogate_loss"],
        abs=1.0e-5,
    )
    assert new_metrics.final_flops_best == pytest.approx(
        legacy_metrics["final_flops_best"]
    )
    assert new_metrics.params_changed is legacy_metrics["params_changed"]
