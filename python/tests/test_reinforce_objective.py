import numpy as np
from flax import nnx

from transformer_policy.batch import pad_token_choice_events
from transformer_policy.decoder import sample_step_with_events
from transformer_policy.sequence_model import CausalTransformerScorer
from reinforce_training.objective import (
    TrainConfig,
    create_optimizer,
    rewards_and_advantages,
    reinforce_loss,
    train_step,
)

from .test_transformer_policy_decoder import PreferenceScorer
from .transformer_policy_fixtures import actionable_state


def _event_batch():
    events = sample_step_with_events(
        actionable_state(),
        PreferenceScorer(),
        np.random.default_rng(0),
    ).events
    return pad_token_choice_events(events, episode_ids=np.zeros(len(events), dtype=np.int32))


def _flat_param_values(scorer):
    state = nnx.state(scorer, nnx.Param)
    values = []
    for leaf in __import__("jax").tree_util.tree_leaves(state):
        value = getattr(leaf, "value", leaf)
        values.append(np.asarray(value).copy())
    return values


def test_rewards_and_advantages_use_negative_final_log_flops_and_batch_mean():
    rewards, advantages = rewards_and_advantages(np.asarray([2.0, 4.0], dtype=np.float32))

    np.testing.assert_allclose(rewards, [-2.0, -4.0])
    np.testing.assert_allclose(advantages, [1.0, -1.0])
    assert np.sum(advantages) == np.float32(0.0)


def test_reinforce_loss_is_finite():
    scorer = CausalTransformerScorer(
        hidden_dim=16,
        num_heads=4,
        num_layers=1,
        mlp_dim=32,
        rngs=nnx.Rngs(0),
    )
    loss, aux = reinforce_loss(
        scorer,
        _event_batch(),
        advantages=np.asarray([1.0], dtype=np.float32),
        episode_count=1,
    )

    assert np.isfinite(float(loss))
    assert np.isfinite(float(aux["mean_trajectory_log_prob"]))


def test_train_step_changes_params_for_nonzero_advantage():
    scorer = CausalTransformerScorer(
        hidden_dim=16,
        num_heads=4,
        num_layers=1,
        mlp_dim=32,
        rngs=nnx.Rngs(0),
    )
    before = _flat_param_values(scorer)
    optimizer = create_optimizer(scorer, TrainConfig(learning_rate=1e-2))

    metrics = train_step(
        scorer,
        optimizer=optimizer,
        batch=_event_batch(),
        advantages=np.asarray([1.0], dtype=np.float32),
        episode_count=1,
    )

    after = _flat_param_values(scorer)
    assert metrics["params_changed"]
    assert np.isfinite(metrics["loss"])
    assert any(
        not np.array_equal(left, right)
        for left, right in zip(before, after, strict=True)
    )
