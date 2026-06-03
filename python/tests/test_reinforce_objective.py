from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from transformer_policy.batch import (
    PaddedTokenChoiceBatch,
    chosen_event_log_probs,
    pad_token_choice_events,
    trajectory_log_probs,
)
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


def _scorer():
    return CausalTransformerScorer(
        hidden_dim=16,
        num_heads=4,
        num_layers=1,
        mlp_dim=32,
        rngs=nnx.Rngs(0),
    )


def _flat_param_values(scorer):
    state = nnx.state(scorer, nnx.Param)
    values = []
    for leaf in __import__("jax").tree_util.tree_leaves(state):
        value = getattr(leaf, "value", leaf)
        values.append(np.asarray(value).copy())
    return values


def _batch_with_masked_chosen():
    batch = _event_batch()
    legal_mask = np.asarray(batch.legal_mask, dtype=bool).copy()
    chosen_index = np.asarray(batch.chosen_index, dtype=np.int32).copy()
    row = 0
    chosen_index[row] = 0
    legal_mask[row] = False
    legal_mask[row, 1] = True
    return replace(batch, legal_mask=legal_mask, chosen_index=chosen_index)


class StaticScorer:
    def score_next_features(
        self,
        sequence_features,
        sequence_mask,
        legal_features,
        legal_mask,
    ):
        return jnp.where(legal_mask, legal_features[:, 0], -jnp.inf)


class NonfiniteScorer(nnx.Module):
    def __init__(self):
        self.scale = nnx.Param(jnp.asarray(np.nan, dtype=jnp.float32))

    def score_next_features(
        self,
        sequence_features,
        sequence_mask,
        legal_features,
        legal_mask,
    ):
        return jnp.ones(legal_mask.shape, dtype=jnp.float32) * self.scale[...]


@jax.custom_gradient
def _finite_value_with_nan_gradient(value):
    def gradient(cotangent):
        return jnp.full_like(value, jnp.nan)

    return jnp.asarray(0.0, dtype=value.dtype), gradient


class NonfiniteGradientScorer(nnx.Module):
    def __init__(self):
        self.scale = nnx.Param(jnp.asarray(1.0, dtype=jnp.float32))

    def score_next_features(
        self,
        sequence_features,
        sequence_mask,
        legal_features,
        legal_mask,
    ):
        return legal_features[:, 0] + _finite_value_with_nan_gradient(self.scale[...])


def _synthetic_batch():
    logits = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [2.0, -2.0, -1.0],
            [1.0, 3.0, 0.0],
        ],
        dtype=np.float32,
    )
    legal_mask = np.asarray(
        [
            [True, True, False],
            [True, True, True],
            [True, True, True],
        ],
        dtype=bool,
    )
    return PaddedTokenChoiceBatch(
        sequence_features=np.ones((3, 1, 1), dtype=np.float32),
        sequence_mask=np.ones((3, 1), dtype=bool),
        legal_features=logits[:, :, None],
        legal_mask=legal_mask,
        next_position=np.zeros(3, dtype=np.int32),
        chosen_index=np.asarray([1, 0, 0], dtype=np.int32),
        episode_id=np.asarray([0, 1, 0], dtype=np.int32),
        event_mask=np.ones(3, dtype=bool),
    )


def test_rewards_and_advantages_use_negative_final_log_flops_and_batch_mean():
    rewards, advantages = rewards_and_advantages(np.asarray([2.0, 4.0], dtype=np.float32))

    np.testing.assert_allclose(rewards, [-2.0, -4.0])
    np.testing.assert_allclose(advantages, [1.0, -1.0])
    assert np.sum(advantages) == np.float32(0.0)


def test_reinforce_loss_is_finite():
    scorer = _scorer()
    loss, aux = reinforce_loss(
        scorer,
        _event_batch(),
        advantages=np.asarray([1.0], dtype=np.float32),
        episode_count=1,
    )

    assert np.isfinite(float(loss))
    assert np.isfinite(float(aux["mean_trajectory_log_prob"]))


def test_train_step_changes_params_for_nonzero_advantage():
    scorer = _scorer()
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


def test_reinforce_loss_rejects_non_1d_advantages():
    with pytest.raises(ValueError, match="advantages must be a finite 1-D array"):
        reinforce_loss(
            _scorer(),
            _event_batch(),
            advantages=np.asarray([[1.0]], dtype=np.float32),
            episode_count=1,
        )


def test_reinforce_loss_rejects_advantage_length_mismatch():
    with pytest.raises(ValueError, match="advantages length must match episode_count"):
        reinforce_loss(
            _scorer(),
            _event_batch(),
            advantages=np.asarray([1.0, 0.0], dtype=np.float32),
            episode_count=1,
        )


def test_reinforce_loss_rejects_nonfinite_advantage():
    with pytest.raises(ValueError, match="advantages must be a finite 1-D array"):
        reinforce_loss(
            _scorer(),
            _event_batch(),
            advantages=np.asarray([np.nan], dtype=np.float32),
            episode_count=1,
        )


def test_reinforce_loss_rejects_nonpositive_episode_count():
    with pytest.raises(ValueError, match="episode_count must be positive"):
        reinforce_loss(
            _scorer(),
            _event_batch(),
            advantages=np.asarray([], dtype=np.float32),
            episode_count=0,
        )


def test_reinforce_loss_rejects_negative_episode_id():
    batch = _event_batch()
    episode_id = np.zeros_like(batch.episode_id)
    episode_id[0] = -1

    with pytest.raises(ValueError, match="episode_id"):
        reinforce_loss(
            _scorer(),
            replace(batch, episode_id=episode_id),
            advantages=np.asarray([1.0], dtype=np.float32),
            episode_count=1,
        )


def test_reinforce_loss_rejects_episode_id_length_mismatch():
    batch = _event_batch()

    with pytest.raises(ValueError, match="episode_id must be a 1-D array"):
        reinforce_loss(
            _scorer(),
            replace(batch, episode_id=batch.episode_id[:-1]),
            advantages=np.asarray([1.0], dtype=np.float32),
            episode_count=1,
        )


def test_reinforce_loss_rejects_out_of_range_episode_id():
    batch = _event_batch()
    episode_id = np.zeros_like(batch.episode_id)
    episode_id[0] = 1

    with pytest.raises(ValueError, match="episode_id"):
        reinforce_loss(
            _scorer(),
            replace(batch, episode_id=episode_id),
            advantages=np.asarray([1.0], dtype=np.float32),
            episode_count=1,
        )


def test_reinforce_loss_rejects_masked_chosen_index():
    with pytest.raises(ValueError, match="chosen_index must point to a legal token"):
        reinforce_loss(
            _scorer(),
            _batch_with_masked_chosen(),
            advantages=np.asarray([1.0], dtype=np.float32),
            episode_count=1,
        )


def test_train_step_rejects_masked_chosen_index():
    scorer = _scorer()
    optimizer = create_optimizer(scorer, TrainConfig(learning_rate=1e-2))

    with pytest.raises(ValueError, match="chosen_index must point to a legal token"):
        train_step(
            scorer,
            optimizer=optimizer,
            batch=_batch_with_masked_chosen(),
            advantages=np.asarray([1.0], dtype=np.float32),
            episode_count=1,
        )


def test_train_step_zero_advantages_does_not_change_params():
    scorer = _scorer()
    before = _flat_param_values(scorer)
    optimizer = create_optimizer(scorer, TrainConfig(learning_rate=1e-2))

    metrics = train_step(
        scorer,
        optimizer=optimizer,
        batch=_event_batch(),
        advantages=np.asarray([0.0], dtype=np.float32),
        episode_count=1,
    )

    after = _flat_param_values(scorer)
    assert not metrics["params_changed"]
    assert np.isfinite(metrics["loss"])
    assert all(
        np.array_equal(left, right)
        for left, right in zip(before, after, strict=True)
    )


def test_train_step_rejects_nonfinite_loss_before_update():
    scorer = NonfiniteScorer()
    before = _flat_param_values(scorer)
    optimizer = create_optimizer(scorer, TrainConfig(learning_rate=1e-2))

    with pytest.raises(ValueError, match="loss must be finite"):
        train_step(
            scorer,
            optimizer=optimizer,
            batch=_synthetic_batch(),
            advantages=np.asarray([1.0, 1.0], dtype=np.float32),
            episode_count=2,
        )

    after = _flat_param_values(scorer)
    assert all(
        np.array_equal(left, right, equal_nan=True)
        for left, right in zip(before, after, strict=True)
    )


def test_train_step_rejects_nonfinite_gradients_before_update():
    scorer = NonfiniteGradientScorer()
    before = _flat_param_values(scorer)
    optimizer = create_optimizer(scorer, TrainConfig(learning_rate=1e-2))

    with pytest.raises(ValueError, match="gradients must be finite"):
        train_step(
            scorer,
            optimizer=optimizer,
            batch=_synthetic_batch(),
            advantages=np.asarray([1.0, 1.0], dtype=np.float32),
            episode_count=2,
        )

    after = _flat_param_values(scorer)
    assert all(
        np.array_equal(left, right)
        for left, right in zip(before, after, strict=True)
    )


def test_reinforce_loss_matches_synthetic_multi_episode_log_prob():
    batch = _synthetic_batch()
    advantages = np.asarray([0.5, -1.0], dtype=np.float32)

    loss, aux = reinforce_loss(
        StaticScorer(),
        batch,
        advantages=advantages,
        episode_count=2,
    )

    logits = jnp.asarray(batch.legal_features[:, :, 0])
    chosen = chosen_event_log_probs(
        logits,
        jnp.asarray(batch.legal_mask),
        jnp.asarray(batch.chosen_index),
    )
    per_episode = trajectory_log_probs(
        chosen,
        jnp.asarray(batch.episode_id),
        episode_count=2,
    )
    expected = -jnp.mean(jnp.asarray(advantages) * per_episode)
    np.testing.assert_allclose(float(loss), float(expected), rtol=1e-6)
    np.testing.assert_allclose(
        float(aux["mean_trajectory_log_prob"]),
        float(jnp.mean(per_episode)),
        rtol=1e-6,
    )
