import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from transformer_policy.batch import (
    chosen_event_log_probs,
    pad_token_choice_events,
    score_event_batch,
    trajectory_log_probs,
)
from transformer_policy.decoder import sample_step_with_events
from transformer_policy.sequence_model import CausalTransformerScorer
from transformer_policy.trace import TokenChoiceEvent
from transformer_policy.types import T

from .test_transformer_policy_decoder import PreferenceScorer
from .transformer_policy_fixtures import actionable_state


def test_pad_token_choice_events_shapes_and_masks():
    events = (
        TokenChoiceEvent(
            sequence_tokens=(T("STATE_START"), T("STATE_END")),
            legal_next_tokens=(T("STOP"), T("DEF", def_index=0)),
            chosen_index=1,
            phase="def",
            step_index=0,
        ),
        TokenChoiceEvent(
            sequence_tokens=(T("STATE_START"), T("STATE_END"), T("DEF", def_index=0)),
            legal_next_tokens=(T("CAND", candidate_index=0),),
            chosen_index=0,
            phase="candidate",
            step_index=0,
        ),
    )

    batch = pad_token_choice_events(events, episode_ids=np.asarray([0, 0]))

    assert batch.sequence_features.shape[0] == 2
    assert batch.sequence_mask.tolist() == [[True, True, False], [True, True, True]]
    assert batch.legal_mask.tolist() == [[True, True], [True, False]]
    assert batch.chosen_index.tolist() == [1, 0]
    assert batch.episode_id.tolist() == [0, 0]
    assert batch.next_position.tolist() == [2, 3]


def test_pad_token_choice_events_rejects_empty_events():
    with pytest.raises(ValueError, match="events must not be empty"):
        pad_token_choice_events((), episode_ids=np.asarray([], dtype=np.int32))


def test_score_event_batch_matches_score_next_for_each_event():
    scorer = CausalTransformerScorer(
        hidden_dim=16,
        num_heads=4,
        num_layers=1,
        mlp_dim=32,
        rngs=nnx.Rngs(0),
    )
    events = sample_step_with_events(
        actionable_state(),
        PreferenceScorer(),
        np.random.default_rng(0),
    ).events
    batch = pad_token_choice_events(events, episode_ids=np.zeros(len(events), dtype=np.int32))

    batched_logits = np.asarray(score_event_batch(scorer, batch))

    for index, event in enumerate(events):
        context_prefix = event.sequence_tokens
        expected = np.asarray(scorer.score_next(context_prefix, (), event.legal_next_tokens))
        actual = batched_logits[index, : len(event.legal_next_tokens)]
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_score_event_batch_masks_padded_legal_columns_with_negative_infinity():
    scorer = CausalTransformerScorer(
        hidden_dim=16,
        num_heads=4,
        num_layers=1,
        mlp_dim=32,
        rngs=nnx.Rngs(0),
    )
    events = (
        TokenChoiceEvent(
            sequence_tokens=(T("STATE_START"), T("STATE_END")),
            legal_next_tokens=(T("STOP"), T("DEF", def_index=0)),
            chosen_index=0,
            phase="def",
            step_index=0,
        ),
        TokenChoiceEvent(
            sequence_tokens=(T("STATE_START"), T("STATE_END"), T("DEF", def_index=0)),
            legal_next_tokens=(T("CAND", candidate_index=0),),
            chosen_index=0,
            phase="candidate",
            step_index=0,
        ),
    )
    batch = pad_token_choice_events(events, episode_ids=np.zeros(len(events), dtype=np.int32))

    batched_logits = np.asarray(score_event_batch(scorer, batch))

    assert np.isneginf(batched_logits[1, 1])


def test_score_next_features_uses_last_true_token_for_non_prefix_mask():
    scorer = CausalTransformerScorer(
        hidden_dim=16,
        num_heads=4,
        num_layers=1,
        mlp_dim=32,
        rngs=nnx.Rngs(1),
    )
    scorer.blocks = nnx.List()
    sequence_features = jnp.asarray(
        [
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.3, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    legal_features = jnp.asarray(
        [
            [0.4, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    legal_mask = jnp.asarray([True, True])

    non_prefix = np.asarray(
        scorer.score_next_features(
            sequence_features,
            jnp.asarray([True, False, True]),
            legal_features,
            legal_mask,
        )
    )
    prefix_last_true = np.asarray(
        scorer.score_next_features(
            sequence_features,
            jnp.asarray([False, False, True]),
            legal_features,
            legal_mask,
        )
    )

    np.testing.assert_allclose(non_prefix, prefix_last_true, rtol=1e-5, atol=1e-5)


def test_chosen_event_and_trajectory_log_probs():
    logits = jnp.asarray([[0.0, 1.0, -1.0], [2.0, -2.0, -1.0]], dtype=jnp.float32)
    legal_mask = jnp.asarray([[True, True, False], [True, True, True]])
    chosen = jnp.asarray([1, 0], dtype=jnp.int32)
    episode_id = jnp.asarray([0, 1], dtype=jnp.int32)

    chosen_logp = chosen_event_log_probs(logits, legal_mask, chosen)
    per_episode = trajectory_log_probs(chosen_logp, episode_id, episode_count=2)

    assert chosen_logp.shape == (2,)
    assert per_episode.shape == (2,)
    assert np.isfinite(np.asarray(chosen_logp)).all()
    assert np.isfinite(np.asarray(per_episode)).all()


def test_chosen_event_log_probs_rejects_invalid_chosen_index():
    logits = jnp.asarray([[0.0, 1.0]], dtype=jnp.float32)
    legal_mask = jnp.asarray([[True, True]])
    chosen = jnp.asarray([2], dtype=jnp.int32)

    with pytest.raises(ValueError, match="chosen_index must be within logits width"):
        chosen_event_log_probs(logits, legal_mask, chosen)


def test_chosen_event_log_probs_rejects_masked_chosen_column():
    logits = jnp.asarray([[0.0, 1.0]], dtype=jnp.float32)
    legal_mask = jnp.asarray([[True, False]])
    chosen = jnp.asarray([1], dtype=jnp.int32)

    with pytest.raises(ValueError, match="chosen_index must point to a legal token"):
        chosen_event_log_probs(logits, legal_mask, chosen)


def test_chosen_event_log_probs_rejects_all_false_legal_mask_row():
    logits = jnp.asarray([[0.0, 1.0]], dtype=jnp.float32)
    legal_mask = jnp.asarray([[False, False]])
    chosen = jnp.asarray([0], dtype=jnp.int32)

    with pytest.raises(ValueError, match="each row must have at least one legal token"):
        chosen_event_log_probs(logits, legal_mask, chosen)


def test_score_event_batch_rejects_empty_sequence_mask_row():
    scorer = CausalTransformerScorer(
        hidden_dim=16,
        num_heads=4,
        num_layers=1,
        mlp_dim=32,
        rngs=nnx.Rngs(2),
    )
    batch = pad_token_choice_events(
        (
            TokenChoiceEvent(
                sequence_tokens=(T("STATE_START"), T("STATE_END")),
                legal_next_tokens=(T("STOP"),),
                chosen_index=0,
                phase="def",
                step_index=0,
            ),
        ),
        episode_ids=np.asarray([0], dtype=np.int32),
    )
    empty_mask_batch = batch.__class__(
        sequence_features=batch.sequence_features,
        sequence_mask=np.asarray([[False, False]], dtype=bool),
        legal_features=batch.legal_features,
        legal_mask=batch.legal_mask,
        next_position=batch.next_position,
        chosen_index=batch.chosen_index,
        episode_id=batch.episode_id,
        event_mask=batch.event_mask,
    )

    with pytest.raises(ValueError, match="each event must contain at least one sequence token"):
        score_event_batch(scorer, empty_mask_batch)


def test_score_event_batch_gradient_smoke():
    scorer = CausalTransformerScorer(
        hidden_dim=16,
        num_heads=4,
        num_layers=1,
        mlp_dim=32,
        rngs=nnx.Rngs(3),
    )
    events = sample_step_with_events(
        actionable_state(),
        PreferenceScorer(),
        np.random.default_rng(0),
    ).events
    batch = pad_token_choice_events(events, episode_ids=np.zeros(len(events), dtype=np.int32))

    def loss_fn(sequence_features, legal_features):
        logits = jax.vmap(scorer.score_next_features)(
            sequence_features,
            jnp.asarray(batch.sequence_mask, dtype=bool),
            legal_features,
            jnp.asarray(batch.legal_mask, dtype=bool),
        )
        return jnp.sum(jnp.where(jnp.isfinite(logits), logits, 0.0))

    grads = jax.grad(loss_fn, argnums=(0, 1))(
        jnp.asarray(batch.sequence_features, dtype=jnp.float32),
        jnp.asarray(batch.legal_features, dtype=jnp.float32),
    )

    assert grads[0].shape == batch.sequence_features.shape
    assert grads[1].shape == batch.legal_features.shape
