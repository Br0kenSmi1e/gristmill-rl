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
