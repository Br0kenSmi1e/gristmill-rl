from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from transformer_policy.embed import TOKEN_FEATURE_DIM, token_features
from transformer_policy.trace import TokenChoiceEvent


@dataclass(frozen=True)
class PaddedTokenChoiceBatch:
    sequence_features: np.ndarray
    sequence_mask: np.ndarray
    legal_features: np.ndarray
    legal_mask: np.ndarray
    next_position: np.ndarray
    chosen_index: np.ndarray
    episode_id: np.ndarray
    event_mask: np.ndarray


def _validate_episode_ids(
    events: tuple[TokenChoiceEvent, ...], episode_ids: np.ndarray
) -> np.ndarray:
    ids = np.asarray(episode_ids, dtype=np.int32)
    if ids.ndim != 1 or len(ids) != len(events):
        raise ValueError("episode_ids must be a 1-D array matching events")
    if np.any(ids < 0):
        raise ValueError("episode_ids must be non-negative")
    return ids


def pad_token_choice_events(
    events: tuple[TokenChoiceEvent, ...],
    *,
    episode_ids: np.ndarray,
) -> PaddedTokenChoiceBatch:
    if not events:
        raise ValueError("events must not be empty")
    ids = _validate_episode_ids(events, episode_ids)
    max_sequence_len = max(len(event.sequence_tokens) for event in events)
    max_legal = max(len(event.legal_next_tokens) for event in events)
    event_count = len(events)
    sequence_features = np.zeros(
        (event_count, max_sequence_len, TOKEN_FEATURE_DIM),
        dtype=np.float32,
    )
    sequence_mask = np.zeros((event_count, max_sequence_len), dtype=bool)
    legal_features = np.zeros(
        (event_count, max_legal, TOKEN_FEATURE_DIM),
        dtype=np.float32,
    )
    legal_mask = np.zeros((event_count, max_legal), dtype=bool)
    next_position = np.zeros(event_count, dtype=np.int32)
    chosen_index = np.zeros(event_count, dtype=np.int32)
    for row, event in enumerate(events):
        sequence = token_features(event.sequence_tokens)
        legal = token_features(event.legal_next_tokens)
        next_pos = len(event.sequence_tokens)
        legal[:, 1] = float(next_pos)
        sequence_features[row, : len(event.sequence_tokens), :] = sequence
        sequence_mask[row, : len(event.sequence_tokens)] = True
        legal_features[row, : len(event.legal_next_tokens), :] = legal
        legal_mask[row, : len(event.legal_next_tokens)] = True
        next_position[row] = next_pos
        chosen_index[row] = event.chosen_index
    return PaddedTokenChoiceBatch(
        sequence_features=sequence_features,
        sequence_mask=sequence_mask,
        legal_features=legal_features,
        legal_mask=legal_mask,
        next_position=next_position,
        chosen_index=chosen_index,
        episode_id=ids,
        event_mask=np.ones(event_count, dtype=bool),
    )


def score_event_batch(scorer, batch: PaddedTokenChoiceBatch) -> jax.Array:
    return jax.vmap(scorer.score_next_features)(
        jnp.asarray(batch.sequence_features, dtype=jnp.float32),
        jnp.asarray(batch.sequence_mask, dtype=bool),
        jnp.asarray(batch.legal_features, dtype=jnp.float32),
        jnp.asarray(batch.legal_mask, dtype=bool),
    )


def chosen_event_log_probs(
    logits: jax.Array,
    legal_mask: jax.Array,
    chosen_index: jax.Array,
) -> jax.Array:
    masked_logits = jnp.where(legal_mask, logits, -jnp.inf)
    log_probs = jax.nn.log_softmax(masked_logits, axis=-1)
    return jnp.take_along_axis(log_probs, chosen_index[:, None], axis=-1).squeeze(-1)


def trajectory_log_probs(
    chosen_log_probs: jax.Array,
    episode_id: jax.Array,
    *,
    episode_count: int,
) -> jax.Array:
    return jnp.zeros((episode_count,), dtype=chosen_log_probs.dtype).at[episode_id].add(
        chosen_log_probs
    )
