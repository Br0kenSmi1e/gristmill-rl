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


def _validate_non_empty_sequence_masks(sequence_mask: np.ndarray) -> None:
    if sequence_mask.ndim != 2:
        raise ValueError("sequence_mask must be a 2-D array")
    if np.any(~np.any(sequence_mask, axis=1)):
        raise ValueError("each event must contain at least one sequence token")


def _validate_choice_inputs(
    logits: jax.Array,
    legal_mask: jax.Array,
    chosen_index: jax.Array,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits_np = np.asarray(logits)
    legal_mask_np = np.asarray(legal_mask, dtype=bool)
    chosen_index_np = np.asarray(chosen_index)
    if logits_np.ndim != 2:
        raise ValueError("logits must be a 2-D matrix")
    if legal_mask_np.shape != logits_np.shape:
        raise ValueError("legal_mask must match logits shape")
    if chosen_index_np.ndim != 1 or chosen_index_np.shape[0] != logits_np.shape[0]:
        raise ValueError("chosen_index must be a 1-D array matching logits rows")
    if np.any(~np.any(legal_mask_np, axis=1)):
        raise ValueError("each row must have at least one legal token")
    if np.any(chosen_index_np < 0) or np.any(chosen_index_np >= logits_np.shape[1]):
        raise ValueError("chosen_index must be within logits width")
    row_index = np.arange(logits_np.shape[0], dtype=np.int32)
    if np.any(~legal_mask_np[row_index, chosen_index_np]):
        raise ValueError("chosen_index must point to a legal token")
    return logits_np, legal_mask_np, chosen_index_np.astype(np.int32, copy=False)


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
    _validate_non_empty_sequence_masks(np.asarray(batch.sequence_mask, dtype=bool))
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
    logits_np, legal_mask_np, chosen_index_np = _validate_choice_inputs(
        logits, legal_mask, chosen_index
    )
    masked_logits = jnp.where(jnp.asarray(legal_mask_np), jnp.asarray(logits_np), -jnp.inf)
    log_probs = jax.nn.log_softmax(masked_logits, axis=-1)
    return jnp.take_along_axis(
        log_probs, jnp.asarray(chosen_index_np)[:, None], axis=-1
    ).squeeze(-1)


def trajectory_log_probs(
    chosen_log_probs: jax.Array,
    episode_id: jax.Array,
    *,
    episode_count: int,
) -> jax.Array:
    return jnp.zeros((episode_count,), dtype=chosen_log_probs.dtype).at[episode_id].add(
        chosen_log_probs
    )
