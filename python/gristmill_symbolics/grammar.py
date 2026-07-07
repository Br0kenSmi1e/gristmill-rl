from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp

from .tokenizer import FlatDefinitionTokenizer

__all__ = ("FlatDefinitionGrammar",)

_CAT_PAD = 0
_CAT_BOS = 1
_CAT_EOS = 2
_CAT_DEF_START = 3
_CAT_DEF_END = 4
_CAT_RANGEID = 5
_CAT_TENSORID = 6
_CAT_INDEXID = 7
_CAT_COEFF_NUM = 8
_CAT_COEFF_DEN = 9
_CAT_OTHER = 10

_S_BEFORE_BOS = 0
_S_BETWEEN_DEFS = 1
_S_EXPECT_BASE = 2
_S_AFTER_BASE_OR_EXT = 3
_S_EXPECT_EXT_RANGE = 4
_S_EXPECT_COEFF_DEN = 5
_S_AFTER_COEFF_DEN_OR_SUM = 6
_S_EXPECT_SUM_RANGE = 7
_S_IN_FACTOR = 8
_S_DONE = 9
_S_ERROR = 10
_NUM_STATES = 11

_KIND_TO_CATEGORY = {
    "pad": _CAT_PAD,
    "bos": _CAT_BOS,
    "eos": _CAT_EOS,
    "def_start": _CAT_DEF_START,
    "def_end": _CAT_DEF_END,
    "rangeid": _CAT_RANGEID,
    "tensorid": _CAT_TENSORID,
    "indexid": _CAT_INDEXID,
    "coeff_num": _CAT_COEFF_NUM,
    "coeff_den": _CAT_COEFF_DEN,
}


class FlatDefinitionGrammar:
    def __init__(self, tokenizer: FlatDefinitionTokenizer):
        self.vocab_size = tokenizer.vocab_size
        self.pad_token_id = tokenizer.pad_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id

        categories = [
            _KIND_TO_CATEGORY.get(tokenizer.token_kind(token_id), _CAT_OTHER)
            for token_id in range(tokenizer.vocab_size)
        ]
        self.category_by_id = jnp.asarray(categories, dtype=jnp.int32)
        self.allowed_by_state = _build_allowed_by_state(categories)

    def initial_state(self, batch_shape: int | Sequence[int]) -> jax.Array:
        if isinstance(batch_shape, int):
            shape = (batch_shape,)
        else:
            shape = tuple(batch_shape)
        return jnp.full(shape, _S_BEFORE_BOS, dtype=jnp.int32)

    def advance_state(
        self,
        state: jax.Array,
        token_id: jax.Array,
    ) -> jax.Array:
        cat = jnp.take(self.category_by_id, token_id)

        next_state = jnp.full_like(state, _S_ERROR)

        next_state = jnp.where(
            (state == _S_BEFORE_BOS) & (cat == _CAT_BOS),
            _S_BETWEEN_DEFS,
            next_state,
        )

        next_state = jnp.where(
            (state == _S_BETWEEN_DEFS) & (cat == _CAT_DEF_START),
            _S_EXPECT_BASE,
            next_state,
        )
        next_state = jnp.where(
            (state == _S_BETWEEN_DEFS) & (cat == _CAT_EOS),
            _S_DONE,
            next_state,
        )

        next_state = jnp.where(
            (state == _S_EXPECT_BASE) & (cat == _CAT_TENSORID),
            _S_AFTER_BASE_OR_EXT,
            next_state,
        )

        next_state = jnp.where(
            (state == _S_AFTER_BASE_OR_EXT) & (cat == _CAT_INDEXID),
            _S_EXPECT_EXT_RANGE,
            next_state,
        )
        next_state = jnp.where(
            (state == _S_AFTER_BASE_OR_EXT) & (cat == _CAT_COEFF_NUM),
            _S_EXPECT_COEFF_DEN,
            next_state,
        )
        next_state = jnp.where(
            (state == _S_AFTER_BASE_OR_EXT) & (cat == _CAT_DEF_END),
            _S_BETWEEN_DEFS,
            next_state,
        )

        next_state = jnp.where(
            (state == _S_EXPECT_EXT_RANGE) & (cat == _CAT_RANGEID),
            _S_AFTER_BASE_OR_EXT,
            next_state,
        )

        next_state = jnp.where(
            (state == _S_EXPECT_COEFF_DEN) & (cat == _CAT_COEFF_DEN),
            _S_AFTER_COEFF_DEN_OR_SUM,
            next_state,
        )

        next_state = jnp.where(
            (state == _S_AFTER_COEFF_DEN_OR_SUM) & (cat == _CAT_INDEXID),
            _S_EXPECT_SUM_RANGE,
            next_state,
        )
        next_state = jnp.where(
            (state == _S_AFTER_COEFF_DEN_OR_SUM) & (cat == _CAT_TENSORID),
            _S_IN_FACTOR,
            next_state,
        )
        next_state = jnp.where(
            (state == _S_AFTER_COEFF_DEN_OR_SUM) & (cat == _CAT_COEFF_NUM),
            _S_EXPECT_COEFF_DEN,
            next_state,
        )
        next_state = jnp.where(
            (state == _S_AFTER_COEFF_DEN_OR_SUM) & (cat == _CAT_DEF_END),
            _S_BETWEEN_DEFS,
            next_state,
        )

        next_state = jnp.where(
            (state == _S_EXPECT_SUM_RANGE) & (cat == _CAT_RANGEID),
            _S_AFTER_COEFF_DEN_OR_SUM,
            next_state,
        )

        next_state = jnp.where(
            (state == _S_IN_FACTOR) & (cat == _CAT_INDEXID),
            _S_IN_FACTOR,
            next_state,
        )
        next_state = jnp.where(
            (state == _S_IN_FACTOR) & (cat == _CAT_TENSORID),
            _S_IN_FACTOR,
            next_state,
        )
        next_state = jnp.where(
            (state == _S_IN_FACTOR) & (cat == _CAT_COEFF_NUM),
            _S_EXPECT_COEFF_DEN,
            next_state,
        )
        next_state = jnp.where(
            (state == _S_IN_FACTOR) & (cat == _CAT_DEF_END),
            _S_BETWEEN_DEFS,
            next_state,
        )

        next_state = jnp.where(
            (state == _S_DONE) & (cat == _CAT_PAD),
            _S_DONE,
            next_state,
        )

        return next_state

    def valid_next_masks_for_decoder_input(
        self,
        decoder_input_ids: jax.Array,
    ) -> jax.Array:
        batch_size = decoder_input_ids.shape[0]
        init_state = self.initial_state((batch_size,))

        def step(state: jax.Array, token_t: jax.Array) -> tuple[jax.Array, jax.Array]:
            next_state = self.advance_state(state, token_t)
            valid_t = jnp.take(self.allowed_by_state, next_state, axis=0)
            return next_state, valid_t

        _final_state, masks_t_b_v = jax.lax.scan(
            step,
            init_state,
            jnp.swapaxes(decoder_input_ids, 0, 1),
        )
        return jnp.swapaxes(masks_t_b_v, 0, 1)

    def valid_next_mask_from_prefix(self, prefix_ids: jax.Array) -> jax.Array:
        batch_size = prefix_ids.shape[0]
        init_state = self.initial_state((batch_size,))

        def step(state: jax.Array, token_t: jax.Array) -> tuple[jax.Array, None]:
            return self.advance_state(state, token_t), None

        final_state, _ = jax.lax.scan(
            step,
            init_state,
            jnp.swapaxes(prefix_ids, 0, 1),
        )
        return jnp.take(self.allowed_by_state, final_state, axis=0)

def _build_allowed_by_state(categories: Sequence[int]) -> jax.Array:
    rows = [[False] * len(categories) for _ in range(_NUM_STATES)]

    def allow(state: int, allowed_categories: set[int]) -> None:
        rows[state] = [category in allowed_categories for category in categories]

    allow(_S_BEFORE_BOS, {_CAT_BOS})
    allow(_S_BETWEEN_DEFS, {_CAT_DEF_START, _CAT_EOS})
    allow(_S_EXPECT_BASE, {_CAT_TENSORID})
    allow(_S_AFTER_BASE_OR_EXT, {_CAT_INDEXID, _CAT_COEFF_NUM, _CAT_DEF_END})
    allow(_S_EXPECT_EXT_RANGE, {_CAT_RANGEID})
    allow(_S_EXPECT_COEFF_DEN, {_CAT_COEFF_DEN})
    allow(
        _S_AFTER_COEFF_DEN_OR_SUM,
        {_CAT_INDEXID, _CAT_TENSORID, _CAT_COEFF_NUM, _CAT_DEF_END},
    )
    allow(_S_EXPECT_SUM_RANGE, {_CAT_RANGEID})
    allow(_S_IN_FACTOR, {_CAT_INDEXID, _CAT_TENSORID, _CAT_COEFF_NUM, _CAT_DEF_END})
    allow(_S_DONE, {_CAT_PAD})

    return jnp.asarray(rows, dtype=bool)
