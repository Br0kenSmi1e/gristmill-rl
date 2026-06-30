from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from gristmill_symbolics._training import TrainingError
from gristmill_symbolics.model.tokenizer import SENTINEL, TOKEN_FIELDS
from gristmill_symbolics.model.tokenizer import TOKEN_KIND

from .nn import LogitDecoder, TokenEmbedder, TransformerEncoder


@dataclass(frozen=True)
class _NetworkSettings:
    d_model: int
    num_attention_layers: int
    num_attention_heads: int
    id_vocab_size: int
    init_scale: float


def _positive_int(name: str, value: int) -> int:
    if type(value) is not int:
        raise TrainingError(f"{name} must be an int")
    if value <= 0:
        raise TrainingError(f"{name} must be positive")
    return value


def _finite_float(name: str, value: float) -> float:
    if isinstance(value, bool) or isinstance(value, np.bool_):
        raise TrainingError(f"{name} must be a finite float")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise TrainingError(f"{name} must be a finite float") from exc
    if not math.isfinite(parsed):
        raise TrainingError(f"{name} must be a finite float")
    return parsed


def _validate_heads(d_model: int, heads: int) -> int:
    if d_model % heads != 0:
        raise TrainingError("d_model must be divisible by num_attention_heads")
    return heads


def _dummy_tokens(length: int):
    return {
        field: jnp.full(
            (length,),
            _dummy_token_value(field),
            dtype=jnp.int32,
        )
        for field in TOKEN_FIELDS
    }


def _dummy_token_value(field: str) -> int:
    if field == "token_kind":
        return int(TOKEN_KIND.PAD)
    return SENTINEL


class TransformerActionSelectorModel:
    def __init__(
        self,
        *,
        state_token_pad_to: int,
        action_token_pad_to: int,
        d_model: int = 32,
        num_attention_layers: int = 1,
        num_attention_heads: int = 4,
        id_vocab_size: int = 128,
        init_scale: float = 0.02,
    ):
        self._state_token_pad_to = _positive_int(
            "state_token_pad_to",
            state_token_pad_to,
        )
        self._action_token_pad_to = _positive_int(
            "action_token_pad_to",
            action_token_pad_to,
        )
        d_model = _positive_int("d_model", d_model)
        heads = _positive_int("num_attention_heads", num_attention_heads)
        self._settings = _NetworkSettings(
            d_model=d_model,
            num_attention_layers=_positive_int(
                "num_attention_layers",
                num_attention_layers,
            ),
            num_attention_heads=_validate_heads(d_model, heads),
            id_vocab_size=_positive_int("id_vocab_size", id_vocab_size),
            init_scale=_finite_float("init_scale", init_scale),
        )
        self._embedder = TokenEmbedder(
            d_model=self._settings.d_model,
            id_vocab_size=self._settings.id_vocab_size,
            init_scale=self._settings.init_scale,
        )
        self._encoder = TransformerEncoder(
            d_model=self._settings.d_model,
            num_layers=self._settings.num_attention_layers,
            num_heads=self._settings.num_attention_heads,
            init_scale=self._settings.init_scale,
        )
        self._logit_decoder = LogitDecoder(
            d_model=self._settings.d_model,
            init_scale=self._settings.init_scale,
        )

    @property
    def state_token_pad_to(self) -> int:
        return self._state_token_pad_to

    @property
    def action_token_pad_to(self) -> int:
        return self._action_token_pad_to

    @property
    def embedder(self) -> TokenEmbedder:
        return self._embedder

    @property
    def encoder(self) -> TransformerEncoder:
        return self._encoder

    @property
    def logit_decoder(self) -> LogitDecoder:
        return self._logit_decoder

    def constructor_kwargs(self) -> dict[str, object]:
        return {
            "state_token_pad_to": self._state_token_pad_to,
            "action_token_pad_to": self._action_token_pad_to,
            "d_model": self._settings.d_model,
            "num_attention_layers": self._settings.num_attention_layers,
            "num_attention_heads": self._settings.num_attention_heads,
            "id_vocab_size": self._settings.id_vocab_size,
            "init_scale": self._settings.init_scale,
        }

    def init_params(self, rng):
        embedder_key, encoder_key, decoder_key = jax.random.split(rng, 3)
        tokens = _dummy_tokens(1)
        mask = jnp.ones((1,), dtype=jnp.bool_)
        vectors = jnp.zeros((1, self._settings.d_model), dtype=jnp.bfloat16)
        condition = jnp.zeros((self._settings.d_model,), dtype=jnp.float32)
        return {
            "embedder": self._embedder.init(embedder_key, tokens)["params"],
            "encoder": self._encoder.init(
                encoder_key,
                vectors,
                mask,
            )["params"],
            "logit_decoder": self._logit_decoder.init(
                decoder_key,
                vectors,
                condition,
            )["params"],
        }

    def sample_step(self, params, rng, states):
        raise NotImplementedError(
            "TransformerActionSelectorModel is being rebuilt"
        )

    def score_step(self, params, transitions):
        raise NotImplementedError(
            "TransformerActionSelectorModel is being rebuilt"
        )
