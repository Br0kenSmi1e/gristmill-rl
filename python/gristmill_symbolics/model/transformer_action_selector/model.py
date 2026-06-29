from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from gristmill_symbolics._training import TrainingError

from .constants import SENTINEL, TOKEN_KIND
from .types import TokenTree, _PolicySettings

_ID_FIELDS = (
    "def_index",
    "term_index",
    "factor_index",
    "tensor_id",
    "range_id",
    "index_id",
    "candidate_index",
)
_FIELD_EMBEDDING_KEY_COUNT = 3 + len(_ID_FIELDS)
_FIXED_PARAM_KEY_COUNT = _FIELD_EMBEDDING_KEY_COUNT + 1 + 2 + 4
_ATTENTION_KEYS_PER_LAYER = 6


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


def _normal(key, shape, scale):
    return jax.random.normal(key, shape, dtype=jnp.float32) * scale


def _split(key, count):
    return iter(jax.random.split(key, count))


def _init_policy_params(settings: _PolicySettings, rng) -> dict[str, object]:
    key_count = _FIXED_PARAM_KEY_COUNT + (
        _ATTENTION_KEYS_PER_LAYER * settings.num_attention_layers
    )
    keys = _split(rng, key_count)
    d = settings.d_model
    params: dict[str, object] = {
        "field_embeddings": {
            "token_kind": jnp.vstack(
                [
                    jnp.zeros((1, d), dtype=jnp.float32),
                    _normal(next(keys), (int(max(TOKEN_KIND)), d), settings.init_scale),
                ]
            ),
            "segment": jnp.vstack(
                [
                    jnp.zeros((1, d), dtype=jnp.float32),
                    _normal(next(keys), (4, d), settings.init_scale),
                ]
            ),
            "side": jnp.vstack(
                [
                    jnp.zeros((1, d), dtype=jnp.float32),
                    _normal(next(keys), (3, d), settings.init_scale),
                ]
            ),
            "def_index": jnp.vstack(
                [
                    jnp.zeros((1, d), dtype=jnp.float32),
                    _normal(
                        next(keys), (settings.id_vocab_size, d), settings.init_scale
                    ),
                ]
            ),
            "term_index": jnp.vstack(
                [
                    jnp.zeros((1, d), dtype=jnp.float32),
                    _normal(
                        next(keys), (settings.id_vocab_size, d), settings.init_scale
                    ),
                ]
            ),
            "factor_index": jnp.vstack(
                [
                    jnp.zeros((1, d), dtype=jnp.float32),
                    _normal(
                        next(keys), (settings.id_vocab_size, d), settings.init_scale
                    ),
                ]
            ),
            "tensor_id": jnp.vstack(
                [
                    jnp.zeros((1, d), dtype=jnp.float32),
                    _normal(
                        next(keys), (settings.id_vocab_size, d), settings.init_scale
                    ),
                ]
            ),
            "range_id": jnp.vstack(
                [
                    jnp.zeros((1, d), dtype=jnp.float32),
                    _normal(
                        next(keys), (settings.id_vocab_size, d), settings.init_scale
                    ),
                ]
            ),
            "index_id": jnp.vstack(
                [
                    jnp.zeros((1, d), dtype=jnp.float32),
                    _normal(
                        next(keys), (settings.id_vocab_size, d), settings.init_scale
                    ),
                ]
            ),
            "candidate_index": jnp.vstack(
                [
                    jnp.zeros((1, d), dtype=jnp.float32),
                    _normal(
                        next(keys), (settings.id_vocab_size, d), settings.init_scale
                    ),
                ]
            ),
        },
        "numeric_projection": _normal(next(keys), (3, d), settings.init_scale),
        "attention": [
            {
                "wq": _normal(next(keys), (d, d), settings.init_scale),
                "wk": _normal(next(keys), (d, d), settings.init_scale),
                "wv": _normal(next(keys), (d, d), settings.init_scale),
                "wo": _normal(next(keys), (d, d), settings.init_scale),
                "w1": _normal(next(keys), (d, 2 * d), settings.init_scale),
                "b1": jnp.zeros((2 * d,), dtype=jnp.float32),
                "w2": _normal(next(keys), (2 * d, d), settings.init_scale),
                "b2": jnp.zeros((d,), dtype=jnp.float32),
            }
            for _ in range(settings.num_attention_layers)
        ],
        "target": {
            "stop_w": _normal(next(keys), (d,), settings.init_scale),
            "def_w": _normal(next(keys), (d,), settings.init_scale),
            "stop_bias": jnp.asarray(settings.stop_bias_init, dtype=jnp.float32),
            "def_bias": jnp.asarray(0.0, dtype=jnp.float32),
        },
        "action": {
            "candidate_w": _normal(next(keys), (d,), settings.init_scale),
            "candidate_bias": jnp.asarray(0.0, dtype=jnp.float32),
            "left_w": _normal(next(keys), (d,), settings.init_scale),
            "left_bias": jnp.asarray(0.0, dtype=jnp.float32),
            "right_w": _normal(next(keys), (d,), settings.init_scale),
            "right_bias": jnp.asarray(0.0, dtype=jnp.float32),
            "left_context_w": _normal(next(keys), (d,), settings.init_scale),
        },
    }
    return params


def _token_kind_index(values, table):
    return jnp.where(values <= 0, 0, jnp.minimum(values, table.shape[0] - 1))


def _embedding_index(values, table):
    size = table.shape[0] - 1
    return jnp.where(values < 0, 0, (values % size) + 1)


def _masked_embedding(table, index):
    gathered = table[index]
    return gathered * (index != 0).astype(gathered.dtype)[:, None]


def _field(tokens: TokenTree, name: str, length: int):
    return tokens.get(name, jnp.full((length,), SENTINEL, dtype=jnp.int32))


def embed_tokens(params, tokens: TokenTree):
    tables = params["field_embeddings"]
    d = tables["token_kind"].shape[1]
    length = tokens["token_kind"].shape[0]
    out = jnp.zeros((length, d), dtype=jnp.float32)
    if "token_kind" in tokens:
        index = _token_kind_index(tokens["token_kind"], tables["token_kind"])
        out = out + _masked_embedding(tables["token_kind"], index)
    for field in ("segment", "side"):
        if field in tokens:
            index = _embedding_index(tokens[field], tables[field])
            out = out + _masked_embedding(tables[field], index)
    for field in _ID_FIELDS:
        if field in tokens:
            index = _embedding_index(tokens[field], tables[field])
            out = out + _masked_embedding(tables[field], index)
    coeff_num = _field(tokens, "coeff_num", length)
    coeff_den = _field(tokens, "coeff_den", length)
    position = _field(tokens, "position", length)
    coeff_present = coeff_den >= 0
    numeric = jnp.stack(
        [
            jnp.where(coeff_present, coeff_num.astype(jnp.float32), 0.0),
            jnp.where(coeff_present, coeff_den.astype(jnp.float32), 0.0),
            jnp.where(position < 0, 0.0, position.astype(jnp.float32)),
        ],
        axis=-1,
    )
    return out + numeric @ params["numeric_projection"]


def _layer_norm(x, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps)


def _attention_block(layer, x, mask):
    q = x @ layer["wq"]
    k = x @ layer["wk"]
    v = x @ layer["wv"]
    scale = 1.0 / math.sqrt(x.shape[-1])
    scores = (q @ k.T) * scale
    scores = jnp.where(mask[None, :], scores, -1.0e30)
    weights = jax.nn.softmax(scores, axis=-1)
    attended = weights @ v
    x = _layer_norm(x + attended @ layer["wo"])
    hidden = jax.nn.gelu(x @ layer["w1"] + layer["b1"])
    return _layer_norm(x + hidden @ layer["w2"] + layer["b2"])


def encode_tokens(params, embedded, mask):
    x = jnp.where(mask[:, None], embedded, 0.0)
    for layer in params["attention"]:
        x = _attention_block(layer, x, mask)
        x = jnp.where(mask[:, None], x, 0.0)
    return x


def masked_mean(values, mask):
    weights = mask.astype(jnp.float32)
    denom = jnp.maximum(jnp.sum(weights), 1.0)
    return jnp.sum(values * weights[:, None], axis=0) / denom


def pool_by_index(values, item_index, mask, requested_indices):
    def one(index):
        item_mask = mask & (item_index == index)
        return masked_mean(values, item_mask)

    return jax.vmap(one)(requested_indices)


class TransformerActionSelectorModel:
    def __init__(
        self,
        *,
        batch_size: int,
        max_steps: int,
        state_token_pad_to: int,
        action_token_pad_to: int,
        definition_pad_to: int,
        d_model: int = 32,
        num_attention_layers: int = 1,
        id_vocab_size: int = 128,
        init_scale: float = 0.02,
        stop_bias_init: float = -20.0,
    ):
        self._batch_size = _positive_int("batch_size", batch_size)
        self._max_steps = _positive_int("max_steps", max_steps)
        self._state_token_pad_to = _positive_int(
            "state_token_pad_to", state_token_pad_to
        )
        self._action_token_pad_to = _positive_int(
            "action_token_pad_to", action_token_pad_to
        )
        self._definition_pad_to = _positive_int(
            "definition_pad_to", definition_pad_to
        )
        self._settings = _PolicySettings(
            d_model=_positive_int("d_model", d_model),
            num_attention_layers=_positive_int(
                "num_attention_layers", num_attention_layers
            ),
            id_vocab_size=_positive_int("id_vocab_size", id_vocab_size),
            init_scale=_finite_float("init_scale", init_scale),
            stop_bias_init=_finite_float("stop_bias_init", stop_bias_init),
        )

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @property
    def state_token_pad_to(self) -> int:
        return self._state_token_pad_to

    @property
    def action_token_pad_to(self) -> int:
        return self._action_token_pad_to

    @property
    def definition_pad_to(self) -> int:
        return self._definition_pad_to

    def constructor_kwargs(self) -> dict[str, object]:
        return {
            "batch_size": self._batch_size,
            "max_steps": self._max_steps,
            "state_token_pad_to": self._state_token_pad_to,
            "action_token_pad_to": self._action_token_pad_to,
            "definition_pad_to": self._definition_pad_to,
            "d_model": self._settings.d_model,
            "num_attention_layers": self._settings.num_attention_layers,
            "id_vocab_size": self._settings.id_vocab_size,
            "init_scale": self._settings.init_scale,
            "stop_bias_init": self._settings.stop_bias_init,
        }

    def init_params(self, rng):
        return _init_policy_params(self._settings, rng)

    def sample_with_logp_grad(self, params, rng, row):
        from .rollout import _sample_static_model_rollout

        result = _sample_static_model_rollout(params, rng, row, self)
        return result.out_row, result.logp, result.grad_logp, {
            "stopped": result.stopped,
        }
