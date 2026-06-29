import jax
import jax.numpy as jnp

from gristmill_symbolics.model.transformer_action_selector import (
    TransformerActionSelectorModel,
)
from gristmill_symbolics.model.transformer_action_selector.constants import TOKEN_KIND
from gristmill_symbolics.model.transformer_action_selector.model import (
    embed_tokens,
    encode_tokens,
    pool_by_index,
)
from gristmill_symbolics.model.transformer_action_selector.tokenize import (
    tokenize_state_snapshot,
)
from tests.policy_fixtures import actionable_state_snapshot


def _model(**overrides):
    values = {
        "batch_size": 1,
        "max_steps": 1,
        "state_token_pad_to": 512,
        "action_token_pad_to": 512,
        "definition_pad_to": 8,
    }
    values.update(overrides)
    return TransformerActionSelectorModel(**values)


def test_init_params_shapes_and_stop_bias():
    params = _model(d_model=16, id_vocab_size=32).init_params(jax.random.PRNGKey(0))

    assert params["field_embeddings"]["token_kind"].shape[1] == 16
    assert set(params["action"]) == {
        "candidate_w",
        "candidate_bias",
        "left_w",
        "left_bias",
        "right_w",
        "right_bias",
        "left_context_w",
    }
    assert params["action"]["candidate_bias"].shape == ()
    assert params["action"]["left_bias"].shape == ()
    assert params["action"]["right_bias"].shape == ()
    assert params["target"]["stop_bias"].shape == ()
    assert float(params["target"]["stop_bias"]) == -20.0


def test_init_params_supports_configured_attention_layer_count():
    params = _model(
        d_model=8,
        num_attention_layers=8,
        id_vocab_size=16,
    ).init_params(jax.random.PRNGKey(3))

    assert len(params["attention"]) == 8
    assert params["attention"][-1]["w2"].shape == (16, 8)


def test_embed_tokens_and_encoder_return_dense_token_vectors():
    params = _model(d_model=16).init_params(jax.random.PRNGKey(1))
    tokens, mask = tokenize_state_snapshot(actionable_state_snapshot())

    embedded = embed_tokens(params, tokens)
    encoded = encode_tokens(params, embedded, mask)

    assert embedded.shape == (mask.shape[0], 16)
    assert encoded.shape == (mask.shape[0], 16)
    assert jnp.isfinite(encoded).all()


def test_sentinel_only_tokens_embed_to_zero_before_attention():
    params = _model(d_model=8).init_params(jax.random.PRNGKey(2))
    for name, table in params["field_embeddings"].items():
        params["field_embeddings"][name] = table.at[0].set(1.0)
    tokens = {
        "token_kind": jnp.array([0], dtype=jnp.int32),
        "segment": jnp.array([-1], dtype=jnp.int32),
        "side": jnp.array([-1], dtype=jnp.int32),
        "def_index": jnp.array([-1], dtype=jnp.int32),
        "term_index": jnp.array([-1], dtype=jnp.int32),
        "factor_index": jnp.array([-1], dtype=jnp.int32),
        "tensor_id": jnp.array([-1], dtype=jnp.int32),
        "range_id": jnp.array([-1], dtype=jnp.int32),
        "index_id": jnp.array([-1], dtype=jnp.int32),
        "candidate_index": jnp.array([-1], dtype=jnp.int32),
        "coeff_num": jnp.array([-1], dtype=jnp.int32),
        "coeff_den": jnp.array([-1], dtype=jnp.int32),
        "position": jnp.array([-1], dtype=jnp.int32),
    }

    embedded = embed_tokens(params, tokens)

    assert jnp.allclose(embedded, 0.0)


def test_sentinel_and_pad_embedding_rows_have_zero_gradients():
    params = _model(d_model=4, id_vocab_size=8).init_params(jax.random.PRNGKey(4))
    tokens = {
        "token_kind": jnp.array(
            [int(TOKEN_KIND.PAD), int(TOKEN_KIND.RANGE)], dtype=jnp.int32
        ),
        "segment": jnp.array([-1, 0], dtype=jnp.int32),
        "def_index": jnp.array([-1, 0], dtype=jnp.int32),
        "candidate_index": jnp.array([-1, 0], dtype=jnp.int32),
        "coeff_num": jnp.array([-1, -1], dtype=jnp.int32),
        "coeff_den": jnp.array([-1, -1], dtype=jnp.int32),
        "position": jnp.array([-1, -1], dtype=jnp.int32),
    }

    def loss(p):
        return embed_tokens(p, tokens).sum()

    grads = jax.grad(loss)(params)

    for name in ("token_kind", "segment", "def_index", "candidate_index"):
        assert jnp.allclose(grads["field_embeddings"][name][0], 0.0)


def test_negative_coefficient_numerators_contribute_to_embedding():
    params = _model(d_model=4).init_params(jax.random.PRNGKey(5))
    params["numeric_projection"] = jnp.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    base_tokens = {
        "token_kind": jnp.array([int(TOKEN_KIND.COEFF)], dtype=jnp.int32),
        "coeff_den": jnp.array([1], dtype=jnp.int32),
        "position": jnp.array([-1], dtype=jnp.int32),
    }
    negative_tokens = {
        **base_tokens,
        "coeff_num": jnp.array([-2], dtype=jnp.int32),
    }
    zero_tokens = {
        **base_tokens,
        "coeff_num": jnp.array([0], dtype=jnp.int32),
    }

    negative_embedded = embed_tokens(params, negative_tokens)
    zero_embedded = embed_tokens(params, zero_tokens)

    assert not jnp.allclose(negative_embedded, zero_embedded)


def test_pool_by_index_returns_one_embedding_per_requested_index():
    values = jnp.asarray([[1.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    item_index = jnp.asarray([0, 0, 1], dtype=jnp.int32)
    mask = jnp.asarray([True, True, True])
    pooled = pool_by_index(values, item_index, mask, jnp.arange(3, dtype=jnp.int32))

    assert pooled.shape == (3, 2)
    assert jnp.allclose(pooled[0], jnp.asarray([2.0, 0.0]))
    assert jnp.allclose(pooled[1], jnp.asarray([0.0, 4.0]))
    assert jnp.allclose(pooled[2], jnp.asarray([0.0, 0.0]))
