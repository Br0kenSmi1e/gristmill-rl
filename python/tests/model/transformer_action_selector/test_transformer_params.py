import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze

from gristmill_symbolics import TensorComputation
from gristmill_symbolics.model.transformer_action_selector import (
    TransformerActionSelectorModel,
)
from gristmill_symbolics.model.tokenizer import (
    TOKEN_KIND,
    tokenize_computation_snapshot,
)
from tests.test_bindings import actionable_json


def _model(**overrides):
    values = {
        "state_token_pad_to": 512,
        "action_token_pad_to": 512,
    }
    values.update(overrides)
    return TransformerActionSelectorModel(**values)


def _actionable_state_snapshot():
    return TensorComputation.from_json_string(actionable_json()).snapshot()


def test_init_params_shapes_and_network_tables():
    model = _model(d_model=16, id_vocab_size=32)
    params = model.init_params(jax.random.PRNGKey(0))

    assert params["embedder"]["token_kind_embedding"].shape[1] == 16
    assert set(params) == {"embedder", "encoder", "logit_decoder"}
    assert "symmetry_action_embedding" in params["embedder"]
    assert "perm_value_embedding" in params["embedder"]


def test_init_params_supports_configured_attention_layer_count():
    params = _model(
        d_model=8,
        num_attention_layers=8,
        id_vocab_size=16,
    ).init_params(jax.random.PRNGKey(3))

    assert "attention_7" in params["encoder"]
    assert params["encoder"]["mlp_out_7"]["kernel"].shape == (16, 8)


def test_embed_tokens_and_encoder_return_dense_token_vectors():
    model = _model(d_model=16)
    params = model.init_params(jax.random.PRNGKey(1))
    tokens, mask = tokenize_computation_snapshot(_actionable_state_snapshot())

    embedded = model.embedder.apply({"params": params["embedder"]}, tokens)
    encoded = model.encoder.apply(
        {"params": params["encoder"]},
        embedded,
        mask,
    )

    assert embedded.shape == (mask.shape[0], 16)
    assert encoded.shape == (mask.shape[0], 16)
    assert jnp.isfinite(encoded).all()


def test_sentinel_only_tokens_embed_to_zero_before_attention():
    model = _model(d_model=8)
    params = model.init_params(jax.random.PRNGKey(2))
    mutable = unfreeze(params)
    for name, values in mutable["embedder"].items():
        if name.endswith("_embedding"):
            mutable["embedder"][name] = values.at[0].set(1.0)
    params = freeze(mutable)
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

    embedded = model.embedder.apply({"params": params["embedder"]}, tokens)

    assert jnp.allclose(embedded, 0.0)


def test_sentinel_and_pad_embedding_rows_have_zero_gradients():
    model = _model(d_model=4, id_vocab_size=8)
    params = model.init_params(jax.random.PRNGKey(4))
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
        return model.embedder.apply({"params": p["embedder"]}, tokens).sum()

    grads = jax.grad(loss)(params)

    for name in ("token_kind", "segment", "def_index", "candidate_index"):
        assert jnp.allclose(grads["embedder"][f"{name}_embedding"][0], 0.0)


def test_negative_coefficient_numerators_contribute_to_embedding():
    model = _model(d_model=4)
    params = model.init_params(jax.random.PRNGKey(5))
    mutable = unfreeze(params)
    mutable["embedder"]["numeric_projection"]["kernel"] = jnp.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    params = freeze(mutable)
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

    negative_embedded = model.embedder.apply(
        {"params": params["embedder"]},
        negative_tokens,
    )
    zero_embedded = model.embedder.apply(
        {"params": params["embedder"]},
        zero_tokens,
    )

    assert not jnp.allclose(negative_embedded, zero_embedded)


def test_decode_logits_uses_vectors_and_condition_vector():
    model = _model(d_model=2, num_attention_heads=1)
    params = model.init_params(jax.random.PRNGKey(6))
    rows = jnp.asarray([[2.0, 3.0], [5.0, 7.0]], dtype=jnp.float32)
    condition = jnp.asarray([11.0, 13.0], dtype=jnp.float32)

    logits = model.logit_decoder.apply(
        {"params": params["logit_decoder"]},
        rows,
        condition,
    )

    assert logits.shape == (2,)
    assert jnp.all(jnp.isfinite(logits))
    assert not jnp.allclose(logits[0], logits[1])
