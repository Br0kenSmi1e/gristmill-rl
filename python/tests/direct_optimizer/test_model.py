import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from gristmill_symbolics.direct_optimizer.converter import (
    computation_to_source_text,
    computation_to_target_text,
)
from gristmill_symbolics.direct_optimizer import model as direct_model
from gristmill_symbolics.direct_optimizer.model import (
    DirectOptimizerTransformer,
    make_decoder_inputs,
    sample_tokens,
    sequence_log_prob,
    token_log_probs,
)
from gristmill_symbolics.direct_optimizer.tokens import (
    KIND,
    KEYWORD,
    SCALAR_TYPE,
    encode_text,
    pad_tokens,
)
from tests.direct_optimizer.fixtures import source_comp


def _target_tokens(length=32):
    return pad_tokens(
        encode_text(computation_to_target_text(source_comp())),
        length=length,
    )


def _source_tokens(length=64):
    return pad_tokens(
        encode_text(computation_to_source_text(source_comp())),
        length=length,
    )


def _batch(row, batch_size=2):
    return {
        key: jnp.asarray(value[None, :]).repeat(batch_size, axis=0)
        for key, value in row.items()
    }


def test_make_decoder_inputs_shifts_encoded_row_for_teacher_forcing():
    target = _target_tokens(length=32)

    decoder_input, labels, mask = make_decoder_inputs(target)
    real_length = int(sum(target["mask"]))
    label_length = real_length - 1

    assert int(decoder_input["kind"][0]) == KIND["BOS"]
    assert jnp.array_equal(
        decoder_input["kind"][:label_length],
        target["kind"][:label_length],
    )
    assert jnp.array_equal(labels["kind"][:label_length], target["kind"][1:real_length])
    assert int(jnp.sum(labels["kind"] == KIND["BOS"])) == 0
    assert int(labels["kind"][label_length - 1]) == KIND["EOS"]
    assert int(jnp.sum(labels["kind"] == KIND["EOS"])) == 1
    assert int(jnp.sum(decoder_input["kind"] == KIND["BOS"])) == 1
    assert int(jnp.sum(decoder_input["kind"] == KIND["EOS"])) == 0
    assert jnp.array_equal(mask[:label_length], jnp.ones(label_length, dtype=bool))
    assert not bool(mask[label_length])


def test_make_decoder_inputs_accepts_full_encoded_row_ending_in_eos():
    target = _target_tokens(length=24)

    decoder_input, labels, mask = make_decoder_inputs(target)
    label_length = int(sum(target["mask"])) - 1

    assert decoder_input["kind"].shape == target["kind"].shape
    assert int(labels["kind"][label_length - 1]) == KIND["EOS"]
    assert bool(mask[label_length - 1])
    assert int(labels["kind"][-1]) == KIND["PAD"]
    assert not bool(mask[-1])


def test_make_decoder_inputs_rejects_rows_without_encoded_controls():
    target = _target_tokens(length=32)
    target["kind"][0] = KIND["KEYWORD"]

    with pytest.raises(ValueError, match="BOS"):
        make_decoder_inputs(target)

    target = _target_tokens(length=32)
    target["kind"][int(sum(target["mask"])) - 1] = KIND["KEYWORD"]

    with pytest.raises(ValueError, match="EOS"):
        make_decoder_inputs(target)


def test_make_decoder_inputs_rejects_batched_rows():
    target = {
        field: jnp.asarray(value[None, :])
        for field, value in _target_tokens(length=32).items()
    }

    with pytest.raises(ValueError, match="1D encoded token row"):
        make_decoder_inputs(target)


def test_token_log_probs_scores_relevant_heads_only():
    target = {
        "kind": jnp.asarray([[KIND["KEYWORD"], KIND["SCALAR"], KIND["PAD"]]]),
        "keyword": jnp.asarray([[KEYWORD["def"], -1, -1]]),
        "scalar_type": jnp.asarray([[-1, SCALAR_TYPE["tensor_id"], -1]]),
        "scalar_value": jnp.asarray([[-1, 3, -1]]),
        "mask": jnp.asarray([[True, True, False]]),
    }
    logits = {
        "kind": jnp.zeros((1, 3, len(KIND))),
        "keyword": jnp.zeros((1, 3, len(KEYWORD))),
        "scalar_type": jnp.zeros((1, 3, len(SCALAR_TYPE))),
        "scalar_value": jnp.zeros((1, 3, 11)),
        "scalar_value_min": -5,
    }

    values = token_log_probs(logits, target)

    assert values.shape == (1, 3)
    assert values[0, 0] == pytest.approx(
        -jnp.log(len(KIND)) - jnp.log(len(KEYWORD))
    )
    assert values[0, 1] == pytest.approx(
        -jnp.log(len(KIND)) - jnp.log(len(SCALAR_TYPE)) - jnp.log(11)
    )
    assert values[0, 2] == pytest.approx(0.0)


def test_token_log_probs_ignores_inactive_out_of_range_scalar_labels():
    target = {
        "kind": jnp.asarray([[KIND["SCALAR"]]]),
        "keyword": jnp.asarray([[-1]]),
        "scalar_type": jnp.asarray([[SCALAR_TYPE["tensor_id"]]]),
        "scalar_value": jnp.asarray([[999]]),
        "mask": jnp.asarray([[False]]),
    }
    logits = {
        "kind": jnp.zeros((1, 1, len(KIND))),
        "keyword": jnp.zeros((1, 1, len(KEYWORD))),
        "scalar_type": jnp.zeros((1, 1, len(SCALAR_TYPE))),
        "scalar_value": jnp.zeros((1, 1, 3)),
        "scalar_value_min": 0,
    }

    values = token_log_probs(logits, target)
    sequence = sequence_log_prob(logits, target, target["mask"])

    assert values.shape == (1, 1)
    assert jnp.isfinite(values[0, 0])
    assert values[0, 0] == pytest.approx(0.0)
    assert jnp.isfinite(sequence[0])
    assert sequence[0] == pytest.approx(0.0)


def test_token_log_probs_uses_shifted_scalar_value_index():
    target = {
        "kind": jnp.asarray([[KIND["SCALAR"]]]),
        "keyword": jnp.asarray([[-1]]),
        "scalar_type": jnp.asarray([[SCALAR_TYPE["tensor_id"]]]),
        "scalar_value": jnp.asarray([[3]]),
        "mask": jnp.asarray([[True]]),
    }
    scalar_value_logits = jnp.full((1, 1, 11), -20.0)
    scalar_value_logits = scalar_value_logits.at[0, 0, 8].set(20.0)
    logits = {
        "kind": jnp.zeros((1, 1, len(KIND))),
        "keyword": jnp.zeros((1, 1, len(KEYWORD))),
        "scalar_type": jnp.zeros((1, 1, len(SCALAR_TYPE))),
        "scalar_value": scalar_value_logits,
        "scalar_value_min": -5,
    }

    values = token_log_probs(logits, target)

    expected_scalar_value_logp = jax.nn.log_softmax(scalar_value_logits, axis=-1)[
        0, 0, 8
    ]
    assert values[0, 0] == pytest.approx(
        -jnp.log(len(KIND))
        - jnp.log(len(SCALAR_TYPE))
        + expected_scalar_value_logp
    )


def test_token_log_probs_rejects_scalar_labels_outside_logits_bounds():
    target = {
        "kind": jnp.asarray([[KIND["SCALAR"]]]),
        "keyword": jnp.asarray([[-1]]),
        "scalar_type": jnp.asarray([[SCALAR_TYPE["tensor_id"]]]),
        "scalar_value": jnp.asarray([[9]]),
        "mask": jnp.asarray([[True]]),
    }
    logits = {
        "kind": jnp.zeros((1, 1, len(KIND))),
        "keyword": jnp.zeros((1, 1, len(KEYWORD))),
        "scalar_type": jnp.zeros((1, 1, len(SCALAR_TYPE))),
        "scalar_value": jnp.zeros((1, 1, 3)),
        "scalar_value_min": 0,
    }

    with pytest.raises(ValueError, match="scalar_value out of bounds"):
        token_log_probs(logits, target)


def test_token_log_probs_can_be_jitted_with_traced_scalar_value_min():
    target = {
        "kind": jnp.asarray([[KIND["SCALAR"]]]),
        "keyword": jnp.asarray([[-1]]),
        "scalar_type": jnp.asarray([[SCALAR_TYPE["tensor_id"]]]),
        "scalar_value": jnp.asarray([[3]]),
        "mask": jnp.asarray([[True]]),
    }
    logits = {
        "kind": jnp.zeros((1, 1, len(KIND))),
        "keyword": jnp.zeros((1, 1, len(KEYWORD))),
        "scalar_type": jnp.zeros((1, 1, len(SCALAR_TYPE))),
        "scalar_value": jnp.zeros((1, 1, 11)),
        "scalar_value_min": jnp.asarray(-5),
    }

    values = jax.jit(token_log_probs)(logits, target)

    assert values.shape == (1, 1)
    assert jnp.isfinite(values[0, 0])


def test_token_log_probs_can_be_differentiated_with_closed_over_labels(monkeypatch):
    target = {
        "kind": jnp.asarray([[KIND["SCALAR"]]]),
        "keyword": jnp.asarray([[-1]]),
        "scalar_type": jnp.asarray([[SCALAR_TYPE["tensor_id"]]]),
        "scalar_value": jnp.asarray([[3]]),
        "mask": jnp.asarray([[True]]),
    }
    scalar_value_min = jnp.asarray(-5)
    constant_logits = {
        "kind": jnp.zeros((1, 1, len(KIND))),
        "keyword": jnp.zeros((1, 1, len(KEYWORD))),
        "scalar_type": jnp.zeros((1, 1, len(SCALAR_TYPE))),
    }

    def fail_if_eager_validation_runs(*_args, **_kwargs):
        raise AssertionError("eager scalar-bound validation ran during tracing")

    monkeypatch.setattr(
        direct_model,
        "validate_scalar_bounds",
        fail_if_eager_validation_runs,
    )

    def loss(scalar_value_logits):
        logits = {
            **constant_logits,
            "scalar_value": scalar_value_logits,
            "scalar_value_min": scalar_value_min,
        }
        return -jnp.sum(token_log_probs(logits, target))

    value, grad = jax.jit(jax.value_and_grad(loss))(jnp.zeros((1, 1, 11)))

    assert jnp.isfinite(value)
    assert grad.shape == (1, 1, 11)


def test_sequence_log_prob_ignores_padding_mask():
    labels = {
        "kind": jnp.asarray([[KIND["KEYWORD"], KIND["KEYWORD"], KIND["PAD"]]]),
        "keyword": jnp.asarray([[KEYWORD["def"], KEYWORD["def"], -1]]),
        "scalar_type": jnp.asarray([[-1, -1, -1]]),
        "scalar_value": jnp.asarray([[-1, -1, -1]]),
        "mask": jnp.asarray([[True, True, False]]),
    }
    target_mask = jnp.asarray([[True, False, False]])
    keyword_logits = jnp.zeros((1, 3, len(KEYWORD)))
    keyword_logits = keyword_logits.at[0, 1, KEYWORD["def"]].set(-100.0)
    logits = {
        "kind": jnp.zeros((1, 3, len(KIND))),
        "keyword": keyword_logits,
        "scalar_type": jnp.zeros((1, 3, len(SCALAR_TYPE))),
        "scalar_value": jnp.zeros((1, 3, 21)),
        "scalar_value_min": -10,
    }

    token_scores = token_log_probs(logits, labels)
    seq = sequence_log_prob(logits, labels, target_mask)

    assert seq.shape == (1,)
    assert jnp.isfinite(seq[0])
    assert seq[0] == pytest.approx(token_scores[0, 0])
    assert seq[0] != pytest.approx(jnp.sum(token_scores[0]))


def test_nnx_transformer_returns_static_logits():
    model = DirectOptimizerTransformer(
        source_len=64,
        target_len=32,
        scalar_value_min=-16,
        scalar_value_max=16,
        d_model=16,
        num_layers=1,
        num_heads=2,
        rngs=nnx.Rngs(0),
    )
    target = _target_tokens(length=32)
    decoder_input, _labels, _mask = make_decoder_inputs(target)

    logits = model(_batch(_source_tokens()), _batch(decoder_input))

    assert logits["kind"].shape == (2, 32, len(KIND))
    assert logits["keyword"].shape == (2, 32, len(KEYWORD))
    assert logits["scalar_type"].shape == (2, 32, len(SCALAR_TYPE))
    assert logits["scalar_value"].shape == (2, 32, 33)
    assert logits["scalar_value_min"] == -16


def test_encoder_layer_gpu_attention_uses_cudnn_lengths_without_dense_mask(
    monkeypatch,
):
    monkeypatch.setattr(direct_model.jax, "default_backend", lambda: "gpu")

    def fail_make_attention_mask(*_args, **_kwargs):
        raise AssertionError("cuDNN attention must not build dense masks")

    calls = []

    def fake_dot_product_attention(query, key, value, **kwargs):
        calls.append(kwargs)
        return query

    monkeypatch.setattr(direct_model.nnx, "make_attention_mask", fail_make_attention_mask)
    monkeypatch.setattr(
        direct_model.jax.nn,
        "dot_product_attention",
        fake_dot_product_attention,
    )
    layer = direct_model.EncoderLayer(
        d_model=8,
        num_heads=2,
        dropout=0.0,
        kernel_init=jax.nn.initializers.normal(0.02),
        rngs=nnx.Rngs(0),
    )
    source_mask = jnp.asarray(
        [
            [True, True, False, False],
            [True, True, True, False],
        ]
    )

    out = layer(
        jnp.ones((2, 4, 8)),
        source_mask=source_mask,
        deterministic=True,
    )

    assert out.shape == (2, 4, 8)
    assert len(calls) == 1
    assert calls[0]["implementation"] == "cudnn"
    assert calls[0]["mask"] is None
    assert calls[0]["bias"] is None
    assert not calls[0]["is_causal"]
    assert calls[0]["query_seq_lengths"].tolist() == [2, 3]
    assert calls[0]["key_value_seq_lengths"].tolist() == [2, 3]


def test_decoder_layer_gpu_attention_uses_cudnn_for_self_and_cross_attention(
    monkeypatch,
):
    monkeypatch.setattr(direct_model.jax, "default_backend", lambda: "gpu")

    def fail_dense_mask(*_args, **_kwargs):
        raise AssertionError("cuDNN attention must not build dense masks")

    calls = []

    def fake_dot_product_attention(query, key, value, **kwargs):
        calls.append(kwargs)
        return query

    monkeypatch.setattr(direct_model.nnx, "make_attention_mask", fail_dense_mask)
    monkeypatch.setattr(direct_model.nnx, "make_causal_mask", fail_dense_mask)
    monkeypatch.setattr(direct_model.nnx, "combine_masks", fail_dense_mask)
    monkeypatch.setattr(
        direct_model.jax.nn,
        "dot_product_attention",
        fake_dot_product_attention,
    )
    layer = direct_model.DecoderLayer(
        d_model=8,
        num_heads=2,
        dropout=0.0,
        kernel_init=jax.nn.initializers.normal(0.02),
        rngs=nnx.Rngs(1),
    )
    target_mask = jnp.asarray(
        [
            [True, True, False],
            [True, True, True],
        ]
    )
    source_mask = jnp.asarray(
        [
            [True, True, True, False],
            [True, False, False, False],
        ]
    )

    out = layer(
        jnp.ones((2, 3, 8)),
        jnp.ones((2, 4, 8)),
        target_mask=target_mask,
        source_mask=source_mask,
        deterministic=True,
    )

    assert out.shape == (2, 3, 8)
    assert len(calls) == 2
    assert calls[0]["implementation"] == "cudnn"
    assert calls[0]["is_causal"]
    assert calls[0]["query_seq_lengths"].tolist() == [2, 3]
    assert calls[0]["key_value_seq_lengths"].tolist() == [2, 3]
    assert calls[1]["implementation"] == "cudnn"
    assert not calls[1]["is_causal"]
    assert calls[1]["query_seq_lengths"].tolist() == [2, 3]
    assert calls[1]["key_value_seq_lengths"].tolist() == [3, 1]


def test_structured_embedder_distinguishes_same_scalar_value_by_type():
    model = DirectOptimizerTransformer(
        source_len=4,
        target_len=4,
        scalar_value_min=-16,
        scalar_value_max=16,
        d_model=8,
        num_layers=1,
        num_heads=1,
        rngs=nnx.Rngs(1),
    )
    tensor_token = {
        "kind": jnp.asarray([[KIND["SCALAR"]]]),
        "keyword": jnp.asarray([[-1]]),
        "scalar_type": jnp.asarray([[SCALAR_TYPE["tensor_id"]]]),
        "scalar_value": jnp.asarray([[3]]),
        "mask": jnp.asarray([[True]]),
    }
    index_token = {
        **tensor_token,
        "scalar_type": jnp.asarray([[SCALAR_TYPE["index_id"]]]),
    }

    assert not bool(
        jnp.allclose(
            model.embed_tokens(tensor_token, length=1),
            model.embed_tokens(index_token, length=1),
        )
    )


def test_keyword_embedding_ignores_inactive_scalar_fields():
    model = DirectOptimizerTransformer(
        source_len=4,
        target_len=4,
        scalar_value_min=-16,
        scalar_value_max=16,
        d_model=8,
        num_layers=1,
        num_heads=1,
        rngs=nnx.Rngs(2),
    )
    keyword_token = {
        "kind": jnp.asarray([[KIND["KEYWORD"]]]),
        "keyword": jnp.asarray([[KEYWORD["def"]]]),
        "scalar_type": jnp.asarray([[SCALAR_TYPE["tensor_id"]]]),
        "scalar_value": jnp.asarray([[3]]),
        "mask": jnp.asarray([[True]]),
    }
    changed_inactive = {
        **keyword_token,
        "scalar_type": jnp.asarray([[SCALAR_TYPE["index_id"]]]),
        "scalar_value": jnp.asarray([[-7]]),
    }

    embedded = model.embed_tokens(keyword_token, length=1)
    expected = (
        model.kind_embed(keyword_token["kind"])
        + model.keyword_embed(keyword_token["keyword"])
        + model.source_position_embed(jnp.asarray([0]))[None, :, :]
    )

    assert jnp.allclose(
        embedded,
        model.embed_tokens(changed_inactive, length=1),
    )
    assert jnp.allclose(embedded, expected)


@pytest.mark.parametrize("kind_name", ["BOS", "EOS"])
def test_control_embedding_ignores_inactive_keyword_and_scalar_fields(kind_name):
    model = DirectOptimizerTransformer(
        source_len=4,
        target_len=4,
        scalar_value_min=-16,
        scalar_value_max=16,
        d_model=8,
        num_layers=1,
        num_heads=1,
        rngs=nnx.Rngs(3),
    )
    control_token = {
        "kind": jnp.asarray([[KIND[kind_name]]]),
        "keyword": jnp.asarray([[KEYWORD["def"]]]),
        "scalar_type": jnp.asarray([[SCALAR_TYPE["tensor_id"]]]),
        "scalar_value": jnp.asarray([[3]]),
        "mask": jnp.asarray([[True]]),
    }
    changed_inactive = {
        **control_token,
        "keyword": jnp.asarray([[KEYWORD["term"]]]),
        "scalar_type": jnp.asarray([[SCALAR_TYPE["index_id"]]]),
        "scalar_value": jnp.asarray([[-7]]),
    }

    embedded = model.embed_tokens(control_token, length=1)
    expected = (
        model.kind_embed(control_token["kind"])
        + model.source_position_embed(jnp.asarray([0]))[None, :, :]
    )

    assert jnp.allclose(
        embedded,
        model.embed_tokens(changed_inactive, length=1),
    )
    assert jnp.allclose(embedded, expected)


def test_sample_tokens_returns_static_padded_batch():
    model = DirectOptimizerTransformer(
        source_len=64,
        target_len=16,
        scalar_value_min=-8,
        scalar_value_max=8,
        d_model=16,
        num_layers=1,
        num_heads=2,
        rngs=nnx.Rngs(2),
    )
    source = _batch(_source_tokens(length=64), batch_size=3)

    generated, mask = sample_tokens(
        model,
        jax.random.PRNGKey(2),
        source,
        max_length=16,
        temperature=1.0,
    )

    assert generated["kind"].shape == (3, 16)
    assert generated["keyword"].shape == (3, 16)
    assert generated["scalar_type"].shape == (3, 16)
    assert generated["scalar_value"].shape == (3, 16)
    assert generated["mask"].shape == (3, 16)
    assert mask.shape == (3, 16)


def test_sample_tokens_is_deterministic_for_fixed_rng_and_state():
    model = DirectOptimizerTransformer(
        source_len=64,
        target_len=8,
        scalar_value_min=-4,
        scalar_value_max=4,
        d_model=8,
        num_layers=1,
        num_heads=1,
        rngs=nnx.Rngs(3),
    )
    source = _batch(_source_tokens(length=64), batch_size=2)

    left, left_mask = sample_tokens(
        model,
        jax.random.PRNGKey(5),
        source,
        max_length=8,
        temperature=1.0,
    )
    right, right_mask = sample_tokens(
        model,
        jax.random.PRNGKey(5),
        source,
        max_length=8,
        temperature=1.0,
    )

    assert all(jnp.array_equal(left[field], right[field]) for field in left)
    assert jnp.array_equal(left_mask, right_mask)
