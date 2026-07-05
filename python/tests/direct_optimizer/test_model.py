import jax
import jax.numpy as jnp
import pytest

from gristmill_symbolics.direct_optimizer.converter import computation_to_target_text
from gristmill_symbolics.direct_optimizer import model as direct_model
from gristmill_symbolics.direct_optimizer.model import (
    make_decoder_inputs,
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
    assert values[0, 2] == pytest.approx(-jnp.log(len(KIND)))


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
