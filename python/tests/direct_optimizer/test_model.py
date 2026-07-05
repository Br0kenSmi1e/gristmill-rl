import jax
import jax.numpy as jnp
import pytest

from gristmill_symbolics.direct_optimizer.converter import computation_to_target_text
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


def test_make_decoder_inputs_adds_bos_prefix_and_eos_label():
    target = _target_tokens(length=32)

    decoder_input, labels, mask = make_decoder_inputs(target)
    real_length = int(sum(target["mask"]))

    assert int(decoder_input["kind"][0]) == KIND["BOS"]
    assert jnp.array_equal(decoder_input["kind"][1:], target["kind"][:-1])
    assert jnp.array_equal(labels["kind"][:real_length], target["kind"][:real_length])
    assert int(labels["kind"][real_length]) == KIND["EOS"]
    assert bool(mask[real_length])
    assert not bool(mask[real_length + 1])


def test_make_decoder_inputs_rejects_full_target_without_eos_room():
    target = _target_tokens(length=24)

    with pytest.raises(ValueError, match="no room for EOS"):
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
