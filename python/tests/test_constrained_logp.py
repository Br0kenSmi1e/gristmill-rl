import jax
import jax.numpy as jnp

from gristmill_symbolics.grammar import FlatDefinitionGrammar
from gristmill_symbolics.scoring import (
    constrained_sequence_log_prob,
    constrained_token_log_probs,
)
from gristmill_symbolics.tokenizer import FlatDefinitionTokenizer


def _tokenizer() -> FlatDefinitionTokenizer:
    return FlatDefinitionTokenizer(
        max_range_id=2,
        max_tensor_id=3,
        max_index_id=4,
        coeff_nums=(-1, 1, 2),
        coeff_dens=(1, 2),
    )


def _id(tokenizer: FlatDefinitionTokenizer, kind: str, offset: int = 0) -> int:
    return tokenizer.token_ids_for_kind(kind)[offset]


def test_constrained_token_log_probs_score_only_grammar_valid_logits():
    tokenizer = _tokenizer()
    grammar = FlatDefinitionGrammar(tokenizer)
    bos = tokenizer.bos_token_id
    def_start = _id(tokenizer, "def_start")
    eos = tokenizer.eos_token_id
    tensor0 = _id(tokenizer, "tensorid")
    tensor1 = _id(tokenizer, "tensorid", 1)
    logits = jnp.zeros((1, 2, tokenizer.vocab_size), dtype=jnp.float32)
    logits = logits.at[0, 0, def_start].set(1.5)
    logits = logits.at[0, 0, eos].set(-0.5)
    logits = logits.at[0, 0, tensor0].set(100.0)
    logits = logits.at[0, 1, tensor0].set(0.25)
    logits = logits.at[0, 1, tensor1].set(1.25)
    decoder_input_ids = jnp.asarray([[bos, def_start]], dtype=jnp.int32)
    labels = jnp.asarray([[def_start, tensor1]], dtype=jnp.int32)
    label_mask = jnp.asarray([[True, True]])

    token_logp = constrained_token_log_probs(
        logits,
        decoder_input_ids,
        labels,
        label_mask,
        grammar,
    )

    expected_first = jax.nn.log_softmax(
        jnp.asarray([1.5, -0.5], dtype=jnp.float32)
    )[0]
    tensor_logits = logits[0, 1, jnp.asarray(tokenizer.token_ids_for_kind("tensorid"))]
    expected_second = jax.nn.log_softmax(tensor_logits)[1]
    assert jnp.allclose(token_logp[0, 0], expected_first)
    assert jnp.allclose(token_logp[0, 1], expected_second)


def test_masked_positions_return_zero_and_sequence_logp_sums_active_tokens():
    tokenizer = _tokenizer()
    grammar = FlatDefinitionGrammar(tokenizer)
    bos = tokenizer.bos_token_id
    def_start = _id(tokenizer, "def_start")
    tensor0 = _id(tokenizer, "tensorid")
    def_end = _id(tokenizer, "def_end")
    logits = jnp.zeros((1, 4, tokenizer.vocab_size), dtype=jnp.float32)
    decoder_input_ids = jnp.asarray(
        [[bos, def_start, tensor0, def_end]],
        dtype=jnp.int32,
    )
    labels = jnp.asarray(
        [[def_start, tensor0, def_end, tokenizer.eos_token_id]],
        dtype=jnp.int32,
    )
    label_mask = jnp.asarray([[True, True, False, False]])

    token_logp = constrained_token_log_probs(
        logits,
        decoder_input_ids,
        labels,
        label_mask,
        grammar,
    )
    sequence_logp = constrained_sequence_log_prob(
        logits,
        decoder_input_ids,
        labels,
        label_mask,
        grammar,
    )

    assert token_logp.shape == (1, 4)
    assert jnp.allclose(token_logp[0, 2:], 0.0)
    assert jnp.allclose(sequence_logp, jnp.sum(token_logp, axis=-1))


def test_active_grammar_invalid_labels_get_large_negative_logp():
    tokenizer = _tokenizer()
    grammar = FlatDefinitionGrammar(tokenizer)
    bos = tokenizer.bos_token_id
    invalid_label = _id(tokenizer, "tensorid")
    logits = jnp.zeros((1, 1, tokenizer.vocab_size), dtype=jnp.float32)
    decoder_input_ids = jnp.asarray([[bos]], dtype=jnp.int32)
    labels = jnp.asarray([[invalid_label]], dtype=jnp.int32)
    label_mask = jnp.asarray([[True]])

    token_logp = constrained_token_log_probs(
        logits,
        decoder_input_ids,
        labels,
        label_mask,
        grammar,
    )

    assert bool(jnp.isneginf(token_logp[0, 0]))


def test_constrained_sequence_log_prob_is_jittable():
    tokenizer = _tokenizer()
    grammar = FlatDefinitionGrammar(tokenizer)
    bos = tokenizer.bos_token_id
    def_start = _id(tokenizer, "def_start")
    logits = jnp.zeros((1, 1, tokenizer.vocab_size), dtype=jnp.float32)
    decoder_input_ids = jnp.asarray([[bos]], dtype=jnp.int32)
    labels = jnp.asarray([[def_start]], dtype=jnp.int32)
    label_mask = jnp.asarray([[True]])

    jitted = jax.jit(
        lambda x: constrained_sequence_log_prob(
            x,
            decoder_input_ids,
            labels,
            label_mask,
            grammar,
        )
    )

    result = jitted(logits)

    assert result.shape == (1,)
    assert jnp.isfinite(result[0])


def test_gradients_flow_through_valid_active_logits():
    tokenizer = _tokenizer()
    grammar = FlatDefinitionGrammar(tokenizer)
    bos = tokenizer.bos_token_id
    def_start = _id(tokenizer, "def_start")
    eos = tokenizer.eos_token_id
    tensor0 = _id(tokenizer, "tensorid")
    logits = jnp.zeros((1, 1, tokenizer.vocab_size), dtype=jnp.float32)
    decoder_input_ids = jnp.asarray([[bos]], dtype=jnp.int32)
    labels = jnp.asarray([[def_start]], dtype=jnp.int32)
    label_mask = jnp.asarray([[True]])

    def score(x):
        return jnp.sum(
            constrained_sequence_log_prob(
                x,
                decoder_input_ids,
                labels,
                label_mask,
                grammar,
            )
        )

    grad = jax.grad(score)(logits)

    assert abs(float(grad[0, 0, def_start])) > 0.0
    assert abs(float(grad[0, 0, eos])) > 0.0
    assert float(grad[0, 0, tensor0]) == 0.0


def test_invalid_active_label_has_no_gradient_signal():
    tokenizer = _tokenizer()
    grammar = FlatDefinitionGrammar(tokenizer)
    bos = tokenizer.bos_token_id
    invalid_label = _id(tokenizer, "tensorid")
    logits = jnp.zeros((1, 1, tokenizer.vocab_size), dtype=jnp.float32)
    decoder_input_ids = jnp.asarray([[bos]], dtype=jnp.int32)
    labels = jnp.asarray([[invalid_label]], dtype=jnp.int32)
    label_mask = jnp.asarray([[True]])

    def score(x):
        return jnp.sum(
            constrained_sequence_log_prob(
                x,
                decoder_input_ids,
                labels,
                label_mask,
                grammar,
            )
        )

    grad = jax.grad(score)(logits)

    assert jnp.allclose(grad, 0.0)
