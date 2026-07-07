import jax
import jax.numpy as jnp

from gristmill_symbolics.grammar import FlatDefinitionGrammar
from gristmill_symbolics.scoring import constrained_sequence_log_prob
from gristmill_symbolics.supervised import weighted_supervised_nll_totals
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


def _simple_batch(
    tokenizer: FlatDefinitionTokenizer,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    bos = tokenizer.bos_token_id
    def_start = _id(tokenizer, "def_start")
    tensor0 = _id(tokenizer, "tensorid")
    def_end = _id(tokenizer, "def_end")
    eos = tokenizer.eos_token_id
    decoder_input_ids = jnp.asarray(
        [
            [bos, def_start, tensor0, def_end],
            [bos, def_start, tensor0, def_end],
        ],
        dtype=jnp.int32,
    )
    target_ids = jnp.asarray(
        [
            [def_start, tensor0, def_end, eos],
            [def_start, tensor0, def_end, eos],
        ],
        dtype=jnp.int32,
    )
    target_mask = jnp.asarray(
        [
            [True, True, True, True],
            [True, True, True, True],
        ]
    )
    return decoder_input_ids, target_ids, target_mask


def test_weighted_supervised_nll_totals_matches_manual_sequence_nll():
    tokenizer = _tokenizer()
    grammar = FlatDefinitionGrammar(tokenizer)
    decoder_input_ids, target_ids, target_mask = _simple_batch(tokenizer)
    logits = jnp.zeros((2, 4, tokenizer.vocab_size), dtype=jnp.float32)
    logits = logits.at[0, 0, _id(tokenizer, "def_start")].set(1.0)
    logits = logits.at[0, 1, _id(tokenizer, "tensorid")].set(2.0)
    logits = logits.at[0, 2, _id(tokenizer, "def_end")].set(3.0)
    logits = logits.at[0, 3, tokenizer.eos_token_id].set(4.0)
    logits = logits.at[1, 0, _id(tokenizer, "def_start")].set(0.5)
    logits = logits.at[1, 1, _id(tokenizer, "tensorid")].set(1.5)
    logits = logits.at[1, 2, _id(tokenizer, "def_end")].set(2.5)
    logits = logits.at[1, 3, tokenizer.eos_token_id].set(3.5)
    example_weight = jnp.asarray([0.25, 1.75], dtype=jnp.float32)

    weighted_nll_sum, weight_sum = weighted_supervised_nll_totals(
        logits,
        decoder_input_ids,
        target_ids,
        target_mask,
        example_weight,
        grammar,
    )

    manual_logp = constrained_sequence_log_prob(
        logits,
        decoder_input_ids,
        target_ids,
        target_mask,
        grammar,
    )
    expected = jnp.sum(example_weight * -manual_logp)
    assert jnp.allclose(weighted_nll_sum, expected)
    assert jnp.allclose(weight_sum, jnp.sum(example_weight))


def test_masked_target_positions_do_not_affect_weighted_nll():
    tokenizer = _tokenizer()
    grammar = FlatDefinitionGrammar(tokenizer)
    decoder_input_ids, target_ids, target_mask = _simple_batch(tokenizer)
    logits = jnp.zeros((2, 4, tokenizer.vocab_size), dtype=jnp.float32)
    example_weight = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    masked_target_ids = target_ids.at[:, 2:].set(_id(tokenizer, "tensorid", 2))
    masked_target_mask = target_mask.at[:, 2:].set(False)
    expected_target_ids = target_ids.at[:, 2:].set(tokenizer.pad_token_id)

    weighted_nll_sum, weight_sum = weighted_supervised_nll_totals(
        logits,
        decoder_input_ids,
        masked_target_ids,
        masked_target_mask,
        example_weight,
        grammar,
    )

    expected_logp = constrained_sequence_log_prob(
        logits,
        decoder_input_ids,
        expected_target_ids,
        masked_target_mask,
        grammar,
    )
    assert jnp.allclose(weighted_nll_sum, jnp.sum(example_weight * -expected_logp))
    assert jnp.allclose(weight_sum, jnp.sum(example_weight))


def test_zero_total_example_weight_returns_zero_totals():
    tokenizer = _tokenizer()
    grammar = FlatDefinitionGrammar(tokenizer)
    decoder_input_ids, target_ids, target_mask = _simple_batch(tokenizer)
    logits = jnp.zeros((2, 4, tokenizer.vocab_size), dtype=jnp.float32)
    example_weight = jnp.asarray([0.0, 0.0], dtype=jnp.float32)

    weighted_nll_sum, weight_sum = weighted_supervised_nll_totals(
        logits,
        decoder_input_ids,
        target_ids,
        target_mask,
        example_weight,
        grammar,
    )

    assert jnp.allclose(weight_sum, 0.0)
    assert jnp.allclose(weighted_nll_sum, 0.0)


def test_weighted_supervised_nll_totals_is_jittable_for_fixed_shapes():
    tokenizer = _tokenizer()
    grammar = FlatDefinitionGrammar(tokenizer)
    decoder_input_ids, target_ids, target_mask = _simple_batch(tokenizer)
    logits = jnp.zeros((2, 4, tokenizer.vocab_size), dtype=jnp.float32)
    example_weight = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    jitted = jax.jit(
        lambda x: weighted_supervised_nll_totals(
            x,
            decoder_input_ids,
            target_ids,
            target_mask,
            example_weight,
            grammar,
        )
    )

    weighted_nll_sum, weight_sum = jitted(logits)

    assert weighted_nll_sum.shape == ()
    assert weight_sum.shape == ()
    assert jnp.isfinite(weighted_nll_sum)


def test_weighted_nll_sum_gradients_flow_to_valid_active_logits():
    tokenizer = _tokenizer()
    grammar = FlatDefinitionGrammar(tokenizer)
    bos = tokenizer.bos_token_id
    def_start = _id(tokenizer, "def_start")
    eos = tokenizer.eos_token_id
    tensor0 = _id(tokenizer, "tensorid")
    logits = jnp.zeros((1, 1, tokenizer.vocab_size), dtype=jnp.float32)
    decoder_input_ids = jnp.asarray([[bos]], dtype=jnp.int32)
    target_ids = jnp.asarray([[def_start]], dtype=jnp.int32)
    target_mask = jnp.asarray([[True]])
    example_weight = jnp.asarray([2.0], dtype=jnp.float32)

    def loss(x):
        weighted_nll_sum, _weight_sum = weighted_supervised_nll_totals(
            x,
            decoder_input_ids,
            target_ids,
            target_mask,
            example_weight,
            grammar,
        )
        return weighted_nll_sum

    grad = jax.grad(loss)(logits)

    assert abs(float(grad[0, 0, def_start])) > 0.0
    assert abs(float(grad[0, 0, eos])) > 0.0
    assert float(grad[0, 0, tensor0]) == 0.0
