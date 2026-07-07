import jax
import jax.numpy as jnp

from gristmill_symbolics.grammar import FlatDefinitionGrammar
from gristmill_symbolics.sampling import sample_token_ids
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


def _scripted_model(tokenizer: FlatDefinitionTokenizer, choices: list[int]):
    def model(source_ids, decoder_input_ids, *, deterministic=True):
        assert deterministic is True
        batch_size = source_ids.shape[0]
        target_len = decoder_input_ids.shape[1]
        logits = jnp.full(
            (batch_size, target_len, tokenizer.vocab_size),
            -1000.0,
            dtype=jnp.float32,
        )
        for position, token_id in enumerate(choices):
            logits = logits.at[:, position, token_id].set(1000.0)
        return logits

    return model


def _assert_grammar_valid_prefix(grammar: FlatDefinitionGrammar, row: jax.Array):
    state = grammar.initial_state(())
    for token_id in list(row):
        mask = grammar.allowed_by_state[state]
        assert bool(mask[int(token_id)])
        state = grammar.advance_state(
            state,
            jnp.asarray(token_id, dtype=jnp.int32),
        )


def test_sample_token_ids_generates_grammar_valid_sequence_with_logp():
    tokenizer = _tokenizer()
    grammar = FlatDefinitionGrammar(tokenizer)
    choices = [
        _id(tokenizer, "def_start"),
        _id(tokenizer, "tensorid"),
        _id(tokenizer, "def_end"),
        tokenizer.eos_token_id,
    ]
    source_ids = jnp.asarray([[1, 2, 0], [1, 0, 0]], dtype=jnp.int32)

    result = sample_token_ids(
        _scripted_model(tokenizer, choices),
        jax.random.key(0),
        source_ids,
        grammar,
        target_len=6,
    )

    expected = jnp.asarray(
        [
            [
                tokenizer.bos_token_id,
                choices[0],
                choices[1],
                choices[2],
                choices[3],
                tokenizer.pad_token_id,
            ],
            [
                tokenizer.bos_token_id,
                choices[0],
                choices[1],
                choices[2],
                choices[3],
                tokenizer.pad_token_id,
            ],
        ],
        dtype=jnp.int32,
    )
    assert result.generated_ids.shape == (2, 6)
    assert result.token_log_probs.shape == (2, 6)
    assert result.sequence_log_prob.shape == (2,)
    assert jnp.array_equal(result.generated_ids, expected)
    assert jnp.allclose(result.token_log_probs[:, 0], 0.0)
    assert jnp.allclose(result.token_log_probs[:, 5], 0.0)
    assert jnp.allclose(
        result.sequence_log_prob,
        jnp.sum(result.token_log_probs, axis=-1),
    )
    for row in result.generated_ids:
        _assert_grammar_valid_prefix(grammar, row)


def test_sample_token_ids_does_not_force_eos_at_max_length():
    tokenizer = _tokenizer()
    grammar = FlatDefinitionGrammar(tokenizer)
    choices = [
        _id(tokenizer, "def_start"),
        _id(tokenizer, "tensorid"),
        _id(tokenizer, "def_end"),
        _id(tokenizer, "def_start"),
        _id(tokenizer, "tensorid", 1),
    ]
    source_ids = jnp.asarray([[1, 0, 0]], dtype=jnp.int32)

    result = sample_token_ids(
        _scripted_model(tokenizer, choices),
        jax.random.key(1),
        source_ids,
        grammar,
        target_len=6,
    )

    assert result.generated_ids[0, 0] == tokenizer.bos_token_id
    assert tokenizer.eos_token_id not in set(map(int, result.generated_ids[0]))
    assert result.generated_ids[0, -1] == choices[-1]
    _assert_grammar_valid_prefix(grammar, result.generated_ids[0])


def test_sample_token_ids_is_jittable_for_fixed_shapes():
    tokenizer = _tokenizer()
    grammar = FlatDefinitionGrammar(tokenizer)
    choices = [
        _id(tokenizer, "def_start"),
        _id(tokenizer, "tensorid"),
        _id(tokenizer, "def_end"),
        tokenizer.eos_token_id,
    ]
    model = _scripted_model(tokenizer, choices)
    source_ids = jnp.asarray([[1, 2, 0]], dtype=jnp.int32)

    @jax.jit
    def run(rng, source):
        return sample_token_ids(
            model,
            rng,
            source,
            grammar,
            target_len=6,
        ).generated_ids

    generated = run(jax.random.key(2), source_ids)

    assert generated.shape == (1, 6)
    assert generated[0, 0] == tokenizer.bos_token_id
    assert generated[0, 4] == tokenizer.eos_token_id
    assert generated[0, 5] == tokenizer.pad_token_id


def test_sampled_logp_is_differentiable_to_logits():
    tokenizer = _tokenizer()
    grammar = FlatDefinitionGrammar(tokenizer)
    source_ids = jnp.asarray([[1, 0, 0]], dtype=jnp.int32)
    logits = jnp.zeros((1, 4, tokenizer.vocab_size), dtype=jnp.float32)

    def score(x):
        def model(_source_ids, _decoder_input_ids, *, deterministic=True):
            assert deterministic is True
            return x

        result = sample_token_ids(
            model,
            jax.random.key(3),
            source_ids,
            grammar,
            target_len=4,
        )
        return result.sequence_log_prob[0]

    grad = jax.grad(score)(logits)

    assert grad.shape == logits.shape
    assert bool(jnp.any(jnp.abs(grad) > 0.0))
