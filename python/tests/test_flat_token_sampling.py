import jax
import jax.numpy as jnp
from flax import nnx

from gristmill_symbolics.grammar import FlatDefinitionGrammar
from gristmill_symbolics.nn import FlatDefinitionSeq2SeqTransformer
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


class _ScriptedCachedModel(nnx.Module):
    def __init__(
        self,
        tokenizer: FlatDefinitionTokenizer,
        choices: list[int],
        *,
        logits: jax.Array | None = None,
    ):
        self.vocab_size = tokenizer.vocab_size
        self.choice_ids = nnx.data(jnp.asarray(choices, dtype=jnp.int32))
        self.logits = nnx.data(logits)

    def encode(self, source_ids, *, deterministic=True):
        assert deterministic is True
        return source_ids[..., None].astype(jnp.float32), source_ids != 0

    def init_decode_cache(self, *, batch_size: int, target_len: int):
        del batch_size, target_len

    def decode_step(
        self,
        token_ids_t,
        memory,
        *,
        source_mask=None,
        step,
        deterministic=True,
    ):
        assert deterministic is True
        del memory, source_mask
        if self.logits is not None:
            return jnp.take(self.logits, step, axis=1)

        batch_size = token_ids_t.shape[0]
        logits = jnp.full(
            (batch_size, self.vocab_size),
            -1000.0,
            dtype=jnp.float32,
        )
        token_id = jnp.take(self.choice_ids, step)
        return logits.at[:, token_id].set(1000.0)

    def __call__(self, *args, **kwargs):
        raise AssertionError("sample_token_ids must not call full model")


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

    model = _ScriptedCachedModel(tokenizer, choices)

    result = sample_token_ids(
        model,
        jax.random.key(0),
        source_ids,
        grammar,
        target_len=6,
    )
    assert type(result) is tuple
    generated_ids, token_log_probs, sequence_log_prob = result

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
    assert generated_ids.shape == (2, 6)
    assert token_log_probs.shape == (2, 6)
    assert sequence_log_prob.shape == (2,)
    assert jnp.array_equal(generated_ids, expected)
    assert jnp.allclose(token_log_probs[:, 0], 0.0)
    assert jnp.allclose(token_log_probs[:, 5], 0.0)
    assert jnp.allclose(
        sequence_log_prob,
        jnp.sum(token_log_probs, axis=-1),
    )
    for row in generated_ids:
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

    generated_ids, _token_log_probs, _sequence_log_prob = sample_token_ids(
        _ScriptedCachedModel(tokenizer, choices),
        jax.random.key(1),
        source_ids,
        grammar,
        target_len=6,
    )

    assert generated_ids[0, 0] == tokenizer.bos_token_id
    assert tokenizer.eos_token_id not in set(map(int, generated_ids[0]))
    assert generated_ids[0, -1] == choices[-1]
    _assert_grammar_valid_prefix(grammar, generated_ids[0])


def test_sample_token_ids_runs_with_real_bfloat16_cached_model():
    tokenizer = FlatDefinitionTokenizer(
        max_range_id=0,
        max_tensor_id=1,
        max_index_id=0,
        coeff_nums=(1,),
        coeff_dens=(1,),
    )
    grammar = FlatDefinitionGrammar(tokenizer)
    model = FlatDefinitionSeq2SeqTransformer(
        source_len=4,
        target_len=5,
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        d_model=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
        rngs=nnx.Rngs(4),
    )

    generated_ids, token_log_probs, sequence_log_prob = sample_token_ids(
        model,
        jax.random.key(4),
        jnp.asarray([[1, 2, 0, 0]], dtype=jnp.int32),
        grammar,
        target_len=5,
    )

    assert generated_ids.shape == (1, 5)
    assert token_log_probs.shape == (1, 5)
    assert sequence_log_prob.shape == (1,)
    assert generated_ids[0, 0] == tokenizer.bos_token_id


def test_sampled_logp_is_differentiable_to_logits():
    tokenizer = _tokenizer()
    grammar = FlatDefinitionGrammar(tokenizer)
    source_ids = jnp.asarray([[1, 0, 0]], dtype=jnp.int32)
    logits = jnp.zeros((1, 4, tokenizer.vocab_size), dtype=jnp.float32)

    def score(x):
        model = _ScriptedCachedModel(tokenizer, [], logits=x)

        _generated_ids, _token_log_probs, sequence_log_prob = sample_token_ids(
            model,
            jax.random.key(3),
            source_ids,
            grammar,
            target_len=4,
        )
        return sequence_log_prob[0]

    grad = jax.grad(score)(logits)

    assert grad.shape == logits.shape
    assert bool(jnp.any(jnp.abs(grad) > 0.0))
