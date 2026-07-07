import jax
import jax.numpy as jnp
from flax import nnx

from gristmill_symbolics.grammar import FlatDefinitionGrammar
from gristmill_symbolics.scoring import constrained_sequence_log_prob
from gristmill_symbolics.supervised import weighted_nll
from gristmill_symbolics.tokenizer import FlatDefinitionTokenizer


class _LogitModel(nnx.Module):
    def __init__(self, logits: jax.Array):
        self.logits = nnx.Param(logits)

    def __call__(
        self,
        source_ids: jax.Array,
        decoder_input_ids: jax.Array,
        *,
        deterministic: bool = True,
    ) -> jax.Array:
        del source_ids, decoder_input_ids, deterministic
        return self.logits[...]


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
    *,
    example_weight: jax.Array | None = None,
) -> dict[str, jax.Array]:
    bos = tokenizer.bos_token_id
    def_start = _id(tokenizer, "def_start")
    tensor0 = _id(tokenizer, "tensorid")
    def_end = _id(tokenizer, "def_end")
    eos = tokenizer.eos_token_id
    if example_weight is None:
        example_weight = jnp.asarray([0.25, 1.75], dtype=jnp.float32)
    return {
        "source_ids": jnp.asarray(
            [
                [1, 2, 0],
                [1, 0, 0],
            ],
            dtype=jnp.int32,
        ),
        "decoder_input_ids": jnp.asarray(
            [
                [bos, def_start, tensor0, def_end],
                [bos, def_start, tensor0, def_end],
            ],
            dtype=jnp.int32,
        ),
        "target_ids": jnp.asarray(
            [
                [def_start, tensor0, def_end, eos],
                [def_start, tensor0, def_end, eos],
            ],
            dtype=jnp.int32,
        ),
        "target_mask": jnp.asarray(
            [
                [True, True, True, True],
                [True, True, True, True],
            ]
        ),
        "example_weight": example_weight,
    }


def _logits(tokenizer: FlatDefinitionTokenizer) -> jax.Array:
    logits = jnp.zeros((2, 4, tokenizer.vocab_size), dtype=jnp.float32)
    logits = logits.at[0, 0, _id(tokenizer, "def_start")].set(1.0)
    logits = logits.at[0, 1, _id(tokenizer, "tensorid")].set(2.0)
    logits = logits.at[0, 2, _id(tokenizer, "def_end")].set(3.0)
    logits = logits.at[0, 3, tokenizer.eos_token_id].set(4.0)
    logits = logits.at[1, 0, _id(tokenizer, "def_start")].set(0.5)
    logits = logits.at[1, 1, _id(tokenizer, "tensorid")].set(1.5)
    logits = logits.at[1, 2, _id(tokenizer, "def_end")].set(2.5)
    logits = logits.at[1, 3, tokenizer.eos_token_id].set(3.5)
    return logits


def test_weighted_nll_matches_manual_sequence_nll():
    tokenizer = _tokenizer()
    grammar = FlatDefinitionGrammar(tokenizer)
    batch = _simple_batch(tokenizer)
    logits = _logits(tokenizer)
    model = _LogitModel(logits)

    weighted_nll_sum, weight_sum = weighted_nll(model, batch, grammar)

    manual_logp = constrained_sequence_log_prob(
        logits,
        batch["decoder_input_ids"],
        batch["target_ids"],
        batch["target_mask"],
        grammar,
    )
    expected = jnp.sum(batch["example_weight"] * -manual_logp)
    assert jnp.allclose(weighted_nll_sum, expected)
    assert jnp.allclose(weight_sum, jnp.sum(batch["example_weight"]))


def test_masked_target_positions_do_not_affect_weighted_nll():
    tokenizer = _tokenizer()
    grammar = FlatDefinitionGrammar(tokenizer)
    batch = _simple_batch(tokenizer, example_weight=jnp.asarray([1.0, 2.0]))
    logits = jnp.zeros((2, 4, tokenizer.vocab_size), dtype=jnp.float32)
    model = _LogitModel(logits)
    masked_target_ids = batch["target_ids"].at[:, 2:].set(
        _id(tokenizer, "tensorid", 2)
    )
    masked_target_mask = batch["target_mask"].at[:, 2:].set(False)
    batch = {
        **batch,
        "target_ids": masked_target_ids,
        "target_mask": masked_target_mask,
    }
    expected_target_ids = batch["target_ids"].at[:, 2:].set(tokenizer.pad_token_id)

    weighted_nll_sum, weight_sum = weighted_nll(model, batch, grammar)

    expected_logp = constrained_sequence_log_prob(
        logits,
        batch["decoder_input_ids"],
        expected_target_ids,
        batch["target_mask"],
        grammar,
    )
    assert jnp.allclose(
        weighted_nll_sum,
        jnp.sum(batch["example_weight"] * -expected_logp),
    )
    assert jnp.allclose(weight_sum, jnp.sum(batch["example_weight"]))


def test_zero_total_example_weight_returns_zero_totals():
    tokenizer = _tokenizer()
    grammar = FlatDefinitionGrammar(tokenizer)
    batch = _simple_batch(tokenizer, example_weight=jnp.asarray([0.0, 0.0]))
    logits = jnp.zeros((2, 4, tokenizer.vocab_size), dtype=jnp.float32)
    model = _LogitModel(logits)

    weighted_nll_sum, weight_sum = weighted_nll(model, batch, grammar)

    assert jnp.allclose(weight_sum, 0.0)
    assert jnp.allclose(weighted_nll_sum, 0.0)


def test_weighted_nll_is_compatible_with_nnx_jitted_value_and_grad():
    tokenizer = _tokenizer()
    grammar = FlatDefinitionGrammar(tokenizer)
    batch = _simple_batch(tokenizer, example_weight=jnp.asarray([1.0, 2.0]))
    model = _LogitModel(jnp.zeros((2, 4, tokenizer.vocab_size), dtype=jnp.float32))

    def objective(model: _LogitModel, batch: dict[str, jax.Array]):
        return weighted_nll(model, batch, grammar)

    loss_and_grad = nnx.jit(nnx.value_and_grad(objective, has_aux=True))

    (weighted_nll_sum, weight_sum), grads = loss_and_grad(model, batch)

    grad_leaves = jax.tree.leaves(grads)
    assert weighted_nll_sum.shape == ()
    assert weight_sum.shape == ()
    assert jnp.isfinite(weighted_nll_sum)
    assert any(bool(jnp.any(jnp.abs(leaf) > 0.0)) for leaf in grad_leaves)


def test_jitted_value_and_grad_uses_batch_as_runtime_argument():
    tokenizer = _tokenizer()
    grammar = FlatDefinitionGrammar(tokenizer)
    model = _LogitModel(jnp.zeros((2, 4, tokenizer.vocab_size), dtype=jnp.float32))

    def objective(model: _LogitModel, batch: dict[str, jax.Array]):
        return weighted_nll(model, batch, grammar)

    loss_and_grad = nnx.jit(nnx.value_and_grad(objective, has_aux=True))
    batch_a = _simple_batch(tokenizer, example_weight=jnp.asarray([1.0, 2.0]))
    batch_b = _simple_batch(tokenizer, example_weight=jnp.asarray([3.0, 4.0]))

    (_nll_a, weight_sum_a), _grads_a = loss_and_grad(model, batch_a)
    (_nll_b, weight_sum_b), _grads_b = loss_and_grad(model, batch_b)

    assert jnp.allclose(weight_sum_a, 3.0)
    assert jnp.allclose(weight_sum_b, 7.0)
