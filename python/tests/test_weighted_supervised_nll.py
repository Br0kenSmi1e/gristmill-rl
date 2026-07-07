import jax
import jax.numpy as jnp
import optax
from flax import nnx

import gristmill_symbolics.supervised as supervised
from gristmill_symbolics.grammar import FlatDefinitionGrammar
from gristmill_symbolics.scoring import constrained_sequence_log_prob
from gristmill_symbolics.supervised import (
    accumulate_weighted_nll_grad,
    SupervisedTrainer,
    weighted_nll,
)
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


def test_accumulate_weighted_nll_grad_adds_totals_and_grad_trees():
    first_grads = {"w": jnp.asarray([1.0, 2.0])}
    second_grads = {"w": jnp.asarray([3.0, 4.0])}

    accumulated = accumulate_weighted_nll_grad(
        None,
        jnp.asarray(2.0),
        jnp.asarray(5.0),
        first_grads,
    )
    accumulated = accumulate_weighted_nll_grad(
        accumulated,
        jnp.asarray(3.0),
        jnp.asarray(7.0),
        second_grads,
    )

    total_nll, total_weight, total_grads = accumulated
    assert jnp.allclose(total_nll, 5.0)
    assert jnp.allclose(total_weight, 12.0)
    assert jnp.allclose(total_grads["w"], jnp.asarray([4.0, 6.0]))


def test_supervised_trainer_update_accumulates_batches_and_updates_once():
    tokenizer = _tokenizer()
    grammar = FlatDefinitionGrammar(tokenizer)
    trainer = SupervisedTrainer(grammar)
    model = _LogitModel(jnp.zeros((2, 4, tokenizer.vocab_size), dtype=jnp.float32))
    optimizer = nnx.Optimizer(model, optax.sgd(0.1), wrt=nnx.Param)
    batches = (
        _simple_batch(tokenizer, example_weight=jnp.asarray([1.0, 2.0])),
        _simple_batch(tokenizer, example_weight=jnp.asarray([3.0, 4.0])),
    )
    expected_totals = [weighted_nll(model, batch, grammar) for batch in batches]
    expected_nll = sum(total[0] for total in expected_totals)
    expected_weight = sum(total[1] for total in expected_totals)
    before_logits = model.logits[...]

    metrics = trainer.update(model, optimizer, batches)

    assert jnp.allclose(metrics["weighted_nll_sum"], expected_nll)
    assert jnp.allclose(metrics["weight_sum"], expected_weight)
    assert jnp.allclose(metrics["mean_nll"], expected_nll / expected_weight)
    assert metrics["num_batches"] == 2
    assert int(optimizer.step[...]) == 1
    assert not jnp.allclose(model.logits[...], before_logits)


def test_supervised_trainer_update_uses_mean_nll_gradient():
    tokenizer = _tokenizer()
    grammar = FlatDefinitionGrammar(tokenizer)
    trainer = SupervisedTrainer(grammar)
    model = _LogitModel(jnp.zeros((2, 4, tokenizer.vocab_size), dtype=jnp.float32))
    optimizer = nnx.Optimizer(model, optax.sgd(0.1), wrt=nnx.Param)
    batches = (
        _simple_batch(tokenizer, example_weight=jnp.asarray([1.0, 2.0])),
        _simple_batch(tokenizer, example_weight=jnp.asarray([3.0, 4.0])),
    )
    initial_logits = model.logits[...]
    expected_weight = 0.0

    def objective(model: _LogitModel, batch: dict[str, jax.Array]):
        return weighted_nll(model, batch, grammar)

    loss_and_grad = nnx.value_and_grad(
        objective,
        argnums=nnx.DiffState(0, nnx.Param),
        has_aux=True,
    )
    expected_accumulated = None
    for batch in batches:
        (batch_nll, batch_weight), grads = loss_and_grad(model, batch)
        expected_weight = expected_weight + batch_weight
        expected_accumulated = accumulate_weighted_nll_grad(
            expected_accumulated,
            batch_nll,
            batch_weight,
            grads,
        )
    _, _, expected_grads = expected_accumulated
    expected_scaled_grads = jax.tree.map(
        lambda grad: grad / expected_weight,
        expected_grads,
    )

    trainer.update(model, optimizer, batches)

    expected_logits = initial_logits - 0.1 * expected_scaled_grads["logits"][...]
    assert jnp.allclose(model.logits[...], expected_logits)


def test_supervised_trainer_update_zero_weight_mean_is_finite():
    tokenizer = _tokenizer()
    grammar = FlatDefinitionGrammar(tokenizer)
    trainer = SupervisedTrainer(grammar)
    model = _LogitModel(jnp.zeros((2, 4, tokenizer.vocab_size), dtype=jnp.float32))
    optimizer = nnx.Optimizer(model, optax.sgd(0.0), wrt=nnx.Param)
    batches = (
        _simple_batch(tokenizer, example_weight=jnp.asarray([0.0, 0.0])),
        _simple_batch(tokenizer, example_weight=jnp.asarray([0.0, 0.0])),
    )

    metrics = trainer.update(model, optimizer, batches)

    assert jnp.allclose(metrics["weighted_nll_sum"], 0.0)
    assert jnp.allclose(metrics["weight_sum"], 0.0)
    assert jnp.isfinite(metrics["mean_nll"])
    assert jnp.allclose(metrics["mean_nll"], 0.0)


def test_supervised_trainer_epoch_aggregates_update_metrics_by_sums():
    tokenizer = _tokenizer()
    grammar = FlatDefinitionGrammar(tokenizer)
    trainer = SupervisedTrainer(grammar)
    model = _LogitModel(jnp.zeros((2, 4, tokenizer.vocab_size), dtype=jnp.float32))
    optimizer = nnx.Optimizer(model, optax.sgd(0.0), wrt=nnx.Param)
    update_batches = (
        (_simple_batch(tokenizer, example_weight=jnp.asarray([1.0, 2.0])),),
        (
            _simple_batch(tokenizer, example_weight=jnp.asarray([3.0, 4.0])),
            _simple_batch(tokenizer, example_weight=jnp.asarray([5.0, 6.0])),
        ),
    )
    expected_nll = 0.0
    expected_weight = 0.0
    for batches in update_batches:
        for batch in batches:
            nll, weight = weighted_nll(model, batch, grammar)
            expected_nll = expected_nll + nll
            expected_weight = expected_weight + weight

    metrics = trainer.epoch(model, optimizer, update_batches)

    assert jnp.allclose(metrics["weighted_nll_sum"], expected_nll)
    assert jnp.allclose(metrics["weight_sum"], expected_weight)
    assert jnp.allclose(metrics["mean_nll"], expected_nll / expected_weight)
    assert metrics["num_updates"] == 2
    assert metrics["num_batches"] == 3
    assert int(optimizer.step[...]) == 2


def test_supervised_trainer_builds_value_and_grad_once(monkeypatch):
    tokenizer = _tokenizer()
    grammar = FlatDefinitionGrammar(tokenizer)
    original_value_and_grad = supervised.nnx.value_and_grad
    calls = {"count": 0}

    def counting_value_and_grad(*args, **kwargs):
        calls["count"] += 1
        return original_value_and_grad(*args, **kwargs)

    monkeypatch.setattr(supervised.nnx, "value_and_grad", counting_value_and_grad)
    trainer = SupervisedTrainer(grammar)
    monkeypatch.setattr(
        supervised.nnx,
        "value_and_grad",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("value_and_grad was rebuilt after trainer creation")
        ),
    )
    model = _LogitModel(jnp.zeros((2, 4, tokenizer.vocab_size), dtype=jnp.float32))
    optimizer = nnx.Optimizer(model, optax.sgd(0.1), wrt=nnx.Param)
    batches = (
        _simple_batch(tokenizer, example_weight=jnp.asarray([1.0, 2.0])),
        _simple_batch(tokenizer, example_weight=jnp.asarray([3.0, 4.0])),
    )

    trainer.update(model, optimizer, batches)
    trainer.update(model, optimizer, batches)

    assert calls["count"] == 1
