import math

from flax import nnx
import jax
import jax.numpy as jnp
import pytest

from gristmill_symbolics.direct_optimizer.dataset import (
    BuildConfig,
    build_processed_dataset,
)
from gristmill_symbolics.direct_optimizer.model import DirectOptimizerTransformer
from gristmill_symbolics.direct_optimizer.trainer import (
    DirectOptimizerTrainer,
    collate_processed_rows,
    weighted_sequence_loss,
)
from tests.direct_optimizer.fixtures import source_comp_json


def _raw_record(*, outputs, candidate_log_flops=1.0):
    return {
        "input_computation": source_comp_json(),
        "candidate_computation": source_comp_json(),
        "outputs": outputs,
        "initial_log_flops": 4.0,
        "candidate_log_flops": candidate_log_flops,
    }


def _processed_rows(count: int):
    records = [
        _raw_record(outputs=[index], candidate_log_flops=float(index + 1))
        for index in range(count)
    ]
    return build_processed_dataset(records, BuildConfig())


def _tiny_model():
    return DirectOptimizerTransformer(
        source_len=128,
        target_len=128,
        scalar_value_min=-128,
        scalar_value_max=128,
        d_model=16,
        num_layers=1,
        num_heads=2,
        rngs=nnx.Rngs(0),
    )


def _tiny_batch():
    return collate_processed_rows(
        _processed_rows(2),
        batch_size=2,
        source_len=128,
        target_len=128,
        scalar_value_min=-128,
        scalar_value_max=128,
    )[0]


def _model_state_array_copies(model):
    copies = []
    for leaf in jax.tree_util.tree_leaves(nnx.state(model)):
        if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
            copies.append(jnp.asarray(leaf).copy())
    return copies


def _same_shape_dtype_arrays(before, after):
    for before_leaf, after_leaf in zip(before, after, strict=True):
        if (
            before_leaf.shape == after_leaf.shape
            and before_leaf.dtype == after_leaf.dtype
        ):
            yield before_leaf, after_leaf


def test_collate_processed_rows_uses_static_shapes_and_drops_remainder():
    rows = _processed_rows(3)

    batches = collate_processed_rows(
        rows,
        batch_size=2,
        source_len=128,
        target_len=128,
        scalar_value_min=-128,
        scalar_value_max=128,
    )

    assert len(batches) == 1
    batch = batches[0]
    assert set(batch) == {
        "source_tokens",
        "decoder_input_tokens",
        "target_tokens",
        "target_mask",
        "example_weight",
    }
    for field in ("kind", "keyword", "scalar_type", "scalar_value", "mask"):
        assert batch["source_tokens"][field].shape == (2, 128)
        assert batch["decoder_input_tokens"][field].shape == (2, 128)
        assert batch["target_tokens"][field].shape == (2, 128)
    assert batch["target_mask"].shape == (2, 128)
    assert batch["example_weight"].shape == (2,)
    assert batch["source_tokens"]["kind"].dtype == jnp.int32
    assert batch["source_tokens"]["mask"].dtype == bool
    assert batch["target_mask"].dtype == bool
    assert batch["example_weight"].dtype == jnp.float32
    assert jax.jit(lambda value: value["example_weight"].sum())(batch) == pytest.approx(
        sum(row["weight"] for row in rows[:2])
    )


def test_collate_processed_rows_rejects_dataset_smaller_than_batch_size():
    rows = _processed_rows(1)

    with pytest.raises(ValueError, match="fewer compatible rows than batch_size"):
        collate_processed_rows(
            rows,
            batch_size=2,
            source_len=128,
            target_len=128,
            scalar_value_min=-128,
            scalar_value_max=128,
        )


def test_collate_processed_rows_skips_negative_weight():
    rows = _processed_rows(3)
    rows[1] = {**rows[1], "weight": -0.5}

    with pytest.raises(ValueError, match="fewer compatible rows than batch_size"):
        collate_processed_rows(
            rows,
            batch_size=3,
            source_len=128,
            target_len=128,
            scalar_value_min=-128,
            scalar_value_max=128,
        )


def test_weighted_sequence_loss_normalizes_by_weight_sum():
    loss = weighted_sequence_loss(
        jnp.asarray([-1.0, -3.0]),
        jnp.asarray([0.25, 0.75]),
    )

    assert loss == pytest.approx(2.5)


def test_direct_optimizer_trainer_import_surface():
    trainer = DirectOptimizerTrainer(batch_size=4, learning_rate=1.0e-3)

    assert trainer.batch_size == 4
    assert trainer.learning_rate == pytest.approx(1.0e-3)


def test_train_step_changes_model_state_and_returns_finite_loss():
    trainer = DirectOptimizerTrainer(batch_size=2, learning_rate=1.0e-3)
    model = _tiny_model()
    optimizer = trainer.init_optimizer(model)
    batch = _tiny_batch()
    before = _model_state_array_copies(model)

    metrics = trainer.train_step(model, optimizer, batch)

    assert math.isfinite(float(metrics["train_loss"]))
    after = _model_state_array_copies(model)
    assert any(
        not bool(jnp.array_equal(before_leaf, after_leaf))
        for before_leaf, after_leaf in _same_shape_dtype_arrays(before, after)
    )


def test_eval_step_returns_finite_loss_without_mutating_model():
    trainer = DirectOptimizerTrainer(batch_size=2, learning_rate=1.0e-3)
    model = _tiny_model()
    batch = _tiny_batch()
    before = _model_state_array_copies(model)

    metrics = trainer.eval_step(model, batch, metric_name="valid_loss")

    assert math.isfinite(float(metrics["valid_loss"]))
    after = _model_state_array_copies(model)
    assert all(
        bool(jnp.array_equal(before_leaf, after_leaf))
        for before_leaf, after_leaf in _same_shape_dtype_arrays(before, after)
    )


def test_direct_optimizer_trainer_validates_constructor_and_round_trips_kwargs():
    trainer = DirectOptimizerTrainer(
        batch_size=4,
        learning_rate=1.0e-3,
        b1=0.8,
        b2=0.95,
        eps=1.0e-7,
    )

    kwargs = trainer.constructor_kwargs()
    assert kwargs == {
        "batch_size": 4,
        "learning_rate": pytest.approx(1.0e-3),
        "b1": pytest.approx(0.8),
        "b2": pytest.approx(0.95),
        "eps": pytest.approx(1.0e-7),
    }
    round_tripped = DirectOptimizerTrainer(**kwargs)
    assert round_tripped.constructor_kwargs() == kwargs

    invalid_kwargs = [
        {"batch_size": 0},
        {"batch_size": True},
        {"batch_size": 1.5},
        {"learning_rate": 0.0},
        {"learning_rate": math.inf},
        {"b1": -0.1},
        {"b1": 1.0},
        {"b2": math.nan},
        {"eps": 0.0},
    ]
    for overrides in invalid_kwargs:
        kwargs = {"batch_size": 4, **overrides}
        with pytest.raises(ValueError):
            DirectOptimizerTrainer(**kwargs)
