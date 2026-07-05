import jax
import jax.numpy as jnp
import pytest

from gristmill_symbolics.direct_optimizer.dataset import (
    BuildConfig,
    build_processed_dataset,
)
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
