import json

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from gristmill_symbolics.direct_optimizer.checkpoint import save_checkpoint
from gristmill_symbolics.direct_optimizer.converter import computation_to_target_text
from gristmill_symbolics.direct_optimizer.model import DirectOptimizerTransformer
from gristmill_symbolics.direct_optimizer.sample import (
    main as sample_main,
    optimize_from_checkpoint,
    optimize_with_model,
)
from gristmill_symbolics.direct_optimizer.tokens import encode_text, pad_tokens
from gristmill_symbolics.direct_optimizer.trainer import DirectOptimizerTrainer
from tests.direct_optimizer.fixtures import source_comp


class FakeModel:
    target_len = 64

    def __init__(self, rows):
        self.rows = list(rows)
        self.calls = 0


def _token_row(text: str, length: int = 64):
    return pad_tokens(encode_text(text), length=length)


def _stack(rows):
    return {field: jnp.stack([row[field] for row in rows]) for field in rows[0]}


def fake_sample_tokens(
    model,
    rng,
    source_tokens,
    *,
    max_length,
    temperature,
    mask_provider=None,
):
    del rng, max_length, temperature, mask_provider
    batch_size = source_tokens["kind"].shape[0]
    start = model.calls
    model.calls += batch_size
    rows = model.rows[start : start + batch_size]
    invalid_filler = _token_row("def base tensor_id:99\nenddef")
    rows = rows + [invalid_filler] * (batch_size - len(rows))
    tokens = _stack(rows)
    return tokens, tokens["mask"]


def test_sampler_rejects_invalid_and_non_equivalent_candidates(monkeypatch):
    valid_text = computation_to_target_text(source_comp())
    non_equivalent_text = valid_text.replace("coeff_num:1", "coeff_num:2", 1)
    invalid_text = "def base tensor_id:1\nterm\nenddef"
    model = FakeModel(
        [
            _token_row(invalid_text),
            _token_row(non_equivalent_text),
            _token_row(valid_text),
        ]
    )
    monkeypatch.setattr(
        "gristmill_symbolics.direct_optimizer.sample.sample_tokens",
        fake_sample_tokens,
    )

    candidate, metrics = optimize_with_model(
        model,
        None,
        source_comp(),
        [1],
        num_samples=3,
        sample_batch_size=2,
        source_len=128,
        target_len=64,
        temperature=1.0,
        seed=0,
    )

    assert candidate is not None
    assert metrics["total_samples"] == 3
    assert metrics["parse_failures"] == 1
    assert metrics["verifier_failures"] == 1
    assert metrics["valid_samples"] == 1
    assert metrics["best_log_flops"] is not None


def test_sampler_ignores_padded_extra_rows(monkeypatch):
    valid_text = computation_to_target_text(source_comp())
    invalid_filler = "def base tensor_id:1\nterm\nenddef"
    model = FakeModel(
        [
            _token_row(valid_text),
            _token_row(valid_text),
            _token_row(valid_text),
            _token_row(invalid_filler),
        ]
    )
    monkeypatch.setattr(
        "gristmill_symbolics.direct_optimizer.sample.sample_tokens",
        fake_sample_tokens,
    )

    candidate, metrics = optimize_with_model(
        model,
        None,
        source_comp(),
        [1],
        num_samples=3,
        sample_batch_size=2,
        source_len=128,
        target_len=64,
        temperature=1.0,
        seed=0,
    )

    assert candidate is not None
    assert metrics["total_samples"] == 3
    assert metrics["valid_samples"] == 3
    assert metrics["parse_failures"] == 0


def test_sampler_rejects_non_none_params(monkeypatch):
    valid_text = computation_to_target_text(source_comp())
    model = FakeModel([_token_row(valid_text)])
    monkeypatch.setattr(
        "gristmill_symbolics.direct_optimizer.sample.sample_tokens",
        fake_sample_tokens,
    )

    with pytest.raises(ValueError, match="params"):
        optimize_with_model(
            model,
            {"state": object()},
            source_comp(),
            [1],
            num_samples=1,
            sample_batch_size=1,
            source_len=128,
            target_len=64,
            temperature=1.0,
            seed=0,
        )


def test_sampler_counts_empty_decoded_output_as_parse_failure(monkeypatch):
    model = FakeModel([_token_row("")])
    monkeypatch.setattr(
        "gristmill_symbolics.direct_optimizer.sample.sample_tokens",
        fake_sample_tokens,
    )

    candidate, metrics = optimize_with_model(
        model,
        None,
        source_comp(),
        [1],
        num_samples=1,
        sample_batch_size=1,
        source_len=128,
        target_len=64,
        temperature=1.0,
        seed=0,
    )

    assert candidate is None
    assert metrics["parse_failures"] == 1
    assert metrics["verifier_failures"] == 0
    assert metrics["valid_samples"] == 0


@pytest.mark.parametrize(
    "outputs",
    [
        [],
        [1, 1],
        [True],
        [-1],
    ],
)
def test_sampler_rejects_invalid_outputs(outputs):
    with pytest.raises(ValueError, match="outputs"):
        optimize_with_model(
            FakeModel([]),
            None,
            source_comp(),
            outputs,
            num_samples=1,
            sample_batch_size=1,
            source_len=128,
            target_len=64,
            temperature=1.0,
            seed=0,
        )


def test_sampler_returns_none_when_no_valid_candidates(monkeypatch):
    invalid_text = "def base tensor_id:1\nterm\nenddef"
    model = FakeModel([_token_row(invalid_text), _token_row(invalid_text)])
    monkeypatch.setattr(
        "gristmill_symbolics.direct_optimizer.sample.sample_tokens",
        fake_sample_tokens,
    )

    candidate, metrics = optimize_with_model(
        model,
        None,
        source_comp(),
        [1],
        num_samples=2,
        sample_batch_size=2,
        source_len=128,
        target_len=64,
        temperature=1.0,
        seed=0,
    )

    assert candidate is None
    assert metrics == {
        "total_samples": 2,
        "decode_failures": 0,
        "parse_failures": 2,
        "reconstruction_failures": 0,
        "verifier_failures": 0,
        "valid_samples": 0,
        "best_log_flops": None,
    }


def _checkpoint_model(seed: int = 4) -> DirectOptimizerTransformer:
    return DirectOptimizerTransformer(
        source_len=128,
        target_len=64,
        scalar_value_min=-128,
        scalar_value_max=128,
        d_model=8,
        num_layers=1,
        num_heads=1,
        rngs=nnx.Rngs(seed),
    )


def _save_checkpoint(path, model: DirectOptimizerTransformer) -> None:
    trainer = DirectOptimizerTrainer(batch_size=2)
    optimizer = trainer.init_optimizer(model)
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        trainer=trainer,
        epoch=0,
        updates=0,
        last_train_loss=1.0,
    )


def test_optimize_from_checkpoint_rejects_static_shape_mismatch(tmp_path):
    model = _checkpoint_model()
    _save_checkpoint(tmp_path, model)

    with pytest.raises(ValueError, match="source_len"):
        optimize_from_checkpoint(
            tmp_path,
            source_comp(),
            [1],
            num_samples=1,
            sample_batch_size=1,
            source_len=64,
            target_len=64,
            temperature=1.0,
            seed=0,
        )


def test_sample_cli_writes_output_for_valid_monkeypatched_candidate(
    tmp_path,
    monkeypatch,
    capsys,
):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "optimized.json"
    ckpt_path = tmp_path / "ckpt"
    source_comp().write_json(input_path)
    model = _checkpoint_model()
    _save_checkpoint(ckpt_path, model)

    valid = _token_row(computation_to_target_text(source_comp()))
    fake = FakeModel([valid])
    monkeypatch.setattr(
        "gristmill_symbolics.direct_optimizer.sample.load_model_for_inference",
        lambda path: (fake, {"model_kwargs": model.model_kwargs()}),
    )
    monkeypatch.setattr(
        "gristmill_symbolics.direct_optimizer.sample.sample_tokens",
        fake_sample_tokens,
    )

    exit_code = sample_main(
        [
            "--checkpoint",
            str(ckpt_path),
            "--input",
            str(input_path),
            "--outputs",
            "1",
            "--samples",
            "1",
            "--sample-batch-size",
            "1",
            "--temperature",
            "1.0",
            "--output",
            str(output_path),
        ]
    )

    metrics = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert metrics["valid_samples"] == 1
    assert output_path.exists()
