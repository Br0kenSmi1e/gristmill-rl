import json
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx
import orbax.checkpoint as ocp

import gristmill_symbolics.sample_tensor_computations as sample_cli
from gristmill_symbolics import TensorComputation
from gristmill_symbolics.nn import FlatDefinitionSeq2SeqTransformer


def _source_json() -> dict[str, object]:
    return {
        "ranges": [{"id": 0, "size": 3}],
        "tensors": [{"id": 0, "symmetry": []}, {"id": 1, "symmetry": []}],
        "definitions": [
            {
                "base": 0,
                "ext_indices": [{"id": 0, "range": 0}],
                "terms": [
                    {
                        "coeff": [1, 1],
                        "sum_indices": [],
                        "factors": [{"tensor": 0, "indices": [0]}],
                    }
                ],
            }
        ],
    }


def _candidate() -> TensorComputation:
    return TensorComputation.from_json_string(
        json.dumps(
            {
                "ranges": [{"id": 0, "size": 3}],
                "tensors": [{"id": 0, "symmetry": []}, {"id": 1, "symmetry": []}],
                "definitions": [
                    {
                        "base": 1,
                        "ext_indices": [{"id": 0, "range": 0}],
                        "terms": [
                            {
                                "coeff": [1, 1],
                                "sum_indices": [],
                                "factors": [{"tensor": 0, "indices": [0]}],
                            }
                        ],
                    }
                ],
            }
        )
    )


def _save_checkpoint(path: Path, *, vocab_size: int) -> None:
    model = FlatDefinitionSeq2SeqTransformer(
        source_len=12,
        target_len=12,
        vocab_size=vocab_size,
        pad_token_id=0,
        d_model=4,
        num_layers=1,
        num_heads=1,
        mlp_hidden_dim=8,
        dropout=0.0,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=nnx.Rngs(0),
    )
    optimizer = nnx.Optimizer(
        model,
        optax.adamw(1e-3),
        wrt=nnx.Param,
    )
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(
        path,
        {"model": nnx.state(model), "optimizer": nnx.state(optimizer)},
        force=True,
    )
    if hasattr(checkpointer, "wait_until_finished"):
        checkpointer.wait_until_finished()


def _argv(tmp_path: Path, checkpoint_path: Path, output_path: Path) -> list[str]:
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(_source_json()))
    return [
        "--checkpoint",
        str(checkpoint_path),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--samples",
        "3",
        "--sample-batch-size",
        "2",
        "--source-len",
        "12",
        "--target-len",
        "12",
        "--max-range-id",
        "0",
        "--max-tensor-id",
        "1",
        "--max-index-id",
        "0",
        "--coeff-nums",
        "1",
        "--coeff-dens",
        "1",
        "--d-model",
        "4",
        "--num-layers",
        "1",
        "--num-heads",
        "1",
        "--mlp-hidden-dim",
        "8",
        "--dropout",
        "0.0",
        "--dtype",
        "float32",
        "--attention-implementation",
        "default",
        "--temperature",
        "0.5",
        "--verify-outputs",
        "1",
    ]


def test_main_restores_checkpoint_and_writes_candidate_jsonl(
    tmp_path,
    monkeypatch,
    capsys,
):
    checkpoint_path = tmp_path / "checkpoint"
    output_path = tmp_path / "candidates.jsonl"
    _save_checkpoint(checkpoint_path, vocab_size=11)
    calls = []

    def fake_sample(
        model,
        rng,
        input_computation,
        source_ids,
        tokenizer,
        grammar,
        *,
        target_len,
        outputs,
        temperature,
    ):
        del model, input_computation, tokenizer, grammar
        calls.append(
            {
                "rng": rng,
                "source_shape": tuple(source_ids.shape),
                "target_len": target_len,
                "outputs": outputs,
                "temperature": temperature,
            }
        )
        return [_candidate()], {
            "total_samples": source_ids.shape[0],
            "decode_failures": 0,
            "reconstruction_failures": 0,
            "verifier_failures": 0,
            "valid_samples": 1,
        }

    monkeypatch.setattr(sample_cli, "sample_tensor_computations", fake_sample)

    result = sample_cli.main(_argv(tmp_path, checkpoint_path, output_path))

    assert result == 0
    assert [call["source_shape"] for call in calls] == [(2, 12), (1, 12)]
    assert all(call["target_len"] == 12 for call in calls)
    assert all(call["outputs"] == [1] for call in calls)
    assert all(call["temperature"] == 0.5 for call in calls)
    records = [
        TensorComputation.from_json_string(line).snapshot()
        for line in output_path.read_text().splitlines()
    ]
    assert len(records) == 2
    assert [record["definitions"][0]["base"] for record in records] == [1, 1]
    assert json.loads(capsys.readouterr().out) == {
        "decode_failures": 0,
        "reconstruction_failures": 0,
        "total_samples": 3,
        "valid_samples": 2,
        "verifier_failures": 0,
    }


def test_main_reports_checkpoint_shape_mismatch(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "checkpoint"
    output_path = tmp_path / "candidates.jsonl"
    _save_checkpoint(checkpoint_path, vocab_size=11)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("sampling should not run after restore mismatch")

    monkeypatch.setattr(sample_cli, "sample_tensor_computations", fail_if_called)
    argv = _argv(tmp_path, checkpoint_path, output_path)
    argv[argv.index("--max-tensor-id") + 1] = "2"

    with pytest.raises(ValueError, match="checkpoint model state does not match"):
        sample_cli.main(argv)
