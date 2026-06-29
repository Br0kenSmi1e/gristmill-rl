from __future__ import annotations

import json
from pathlib import Path

from gristmill_symbolics.profiling.attention_backend_profile import (
    AttentionProfileConfig,
    _profile_env,
    run_attention_profile,
)
from gristmill_symbolics.profiling.profile_summary import summarize_run


def _config(tmp_path: Path, **overrides) -> AttentionProfileConfig:
    values = {
        "backend": "explicit",
        "run_root": tmp_path,
        "run_name": "attention-explicit-small",
        "batch_size": 1,
        "seq_len": 8,
        "d_model": 4,
        "seed": 42,
        "dtype": "float32",
        "sample_ms": 250,
        "xla_dump": False,
        "xla_dump_all_passes": False,
        "repo_root": tmp_path,
    }
    values.update(overrides)
    return AttentionProfileConfig(**values)


def test_attention_profile_env_can_request_all_xla_pass_dumps(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setenv("XLA_FLAGS", "--existing_flag=true")
    config = _config(
        tmp_path,
        backend="cudnn",
        xla_dump=True,
        xla_dump_all_passes=True,
    )

    env = _profile_env(config, tmp_path / "run")

    assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert env["JAX_LOG_COMPILES"] == "1"
    assert env["XLA_FLAGS"] == (
        "--existing_flag=true "
        f"--xla_dump_to={tmp_path / 'run' / 'xla'} "
        "--xla_dump_hlo_as_text "
        "--xla_dump_hlo_pass_re=.*"
    )


def test_explicit_attention_profile_writes_summarizable_run(tmp_path: Path):
    run_dir = run_attention_profile(_config(tmp_path))

    assert run_dir == tmp_path / "attention-explicit-small"
    assert (run_dir / "status.txt").read_text(encoding="utf-8") == "0\n"
    metrics = json.loads((run_dir / "stdout.jsonl").read_text(encoding="utf-8"))
    assert metrics["backend"] == "explicit"
    assert metrics["batch_size"] == 1
    assert metrics["seq_len"] == 8
    assert metrics["d_model"] == 4
    assert metrics["dtype"] == "float32"
    assert metrics["loss"] > 0.0
    assert metrics["grad_abs_sum"] > 0.0

    summary = summarize_run(run_dir)
    assert summary.status == 0
    assert summary.rollout_phase_counts == {"attention_value_and_grad": 1}
    assert summary.max_state_token_len == 8
