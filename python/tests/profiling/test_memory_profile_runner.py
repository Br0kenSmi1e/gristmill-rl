from __future__ import annotations

from pathlib import Path

from gristmill_symbolics.profiling.memory_profile_runner import (
    MemoryProfileConfig,
    _profile_env,
    build_train_command,
)


def test_build_train_command_uses_current_python_and_records_cprofile(tmp_path: Path):
    command = build_train_command(
        python_executable="/venv/bin/python",
        input_path=Path("../tmp/ccsd/working_eqn.json"),
        profile_path=tmp_path / "profile.prof",
        updates=1,
        batch_size=2,
        max_steps=64,
        seed=42,
        state_token_pad_to=3072,
        action_token_pad_to=4096,
        definition_pad_to=128,
        cprofile=True,
    )

    assert command == [
        "/usr/bin/time",
        "-v",
        "/venv/bin/python",
        "-m",
        "cProfile",
        "-o",
        str(tmp_path / "profile.prof"),
        "-m",
        "gristmill_symbolics.cli.train",
        "--input",
        "../tmp/ccsd/working_eqn.json",
        "--updates",
        "1",
        "--batch-size",
        "2",
        "--max-steps",
        "64",
        "--seed",
        "42",
        "--state-token-pad-to",
        "3072",
        "--action-token-pad-to",
        "4096",
        "--definition-pad-to",
        "128",
    ]


def test_build_train_command_can_skip_cprofile_for_oom_boundary(tmp_path: Path):
    command = build_train_command(
        python_executable="/venv/bin/python",
        input_path=Path("../tmp/ccsd/working_eqn.json"),
        profile_path=tmp_path / "profile.prof",
        updates=1,
        batch_size=8,
        max_steps=64,
        seed=42,
        state_token_pad_to=3072,
        action_token_pad_to=4096,
        definition_pad_to=128,
        cprofile=False,
    )

    assert "-m" in command
    assert "cProfile" not in command
    assert "gristmill_symbolics.cli.train" in command
    assert command[0:4] == ["/usr/bin/time", "-v", "/venv/bin/python", "-m"]


def test_profile_env_can_request_all_xla_pass_dumps(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("XLA_FLAGS", "--existing_flag=true")
    config = MemoryProfileConfig(
        input_path=Path("../tmp/ccsd/working_eqn.json"),
        run_root=tmp_path,
        run_name="run",
        updates=1,
        batch_size=8,
        max_steps=64,
        seed=42,
        state_token_pad_to=3072,
        action_token_pad_to=4096,
        definition_pad_to=128,
        sample_ms=250,
        cprofile=False,
        xla_dump=True,
        xla_dump_all_passes=True,
        rollout_sync=True,
        python_executable="/venv/bin/python",
        repo_root=tmp_path,
    )

    env = _profile_env(config, tmp_path / "run")

    assert env["XLA_FLAGS"] == (
        "--existing_flag=true "
        f"--xla_dump_to={tmp_path / 'run' / 'xla'} "
        "--xla_dump_hlo_as_text "
        "--xla_dump_hlo_pass_re=.*"
    )
