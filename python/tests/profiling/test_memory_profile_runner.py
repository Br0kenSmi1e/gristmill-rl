from __future__ import annotations

from pathlib import Path

from gristmill_symbolics.profiling.memory_profile_runner import build_train_command


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
