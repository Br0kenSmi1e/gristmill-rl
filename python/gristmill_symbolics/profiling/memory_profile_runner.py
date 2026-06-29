from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class MemoryProfileConfig:
    input_path: Path
    run_root: Path
    run_name: str | None
    updates: int
    batch_size: int
    max_steps: int
    seed: int
    state_token_pad_to: int
    action_token_pad_to: int
    definition_pad_to: int
    sample_ms: int
    cprofile: bool
    xla_dump: bool
    xla_dump_all_passes: bool
    rollout_sync: bool
    python_executable: str
    repo_root: Path


def build_train_command(
    *,
    python_executable: str,
    input_path: Path,
    profile_path: Path,
    updates: int,
    batch_size: int,
    max_steps: int,
    seed: int,
    state_token_pad_to: int,
    action_token_pad_to: int,
    definition_pad_to: int,
    cprofile: bool,
) -> list[str]:
    command = [
        "/usr/bin/time",
        "-v",
        python_executable,
    ]
    if cprofile:
        command.extend(["-m", "cProfile", "-o", str(profile_path)])
    command.extend(
        [
            "-m",
            "gristmill_symbolics.cli.train",
            "--input",
            str(input_path),
            "--updates",
            str(updates),
            "--batch-size",
            str(batch_size),
            "--max-steps",
            str(max_steps),
            "--seed",
            str(seed),
            "--state-token-pad-to",
            str(state_token_pad_to),
            "--action-token-pad-to",
            str(action_token_pad_to),
            "--definition-pad-to",
            str(definition_pad_to),
        ]
    )
    return command


def run_memory_profile(config: MemoryProfileConfig) -> Path:
    run_dir = _make_run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=False)

    _write_git_artifacts(config.repo_root, run_dir)
    _write_jax_env(config.python_executable, run_dir)
    _run_nvidia_smi_snapshot(run_dir / "nvidia-before.txt")

    sampler = _start_nvidia_smi_sampler(run_dir / "nvidia-smi.csv", config.sample_ms)
    status = 127
    try:
        env = _profile_env(config, run_dir)
        command = build_train_command(
            python_executable=config.python_executable,
            input_path=config.input_path,
            profile_path=run_dir / "profile.prof",
            updates=config.updates,
            batch_size=config.batch_size,
            max_steps=config.max_steps,
            seed=config.seed,
            state_token_pad_to=config.state_token_pad_to,
            action_token_pad_to=config.action_token_pad_to,
            definition_pad_to=config.definition_pad_to,
            cprofile=config.cprofile,
        )
        (run_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
        with (run_dir / "stdout.jsonl").open("w", encoding="utf-8") as stdout:
            with (run_dir / "stderr.log").open("w", encoding="utf-8") as stderr:
                completed = subprocess.run(
                    command,
                    cwd=config.repo_root / "python",
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                )
        status = int(completed.returncode)
    finally:
        _stop_sampler(sampler)
        (run_dir / "status.txt").write_text(f"{status}\n", encoding="utf-8")
        _run_nvidia_smi_snapshot(run_dir / "nvidia-after.txt")

    return run_dir


def _make_run_dir(config: MemoryProfileConfig) -> Path:
    if config.run_name:
        name = config.run_name
    else:
        now = datetime.now(ZoneInfo("UTC")).strftime("%Y%m%d-%H%M%S")
        suffix = f"bs{config.batch_size}"
        if config.xla_dump:
            suffix += "-xla"
        if config.xla_dump_all_passes:
            suffix += "-all-passes"
        name = f"{now}-{suffix}"
    return config.run_root / name


def _write_git_artifacts(repo_root: Path, run_dir: Path) -> None:
    _capture_command(["git", "-C", str(repo_root), "rev-parse", "HEAD"], run_dir / "git-sha.txt")
    _capture_command(["git", "-C", str(repo_root), "status", "--short"], run_dir / "git-status.txt")


def _write_jax_env(python_executable: str, run_dir: Path) -> None:
    code = (
        "import jax\n"
        "print('jax', jax.__version__)\n"
        "print('devices', jax.devices())\n"
    )
    _capture_command([python_executable, "-c", code], run_dir / "jax-env.txt")


def _capture_command(command: list[str], output_path: Path) -> None:
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    output_path.write_text(completed.stdout, encoding="utf-8")


def _run_nvidia_smi_snapshot(output_path: Path) -> None:
    if shutil.which("nvidia-smi") is None:
        output_path.write_text("nvidia-smi not found\n", encoding="utf-8")
        return
    _capture_command(["nvidia-smi", "-q", "-d", "MEMORY"], output_path)


def _start_nvidia_smi_sampler(output_path: Path, sample_ms: int) -> subprocess.Popen[str] | None:
    if shutil.which("nvidia-smi") is None:
        output_path.write_text("nvidia-smi not found\n", encoding="utf-8")
        return None
    handle = output_path.open("w", encoding="utf-8")
    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,power.draw",
        "--format=csv",
        "-lms",
        str(sample_ms),
    ]
    process = subprocess.Popen(
        command,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    process._gristmill_output_handle = handle  # type: ignore[attr-defined]
    return process


def _stop_sampler(process: subprocess.Popen[str] | None) -> None:
    if process is None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    handle = getattr(process, "_gristmill_output_handle", None)
    if handle is not None:
        handle.close()


def _profile_env(config: MemoryProfileConfig, run_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["JAX_LOG_COMPILES"] = "1"
    env["GRISTMILL_PROFILE_ROLLOUT"] = "1"
    env["GRISTMILL_PROFILE_ROLLOUT_SYNC"] = "1" if config.rollout_sync else "0"
    if config.xla_dump:
        xla_dir = run_dir / "xla"
        xla_dir.mkdir(parents=True, exist_ok=True)
        flags = [f"--xla_dump_to={xla_dir}", "--xla_dump_hlo_as_text"]
        if config.xla_dump_all_passes:
            flags.append("--xla_dump_hlo_pass_re=.*")
        dump_flags = " ".join(flags)
        existing = env.get("XLA_FLAGS")
        env["XLA_FLAGS"] = f"{existing} {dump_flags}" if existing else dump_flags
    return env


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _default_repo_root() -> Path:
    cwd = Path.cwd()
    if cwd.name == "python":
        return cwd.parent
    return cwd


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture a CCSD training memory profile run."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("/tmp/gristmill-mem-baseline"),
    )
    parser.add_argument("--run-name")
    parser.add_argument("--updates", type=_positive_int, default=1)
    parser.add_argument("--batch-size", type=_positive_int, default=2)
    parser.add_argument("--max-steps", type=_positive_int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--state-token-pad-to", type=_positive_int, default=3072)
    parser.add_argument("--action-token-pad-to", type=_positive_int, default=4096)
    parser.add_argument("--definition-pad-to", type=_positive_int, default=128)
    parser.add_argument("--sample-ms", type=_positive_int, default=250)
    parser.add_argument("--no-cprofile", action="store_true")
    parser.add_argument("--xla-dump", action="store_true")
    parser.add_argument(
        "--xla-dump-all-passes",
        action="store_true",
        help=(
            "Dump text for every XLA HLO pass. This implies --xla-dump and can "
            "write a large xla/ directory."
        ),
    )
    parser.add_argument("--no-rollout-sync", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=_default_repo_root())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = MemoryProfileConfig(
        input_path=args.input,
        run_root=args.run_root,
        run_name=args.run_name,
        updates=args.updates,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        seed=args.seed,
        state_token_pad_to=args.state_token_pad_to,
        action_token_pad_to=args.action_token_pad_to,
        definition_pad_to=args.definition_pad_to,
        sample_ms=args.sample_ms,
        cprofile=not args.no_cprofile,
        xla_dump=args.xla_dump or args.xla_dump_all_passes,
        xla_dump_all_passes=args.xla_dump_all_passes,
        rollout_sync=not args.no_rollout_sync,
        python_executable=sys.executable,
        repo_root=args.repo_root.resolve(),
    )
    run_dir = run_memory_profile(config)
    status = int((run_dir / "status.txt").read_text(encoding="utf-8").strip())
    print(run_dir)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
