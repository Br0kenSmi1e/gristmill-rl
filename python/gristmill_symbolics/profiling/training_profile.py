from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo


_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_TIME_RE = re.compile(
    r"Elapsed \(wall clock\) time(?: \([^)]*\))?:\s*(?P<value>\S+)"
)
_RSS_RE = re.compile(r"Maximum resident set size.*:\s*(?P<value>\d+)")


@dataclass(frozen=True)
class TrainingProfileCase:
    input_path: Path
    run_root: Path
    run_name: str
    updates: int
    batch_size: int
    max_steps: int
    seed: int
    state_token_pad_to: int
    action_token_pad_to: int
    definition_pad_to: int
    candidate_pad_to: int
    side_term_pad_to: int
    d_model: int
    num_attention_layers: int
    num_attention_heads: int
    sample_ms: int
    cprofile: bool
    xla_dump: bool
    require_gpu: bool
    python_executable: str
    repo_root: Path


@dataclass(frozen=True)
class TrainingProfileSummary:
    run_name: str
    updates: int
    batch_size: int
    max_steps: int
    status: int | None
    elapsed_seconds: float | None
    max_rss_kbytes: int | None
    peak_gpu_memory_mib: float | None
    avg_gpu_util_percent: float | None
    peak_gpu_util_percent: float | None
    compile_count: int
    cudnn_hlo_count: int
    final_metrics: str | None


def build_train_command(case: TrainingProfileCase) -> list[str]:
    command = [
        "/usr/bin/time",
        "-v",
        case.python_executable,
    ]
    if case.cprofile:
        profile_path = case.run_root / case.run_name / "profile.prof"
        command.extend(["-m", "cProfile", "-o", str(profile_path)])
    command.extend(
        [
            "-m",
            "gristmill_symbolics.cli.train",
            "--input",
            str(case.input_path),
            "--updates",
            str(case.updates),
            "--batch-size",
            str(case.batch_size),
            "--max-steps",
            str(case.max_steps),
            "--seed",
            str(case.seed),
            "--state-token-pad-to",
            str(case.state_token_pad_to),
            "--action-token-pad-to",
            str(case.action_token_pad_to),
            "--definition-pad-to",
            str(case.definition_pad_to),
            "--candidate-pad-to",
            str(case.candidate_pad_to),
            "--side-term-pad-to",
            str(case.side_term_pad_to),
            "--d-model",
            str(case.d_model),
            "--num-attention-layers",
            str(case.num_attention_layers),
            "--num-attention-heads",
            str(case.num_attention_heads),
        ]
    )
    return command


def run_matrix(cases: list[TrainingProfileCase]) -> list[TrainingProfileSummary]:
    summaries = [run_case(case) for case in cases]
    if cases:
        write_matrix_csv(cases[0].run_root / "matrix.csv", summaries)
    return summaries


def run_case(case: TrainingProfileCase) -> TrainingProfileSummary:
    run_dir = case.run_root / case.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_text(run_dir / "case.json", json.dumps(_case_json(case), indent=2))
    _write_git_artifacts(case.repo_root, run_dir)
    _write_jax_env(case.python_executable, run_dir)
    if case.require_gpu and not _jax_has_gpu(case.python_executable):
        raise RuntimeError("JAX did not report a GPU backend")

    command = build_train_command(case)
    _write_text(run_dir / "command.txt", shlex.join(command) + "\n")
    sampler = _start_nvidia_smi_sampler(
        run_dir / "nvidia-smi.csv",
        case.sample_ms,
    )
    status = 127
    try:
        with (
            (run_dir / "stdout.jsonl").open("w", encoding="utf-8") as stdout,
            (run_dir / "stderr.log").open("w", encoding="utf-8") as stderr,
        ):
            completed = subprocess.run(
                command,
                cwd=case.repo_root / "python",
                env=_profile_env(case, run_dir),
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        status = int(completed.returncode)
    finally:
        _stop_sampler(sampler)
        _write_text(run_dir / "status.txt", f"{status}\n")

    summary = summarize_case(case, run_dir)
    _write_text(
        run_dir / "summary.json",
        json.dumps(asdict(summary), indent=2, sort_keys=True),
    )
    return summary


def summarize_case(
    case: TrainingProfileCase,
    run_dir: Path,
) -> TrainingProfileSummary:
    stderr = _read_text(run_dir / "stderr.log")
    stdout = _read_text(run_dir / "stdout.jsonl")
    nvidia = _read_nvidia_summary(run_dir / "nvidia-smi.csv")
    return TrainingProfileSummary(
        run_name=case.run_name,
        updates=case.updates,
        batch_size=case.batch_size,
        max_steps=case.max_steps,
        status=_read_status(run_dir / "status.txt"),
        elapsed_seconds=parse_elapsed_seconds(stderr),
        max_rss_kbytes=parse_max_rss_kbytes(stderr),
        peak_gpu_memory_mib=nvidia["peak_memory_mib"],
        avg_gpu_util_percent=nvidia["avg_gpu_util_percent"],
        peak_gpu_util_percent=nvidia["peak_gpu_util_percent"],
        compile_count=sum(
            1 for line in stderr.splitlines() if "Compiling " in line
        ),
        cudnn_hlo_count=_count_xla_mentions(run_dir / "xla", "cudnn"),
        final_metrics=_last_nonempty_line(stdout),
    )


def write_matrix_csv(
    path: Path,
    summaries: list[TrainingProfileSummary],
) -> None:
    if not summaries:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(asdict(summaries[0]).keys()),
        )
        writer.writeheader()
        for summary in summaries:
            writer.writerow(asdict(summary))


def parse_elapsed_seconds(stderr: str) -> float | None:
    for line in stderr.splitlines():
        match = _TIME_RE.search(line)
        if match:
            return _parse_time_value(match.group("value"))
    return None


def parse_max_rss_kbytes(stderr: str) -> int | None:
    for line in stderr.splitlines():
        match = _RSS_RE.search(line)
        if match:
            return int(match.group("value"))
    return None


def _parse_time_value(value: str) -> float:
    parts = value.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    return float(value)


def _read_nvidia_summary(path: Path) -> dict[str, float | None]:
    try:
        rows = _read_nvidia_rows(path)
    except FileNotFoundError:
        return _empty_nvidia_summary()
    memory_values: list[float] = []
    gpu_values: list[float] = []
    for row in rows:
        for key, value in row.items():
            if key.startswith("memory.used"):
                _append_number(memory_values, value)
            elif key.startswith("utilization.gpu"):
                _append_number(gpu_values, value)
    return {
        "peak_memory_mib": max(memory_values) if memory_values else None,
        "avg_gpu_util_percent": _average(gpu_values),
        "peak_gpu_util_percent": max(gpu_values) if gpu_values else None,
    }


def _read_nvidia_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            {key.strip().lower(): value.strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _empty_nvidia_summary() -> dict[str, float | None]:
    return {
        "peak_memory_mib": None,
        "avg_gpu_util_percent": None,
        "peak_gpu_util_percent": None,
    }


def _start_nvidia_smi_sampler(
    output_path: Path,
    sample_ms: int,
) -> subprocess.Popen[str] | None:
    if shutil.which("nvidia-smi") is None:
        _write_text(output_path, "nvidia-smi not found\n")
        return None
    handle = output_path.open("w", encoding="utf-8")
    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,name,utilization.gpu,"
        "utilization.memory,memory.used,memory.total,memory.free,power.draw",
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


def _profile_env(case: TrainingProfileCase, run_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("JAX_LOG_COMPILES", "1")
    if case.xla_dump:
        xla_dir = run_dir / "xla"
        xla_dir.mkdir(parents=True, exist_ok=True)
        flags = f"--xla_dump_to={xla_dir} --xla_dump_hlo_as_text"
        existing = env.get("XLA_FLAGS")
        env["XLA_FLAGS"] = f"{existing} {flags}" if existing else flags
    return env


def _write_git_artifacts(repo_root: Path, run_dir: Path) -> None:
    _write_text(
        run_dir / "git-sha.txt",
        _run_text(["git", "-C", str(repo_root), "rev-parse", "HEAD"]),
    )
    _write_text(
        run_dir / "git-status.txt",
        _run_text(["git", "-C", str(repo_root), "status", "--short"]),
    )


def _run_text(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return completed.stdout


def _write_jax_env(python_executable: str, run_dir: Path) -> None:
    code = (
        "import json, jax\n"
        "print(json.dumps({"
        "'jax_version': jax.__version__, "
        "'default_backend': jax.default_backend(), "
        "'devices': [str(device) for device in jax.devices()]"
        "}, indent=2))\n"
    )
    completed = subprocess.run(
        [python_executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    _write_text(run_dir / "jax-env.json", completed.stdout)


def _jax_has_gpu(python_executable: str) -> bool:
    code = "import jax; print(jax.default_backend())"
    completed = subprocess.run(
        [python_executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return completed.stdout.strip() == "gpu"


def _case_json(case: TrainingProfileCase) -> dict[str, object]:
    payload = asdict(case)
    for key in ("input_path", "run_root", "repo_root"):
        payload[key] = str(payload[key])
    return payload


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _read_status(path: Path) -> int | None:
    text = _read_text(path).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _last_nonempty_line(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else None


def _count_xla_mentions(xla_dir: Path, needle: str) -> int:
    if not xla_dir.is_dir():
        return 0
    count = 0
    lowered = needle.lower()
    for path in xla_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        count += text.lower().count(lowered)
    return count


def _append_number(values: list[float], text: str) -> None:
    match = _NUMBER_RE.search(text)
    if match:
        values.append(float(match.group(0)))


def _average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _int_list(value: str) -> tuple[int, ...]:
    values = tuple(_positive_int(part.strip()) for part in value.split(","))
    if not values:
        raise argparse.ArgumentTypeError("must include at least one integer")
    return values


def _default_repo_root() -> Path:
    cwd = Path.cwd()
    return cwd.parent if cwd.name == "python" else cwd


def _run_name(prefix: str, updates: int, batch_size: int) -> str:
    stem = f"updates{updates}-batch{batch_size}"
    return f"{prefix}-{stem}" if prefix else stem


def _build_cases(args) -> list[TrainingProfileCase]:
    now = datetime.now(ZoneInfo("UTC")).strftime("%Y%m%d-%H%M%S")
    prefix = args.run_prefix or now
    return [
        TrainingProfileCase(
            input_path=args.input.resolve(),
            run_root=args.run_root.resolve(),
            run_name=_run_name(prefix, updates, batch_size),
            updates=updates,
            batch_size=batch_size,
            max_steps=args.max_steps,
            seed=args.seed,
            state_token_pad_to=args.state_token_pad_to,
            action_token_pad_to=args.action_token_pad_to,
            definition_pad_to=args.definition_pad_to,
            candidate_pad_to=args.candidate_pad_to,
            side_term_pad_to=args.side_term_pad_to,
            d_model=args.d_model,
            num_attention_layers=args.num_attention_layers,
            num_attention_heads=args.num_attention_heads,
            sample_ms=args.sample_ms,
            cprofile=args.cprofile,
            xla_dump=args.xla_dump,
            require_gpu=not args.allow_cpu,
            python_executable=sys.executable,
            repo_root=args.repo_root.resolve(),
        )
        for updates in args.updates
        for batch_size in args.batch_sizes
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile deep-cleanup CCSD training over update/batch grids."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("/tmp/ccsd-profile/deep-cleanup-4060ti"),
    )
    parser.add_argument("--run-prefix")
    parser.add_argument("--updates", type=_int_list, default=(1, 2, 4))
    parser.add_argument("--batch-sizes", type=_int_list, default=(1, 2, 4))
    parser.add_argument("--max-steps", type=_positive_int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--state-token-pad-to", type=_positive_int, default=5000)
    parser.add_argument("--action-token-pad-to", type=_positive_int, default=5000)
    parser.add_argument("--definition-pad-to", type=_positive_int, default=128)
    parser.add_argument("--candidate-pad-to", type=_positive_int, default=2048)
    parser.add_argument("--side-term-pad-to", type=_positive_int, default=256)
    parser.add_argument("--d-model", type=_positive_int, default=32)
    parser.add_argument("--num-attention-layers", type=_positive_int, default=1)
    parser.add_argument("--num-attention-heads", type=_positive_int, default=4)
    parser.add_argument("--sample-ms", type=_positive_int, default=250)
    parser.add_argument("--cprofile", action="store_true")
    parser.add_argument("--xla-dump", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=_default_repo_root())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summaries = run_matrix(_build_cases(args))
    for summary in summaries:
        print(json.dumps(asdict(summary), sort_keys=True))
    return 0 if all(summary.status == 0 for summary in summaries) else 1
