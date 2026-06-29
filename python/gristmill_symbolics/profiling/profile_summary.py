from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re


_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_HLO_SHAPE_RE = re.compile(r"\b(?:bf16|f16|f32|f64|s\d+|u\d+|i\d+|pred)\[[0-9,]+\]")


@dataclass(frozen=True)
class ProfileRunSummary:
    run_dir: Path
    status: int | None
    final_metrics: str | None
    compile_count: int
    max_rss_kbytes: int | None
    peak_gpu_memory_mib: float | None
    avg_gpu_util_percent: float | None
    peak_gpu_util_percent: float | None
    avg_power_draw_w: float | None
    rollout_phase_totals_ms: dict[str, float]
    rollout_phase_counts: dict[str, int]
    max_state_token_len: int
    max_action_token_len: int
    max_definition_count: int
    oom_hlo_shapes: tuple[str, ...]


def summarize_run(run_dir: Path) -> ProfileRunSummary:
    stderr = _read_text(run_dir / "stderr.log")
    stdout = _read_text(run_dir / "stdout.jsonl")
    status = _read_status(run_dir / "status.txt")
    rollout_totals, rollout_counts, max_state, max_action, max_defs = _rollout_summary(stderr)
    nvidia = _nvidia_summary(run_dir / "nvidia-smi.csv")
    return ProfileRunSummary(
        run_dir=run_dir,
        status=status,
        final_metrics=_last_nonempty_line(stdout),
        compile_count=sum(1 for line in stderr.splitlines() if line.startswith("Compiling")),
        max_rss_kbytes=_time_int(stderr, "Maximum resident set size"),
        peak_gpu_memory_mib=nvidia["peak_memory_mib"],
        avg_gpu_util_percent=nvidia["avg_gpu_util_percent"],
        peak_gpu_util_percent=nvidia["peak_gpu_util_percent"],
        avg_power_draw_w=nvidia["avg_power_draw_w"],
        rollout_phase_totals_ms=dict(sorted(rollout_totals.items())),
        rollout_phase_counts=dict(sorted(rollout_counts.items())),
        max_state_token_len=max_state,
        max_action_token_len=max_action,
        max_definition_count=max_defs,
        oom_hlo_shapes=tuple(sorted(set(_HLO_SHAPE_RE.findall(stderr)))),
    )


def format_summary(summary: ProfileRunSummary) -> str:
    lines = [
        f"RUN={summary.run_dir}",
        "",
        "== status ==",
        f"exit_status={_format_optional(summary.status)}",
        f"final_metrics={summary.final_metrics or '(missing)'}",
        "",
        "== memory ==",
        f"peak_gpu_memory_mib={_format_optional(summary.peak_gpu_memory_mib)}",
        f"max_rss_kbytes={_format_optional(summary.max_rss_kbytes)}",
        "",
        "== gpu telemetry ==",
        f"avg_gpu_util_percent={_format_optional(summary.avg_gpu_util_percent)}",
        f"peak_gpu_util_percent={_format_optional(summary.peak_gpu_util_percent)}",
        f"avg_power_draw_w={_format_optional(summary.avg_power_draw_w)}",
        "",
        "== compile ==",
        f"compile_count={summary.compile_count}",
        "",
        "== rollout phases ==",
    ]
    if summary.rollout_phase_totals_ms:
        for phase, elapsed_ms in sorted(
            summary.rollout_phase_totals_ms.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            count = summary.rollout_phase_counts.get(phase, 0)
            lines.append(f"{phase:28s} {elapsed_ms:12.3f} ms  count={count}")
    else:
        lines.append("(no rollout_phase events)")
    lines.extend(
        [
            f"max_state_token_len={summary.max_state_token_len}",
            f"max_action_token_len={summary.max_action_token_len}",
            f"max_definition_count={summary.max_definition_count}",
            "",
            "== oom hlo shapes ==",
        ]
    )
    if summary.oom_hlo_shapes:
        lines.extend(summary.oom_hlo_shapes)
    else:
        lines.append("(none found)")
    return "\n".join(lines)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


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


def _time_int(stderr: str, key: str) -> int | None:
    for line in stderr.splitlines():
        if key not in line:
            continue
        number = _number(line)
        if number is not None:
            return int(number)
    return None


def _rollout_summary(
    stderr: str,
) -> tuple[dict[str, float], dict[str, int], int, int, int]:
    totals: defaultdict[str, float] = defaultdict(float)
    counts: Counter[str] = Counter()
    max_state = 0
    max_action = 0
    max_defs = 0
    for line in stderr.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") != "rollout_phase":
            continue
        phase = str(event["phase"])
        totals[phase] += float(event.get("elapsed_ms") or 0.0)
        counts[phase] += 1
        max_state = max(max_state, int(event.get("state_token_len_max") or 0))
        max_action = max(max_action, int(event.get("action_token_len_max") or 0))
        max_defs = max(max_defs, int(event.get("definition_count_max") or 0))
    return dict(totals), dict(counts), max_state, max_action, max_defs


def _nvidia_summary(path: Path) -> dict[str, float | None]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = [
                {key.strip().lower(): value.strip() for key, value in row.items()}
                for row in csv.DictReader(handle)
            ]
    except FileNotFoundError:
        rows = []
    memory_values: list[float] = []
    gpu_values: list[float] = []
    power_values: list[float] = []
    for row in rows:
        for key, value in row.items():
            if key.startswith("memory.used"):
                _append_number(memory_values, value)
            elif key.startswith("utilization.gpu"):
                _append_number(gpu_values, value)
            elif key.startswith("power.draw"):
                _append_number(power_values, value)
    return {
        "peak_memory_mib": max(memory_values) if memory_values else None,
        "avg_gpu_util_percent": _average(gpu_values),
        "peak_gpu_util_percent": max(gpu_values) if gpu_values else None,
        "avg_power_draw_w": _average(power_values),
    }


def _append_number(values: list[float], text: str) -> None:
    value = _number(text)
    if value is not None:
        values.append(value)


def _number(text: str) -> float | None:
    match = _NUMBER_RE.search(text)
    return float(match.group(0)) if match else None


def _average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _format_optional(value: object) -> str:
    if value is None:
        return "(missing)"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize a memory profile run.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = summarize_run(args.run_dir)
    if args.as_json:
        payload = {
            **summary.__dict__,
            "run_dir": str(summary.run_dir),
            "oom_hlo_shapes": list(summary.oom_hlo_shapes),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
