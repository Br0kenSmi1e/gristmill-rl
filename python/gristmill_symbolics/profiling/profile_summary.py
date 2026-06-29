from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re


_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_HLO_SHAPE_RE = re.compile(
    r"\b(?P<dtype>bf16|f16|f32|f64|s\d+|u\d+|i\d+|pred)\[(?P<dims>[0-9,]+)\]"
)
_FAILED_ALLOCATION_RE = re.compile(
    r"Out of memory while trying to allocate\s+"
    r"(?P<amount>\d+(?:\.\d+)?)\s*(?P<unit>TiB|GiB|MiB|KiB|TB|GB|MB|KB|B|bytes?)",
    re.IGNORECASE,
)
_EXECUTABLE_RE = re.compile(r"executable_name='(?P<name>[^']+)'")
_SIZE_RE = re.compile(r"\bsize\s*[=:]?\s*(?P<size>\d+)\b", re.IGNORECASE)
_TOTAL_BYTES_RE = re.compile(
    r"\b(?:total\s+)?bytes\s+used\s*:\s*(?P<size>\d+)\b",
    re.IGNORECASE,
)
_ALLOCATION_HEADER_RE = re.compile(r"^\s*allocation\s+\d+:", re.IGNORECASE)
_XLA_SHAPE_LIMIT = 20
_XLA_ALLOCATION_LIMIT = 20
_XLA_ALLOCATION_CONTEXT_LIMIT = 8


@dataclass(frozen=True)
class XlaShapeSummary:
    shape: str
    dtype: str
    dims: tuple[int, ...]
    estimated_bytes: int
    count: int
    sources: tuple[str, ...]


@dataclass(frozen=True)
class XlaAllocationSummary:
    size_bytes: int | None
    shape: str | None
    source: str
    text: str
    context: tuple[str, ...] = ()
    context_shapes: tuple[str, ...] = ()
    context_value_count: int = 0


@dataclass(frozen=True)
class _XlaScan:
    files_scanned: int
    shape_counts: Counter[str]
    shape_sources: dict[str, set[str]]
    allocations: list[XlaAllocationSummary]


@dataclass(frozen=True)
class ProfileRunSummary:
    run_dir: Path
    status: int | None
    final_metrics: str | None
    failed_executable_name: str | None
    failed_allocation_bytes: int | None
    failed_allocation_text: str | None
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
    xla_files_scanned: int
    largest_xla_shapes: tuple[XlaShapeSummary, ...]
    largest_xla_allocations: tuple[XlaAllocationSummary, ...]


def summarize_run(run_dir: Path) -> ProfileRunSummary:
    stderr = _read_text(run_dir / "stderr.log")
    stdout = _read_text(run_dir / "stdout.jsonl")
    status = _read_status(run_dir / "status.txt")
    rollout_totals, rollout_counts, max_state, max_action, max_defs = _rollout_summary(stderr)
    nvidia = _nvidia_summary(run_dir / "nvidia-smi.csv")
    failed_executable, failed_allocation_bytes, failed_allocation_text = (
        _failed_allocation_summary(stderr)
    )
    xla_scan = _scan_xla_dump(run_dir)
    return ProfileRunSummary(
        run_dir=run_dir,
        status=status,
        final_metrics=_last_nonempty_line(stdout),
        failed_executable_name=failed_executable,
        failed_allocation_bytes=failed_allocation_bytes,
        failed_allocation_text=failed_allocation_text,
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
        oom_hlo_shapes=tuple(sorted(set(_shape_strings(stderr)))),
        xla_files_scanned=xla_scan.files_scanned,
        largest_xla_shapes=_largest_xla_shapes(xla_scan),
        largest_xla_allocations=_largest_xla_allocations(xla_scan),
    )


def format_summary(summary: ProfileRunSummary) -> str:
    lines = [
        f"RUN={summary.run_dir}",
        "",
        "== status ==",
        f"exit_status={_format_optional(summary.status)}",
        f"final_metrics={summary.final_metrics or '(missing)'}",
        "",
        "== failure ==",
        f"failed_executable={summary.failed_executable_name or '(missing)'}",
        f"failed_allocation={_format_failed_allocation(summary)}",
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
            "== stderr hlo shapes ==",
        ]
    )
    if summary.oom_hlo_shapes:
        lines.extend(summary.oom_hlo_shapes)
    else:
        lines.append("(none found)")
    lines.extend(
        [
            "",
            "== xla dump ==",
            f"xla_files_scanned={summary.xla_files_scanned}",
            "",
            "== largest xla shapes ==",
        ]
    )
    if summary.largest_xla_shapes:
        for shape in summary.largest_xla_shapes:
            lines.append(
                f"{_format_bytes(shape.estimated_bytes)}  count={shape.count}  "
                f"{shape.shape}  sources={_format_sources(shape.sources)}"
            )
    else:
        lines.append("(none found)")
    lines.extend(["", "== largest xla allocation lines =="])
    if summary.largest_xla_allocations:
        for allocation in summary.largest_xla_allocations:
            size = (
                _format_bytes(allocation.size_bytes)
                if allocation.size_bytes is not None
                else "(size missing)"
            )
            lines.append(
                f"{size}  {allocation.shape or '(shape missing)'}  "
                f"{allocation.source}  {allocation.text}"
            )
            if allocation.context_value_count:
                lines.append(
                    f"  context_values_shown={len(allocation.context)}/"
                    f"{allocation.context_value_count}"
                )
            if allocation.context_shapes:
                lines.append(f"  context_shapes={', '.join(allocation.context_shapes)}")
            for context_line in allocation.context:
                lines.append(f"  {context_line}")
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


def _failed_allocation_summary(stderr: str) -> tuple[str | None, int | None, str | None]:
    executable = None
    allocation_bytes = None
    allocation_text = None
    for line in stderr.splitlines():
        if executable is None:
            executable_match = _EXECUTABLE_RE.search(line)
            if executable_match:
                executable = executable_match.group("name")
        if allocation_bytes is None:
            allocation_match = _FAILED_ALLOCATION_RE.search(line)
            if allocation_match:
                amount = allocation_match.group("amount")
                unit = allocation_match.group("unit")
                allocation_text = f"{amount}{unit}"
                allocation_bytes = _parse_size_to_bytes(amount, unit)
        if executable is not None and allocation_bytes is not None:
            break
    return executable, allocation_bytes, allocation_text


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


def _scan_xla_dump(run_dir: Path) -> _XlaScan:
    xla_dir = run_dir / "xla"
    shape_counts: Counter[str] = Counter()
    shape_sources: defaultdict[str, set[str]] = defaultdict(set)
    allocations: list[XlaAllocationSummary] = []
    files_scanned = 0
    if not xla_dir.is_dir():
        return _XlaScan(files_scanned, shape_counts, dict(shape_sources), allocations)

    for path in sorted(candidate for candidate in xla_dir.rglob("*") if candidate.is_file()):
        rel_source = path.relative_to(run_dir).as_posix()
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        files_scanned += 1
        for line_number, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            for shape in _shape_strings(stripped):
                shape_counts[shape] += 1
                shape_sources[shape].add(rel_source)
            context, context_value_count = _allocation_context(lines, line_number - 1)
            allocation = _allocation_summary(
                stripped,
                f"{rel_source}:{line_number}",
                context,
                context_value_count,
            )
            if allocation is not None:
                allocations.append(allocation)

    return _XlaScan(files_scanned, shape_counts, dict(shape_sources), allocations)


def _shape_strings(text: str) -> tuple[str, ...]:
    return tuple(match.group(0) for match in _HLO_SHAPE_RE.finditer(text))


def _allocation_summary(
    line: str,
    source: str,
    context: tuple[str, ...],
    context_value_count: int,
) -> XlaAllocationSummary | None:
    lower = line.lower()
    if "allocation" not in lower and "bytes used" not in lower:
        return None
    size = _line_size_bytes(line)
    shape_match = _HLO_SHAPE_RE.search(line)
    shape = shape_match.group(0) if shape_match is not None else None
    if size is None and shape is None:
        return None
    context_shapes = tuple(
        dict.fromkeys(shape for item in context for shape in _shape_strings(item))
    )
    return XlaAllocationSummary(
        size_bytes=size,
        shape=shape,
        source=source,
        text=line,
        context=context,
        context_shapes=context_shapes,
        context_value_count=context_value_count,
    )


def _allocation_context(lines: list[str], header_index: int) -> tuple[tuple[str, ...], int]:
    header = lines[header_index].strip()
    if not _ALLOCATION_HEADER_RE.match(header):
        return (), 0
    context: list[str] = []
    for line in lines[header_index + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        if _ALLOCATION_HEADER_RE.match(stripped):
            break
        context.append(stripped)
    ranked = sorted(context, key=_allocation_context_sort_key, reverse=True)
    return tuple(ranked[:_XLA_ALLOCATION_CONTEXT_LIMIT]), len(context)


def _allocation_context_sort_key(line: str) -> tuple[bool, int]:
    size = _line_size_bytes(line)
    return size is not None, size or 0


def _line_size_bytes(line: str) -> int | None:
    match = _SIZE_RE.search(line)
    if match is None:
        match = _TOTAL_BYTES_RE.search(line)
    return int(match.group("size")) if match is not None else None


def _largest_xla_shapes(scan: _XlaScan) -> tuple[XlaShapeSummary, ...]:
    summaries = [
        summary
        for shape, count in scan.shape_counts.items()
        if (summary := _xla_shape_summary(shape, count, scan.shape_sources.get(shape, set())))
        is not None
    ]
    summaries.sort(
        key=lambda item: (item.estimated_bytes, item.count, item.shape),
        reverse=True,
    )
    return tuple(summaries[:_XLA_SHAPE_LIMIT])


def _xla_shape_summary(
    shape: str,
    count: int,
    sources: set[str],
) -> XlaShapeSummary | None:
    match = _HLO_SHAPE_RE.fullmatch(shape)
    if match is None:
        return None
    dtype = match.group("dtype")
    dims = tuple(int(part) for part in match.group("dims").split(",") if part)
    dtype_bytes = _dtype_bytes(dtype)
    if dtype_bytes is None:
        return None
    element_count = 1
    for dim in dims:
        element_count *= dim
    return XlaShapeSummary(
        shape=shape,
        dtype=dtype,
        dims=dims,
        estimated_bytes=element_count * dtype_bytes,
        count=count,
        sources=tuple(sorted(sources)),
    )


def _largest_xla_allocations(scan: _XlaScan) -> tuple[XlaAllocationSummary, ...]:
    allocations = sorted(
        scan.allocations,
        key=lambda item: (
            item.size_bytes is not None,
            item.size_bytes or 0,
            item.source,
        ),
        reverse=True,
    )
    return tuple(allocations[:_XLA_ALLOCATION_LIMIT])


def _dtype_bytes(dtype: str) -> int | None:
    if dtype == "pred":
        return 1
    if dtype in {"bf16", "f16"}:
        return 2
    if dtype == "f32":
        return 4
    if dtype == "f64":
        return 8
    if dtype[0] in {"s", "u", "i"} and dtype[1:].isdigit():
        bits = int(dtype[1:])
        return max(bits // 8, 1)
    return None


def _parse_size_to_bytes(amount: str, unit: str) -> int:
    scale = {
        "b": 1,
        "byte": 1,
        "bytes": 1,
        "kb": 1000,
        "mb": 1000**2,
        "gb": 1000**3,
        "tb": 1000**4,
        "kib": 1024,
        "mib": 1024**2,
        "gib": 1024**3,
        "tib": 1024**4,
    }[unit.lower()]
    return int(float(amount) * scale)


def _append_number(values: list[float], text: str) -> None:
    value = _number(text)
    if value is not None:
        values.append(value)


def _number(text: str) -> float | None:
    match = _NUMBER_RE.search(text)
    return float(match.group(0)) if match else None


def _average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _format_failed_allocation(summary: ProfileRunSummary) -> str:
    if summary.failed_allocation_bytes is None:
        return "(missing)"
    text = summary.failed_allocation_text
    suffix = f" ({text})" if text is not None else ""
    return f"{_format_bytes(summary.failed_allocation_bytes)} requested{suffix}"


def _format_bytes(value: int) -> str:
    units = ["bytes", "KiB", "MiB", "GiB", "TiB"]
    amount = float(value)
    unit = units[0]
    for unit in units:
        if abs(amount) < 1024.0 or unit == units[-1]:
            break
        amount /= 1024.0
    if unit == "bytes":
        return f"{int(amount)} bytes"
    return f"{amount:.2f} {unit}"


def _format_sources(sources: tuple[str, ...]) -> str:
    if not sources:
        return "(missing)"
    shown = ", ".join(sources[:3])
    if len(sources) > 3:
        shown += f", ... (+{len(sources) - 3} more)"
    return shown


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
        payload = asdict(summary)
        payload["run_dir"] = str(summary.run_dir)
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
