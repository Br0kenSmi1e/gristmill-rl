from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from gristmill_symbolics import TensorComputation


@dataclass(frozen=True)
class BaselineMetric:
    name: str
    path: Path
    log_flops: float

    def to_json(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "path": str(self.path),
            "log_flops": self.log_flops,
        }


def parse_baseline_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--baseline must use NAME=PATH")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError("--baseline name must not be empty")
    if not raw_path:
        raise ValueError("--baseline path must not be empty")
    return name, Path(raw_path)


def load_baselines(items: Sequence[tuple[str, Path]]) -> list[BaselineMetric]:
    seen: set[str] = set()
    baselines: list[BaselineMetric] = []
    for name, path in items:
        if name in seen:
            raise ValueError(f"duplicate baseline name: {name}")
        seen.add(name)
        comp = TensorComputation.load_json(path)
        baselines.append(
            BaselineMetric(name=name, path=path, log_flops=float(comp.log_total_flops()))
        )
    return baselines


class MonitorWriter:
    def __init__(self, log_dir: Path, *, baselines: Sequence[BaselineMetric]):
        self.log_dir = Path(log_dir)
        self.baselines = list(baselines)
        self.metrics_path = self.log_dir / "metrics.jsonl"
        self.baselines_path = self.log_dir / "baselines.json"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def write_baselines(self) -> None:
        payload = {"baselines": [baseline.to_json() for baseline in self.baselines]}
        self.baselines_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    def append_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(metrics)
        enriched["flops_improvement"] = float(
            enriched["initial_log_flops"] - enriched["final_log_flops"]
        )
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(enriched, sort_keys=True))
            handle.write("\n")
        return enriched
