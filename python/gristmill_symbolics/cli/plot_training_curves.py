from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_FIELDS = (
    "update_index",
    "batch_size",
    "objective_loss_mean",
    "reward_std",
    "final_flops_best",
)

GRISTMILL_OPTIMIZED_LOG_FLOPS = 49.23057289544251


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot training curves from training JSONL metrics."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title")
    return parser


def load_metrics(path: Path) -> list[dict[str, Any]]:
    metrics = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            metrics.append(_load_metric_line(line, line_number))
    if not metrics:
        raise ValueError(f"{path} does not contain any metric lines")
    return metrics


def plot_training_curves(
    metrics: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str | None = None,
) -> None:
    _validate_metrics(metrics)
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    x = np.asarray([item["update_index"] for item in metrics], dtype=np.int64)
    objective = _float_array(metrics, "objective_loss_mean")
    final_flops = _float_array(metrics, "final_flops_best")
    objective_error = objective_errorbar(metrics)

    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    if title is not None:
        figure.suptitle(title)
    _plot_objective(axes[0], x, objective, objective_error)
    _plot_final_flops(axes[1], x, final_flops)
    axes[2].axis("off")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    metrics = load_metrics(Path(args.input))
    plot_training_curves(
        metrics,
        Path(args.output),
        title=args.title,
    )
    return 0


def _load_metric_line(line: str, line_number: int) -> dict[str, Any]:
    try:
        value = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"line {line_number} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"line {line_number} must contain a JSON object")
    return value


def _validate_metrics(metrics: list[dict[str, Any]]) -> None:
    for row, metric in enumerate(metrics, start=1):
        missing = [field for field in REQUIRED_FIELDS if field not in metric]
        if missing:
            raise ValueError(f"metric line {row} is missing {missing}")
    objective_errorbar(metrics)


def _float_array(metrics: list[dict[str, Any]], field: str) -> np.ndarray:
    try:
        return np.asarray([item[field] for item in metrics], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc


def objective_errorbar(metrics: list[dict[str, Any]]) -> np.ndarray:
    reward_std = _float_array(metrics, "reward_std")
    batch_size = _float_array(metrics, "batch_size")
    if not bool(np.all(batch_size > 0.0)):
        raise ValueError("batch_size must be positive")
    return reward_std / np.sqrt(batch_size)


def _plot_objective(axis, x, objective, objective_error) -> None:
    axis.errorbar(
        x,
        objective,
        yerr=objective_error,
        fmt="-o",
        markersize=3.0,
        linewidth=1.4,
        capsize=2.0,
    )
    axis.set_title("Objective Loss")
    axis.set_xlabel("Update")
    axis.set_ylabel("objective_loss_mean")
    axis.grid(True, alpha=0.3)


def _plot_final_flops(axis, x, final_flops) -> None:
    axis.plot(x, final_flops, "-o", markersize=3.0, linewidth=1.4)
    axis.axhline(
        GRISTMILL_OPTIMIZED_LOG_FLOPS,
        color="tab:red",
        linestyle="--",
        linewidth=1.2,
        label="gristmill optimized",
    )
    axis.set_title("Best Final FLOPs")
    axis.set_xlabel("Update")
    axis.set_ylabel("final_flops_best")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best", fontsize="small")


if __name__ == "__main__":
    raise SystemExit(main())
