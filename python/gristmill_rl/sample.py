from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from gristmill_symbolics import TensorComputation

from gristmill_rl.checkpoint import load_checkpoint
from gristmill_rl.rollout import RolloutConfig, run_policy_rollout


@dataclass(frozen=True)
class SampleConfig:
    checkpoint: Path
    input: Path
    output_dir: Path
    samples: int = 1
    max_steps: int = 4
    simulations: int = 8
    actions_per_node: int = 8
    sample_attempts: int = 64
    temperature: float = 1.0
    c_puct: float = 1.5
    seed: int = 0
    overwrite_output: bool = False


def parse_args(argv: Sequence[str] | None = None) -> SampleConfig:
    parser = argparse.ArgumentParser(
        description="Sample gristmill rewrites from a saved RL checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=SampleConfig.samples)
    parser.add_argument("--max-steps", type=int, default=SampleConfig.max_steps)
    parser.add_argument("--simulations", type=int, default=SampleConfig.simulations)
    parser.add_argument(
        "--actions-per-node", type=int, default=SampleConfig.actions_per_node
    )
    parser.add_argument(
        "--sample-attempts", type=int, default=SampleConfig.sample_attempts
    )
    parser.add_argument("--temperature", type=float, default=SampleConfig.temperature)
    parser.add_argument("--c-puct", type=float, default=SampleConfig.c_puct)
    parser.add_argument("--seed", type=int, default=SampleConfig.seed)
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        default=SampleConfig.overwrite_output,
    )
    args = parser.parse_args(argv)
    return SampleConfig(
        checkpoint=args.checkpoint,
        input=args.input,
        output_dir=args.output_dir,
        samples=args.samples,
        max_steps=args.max_steps,
        simulations=args.simulations,
        actions_per_node=args.actions_per_node,
        sample_attempts=args.sample_attempts,
        temperature=args.temperature,
        c_puct=args.c_puct,
        seed=args.seed,
        overwrite_output=args.overwrite_output,
    )


def _prepare_sample_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"sample output directory already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def run(config: SampleConfig) -> dict[str, float | int | str | None]:
    loaded = load_checkpoint(config.checkpoint)
    rng = np.random.default_rng(config.seed)
    rollout_config = RolloutConfig(
        max_steps=config.max_steps,
        simulations=config.simulations,
        actions_per_node=config.actions_per_node,
        sample_attempts=config.sample_attempts,
        temperature=config.temperature,
        c_puct=config.c_puct,
    )

    total_steps = 0
    best_final_log_flops: float | None = None
    config.output_dir.mkdir(parents=True, exist_ok=True)

    for sample in range(config.samples):
        sample_dir = config.output_dir / f"sample-{sample:03d}"
        _prepare_sample_dir(sample_dir, overwrite=config.overwrite_output)
        rollout = run_policy_rollout(
            TensorComputation.load_json(config.input),
            model=loaded.model,
            feature_config=loaded.feature_config,
            config=rollout_config,
            rng=rng,
        )

        final_path = sample_dir / "final.json"
        metrics_path = sample_dir / "metrics.json"
        rollout.comp.write_json(final_path)

        metrics: dict[str, float | int | bool | str | list[int]] = {
            "sample": sample,
            "steps": rollout.steps,
            "terminal": rollout.terminal,
            "initial_log_flops": rollout.initial_log_flops,
            "final_log_flops": rollout.final_log_flops,
            "valid_action_counts": rollout.valid_action_counts,
            "checkpoint": str(config.checkpoint),
            "input": str(config.input),
        }
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
        print(json.dumps(metrics, sort_keys=True))

        total_steps += rollout.steps
        if (
            best_final_log_flops is None
            or rollout.final_log_flops < best_final_log_flops
        ):
            best_final_log_flops = rollout.final_log_flops

    return {
        "samples": config.samples,
        "output_dir": str(config.output_dir),
        "total_steps": total_steps,
        "best_final_log_flops": best_final_log_flops,
    }


def main(argv: Sequence[str] | None = None) -> None:
    summary = run(parse_args(argv))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
