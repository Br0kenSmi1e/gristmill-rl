from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from gristmill_symbolics import RewriteState, TensorComputation
from gristmill_symbolics.policy import PolicyConfig

from .checkpoint import load_checkpoint, save_checkpoint
from .train_state import init_train_state, train_update
from .types import (
    BaselineConfig,
    LossConfig,
    OptimizerConfig,
    RewardConfig,
    RolloutConfig,
)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke-run on-policy REINFORCE updates."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--updates", type=_positive_int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--checkpoint-in")
    parser.add_argument("--checkpoint-out")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)

    if args.checkpoint_in is None:
        rollout_config = RolloutConfig(
            batch_size=args.batch_size,
            max_steps=args.max_steps,
            seed=args.seed,
        )
        reward_config = RewardConfig()
        baseline_config = BaselineConfig()
        loss_config = LossConfig()
        train_state = init_train_state(
            PolicyConfig(d_model=8, max_candidates=8, max_side_terms=4),
            OptimizerConfig(learning_rate=args.learning_rate),
            seed=args.seed,
        )
        recent_metrics = []
    else:
        checkpoint = load_checkpoint(args.checkpoint_in)
        train_state = checkpoint.train_state
        rollout_config = checkpoint.rollout_config
        reward_config = checkpoint.reward_config
        baseline_config = checkpoint.baseline_config
        loss_config = checkpoint.loss_config
        recent_metrics = list(checkpoint.recent_metrics)

    input_path = Path(args.input)
    for _ in range(args.updates):
        comp = TensorComputation.load_json(input_path)
        initial_states = [
            RewriteState.from_computation(comp) for _ in range(rollout_config.batch_size)
        ]
        train_state, metrics, _table = train_update(
            train_state,
            initial_states,
            rollout_config,
            reward_config,
            baseline_config,
            loss_config,
        )
        recent_metrics.append(metrics)
        print(json.dumps(asdict(metrics), sort_keys=True))

    if args.checkpoint_out is not None:
        save_checkpoint(
            args.checkpoint_out,
            train_state,
            rollout_config=rollout_config,
            reward_config=reward_config,
            baseline_config=baseline_config,
            loss_config=loss_config,
            recent_metrics=tuple(recent_metrics[-10:]),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
