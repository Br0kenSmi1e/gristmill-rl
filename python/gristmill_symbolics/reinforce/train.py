from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from gristmill_symbolics import RewriteState, TensorComputation
from gristmill_symbolics.policy import PolicyConfig

from .checkpoint import load_checkpoint, save_checkpoint
from .model import CurrentTransformerModel
from .trainer import ReinforceTrainer
from .train_state import advance_train_state, init_train_state
from .types import (
    CurrentTransformerModelConfig,
    OptimizerConfig,
    ReinforceTrainerConfig,
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
    parser.add_argument("--state-token-pad-to", type=_positive_int)
    parser.add_argument("--action-token-pad-to", type=_positive_int)
    parser.add_argument("--definition-pad-to", type=_positive_int)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--checkpoint-in")
    parser.add_argument("--checkpoint-out")
    return parser


def main(argv=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)

    if args.checkpoint_in is None:
        missing_pad_flags = [
            flag
            for flag, value in [
                ("--state-token-pad-to", args.state_token_pad_to),
                ("--action-token-pad-to", args.action_token_pad_to),
                ("--definition-pad-to", args.definition_pad_to),
            ]
            if value is None
        ]
        if missing_pad_flags:
            parser.error(
                "fresh training requires static pad flags: "
                + ", ".join(missing_pad_flags)
            )

        policy_config = PolicyConfig(d_model=8)
        optimizer_config = OptimizerConfig(learning_rate=args.learning_rate)
        model_config = CurrentTransformerModelConfig(
            policy_config=policy_config,
            batch_size=args.batch_size,
            max_steps=args.max_steps,
            state_token_pad_to=args.state_token_pad_to,
            action_token_pad_to=args.action_token_pad_to,
            definition_pad_to=args.definition_pad_to,
        )
        trainer_config = ReinforceTrainerConfig(
            batch_size=args.batch_size,
            optimizer_config=optimizer_config,
        )
        train_state = init_train_state(
            policy_config,
            optimizer_config,
            seed=args.seed,
        )
        recent_metrics = []
    else:
        checkpoint = load_checkpoint(args.checkpoint_in)
        train_state = checkpoint.train_state
        model_config = checkpoint.model_config
        trainer_config = checkpoint.trainer_config
        recent_metrics = list(checkpoint.recent_metrics)

    model = CurrentTransformerModel()
    trainer = ReinforceTrainer()
    input_path = Path(args.input)
    for _ in range(args.updates):
        comp = TensorComputation.load_json(input_path)
        initial_states = [
            RewriteState.from_computation(comp)
            for _ in range(trainer_config.batch_size)
        ]
        train_state, metrics = advance_train_state(
            train_state,
            initial_states,
            model=model,
            trainer=trainer,
            model_config=model_config,
            trainer_config=trainer_config,
        )
        recent_metrics.append(metrics)
        print(json.dumps(asdict(metrics), sort_keys=True))

    if args.checkpoint_out is not None:
        save_checkpoint(
            args.checkpoint_out,
            train_state,
            model_config=model_config,
            trainer_config=trainer_config,
            recent_metrics=tuple(recent_metrics[-10:]),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
