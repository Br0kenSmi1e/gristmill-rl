from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from transformer_policy.batch import pad_token_choice_events
from reinforce_training.checkpoint import load_checkpoint, save_checkpoint
from reinforce_training.objective import (
    TrainConfig,
    create_optimizer,
    rewards_and_advantages,
    train_step,
)
from reinforce_training.rollout import (
    PolicyConfig,
    RolloutConfig,
    collect_episode_batch,
)


_POLICY_FIELDS = (
    ("hidden_dim", "--hidden-dim"),
    ("num_heads", "--num-heads"),
    ("num_layers", "--num-layers"),
    ("mlp_dim", "--mlp-dim"),
)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _finite_positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be finite and positive") from exc
    if parsed <= 0.0 or not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run REINFORCE training for transformer rewrite policies."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--updates", type=_positive_int, default=1)
    parser.add_argument("--batch-size", type=_positive_int, default=1)
    parser.add_argument("--max-steps", type=_positive_int, default=None)
    parser.add_argument("--num-workers", type=_positive_int, default=1)
    parser.add_argument(
        "--learning-rate",
        type=_finite_positive_float,
        default=None,
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint-in", type=Path, default=None)
    parser.add_argument("--checkpoint-out", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-overwrite",
        action="store_true",
        default=False,
    )
    parser.add_argument("--hidden-dim", type=_positive_int, default=None)
    parser.add_argument("--num-heads", type=_positive_int, default=None)
    parser.add_argument("--num-layers", type=_positive_int, default=None)
    parser.add_argument("--mlp-dim", type=_positive_int, default=None)
    return parser.parse_args(argv)


def _policy_config_from_args(args: argparse.Namespace, base: PolicyConfig) -> PolicyConfig:
    values = {
        field: (
            getattr(args, field)
            if getattr(args, field) is not None
            else getattr(base, field)
        )
        for field, _ in _POLICY_FIELDS
    }
    return PolicyConfig(**values)


def _reject_policy_config_conflicts(
    args: argparse.Namespace,
    policy_config: PolicyConfig,
) -> None:
    for field, flag in _POLICY_FIELDS:
        value = getattr(args, field)
        loaded_value = getattr(policy_config, field)
        if value is not None and value != loaded_value:
            raise ValueError(
                f"{flag} {value} does not match checkpoint policy_config.{field} "
                f"{loaded_value}"
            )


def _episode_events(episodes):
    events = []
    episode_ids = []
    for episode_id, episode in enumerate(episodes):
        for step in episode.steps:
            for event in step.token_events:
                events.append(event)
                episode_ids.append(episode_id)
    return tuple(events), np.asarray(episode_ids, dtype=np.int32)


def _mean_sample_log_prob(episodes) -> float:
    sample_log_probs = [
        step.sample_log_prob
        for episode in episodes
        for step in episode.steps
    ]
    return float(np.mean(np.asarray(sample_log_probs, dtype=np.float32)))


def _run_configs(args: argparse.Namespace):
    checkpoint_in = None
    seed = 0 if args.seed is None else args.seed

    if args.checkpoint_in is None:
        policy_config = _policy_config_from_args(args, PolicyConfig())
        train_config = TrainConfig(
            learning_rate=(
                args.learning_rate
                if args.learning_rate is not None
                else TrainConfig().learning_rate
            )
        )
        rollout_config = RolloutConfig(
            max_steps=(
                args.max_steps
                if args.max_steps is not None
                else RolloutConfig().max_steps
            )
        )
        scorer = policy_config.create_scorer(seed=seed)
        optimizer = create_optimizer(scorer, train_config)
        start_update = 0
    else:
        checkpoint_in = str(args.checkpoint_in)
        loaded = load_checkpoint(args.checkpoint_in)
        _reject_policy_config_conflicts(args, loaded.policy_config)
        if (
            args.learning_rate is not None
            and args.learning_rate != loaded.train_config.learning_rate
        ):
            raise ValueError(
                f"--learning-rate {args.learning_rate} does not match checkpoint "
                f"train_config.learning_rate {loaded.train_config.learning_rate}"
            )

        policy_config = loaded.policy_config
        train_config = loaded.train_config
        rollout_config = RolloutConfig(
            max_steps=(
                args.max_steps
                if args.max_steps is not None
                else loaded.rollout_config.max_steps
            )
        )
        seed = loaded.seed if args.seed is None else args.seed
        scorer = loaded.scorer
        optimizer = loaded.optimizer
        start_update = loaded.update_count

    return (
        scorer,
        optimizer,
        policy_config,
        train_config,
        rollout_config,
        start_update,
        seed,
        checkpoint_in,
    )


def _update_metrics(
    *,
    update: int,
    updates: int,
    batch_size: int,
    num_workers: int,
    episodes,
    rewards: np.ndarray,
    final_log_flops: np.ndarray,
    train_metrics: dict[str, float | bool],
    checkpoint_in: str | None,
    checkpoint_out: str | None,
) -> dict[str, object]:
    step_counts = np.asarray(
        [len(episode.steps) for episode in episodes],
        dtype=np.float32,
    )
    return {
        "update": update,
        "updates": updates,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "mean_reward": float(np.mean(rewards)),
        "mean_final_log_flops": float(np.mean(final_log_flops)),
        "best_final_log_flops": float(np.min(final_log_flops)),
        "mean_steps": float(np.mean(step_counts)),
        "stop_count": sum(episode.terminal_reason == "stop" for episode in episodes),
        "max_steps_count": sum(
            episode.terminal_reason == "max_steps" for episode in episodes
        ),
        "loss": float(train_metrics["loss"]),
        "mean_sample_log_prob": _mean_sample_log_prob(episodes),
        "mean_trajectory_log_prob": float(train_metrics["mean_trajectory_log_prob"]),
        "params_changed": bool(train_metrics["params_changed"]),
        "checkpoint_in": checkpoint_in,
        "checkpoint_out": checkpoint_out,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    (
        scorer,
        optimizer,
        policy_config,
        train_config,
        rollout_config,
        start_update,
        seed,
        checkpoint_in,
    ) = _run_configs(args)
    input_json = args.input.read_text()
    checkpoint_out = str(args.checkpoint_out) if args.checkpoint_out is not None else None
    last_metrics: dict[str, object] | None = None

    for offset in range(args.updates):
        update_index = start_update + offset
        episodes = collect_episode_batch(
            input_json=input_json,
            scorer=scorer,
            policy_config=policy_config,
            rollout_config=rollout_config,
            update_index=update_index,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=seed,
        )
        final_log_flops = np.asarray(
            [episode.final_log_flops for episode in episodes],
            dtype=np.float32,
        )
        rewards, advantages = rewards_and_advantages(final_log_flops)
        events, episode_ids = _episode_events(episodes)
        batch = pad_token_choice_events(events, episode_ids=episode_ids)
        train_metrics = train_step(
            scorer,
            optimizer=optimizer,
            batch=batch,
            advantages=advantages,
            episode_count=len(episodes),
        )
        last_metrics = _update_metrics(
            update=update_index + 1,
            updates=args.updates,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            episodes=episodes,
            rewards=rewards,
            final_log_flops=final_log_flops,
            train_metrics=train_metrics,
            checkpoint_in=checkpoint_in,
            checkpoint_out=checkpoint_out,
        )
        print(json.dumps(last_metrics, allow_nan=False, sort_keys=True), flush=True)

    if args.checkpoint_out is not None:
        save_checkpoint(
            args.checkpoint_out,
            scorer=scorer,
            optimizer=optimizer,
            policy_config=policy_config,
            train_config=train_config,
            rollout_config=rollout_config,
            update_count=start_update + args.updates,
            seed=seed,
            overwrite=args.checkpoint_overwrite,
        )

    if last_metrics is None:
        raise ValueError("updates must be positive")
    return last_metrics


def main(argv: Sequence[str] | None = None) -> None:
    try:
        run(parse_args(argv))
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"reinforce_training.train: {exc}") from None


if __name__ == "__main__":
    main()
