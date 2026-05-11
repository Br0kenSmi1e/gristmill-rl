from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from gristmill_symbolics import TensorComputation

from gristmill_rl.checkpoint import load_checkpoint, save_checkpoint
from gristmill_rl.features import FeatureConfig, extract_features
from gristmill_rl.model import PolicyValueModel, TrainConfig, train_step
from gristmill_rl.replay import ReplayBuffer, ReplayItem
from gristmill_rl.rollout import RolloutConfig, run_policy_rollout


@dataclass(frozen=True)
class RunnerConfig:
    input: Path
    episodes: int = 2
    max_steps: int = 4
    simulations: int = 8
    actions_per_node: int = 8
    sample_attempts: int = 64
    train_steps: int = 1
    batch_size: int = 4
    replay_capacity: int = 256
    temperature: float = 1.0
    c_puct: float = 1.5
    seed: int = 0
    hidden_dim: int | None = None
    checkpoint_in: Path | None = None
    checkpoint_out: Path | None = None
    checkpoint_overwrite: bool = False


def parse_args(argv: Sequence[str] | None = None) -> RunnerConfig:
    parser = argparse.ArgumentParser(
        description="Run a tiny pure-Python RL training loop for gristmill rewrites."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=RunnerConfig.episodes)
    parser.add_argument("--max-steps", type=int, default=RunnerConfig.max_steps)
    parser.add_argument("--simulations", type=int, default=RunnerConfig.simulations)
    parser.add_argument(
        "--actions-per-node", type=int, default=RunnerConfig.actions_per_node
    )
    parser.add_argument(
        "--sample-attempts", type=int, default=RunnerConfig.sample_attempts
    )
    parser.add_argument("--train-steps", type=int, default=RunnerConfig.train_steps)
    parser.add_argument("--batch-size", type=int, default=RunnerConfig.batch_size)
    parser.add_argument(
        "--replay-capacity", type=int, default=RunnerConfig.replay_capacity
    )
    parser.add_argument("--temperature", type=float, default=RunnerConfig.temperature)
    parser.add_argument("--c-puct", type=float, default=RunnerConfig.c_puct)
    parser.add_argument("--seed", type=int, default=RunnerConfig.seed)
    parser.add_argument("--hidden-dim", type=int, default=RunnerConfig.hidden_dim)
    parser.add_argument("--checkpoint-in", type=Path, default=RunnerConfig.checkpoint_in)
    parser.add_argument(
        "--checkpoint-out", type=Path, default=RunnerConfig.checkpoint_out
    )
    parser.add_argument(
        "--checkpoint-overwrite",
        action="store_true",
        default=RunnerConfig.checkpoint_overwrite,
    )
    args = parser.parse_args(argv)
    return RunnerConfig(
        input=args.input,
        episodes=args.episodes,
        max_steps=args.max_steps,
        simulations=args.simulations,
        actions_per_node=args.actions_per_node,
        sample_attempts=args.sample_attempts,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        temperature=args.temperature,
        c_puct=args.c_puct,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        checkpoint_in=args.checkpoint_in,
        checkpoint_out=args.checkpoint_out,
        checkpoint_overwrite=args.checkpoint_overwrite,
    )


def _item_batch(
    replay_items: Sequence[ReplayItem], feature_config: FeatureConfig
) -> list[dict[str, Any]]:
    return [
        {
            "features": extract_features(
                comp_snapshot=item.state_snapshot,
                action_space_snapshot=item.action_space_snapshot,
                start_from=item.start_from,
                log_total_flops=item.state_log_flops,
                config=feature_config,
            ),
            "actions": item.sampled_actions,
            "policy_target": item.policy_target,
            "value_target": item.value_target,
        }
        for item in replay_items
    ]


def _load_comp(path: Path) -> TensorComputation:
    return TensorComputation.from_json_string(path.read_text())


def run(config: RunnerConfig) -> dict[str, float | int | bool | str | None]:
    rng = np.random.default_rng(config.seed)
    checkpoint_in: str | None = None
    if config.checkpoint_in is None:
        hidden_dim = config.hidden_dim if config.hidden_dim is not None else 32
        model = PolicyValueModel(hidden_dim=hidden_dim, rng_seed=config.seed)
        feature_config = FeatureConfig()
    else:
        loaded = load_checkpoint(config.checkpoint_in)
        hidden_dim = loaded.metadata.hidden_dim
        if config.hidden_dim is not None and config.hidden_dim != hidden_dim:
            raise ValueError(
                f"--hidden-dim {config.hidden_dim} does not match checkpoint "
                f"hidden_dim {hidden_dim}"
            )
        model = loaded.model
        feature_config = loaded.feature_config
        checkpoint_in = str(config.checkpoint_in)
    replay = ReplayBuffer(capacity=config.replay_capacity, seed=config.seed)
    train_config = TrainConfig()
    rollout_config = RolloutConfig(
        max_steps=config.max_steps,
        simulations=config.simulations,
        actions_per_node=config.actions_per_node,
        sample_attempts=config.sample_attempts,
        temperature=config.temperature,
        c_puct=config.c_puct,
    )

    last_total_loss = 0.0
    last_policy_loss = 0.0
    last_value_loss = 0.0
    params_changed = False
    last_initial_log_flops = 0.0
    last_final_log_flops = 0.0
    last_episode_steps = 0
    last_episode_records = 0

    for episode in range(config.episodes):
        rollout = run_policy_rollout(
            _load_comp(config.input),
            model=model,
            feature_config=feature_config,
            config=rollout_config,
            rng=rng,
        )
        last_initial_log_flops = rollout.initial_log_flops
        last_final_log_flops = rollout.final_log_flops
        last_episode_steps = rollout.steps
        completed_items = rollout.trace.complete(final_log_flops=last_final_log_flops)
        replay.extend(completed_items)
        last_episode_records = len(completed_items)

        for _ in range(config.train_steps):
            if len(replay) == 0:
                break
            metrics = train_step(
                model,
                batch=_item_batch(
                    replay.sample(batch_size=config.batch_size), feature_config
                ),
                config=train_config,
            )
            last_policy_loss = float(metrics["policy_loss"])
            last_value_loss = float(metrics["value_loss"])
            last_total_loss = float(metrics["total_loss"])
            params_changed = bool(params_changed or metrics["params_changed"])

        episode_metrics: dict[str, float | int | bool] = {
            "episode": episode + 1,
            "episodes": config.episodes,
            "replay_size": len(replay),
            "episode_steps": last_episode_steps,
            "episode_records": last_episode_records,
            "initial_log_flops": last_initial_log_flops,
            "final_log_flops": last_final_log_flops,
            "last_policy_loss": last_policy_loss,
            "last_value_loss": last_value_loss,
            "last_total_loss": last_total_loss,
            "params_changed": params_changed,
        }
        print(json.dumps(episode_metrics, sort_keys=True))

    checkpoint_out = str(config.checkpoint_out) if config.checkpoint_out else None
    if config.checkpoint_out is not None:
        save_checkpoint(
            config.checkpoint_out,
            model=model,
            feature_config=feature_config,
            hidden_dim=hidden_dim,
            metadata={"seed": config.seed, "episodes": config.episodes},
            overwrite=config.checkpoint_overwrite,
        )

    return {
        "episodes": config.episodes,
        "replay_size": len(replay),
        "last_episode_steps": last_episode_steps,
        "last_episode_records": last_episode_records,
        "initial_log_flops": last_initial_log_flops,
        "final_log_flops": last_final_log_flops,
        "last_policy_loss": last_policy_loss,
        "last_value_loss": last_value_loss,
        "last_total_loss": last_total_loss,
        "params_changed": params_changed,
        "checkpoint_in": checkpoint_in,
        "checkpoint_out": checkpoint_out,
    }


def main(argv: Sequence[str] | None = None) -> None:
    metrics = run(parse_args(argv))
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
