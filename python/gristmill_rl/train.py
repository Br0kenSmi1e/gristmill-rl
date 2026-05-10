from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from gristmill_symbolics import TensorComputation

from gristmill_rl.actions import SampledAction, make_model_proposal_fn, sample_valid_actions
from gristmill_rl.features import FeatureConfig, extract_features
from gristmill_rl.model import PolicyValueModel, TrainConfig, train_step
from gristmill_rl.replay import EpisodeTrace, ReplayBuffer, ReplayItem, RootTraceRecord
from gristmill_rl.search import SearchConfig, SearchNode, run_sampled_puct


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
    )


def _sample_from_visit_counts(
    actions: Sequence[SampledAction],
    visit_distribution: np.ndarray,
    *,
    temperature: float,
    rng: np.random.Generator,
) -> SampledAction:
    if not actions:
        raise ValueError("actions must not be empty")
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative")

    distribution = np.asarray(visit_distribution, dtype=np.float64)
    if distribution.ndim != 1 or len(distribution) != len(actions):
        raise ValueError("visit_distribution must match actions")
    if not np.all(np.isfinite(distribution)) or np.any(distribution < 0.0):
        raise ValueError("visit_distribution must be finite and non-negative")

    if temperature == 0.0:
        return actions[int(np.argmax(distribution))]

    adjusted = np.power(distribution, 1.0 / temperature)
    total = float(np.sum(adjusted))
    if not np.isfinite(total) or total <= 0.0:
        adjusted = np.full(len(actions), 1.0 / len(actions), dtype=np.float64)
    else:
        adjusted = adjusted / total
    return actions[int(rng.choice(len(actions), p=adjusted))]


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


def _proposal_for_node(
    node: SearchNode,
    *,
    model: PolicyValueModel,
    feature_config: FeatureConfig,
    rng: np.random.Generator,
    actions_per_node: int,
    sample_attempts: int,
) -> Callable[[dict[str, Any]], list[SampledAction]]:
    def proposal(action_space_snapshot: dict[str, Any]) -> list[SampledAction]:
        validation_space = node.action_space
        if validation_space is None:
            raise RuntimeError("search node action space is unavailable during proposal")
        features = extract_features(
            comp_snapshot=node.comp.snapshot(),
            action_space_snapshot=action_space_snapshot,
            start_from=node.start_from,
            log_total_flops=node.comp.log_total_flops(),
            config=feature_config,
        )
        model_proposal = make_model_proposal_fn(
            model=model,
            features=features,
            action_space_snapshot=action_space_snapshot,
            rng=rng,
        )
        return sample_valid_actions(
            node.comp,
            validation_space,
            model_proposal,
            actions_per_node=actions_per_node,
            sample_attempts=sample_attempts,
        )

    return proposal


def run(config: RunnerConfig) -> dict[str, float | int | bool]:
    rng = np.random.default_rng(config.seed)
    model = PolicyValueModel(rng_seed=config.seed)
    replay = ReplayBuffer(capacity=config.replay_capacity, seed=config.seed)
    feature_config = FeatureConfig()
    train_config = TrainConfig()
    search_config = SearchConfig(
        simulations=config.simulations,
        actions_per_node=config.actions_per_node,
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
        current_comp = _load_comp(config.input)
        trace = EpisodeTrace()
        start_from = 0
        last_initial_log_flops = float(current_comp.log_total_flops())
        last_episode_steps = 0

        for _ in range(config.max_steps):
            state_log_flops = float(current_comp.log_total_flops())
            root = SearchNode(comp=current_comp.clone(), start_from=start_from)

            def value_fn(node: SearchNode) -> float:
                child_log_flops = float(node.comp.log_total_flops())
                node.expand(
                    proposal_fn=_proposal_for_node(
                        node,
                        model=model,
                        feature_config=feature_config,
                        rng=rng,
                        actions_per_node=config.actions_per_node,
                        sample_attempts=config.sample_attempts,
                    )
                )
                if node.terminal or node.action_space_snapshot is None:
                    return state_log_flops - child_log_flops
                features = extract_features(
                    comp_snapshot=node.comp.snapshot(),
                    action_space_snapshot=node.action_space_snapshot,
                    start_from=node.start_from,
                    log_total_flops=child_log_flops,
                    config=feature_config,
                )
                return float(model(features).value)

            result = run_sampled_puct(
                root,
                config=search_config,
                proposal_fn=_proposal_for_node(
                    root,
                    model=model,
                    feature_config=feature_config,
                    rng=rng,
                    actions_per_node=config.actions_per_node,
                    sample_attempts=config.sample_attempts,
                ),
                value_fn=value_fn,
            )

            if (
                result.selected_action is None
                or root.action_space is None
                or root.action_space_snapshot is None
                or not root.sampled_actions
            ):
                break

            trace.append(
                RootTraceRecord(
                    state_snapshot=current_comp.snapshot(),
                    action_space_snapshot=root.action_space_snapshot,
                    sampled_actions=root.sampled_actions,
                    visit_distribution=result.visit_distribution,
                    state_log_flops=state_log_flops,
                    start_from=start_from,
                )
            )
            chosen = _sample_from_visit_counts(
                root.sampled_actions,
                result.visit_distribution,
                temperature=config.temperature,
                rng=rng,
            )
            current_comp.apply_decision_with_space(root.action_space, chosen.decision)
            start_from = int(root.action_space.def_index)
            last_episode_steps += 1

        last_final_log_flops = float(current_comp.log_total_flops())
        completed_items = trace.complete(final_log_flops=last_final_log_flops)
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
    }


def main(argv: Sequence[str] | None = None) -> None:
    metrics = run(parse_args(argv))
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
