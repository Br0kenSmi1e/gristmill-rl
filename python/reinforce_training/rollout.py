from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import numpy as np
from flax import nnx

from gristmill_symbolics import RewriteState, TensorComputation
from reinforce_training.trace import EpisodeTrace, step_trace_from_traced_sample
from transformer_policy.decoder import sample_step_with_events
from transformer_policy.sequence_model import CausalTransformerScorer


def _validate_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be positive integer")


@dataclass(frozen=True)
class PolicyConfig:
    hidden_dim: int = 32
    num_heads: int = 4
    num_layers: int = 1
    mlp_dim: int = 64

    def __post_init__(self) -> None:
        _validate_positive_integer("hidden_dim", self.hidden_dim)
        _validate_positive_integer("num_heads", self.num_heads)
        _validate_positive_integer("num_layers", self.num_layers)
        _validate_positive_integer("mlp_dim", self.mlp_dim)
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

    def create_scorer(self, *, seed: int) -> CausalTransformerScorer:
        return CausalTransformerScorer(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            mlp_dim=self.mlp_dim,
            rngs=nnx.Rngs(seed),
        )


@dataclass(frozen=True)
class RolloutConfig:
    max_steps: int = 4

    def __post_init__(self) -> None:
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int):
            raise ValueError("max_steps must be an integer")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")


def _model_state(scorer) -> Any:
    return nnx.state(scorer, nnx.Param)


def _restore_scorer(
    policy_config: PolicyConfig,
    state: Any,
) -> CausalTransformerScorer:
    scorer = policy_config.create_scorer(seed=0)
    nnx.update(scorer, state)
    return scorer


def sample_episode(
    *,
    input_json: str,
    scorer,
    config: RolloutConfig,
    episode_index: int,
    episode_seed: int,
) -> EpisodeTrace:
    comp = TensorComputation.from_json_string(input_json)
    state = RewriteState.from_computation(comp)
    rng = np.random.default_rng(episode_seed)
    steps = []
    terminal_reason = "max_steps"

    for step_index in range(config.max_steps):
        state_snapshot = state.snapshot()
        traced = sample_step_with_events(
            state,
            scorer,
            rng,
            step_index=step_index,
        )
        steps.append(
            step_trace_from_traced_sample(
                step_index=step_index,
                state_snapshot=state_snapshot,
                traced=traced,
            )
        )
        if traced.sample.stopped:
            terminal_reason = "stop"
            break
        state.step_with_space(traced.sample.action_space, traced.sample.decision)

    final_log_flops = float(state.log_total_flops())
    reward = -final_log_flops
    return EpisodeTrace(
        episode_index=episode_index,
        episode_seed=episode_seed,
        steps=tuple(steps),
        final_snapshot=state.snapshot(),
        final_log_flops=final_log_flops,
        reward=reward,
        terminal_reason=terminal_reason,
    )


def _episode_job(
    *,
    input_json: str,
    policy_config: PolicyConfig,
    model_state: Any,
    rollout_config: RolloutConfig,
    episode_index: int,
    episode_seed: int,
) -> EpisodeTrace:
    try:
        scorer = _restore_scorer(policy_config, model_state)
        return sample_episode(
            input_json=input_json,
            scorer=scorer,
            config=rollout_config,
            episode_index=episode_index,
            episode_seed=episode_seed,
        )
    except Exception as exc:
        raise RuntimeError(
            "rollout episode failed "
            f"episode_index={episode_index} episode_seed={episode_seed}"
        ) from exc


def _validate_nonnegative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be non-negative")


def collect_episode_batch(
    *,
    input_json: str,
    scorer,
    policy_config: PolicyConfig,
    rollout_config: RolloutConfig,
    update_index: int,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[EpisodeTrace, ...]:
    _validate_positive_integer("batch_size", batch_size)
    _validate_positive_integer("num_workers", num_workers)
    _validate_nonnegative_integer("update_index", update_index)
    model_state = _model_state(scorer)

    def episode_seed(index: int) -> int:
        return int(seed + update_index * batch_size + index)

    if num_workers == 1:
        episodes = [
            _episode_job(
                input_json=input_json,
                policy_config=policy_config,
                model_state=model_state,
                rollout_config=rollout_config,
                episode_index=index,
                episode_seed=episode_seed(index),
            )
            for index in range(batch_size)
        ]
    else:
        episodes = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    _episode_job,
                    input_json=input_json,
                    policy_config=policy_config,
                    model_state=model_state,
                    rollout_config=rollout_config,
                    episode_index=index,
                    episode_seed=episode_seed(index),
                )
                for index in range(batch_size)
            ]
            for future in as_completed(futures):
                episodes.append(future.result())

    return tuple(sorted(episodes, key=lambda episode: episode.episode_index))
