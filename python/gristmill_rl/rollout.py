from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from gristmill_rl.actions import SampledAction, make_model_proposal_fn, sample_valid_actions
from gristmill_rl.features import FeatureConfig, extract_features
from gristmill_rl.model import PolicyValueModel
from gristmill_rl.replay import EpisodeTrace, RootTraceRecord
from gristmill_rl.search import SearchConfig, SearchNode, run_sampled_puct


@dataclass(frozen=True)
class RolloutConfig:
    max_steps: int = 4
    simulations: int = 8
    actions_per_node: int = 8
    sample_attempts: int = 64
    temperature: float = 1.0
    c_puct: float = 1.5


@dataclass(frozen=True)
class RolloutResult:
    comp: Any
    trace: EpisodeTrace
    initial_log_flops: float
    final_log_flops: float
    steps: int
    terminal: bool
    valid_action_counts: list[int]


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


def run_policy_rollout(
    comp: Any,
    *,
    model: PolicyValueModel,
    feature_config: FeatureConfig,
    config: RolloutConfig,
    rng: np.random.Generator,
) -> RolloutResult:
    current_comp = comp.clone()
    trace = EpisodeTrace()
    start_from = 0
    initial_log_flops: float | None = None
    steps = 0
    terminal = False
    valid_action_counts: list[int] = []
    search_config = SearchConfig(
        simulations=config.simulations,
        actions_per_node=config.actions_per_node,
        c_puct=config.c_puct,
    )

    for _ in range(config.max_steps):
        state_log_flops: float | None = None
        root = SearchNode(comp=current_comp.clone(), start_from=start_from)

        def get_state_log_flops() -> float:
            nonlocal initial_log_flops, state_log_flops
            if state_log_flops is None:
                state_log_flops = float(current_comp.log_total_flops())
                if initial_log_flops is None:
                    initial_log_flops = state_log_flops
            return state_log_flops

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
                return get_state_log_flops() - child_log_flops
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
            terminal = bool(root.terminal)
            break

        trace.append(
            RootTraceRecord(
                state_snapshot=current_comp.snapshot(),
                action_space_snapshot=root.action_space_snapshot,
                sampled_actions=root.sampled_actions,
                visit_distribution=result.visit_distribution,
                state_log_flops=get_state_log_flops(),
                start_from=start_from,
            )
        )
        valid_action_counts.append(int(result.valid_action_count))
        chosen = _sample_from_visit_counts(
            root.sampled_actions,
            result.visit_distribution,
            temperature=config.temperature,
            rng=rng,
        )
        current_comp.apply_decision_with_space(root.action_space, chosen.decision)
        start_from = int(root.action_space.def_index)
        steps += 1

    if initial_log_flops is None:
        initial_log_flops = 0.0 if terminal else float(current_comp.log_total_flops())
    final_log_flops = (
        initial_log_flops
        if steps == 0 and terminal
        else float(current_comp.log_total_flops())
    )
    return RolloutResult(
        comp=current_comp,
        trace=trace,
        initial_log_flops=initial_log_flops,
        final_log_flops=final_log_flops,
        steps=steps,
        terminal=terminal,
        valid_action_counts=valid_action_counts,
    )
