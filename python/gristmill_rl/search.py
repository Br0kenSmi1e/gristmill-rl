from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from math import isfinite, sqrt
from typing import Any, Callable

import numpy as np

from gristmill_rl.actions import SampledAction


ProposalFn = Callable[[dict[str, Any]], list[SampledAction]]


@dataclass(frozen=True)
class SearchConfig:
    simulations: int = 8
    actions_per_node: int = 8
    c_puct: float = 1.5


@dataclass
class SearchChild:
    action: SampledAction
    prior: float
    visit_count: int = 0
    total_value: float = 0.0
    node: SearchNode | None = None

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


@dataclass
class SearchNode:
    comp: Any
    start_from: int = 0
    action_space: Any | None = None
    action_space_snapshot: dict[str, Any] | None = None
    sampled_actions: list[SampledAction] | None = None
    children: list[SearchChild] | None = None
    expanded: bool = False
    terminal: bool = False

    def __post_init__(self) -> None:
        if self.sampled_actions is None:
            self.sampled_actions = []
        if self.children is None:
            self.children = []

    def expand(self, proposal_fn: ProposalFn, config: SearchConfig) -> None:
        if self.expanded:
            return
        if config.actions_per_node <= 0:
            raise ValueError("actions_per_node must be positive")

        space = self.comp.next_action_space(self.start_from)
        self.action_space = space
        self.expanded = True
        if space is None:
            self.terminal = True
            return

        snapshot = space.snapshot()
        self.action_space_snapshot = deepcopy(snapshot)
        proposed = list(proposal_fn(self.action_space_snapshot))
        actions = proposed[: config.actions_per_node]
        normalized = _normalize_action_priors(actions)
        self.sampled_actions = normalized
        self.children = [
            SearchChild(action=action, prior=float(action.prior)) for action in normalized
        ]
        if not self.children:
            self.terminal = True


@dataclass(frozen=True)
class SearchResult:
    root: SearchNode
    state_snapshot: dict[str, Any]
    action_space_snapshot: dict[str, Any]
    sampled_actions: list[SampledAction]
    visit_distribution: np.ndarray
    state_log_flops: float
    start_from: int


def _normalize_action_priors(actions: list[SampledAction]) -> list[SampledAction]:
    if not actions:
        return []

    priors = np.asarray([float(action.prior) for action in actions], dtype=np.float64)
    valid = np.isfinite(priors) & (priors >= 0.0)
    priors = np.where(valid, priors, 0.0)
    total = float(priors.sum())
    if not isfinite(total) or total <= 0.0:
        priors = np.full(len(actions), 1.0 / len(actions), dtype=np.float64)
    else:
        priors = priors / total
    return [
        replace(action, prior=float(prior)) for action, prior in zip(actions, priors)
    ]


def puct_score(
    child: SearchChild,
    *,
    parent_visit_count: int,
    c_puct: float,
) -> float:
    exploration = c_puct * child.prior * sqrt(parent_visit_count + 1) / (
        1 + child.visit_count
    )
    return child.q_value + exploration


def _select_child(node: SearchNode, config: SearchConfig) -> SearchChild:
    if not node.children:
        raise ValueError("cannot select from a node with no children")
    parent_visits = sum(child.visit_count for child in node.children)
    return max(
        node.children,
        key=lambda child: puct_score(
            child,
            parent_visit_count=parent_visits,
            c_puct=config.c_puct,
        ),
    )


def _child_node(parent: SearchNode, child: SearchChild) -> SearchNode:
    if parent.action_space is None:
        raise ValueError("parent must store an action space before creating children")
    rewritten = parent.comp.clone()
    rewritten.apply_decision_with_space(parent.action_space, child.action.decision)
    return SearchNode(comp=rewritten, start_from=int(parent.action_space.def_index))


def _visit_distribution(children: list[SearchChild]) -> np.ndarray:
    if not children:
        return np.asarray([], dtype=np.float32)

    counts = np.asarray([child.visit_count for child in children], dtype=np.float64)
    total = float(counts.sum())
    if isfinite(total) and total > 0.0:
        distribution = counts / total
    else:
        priors = np.asarray([child.prior for child in children], dtype=np.float64)
        prior_total = float(priors.sum())
        if isfinite(prior_total) and prior_total > 0.0:
            distribution = priors / prior_total
        else:
            distribution = np.full(len(children), 1.0 / len(children), dtype=np.float64)
    return distribution.astype(np.float32)


def run_sampled_puct(
    comp: Any,
    *,
    start_from: int = 0,
    proposal_fn: ProposalFn,
    config: SearchConfig = SearchConfig(),
) -> SearchResult:
    if config.simulations < 0:
        raise ValueError("simulations must be non-negative")

    root = SearchNode(comp=comp, start_from=start_from)
    root.expand(proposal_fn, config)

    for _ in range(config.simulations):
        node = root
        path: list[SearchChild] = []
        while node.expanded and not node.terminal and node.children:
            child = _select_child(node, config)
            path.append(child)
            if child.node is None:
                child.node = _child_node(node, child)
            node = child.node

        if not node.expanded:
            node.expand(proposal_fn, config)

        value = -float(node.comp.log_total_flops())
        for child in path:
            child.visit_count += 1
            child.total_value += value

    return SearchResult(
        root=root,
        state_snapshot=deepcopy(comp.snapshot()),
        action_space_snapshot=deepcopy(root.action_space_snapshot or {}),
        sampled_actions=list(root.sampled_actions),
        visit_distribution=_visit_distribution(root.children),
        state_log_flops=float(comp.log_total_flops()),
        start_from=start_from,
    )
