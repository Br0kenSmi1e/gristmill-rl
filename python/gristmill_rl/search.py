from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from math import isfinite
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
    visits: int = 0
    value_sum: float = 0.0
    node: SearchNode | None = None

    @property
    def q_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


@dataclass
class SearchNode:
    comp: Any
    start_from: int
    action_space: Any | None = None
    action_space_snapshot: dict[str, Any] | None = None
    sampled_actions: list[SampledAction] = field(default_factory=list)
    children: list[SearchChild] = field(default_factory=list)
    expanded: bool = False
    terminal: bool = False

    def expand(
        self, *, proposal_fn: Callable[[dict[str, Any]], list[SampledAction]]
    ) -> SearchNode:
        if self.expanded:
            return self

        previous_action_space = self.action_space
        previous_action_space_snapshot = (
            deepcopy(self.action_space_snapshot)
            if self.action_space_snapshot is not None
            else None
        )
        previous_sampled_actions = list(self.sampled_actions)
        previous_children = list(self.children)
        previous_expanded = self.expanded
        previous_terminal = self.terminal

        space = self.comp.next_action_space(self.start_from)
        if space is None:
            self.action_space = space
            self.expanded = True
            self.terminal = True
            return self

        snapshot = space.snapshot()
        self.action_space = space
        self.action_space_snapshot = deepcopy(snapshot)
        try:
            proposed = list(proposal_fn(deepcopy(snapshot)))
            normalized = _normalize_action_priors(proposed)
            children = [
                SearchChild(action=action, prior=float(action.prior))
                for action in normalized
            ]
        except Exception:
            self.action_space = previous_action_space
            self.action_space_snapshot = previous_action_space_snapshot
            self.sampled_actions = previous_sampled_actions
            self.children = previous_children
            self.expanded = previous_expanded
            self.terminal = previous_terminal
            raise

        self.sampled_actions = normalized
        self.children = children
        self.action_space = space
        self.action_space_snapshot = deepcopy(snapshot)
        self.expanded = True
        self.terminal = False
        return self


@dataclass(frozen=True)
class SearchResult:
    selected_action: SampledAction | None
    visit_counts: np.ndarray
    visit_distribution: np.ndarray
    valid_action_count: int


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
    *,
    q_value: float,
    prior: float,
    parent_visits: int,
    child_visits: int,
    c_puct: float,
) -> float:
    return float(
        q_value
        + c_puct * prior * np.sqrt(max(parent_visits, 1)) / (1 + child_visits)
    )


def _select_child(node: SearchNode, config: SearchConfig) -> SearchChild:
    if not node.children:
        raise ValueError("cannot select from a node with no children")
    parent_visits = sum(child.visits for child in node.children)
    return max(
        node.children,
        key=lambda child: puct_score(
            q_value=child.q_value,
            prior=child.prior,
            parent_visits=parent_visits,
            child_visits=child.visits,
            c_puct=config.c_puct,
        ),
    )


def _child_node(parent: SearchNode, child: SearchChild) -> SearchNode:
    if parent.action_space is None:
        raise ValueError("parent must store an action space before creating children")
    rewritten = parent.comp.clone()
    rewritten.apply_decision_with_space(parent.action_space, child.action.decision)
    return SearchNode(comp=rewritten, start_from=int(parent.action_space.def_index))


def _visit_distribution(children: list[SearchChild]) -> tuple[np.ndarray, np.ndarray]:
    if not children:
        empty = np.asarray([], dtype=np.float32)
        return empty, empty

    counts = np.asarray([child.visits for child in children], dtype=np.float32)
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
    return counts, distribution.astype(np.float32)


def run_sampled_puct(
    root: SearchNode,
    *,
    config: SearchConfig,
    proposal_fn: ProposalFn,
    value_fn: Callable[[SearchNode], float],
) -> SearchResult:
    if config.simulations < 0:
        raise ValueError("simulations must be non-negative")
    if config.actions_per_node <= 0:
        raise ValueError("actions_per_node must be positive")

    def limited_proposal(snapshot: dict[str, Any]) -> list[SampledAction]:
        return list(proposal_fn(snapshot))[: config.actions_per_node]

    root.expand(proposal_fn=limited_proposal)
    if root.terminal or not root.children:
        empty = np.asarray([], dtype=np.float32)
        return SearchResult(
            selected_action=None,
            visit_counts=empty,
            visit_distribution=empty,
            valid_action_count=0,
        )

    for _ in range(config.simulations):
        child = _select_child(root, config)
        if child.node is None:
            child.node = _child_node(root, child)
        value = float(value_fn(child.node))
        child.visits += 1
        child.value_sum += value

    visit_counts, visit_distribution = _visit_distribution(root.children)
    selected_child = max(root.children, key=lambda child: child.visits)
    return SearchResult(
        selected_action=selected_child.action,
        visit_counts=visit_counts,
        visit_distribution=visit_distribution,
        valid_action_count=len(root.children),
    )
