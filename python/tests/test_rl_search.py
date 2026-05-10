from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from gristmill_rl.actions import SampledAction, first_full_mask_action
from gristmill_rl.search import (
    SearchChild,
    SearchConfig,
    SearchNode,
    SearchResult,
    _child_node,
    _visit_distribution,
    puct_score,
    run_sampled_puct,
)

from .rl_fixtures import actionable_comp


def test_search_child_exposes_spec_counters_and_q_value():
    child = SearchChild(
        action=SampledAction(
            {"candidate_index": 0, "left_mask": [True], "right_mask": [True]},
            prior=0.25,
        ),
        prior=0.25,
    )

    assert child.visits == 0
    assert child.value_sum == 0.0
    assert child.q_value == 0.0

    child.visits = 4
    child.value_sum = 10.0

    assert child.q_value == 2.5
    assert not hasattr(child, "visit_count")
    assert not hasattr(child, "total_value")


def test_search_node_uses_default_factories_for_mutable_fields():
    left = SearchNode(comp=object(), start_from=0)
    right = SearchNode(comp=object(), start_from=0)

    left.sampled_actions.append(
        SampledAction(
            {"candidate_index": 0, "left_mask": [True], "right_mask": [True]},
            prior=1.0,
        )
    )
    left.children.append(SearchChild(action=left.sampled_actions[0], prior=1.0))

    assert right.sampled_actions == []
    assert right.children == []


def test_search_node_expands_once_and_stores_space():
    comp = actionable_comp()
    node = SearchNode(comp=comp, start_from=0)
    calls = 0

    def proposal(snapshot):
        nonlocal calls
        calls += 1
        return [first_full_mask_action(snapshot, prior=2.0)]

    first = node.expand(proposal_fn=proposal)
    first_space = node.action_space
    first_snapshot = node.action_space_snapshot
    second = node.expand(proposal_fn=proposal)

    assert first is node
    assert second is node
    assert calls == 1
    assert node.expanded
    assert not node.terminal
    assert node.action_space is first_space
    assert node.action_space_snapshot == first_snapshot
    assert node.sampled_actions[0].prior == 1.0
    assert len(node.children) == 1


def test_search_result_is_frozen_spec_contract():
    result = SearchResult(
        selected_action=None,
        visit_counts=np.asarray([], dtype=np.float32),
        visit_distribution=np.asarray([], dtype=np.float32),
        valid_action_count=0,
    )

    assert tuple(result.__dataclass_fields__) == (
        "selected_action",
        "visit_counts",
        "visit_distribution",
        "valid_action_count",
    )
    with pytest.raises(FrozenInstanceError):
        result.valid_action_count = 1


def test_puct_score_uses_numeric_spec_formula():
    assert puct_score(
        q_value=2.0,
        prior=0.5,
        parent_visits=4,
        child_visits=1,
        c_puct=1.5,
    ) == 2.75


def test_visit_distribution_returns_float32_counts_and_distribution():
    children = [
        SearchChild(
            action=SampledAction(
                {"candidate_index": 0, "left_mask": [True], "right_mask": [True]},
                prior=0.25,
            ),
            prior=0.25,
            visits=2,
        ),
        SearchChild(
            action=SampledAction(
                {"candidate_index": 1, "left_mask": [True], "right_mask": [True]},
                prior=0.75,
            ),
            prior=0.75,
            visits=6,
        ),
    ]

    counts, distribution = _visit_distribution(children)

    assert counts.dtype == np.float32
    assert distribution.dtype == np.float32
    np.testing.assert_array_equal(counts, np.asarray([2.0, 6.0], dtype=np.float32))
    np.testing.assert_allclose(
        distribution, np.asarray([0.25, 0.75], dtype=np.float32)
    )


def test_run_sampled_puct_returns_visit_counts_distribution_and_best_action():
    comp = actionable_comp()
    root = SearchNode(comp=comp, start_from=0)

    def proposal(snapshot):
        return [
            first_full_mask_action(snapshot, candidate_index=0, prior=0.25),
            first_full_mask_action(snapshot, candidate_index=1, prior=0.75),
        ]

    result = run_sampled_puct(
        root,
        config=SearchConfig(simulations=6, actions_per_node=2),
        proposal_fn=proposal,
        value_fn=lambda node: 1.0,
    )

    assert result.valid_action_count == 2
    assert result.selected_action == root.children[1].action
    assert result.visit_counts.dtype == np.float32
    assert result.visit_distribution.dtype == np.float32
    assert result.visit_counts.sum() == 6.0
    assert root.children[1].visits == int(result.visit_counts.max())
    np.testing.assert_allclose(result.visit_distribution.sum(), 1.0)


def test_run_sampled_puct_limits_root_actions_before_search():
    comp = actionable_comp()
    root = SearchNode(comp=comp, start_from=0)

    def proposal(snapshot):
        return [
            first_full_mask_action(snapshot, candidate_index=0, prior=1.0),
            first_full_mask_action(snapshot, candidate_index=1, prior=1.0),
        ]

    result = run_sampled_puct(
        root,
        config=SearchConfig(simulations=4, actions_per_node=1),
        proposal_fn=proposal,
        value_fn=lambda node: 1.0,
    )

    assert result.valid_action_count == 1
    assert len(root.children) == 1
    assert result.selected_action == root.children[0].action
    np.testing.assert_array_equal(result.visit_counts, np.asarray([4.0]))


def test_run_sampled_puct_evaluates_selected_child_node():
    comp = actionable_comp()
    root = SearchNode(comp=comp, start_from=0)
    evaluated = []

    def proposal(snapshot):
        return [
            first_full_mask_action(snapshot, candidate_index=0, prior=1.0),
        ]

    def value_fn(node):
        evaluated.append(node)
        return 3.0

    run_sampled_puct(
        root,
        config=SearchConfig(simulations=1, actions_per_node=1),
        proposal_fn=proposal,
        value_fn=value_fn,
    )

    assert evaluated == [root.children[0].node]
    assert evaluated[0] is not None
    assert evaluated[0].comp is not root.comp
    assert root.children[0].visits == 1
    assert root.children[0].value_sum == 3.0


def test_run_sampled_puct_terminal_return_is_empty_float32_arrays():
    class TerminalComp:
        def next_action_space(self, start_from):
            return None

    root = SearchNode(comp=TerminalComp(), start_from=0)
    result = run_sampled_puct(
        root,
        config=SearchConfig(),
        proposal_fn=lambda snapshot: [],
        value_fn=lambda node: 1.0,
    )

    assert result.selected_action is None
    assert result.valid_action_count == 0
    assert result.visit_counts.dtype == np.float32
    assert result.visit_distribution.dtype == np.float32
    assert result.visit_counts.tolist() == []
    assert result.visit_distribution.tolist() == []


def test_child_node_owns_rewritten_clone_and_start_cursor():
    comp = actionable_comp()
    node = SearchNode(comp=comp, start_from=0)
    node.expand(
        proposal_fn=lambda snapshot: [first_full_mask_action(snapshot, prior=1.0)]
    )
    child = node.children[0]

    child_node = _child_node(node, child)

    assert child_node.comp is not comp
    assert child_node.comp.snapshot() != comp.snapshot()
    assert child_node.start_from == node.action_space.def_index
    assert comp.snapshot() == node.comp.snapshot()
