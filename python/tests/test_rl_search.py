import numpy as np

from gristmill_rl.actions import SampledAction, first_full_mask_action
from gristmill_rl.search import (
    SearchChild,
    SearchConfig,
    SearchNode,
    _child_node,
    puct_score,
    run_sampled_puct,
)

from .rl_fixtures import actionable_comp


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


def test_puct_score_prefers_unvisited_child_with_prior():
    low_prior = SearchChild(
        action=SampledAction(
            {"candidate_index": 0, "left_mask": [True], "right_mask": [True]},
            prior=0.1,
        ),
        prior=0.1,
        visit_count=5,
        total_value=5.0,
    )
    high_prior = SearchChild(
        action=SampledAction(
            {"candidate_index": 1, "left_mask": [True], "right_mask": [True]},
            prior=0.9,
        ),
        prior=0.9,
    )

    assert puct_score(high_prior, parent_visit_count=5, c_puct=1.5) > puct_score(
        low_prior,
        parent_visit_count=5,
        c_puct=1.5,
    )


def test_puct_score_uses_max_parent_visit_count_formula():
    child = SearchChild(
        action=SampledAction(
            {"candidate_index": 0, "left_mask": [True], "right_mask": [True]},
            prior=0.5,
        ),
        prior=0.5,
    )

    assert puct_score(child, parent_visit_count=4, c_puct=1.5) == 1.5


def test_run_sampled_puct_visit_distribution_sums_to_one():
    comp = actionable_comp()

    def proposal(snapshot):
        return [
            first_full_mask_action(snapshot, candidate_index=0, prior=0.25),
            first_full_mask_action(snapshot, candidate_index=1, prior=0.75),
        ]

    result = run_sampled_puct(
        comp,
        start_from=0,
        proposal_fn=proposal,
        config=SearchConfig(simulations=6, actions_per_node=2),
    )

    assert len(result.sampled_actions) == 2
    assert result.visit_distribution.dtype == np.float32
    np.testing.assert_allclose(result.visit_distribution.sum(), 1.0)


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
