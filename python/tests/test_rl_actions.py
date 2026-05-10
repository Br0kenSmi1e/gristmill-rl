import numpy as np

from gristmill_rl.actions import (
    decision_key,
    first_full_mask_action,
    sample_valid_actions,
    uniform_random_action,
)

from .rl_fixtures import actionable_space


def test_first_full_mask_action_applies_through_stored_space():
    comp, space = actionable_space()
    action = first_full_mask_action(space.snapshot())
    child = comp.clone()

    child.apply_decision_with_space(space, action.decision)

    assert child.snapshot() != comp.snapshot()
    assert action.prior == 1.0


def test_uniform_random_action_has_nonempty_masks():
    _, space = actionable_space()
    rng = np.random.default_rng(3)

    action = uniform_random_action(space.snapshot(), rng=rng, prior=0.25)

    assert any(action.decision["left_mask"])
    assert any(action.decision["right_mask"])
    assert action.prior == 0.25


def test_sample_valid_actions_deduplicates_and_validates():
    comp, space = actionable_space()
    proposal = first_full_mask_action(space.snapshot())

    actions = sample_valid_actions(
        comp=comp,
        space=space,
        proposal_fn=lambda: proposal,
        actions_per_node=2,
        sample_attempts=4,
    )

    assert len(actions) == 1
    assert decision_key(actions[0].decision) == decision_key(proposal.decision)
