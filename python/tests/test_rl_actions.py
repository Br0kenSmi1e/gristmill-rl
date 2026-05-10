import numpy as np

from gristmill_rl.actions import (
    _sample_mask_from_logits,
    decision_key,
    first_full_mask_action,
    make_model_proposal_fn,
    sample_valid_actions,
    uniform_random_action,
)
from gristmill_rl.features import FeatureConfig, extract_features
from gristmill_rl.model import PolicyValueModel

from .rl_fixtures import actionable_space


def test_first_full_mask_action_applies_through_stored_space():
    comp, space = actionable_space()
    action = first_full_mask_action(space.snapshot(), 0, 1.0)
    child = comp.clone()

    child.apply_decision_with_space(space, action.decision)

    assert child.snapshot() != comp.snapshot()
    assert action.prior == 1.0


def test_uniform_random_action_has_nonempty_masks():
    _, space = actionable_space()
    rng = np.random.default_rng(3)

    action = uniform_random_action(space.snapshot(), rng, 0.25)

    assert any(action.decision["left_mask"])
    assert any(action.decision["right_mask"])
    assert action.prior == 0.25


def test_sample_mask_single_valid_term_has_conditioned_prior():
    mask, prior = _sample_mask_from_logits(
        np.asarray([0.0], dtype=np.float64),
        np.asarray([True], dtype=bool),
        np.random.default_rng(0),
    )

    assert mask == [True]
    assert prior == 1.0


def test_sample_valid_actions_deduplicates_and_validates():
    comp, space = actionable_space()
    proposal = first_full_mask_action(space.snapshot())

    actions = sample_valid_actions(comp, space, lambda: proposal, 2, 4)

    assert len(actions) == 1
    assert decision_key(actions[0].decision) == decision_key(proposal.decision)


def test_model_proposal_fn_returns_valid_unique_actions():
    comp, space = actionable_space()
    model = PolicyValueModel(hidden_dim=16, rng_seed=0)
    features = extract_features(
        comp_snapshot=comp.snapshot(),
        action_space_snapshot=space.snapshot(),
        start_from=0,
        log_total_flops=comp.log_total_flops(),
        config=FeatureConfig(max_candidates=4, max_left_terms=3, max_right_terms=3),
    )
    proposal_fn = make_model_proposal_fn(
        model=model,
        features=features,
        action_space_snapshot=space.snapshot(),
        rng=np.random.default_rng(0),
    )

    actions = sample_valid_actions(
        comp=comp,
        space=space,
        proposal_fn=proposal_fn,
        actions_per_node=2,
        sample_attempts=8,
    )

    assert actions
    assert all(action.prior > 0.0 for action in actions)


def test_model_proposal_fn_falls_back_for_unrepresented_nonempty_side():
    comp, space = actionable_space()
    model = PolicyValueModel(hidden_dim=16, rng_seed=0)
    features = extract_features(
        comp_snapshot=comp.snapshot(),
        action_space_snapshot=space.snapshot(),
        start_from=0,
        log_total_flops=comp.log_total_flops(),
        config=FeatureConfig(max_candidates=4, max_left_terms=0, max_right_terms=2),
    )
    proposal_fn = make_model_proposal_fn(
        model=model,
        features=features,
        action_space_snapshot=space.snapshot(),
        rng=np.random.default_rng(0),
    )

    actions = sample_valid_actions(
        comp=comp,
        space=space,
        proposal_fn=proposal_fn,
        actions_per_node=2,
        sample_attempts=8,
    )

    assert actions
    assert all(any(action.decision["left_mask"]) for action in actions)
    assert all(any(action.decision["right_mask"]) for action in actions)
    assert all(action.prior > 0.0 for action in actions)
