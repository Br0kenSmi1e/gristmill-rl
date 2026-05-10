import numpy as np
import pytest

from gristmill_rl.actions import first_full_mask_action
from gristmill_rl.features import FeatureConfig, extract_features
from gristmill_rl.model import PolicyValueModel, TrainConfig, action_log_prob, train_step

from .rl_fixtures import actionable_space


def model_features():
    comp, space = actionable_space()
    return (
        extract_features(
            comp_snapshot=comp.snapshot(),
            action_space_snapshot=space.snapshot(),
            start_from=0,
            log_total_flops=comp.log_total_flops(),
            config=FeatureConfig(max_candidates=4, max_left_terms=1, max_right_terms=2),
        ),
        first_full_mask_action(space.snapshot()),
    )


def model_features_with_config(config: FeatureConfig, candidate_index: int = 0):
    comp, space = actionable_space()
    return (
        extract_features(
            comp_snapshot=comp.snapshot(),
            action_space_snapshot=space.snapshot(),
            start_from=0,
            log_total_flops=comp.log_total_flops(),
            config=config,
        ),
        first_full_mask_action(space.snapshot(), candidate_index=candidate_index),
    )


def test_model_forward_shapes():
    features, _ = model_features()
    model = PolicyValueModel(hidden_dim=16, rng_seed=0)

    outputs = model(features)

    assert outputs.candidate_logits.shape == (4,)
    assert outputs.left_logits.shape == (4, 1)
    assert outputs.right_logits.shape == (4, 2)
    assert outputs.value.shape == ()


def test_action_log_prob_is_finite():
    features, action = model_features()
    model = PolicyValueModel(hidden_dim=16, rng_seed=0)

    value = action_log_prob(model, features, action)

    assert np.isfinite(float(value))


def test_action_log_prob_rejects_unrepresented_candidate():
    features, action = model_features_with_config(
        FeatureConfig(max_candidates=1, max_left_terms=3, max_right_terms=3),
        candidate_index=1,
    )
    model = PolicyValueModel(hidden_dim=16, rng_seed=0)

    with pytest.raises(ValueError, match="candidate_index"):
        action_log_prob(model, features, action)


def test_action_log_prob_rejects_unrepresented_term_mask():
    features, action = model_features_with_config(
        FeatureConfig(max_candidates=4, max_left_terms=0, max_right_terms=2)
    )
    model = PolicyValueModel(hidden_dim=16, rng_seed=0)

    with pytest.raises(ValueError, match="mask length"):
        action_log_prob(model, features, action)


def test_train_step_changes_a_parameter():
    features, action = model_features()
    model = PolicyValueModel(hidden_dim=16, rng_seed=0)
    config = TrainConfig(learning_rate=1e-2)

    metrics = train_step(
        model,
        batch=[
            {
                "features": features,
                "actions": [action],
                "policy_target": np.asarray([1.0], dtype=np.float32),
                "value_target": 0.5,
            }
        ],
        config=config,
    )

    assert metrics["total_loss"] > 0.0
    assert np.isfinite(metrics["policy_loss"])
    assert np.isfinite(metrics["value_loss"])
    assert np.isfinite(metrics["total_loss"])
    assert metrics["params_changed"]
