import numpy as np

from gristmill_rl.features import FeatureConfig, extract_features

from .rl_fixtures import actionable_space


def test_extract_features_produces_padded_shapes_and_masks():
    comp, space = actionable_space()
    config = FeatureConfig(max_candidates=4, max_left_terms=3, max_right_terms=3)

    features = extract_features(
        comp_snapshot=comp.snapshot(),
        action_space_snapshot=space.snapshot(),
        start_from=0,
        log_total_flops=comp.log_total_flops(),
        config=config,
    )

    assert features.state.shape == (8,)
    assert features.candidates.shape == (4, 6)
    assert features.left_terms.shape == (4, 3, 4)
    assert features.right_terms.shape == (4, 3, 4)
    assert features.candidate_mask.shape == (4,)
    assert features.left_term_mask.shape == (4, 3)
    assert features.right_term_mask.shape == (4, 3)
    assert features.candidate_mask.dtype == np.bool_
    assert features.candidate_mask[0]
    np.testing.assert_array_equal(features.candidate_mask, [True, True, False, False])
    np.testing.assert_array_equal(features.candidates[2:], np.zeros((2, 6), dtype=np.float32))
    np.testing.assert_array_equal(
        features.left_term_mask[2:], np.zeros((2, 3), dtype=np.bool_)
    )
    np.testing.assert_array_equal(
        features.right_term_mask[2:], np.zeros((2, 3), dtype=np.bool_)
    )
    np.testing.assert_array_equal(
        features.left_terms[2:], np.zeros((2, 3, 4), dtype=np.float32)
    )
    np.testing.assert_array_equal(
        features.right_terms[2:], np.zeros((2, 3, 4), dtype=np.float32)
    )
    np.testing.assert_array_equal(features.state[:4], [1.0, 4.0, 1.0, 0.0])
    assert features.left_term_mask[0].any()
    assert features.right_term_mask[0].any()


def test_extract_features_counts_truncation():
    comp, space = actionable_space()
    config = FeatureConfig(max_candidates=1, max_left_terms=1, max_right_terms=1)

    features = extract_features(
        comp_snapshot=comp.snapshot(),
        action_space_snapshot=space.snapshot(),
        start_from=0,
        log_total_flops=comp.log_total_flops(),
        config=config,
    )

    assert features.candidates.shape == (1, 6)
    assert features.truncation.candidates == 1
    assert features.truncation.left_terms == 0
    assert features.truncation.right_terms == 1
