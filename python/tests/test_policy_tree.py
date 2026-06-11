import jax.numpy as jnp
import pytest

from gristmill_symbolics.policy import (
    ACTION_TOKEN_FIELDS,
    SENTINEL,
    STATE_TOKEN_FIELDS,
    PolicyConfig,
    action_choice_to_python,
    make_action_choice,
    pad_token_tree,
    stack_token_trees,
)
from gristmill_symbolics.policy.constants import TOKEN_KIND
from gristmill_symbolics.policy.tree import make_token_tree


def test_state_and_action_field_sets_are_concrete_columnar_records():
    assert STATE_TOKEN_FIELDS == (
        "token_kind",
        "segment",
        "def_index",
        "term_index",
        "factor_index",
        "tensor_id",
        "range_id",
        "index_id",
        "coeff_num",
        "coeff_den",
        "position",
    )
    assert ACTION_TOKEN_FIELDS == STATE_TOKEN_FIELDS + ("candidate_index", "side")
    assert SENTINEL == -1


def test_make_token_tree_uses_int32_leaves_and_shared_length():
    rows = [
        {"token_kind": TOKEN_KIND.RANGE, "range_id": 0, "position": 0},
        {"token_kind": TOKEN_KIND.TENSOR, "tensor_id": 7, "position": 1},
    ]

    tokens, mask = make_token_tree(rows, STATE_TOKEN_FIELDS)

    assert mask.dtype == jnp.bool_
    assert mask.tolist() == [True, True]
    assert tokens["token_kind"].dtype == jnp.int32
    assert tokens["tensor_id"].tolist() == [SENTINEL, 7]
    assert set(tokens) == set(STATE_TOKEN_FIELDS)


def test_padding_preserves_real_tokens_and_uses_safe_values():
    rows = [{"token_kind": TOKEN_KIND.RANGE, "range_id": 0, "position": 0}]
    tokens, mask = make_token_tree(rows, STATE_TOKEN_FIELDS)

    padded, padded_mask = pad_token_tree(tokens, mask, 3)

    assert padded_mask.tolist() == [True, False, False]
    assert padded["token_kind"].tolist() == [TOKEN_KIND.RANGE, TOKEN_KIND.PAD, TOKEN_KIND.PAD]
    assert padded["range_id"].tolist() == [0, SENTINEL, SENTINEL]


def test_pad_token_tree_rejects_shrinking():
    tokens, mask = make_token_tree(
        [
            {"token_kind": TOKEN_KIND.RANGE, "range_id": 0, "position": 0},
            {"token_kind": TOKEN_KIND.TENSOR, "tensor_id": 7, "position": 1},
        ],
        STATE_TOKEN_FIELDS,
    )

    with pytest.raises(ValueError, match="shorter length 1"):
        pad_token_tree(tokens, mask, 1)


def test_pad_token_tree_rejects_malformed_leaf_length_with_field_name():
    tokens, mask = make_token_tree(
        [{"token_kind": TOKEN_KIND.RANGE, "range_id": 0, "position": 0}],
        STATE_TOKEN_FIELDS,
    )
    tokens = dict(tokens)
    tokens["range_id"] = jnp.asarray([0, 1], dtype=jnp.int32)

    with pytest.raises(ValueError, match="range_id"):
        pad_token_tree(tokens, mask, 2)


def test_pad_token_tree_rejects_non_1d_mask():
    tokens, mask = make_token_tree(
        [{"token_kind": TOKEN_KIND.RANGE, "range_id": 0, "position": 0}],
        STATE_TOKEN_FIELDS,
    )

    with pytest.raises(ValueError, match="mask must be 1D"):
        pad_token_tree(tokens, mask.reshape((1, 1)), 1)


def test_pad_token_tree_rejects_non_1d_leaf_with_field_name():
    tokens, mask = make_token_tree(
        [{"token_kind": TOKEN_KIND.RANGE, "range_id": 0, "position": 0}],
        STATE_TOKEN_FIELDS,
    )
    tokens = dict(tokens)
    tokens["range_id"] = tokens["range_id"].reshape((1, 1))

    with pytest.raises(ValueError, match="range_id"):
        pad_token_tree(tokens, mask, 1)


def test_stack_token_trees_adds_sample_axis_and_pads_to_row_width():
    left, left_mask = make_token_tree(
        [{"token_kind": TOKEN_KIND.RANGE, "range_id": 0, "position": 0}],
        STATE_TOKEN_FIELDS,
    )
    right, right_mask = make_token_tree(
        [
            {"token_kind": TOKEN_KIND.RANGE, "range_id": 0, "position": 0},
            {"token_kind": TOKEN_KIND.TENSOR, "tensor_id": 1, "position": 1},
        ],
        STATE_TOKEN_FIELDS,
    )

    stacked, stacked_mask = stack_token_trees([(left, left_mask), (right, right_mask)])

    assert stacked_mask.shape == (2, 2)
    assert stacked["token_kind"].shape == (2, 2)
    assert stacked_mask.tolist() == [[True, False], [True, True]]


def test_stack_token_trees_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one token tree"):
        stack_token_trees([])


def test_stack_token_trees_rejects_mismatched_field_sets():
    left, left_mask = make_token_tree(
        [{"token_kind": TOKEN_KIND.RANGE, "range_id": 0, "position": 0}],
        STATE_TOKEN_FIELDS,
    )
    right, right_mask = make_token_tree(
        [{"token_kind": TOKEN_KIND.RANGE, "range_id": 0, "position": 0}],
        STATE_TOKEN_FIELDS,
    )
    right = dict(right)
    del right["range_id"]

    with pytest.raises(ValueError, match="field set mismatch"):
        stack_token_trees([(left, left_mask), (right, right_mask)])


def test_action_choice_tree_round_trips_to_python_padded_choice():
    choice = make_action_choice(
        candidate_index=2,
        left_mask=[True, False],
        left_valid_mask=[True, True],
        right_mask=[False, True],
        right_valid_mask=[True, True],
    )

    assert choice["candidate_index"].shape == ()
    assert action_choice_to_python(choice) == {
        "candidate_index": 2,
        "left_mask": [True, False],
        "left_valid_mask": [True, True],
        "right_mask": [False, True],
        "right_valid_mask": [True, True],
    }


def test_action_choice_rejects_different_left_and_right_widths():
    with pytest.raises(ValueError, match="left and right mask shapes differ"):
        make_action_choice(
            candidate_index=2,
            left_mask=[True],
            left_valid_mask=[True],
            right_mask=[True, False],
            right_valid_mask=[True, True],
        )


def test_action_choice_rejects_left_mask_valid_shape_mismatch():
    with pytest.raises(ValueError, match="left_mask and left_valid_mask shapes differ"):
        make_action_choice(
            candidate_index=2,
            left_mask=[True],
            left_valid_mask=[True, False],
            right_mask=[True],
            right_valid_mask=[True],
        )


def test_action_choice_rejects_right_mask_valid_shape_mismatch():
    with pytest.raises(ValueError, match="right_mask and right_valid_mask shapes differ"):
        make_action_choice(
            candidate_index=2,
            left_mask=[True],
            left_valid_mask=[True],
            right_mask=[True],
            right_valid_mask=[True, False],
        )


def test_action_choice_rejects_non_1d_side_masks():
    with pytest.raises(ValueError, match="left_mask must be 1D"):
        make_action_choice(
            candidate_index=2,
            left_mask=[[True, False]],
            left_valid_mask=[[True, True]],
            right_mask=[[False, True]],
            right_valid_mask=[[True, True]],
        )


def test_action_choice_rejects_non_scalar_candidate_index():
    with pytest.raises(ValueError, match="candidate_index must be scalar"):
        make_action_choice(
            candidate_index=[2, 3],
            left_mask=[True, False],
            left_valid_mask=[True, True],
            right_mask=[False, True],
            right_valid_mask=[True, True],
        )


def test_policy_config_defaults_match_phase_2_small_model():
    config = PolicyConfig()

    assert config.d_model == 32
    assert config.num_attention_layers == 1
    assert config.max_candidates == 32
    assert config.max_side_terms == 32
    assert config.stop_bias_init == -20.0
    assert config.id_vocab_size == 128
    assert config.init_scale == 0.02
