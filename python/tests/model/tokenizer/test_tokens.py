import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gristmill_symbolics.model.tokenizer import (
    SENTINEL,
    TOKEN_FIELDS,
    TOKEN_KIND,
    make_token_arrays,
    pad_token_arrays,
    stack_token_arrays,
)


def test_token_fields_are_concrete_columnar_records():
    assert TOKEN_FIELDS == (
        "token_kind",
        "segment",
        "position",
        "def_index",
        "term_index",
        "factor_index",
        "tensor_id",
        "range_id",
        "index_id",
        "candidate_index",
        "side",
        "coeff_num",
        "coeff_den",
        "symmetry_index",
        "symmetry_action",
        "perm_index",
        "perm_value",
    )
    assert SENTINEL == -1


def test_make_token_arrays_uses_int32_leaves_and_bool_mask():
    rows = [
        {"token_kind": TOKEN_KIND.RANGE, "range_id": 0, "position": 0},
        {"token_kind": TOKEN_KIND.TENSOR_START, "tensor_id": 7},
    ]

    tokens, mask = make_token_arrays(rows)

    assert mask.dtype == jnp.bool_
    assert mask.tolist() == [True, True]
    assert set(tokens) == set(TOKEN_FIELDS)
    assert all(values.dtype == jnp.int32 for values in tokens.values())
    assert tokens["position"].tolist() == [0, 1]
    assert tokens["tensor_id"].tolist() == [SENTINEL, 7]


def test_make_token_arrays_keeps_row_tokenization_host_side():
    rows = [{"token_kind": TOKEN_KIND.RANGE, "range_id": 0}]

    tokens, mask = make_token_arrays(rows)

    assert isinstance(mask, np.ndarray)
    assert all(isinstance(values, np.ndarray) for values in tokens.values())


def test_padding_preserves_real_tokens_and_uses_safe_values():
    tokens, mask = make_token_arrays(
        [{"token_kind": TOKEN_KIND.RANGE, "range_id": 0}]
    )

    padded, padded_mask = pad_token_arrays(tokens, mask, 3)

    assert padded_mask.tolist() == [True, False, False]
    assert padded["token_kind"].tolist() == [
        TOKEN_KIND.RANGE,
        TOKEN_KIND.PAD,
        TOKEN_KIND.PAD,
    ]
    assert padded["range_id"].tolist() == [0, SENTINEL, SENTINEL]


def test_pad_token_arrays_rejects_shrinking():
    tokens, mask = make_token_arrays(
        [
            {"token_kind": TOKEN_KIND.RANGE, "range_id": 0},
            {"token_kind": TOKEN_KIND.TENSOR_START, "tensor_id": 7},
        ]
    )

    with pytest.raises(ValueError, match="shorter length 1"):
        pad_token_arrays(tokens, mask, 1)


def test_pad_token_arrays_rejects_malformed_leaf_length():
    tokens, mask = make_token_arrays(
        [{"token_kind": TOKEN_KIND.RANGE, "range_id": 0}]
    )
    tokens = dict(tokens)
    tokens["range_id"] = jnp.asarray([0, 1], dtype=jnp.int32)

    with pytest.raises(ValueError, match="range_id"):
        pad_token_arrays(tokens, mask, 2)


def test_pad_token_arrays_rejects_missing_fields():
    tokens, mask = make_token_arrays(
        [{"token_kind": TOKEN_KIND.RANGE, "range_id": 0}]
    )
    tokens = dict(tokens)
    del tokens["range_id"]

    with pytest.raises(ValueError, match="field set mismatch"):
        pad_token_arrays(tokens, mask, 2)


def test_pad_token_arrays_rejects_extra_fields():
    tokens, mask = make_token_arrays(
        [{"token_kind": TOKEN_KIND.RANGE, "range_id": 0}]
    )
    tokens = dict(tokens)
    tokens["extra"] = jnp.asarray([0], dtype=jnp.int32)

    with pytest.raises(ValueError, match="field set mismatch"):
        pad_token_arrays(tokens, mask, 2)


def test_pad_token_arrays_rejects_non_int32_leaves():
    tokens, mask = make_token_arrays(
        [{"token_kind": TOKEN_KIND.RANGE, "range_id": 0}]
    )
    tokens = dict(tokens)
    tokens["range_id"] = jnp.asarray([0.0], dtype=jnp.float32)

    with pytest.raises(ValueError, match="range_id"):
        pad_token_arrays(tokens, mask, 2)


def test_pad_token_arrays_rejects_non_bool_mask():
    tokens, mask = make_token_arrays(
        [{"token_kind": TOKEN_KIND.RANGE, "range_id": 0}]
    )
    mask = jnp.asarray([1], dtype=jnp.int32)

    with pytest.raises(ValueError, match="mask"):
        pad_token_arrays(tokens, mask, 2)


def test_stack_token_arrays_adds_batch_axis_and_pads_to_width():
    left, left_mask = make_token_arrays(
        [{"token_kind": TOKEN_KIND.RANGE, "range_id": 0}]
    )
    right, right_mask = make_token_arrays(
        [
            {"token_kind": TOKEN_KIND.RANGE, "range_id": 0},
            {"token_kind": TOKEN_KIND.TENSOR_START, "tensor_id": 1},
        ]
    )

    stacked, stacked_mask = stack_token_arrays(
        [(left, left_mask), (right, right_mask)]
    )

    assert stacked_mask.shape == (2, 2)
    assert stacked["token_kind"].shape == (2, 2)
    assert isinstance(stacked_mask, jax.Array)
    assert all(isinstance(values, jax.Array) for values in stacked.values())
    assert stacked_mask.tolist() == [[True, False], [True, True]]


def test_stack_token_arrays_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one token array set"):
        stack_token_arrays([])


def test_stack_token_arrays_rejects_mismatched_field_sets():
    left, left_mask = make_token_arrays(
        [{"token_kind": TOKEN_KIND.RANGE, "range_id": 0}]
    )
    right, right_mask = make_token_arrays(
        [{"token_kind": TOKEN_KIND.RANGE, "range_id": 0}]
    )
    right = dict(right)
    del right["range_id"]

    with pytest.raises(ValueError, match="field set mismatch"):
        stack_token_arrays([(left, left_mask), (right, right_mask)])
