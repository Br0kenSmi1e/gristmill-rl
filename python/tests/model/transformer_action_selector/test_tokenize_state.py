import jax.numpy as jnp

from gristmill_symbolics.model.transformer_action_selector.constants import (
    SENTINEL,
    STATE_TOKEN_FIELDS,
    SEGMENT,
    TOKEN_KIND,
)
from gristmill_symbolics.model.transformer_action_selector.tokenize import (
    tokenize_state_snapshot,
)
from tests.policy_fixtures import actionable_state_snapshot


def _selected_row_values(tokens, fields):
    return [
        tuple(tokens[field][position].item() for field in fields)
        for position in range(int(tokens["token_kind"].shape[0]))
    ]


def test_state_tokenization_returns_complete_jax_token_tree():
    tokens, mask = tokenize_state_snapshot(actionable_state_snapshot())

    assert set(tokens) == set(STATE_TOKEN_FIELDS)
    assert all(leaf.dtype == jnp.int32 for leaf in tokens.values())
    assert mask.dtype == jnp.bool_
    assert mask.shape == tokens["token_kind"].shape
    assert mask.all()


def test_state_tokenization_is_deterministic_and_positioned():
    snapshot = actionable_state_snapshot()

    left_tokens, left_mask = tokenize_state_snapshot(snapshot)
    right_tokens, right_mask = tokenize_state_snapshot(snapshot)

    assert left_mask.tolist() == right_mask.tolist()
    assert {k: v.tolist() for k, v in left_tokens.items()} == {
        k: v.tolist() for k, v in right_tokens.items()
    }
    assert left_tokens["position"].tolist() == list(range(int(left_mask.shape[0])))


def test_state_tokenization_preserves_snapshot_order_and_structure():
    tokens, _ = tokenize_state_snapshot(actionable_state_snapshot())
    kinds = tokens["token_kind"].tolist()

    assert kinds[0] == TOKEN_KIND.RANGE
    assert kinds[1:5] == [
        TOKEN_KIND.TENSOR,
        TOKEN_KIND.TENSOR,
        TOKEN_KIND.TENSOR,
        TOKEN_KIND.TENSOR,
    ]
    assert TOKEN_KIND.DEF_START in kinds
    assert TOKEN_KIND.TERM_START in kinds
    assert TOKEN_KIND.COEFF in kinds
    assert TOKEN_KIND.FACTOR_START in kinds
    assert TOKEN_KIND.FACTOR_INDEX in kinds
    assert TOKEN_KIND.DEF_END in kinds


def test_state_tokenization_uses_scoped_ids_and_sentinels():
    tokens, _ = tokenize_state_snapshot(actionable_state_snapshot())

    factor_index_positions = [
        i for i, kind in enumerate(tokens["token_kind"].tolist())
        if kind == TOKEN_KIND.FACTOR_INDEX
    ]
    factor_indices = [tokens["index_id"][i].item() for i in factor_index_positions]

    assert 0 in factor_indices
    assert 2 in factor_indices
    assert "candidate_index" not in tokens
    assert tokens["def_index"][0].item() == SENTINEL
    assert all(
        value == SEGMENT.DEFINITIONS
        for value, kind in zip(tokens["segment"].tolist(), tokens["token_kind"].tolist())
        if kind == TOKEN_KIND.DEF_START
    )


def test_state_tokenization_scopes_factor_index_ranges_per_definition():
    snapshot = {
        "ranges": [{"id": 0, "size": 3}, {"id": 1, "size": 5}],
        "tensors": [{"id": 0, "symmetry": []}, {"id": 1, "symmetry": []}],
        "definitions": [
            {
                "base": 0,
                "ext_indices": [{"id": 0, "range": 0}],
                "terms": [
                    {
                        "coeff": {"numer": 1, "denom": 1},
                        "sum_indices": [],
                        "factors": [{"tensor": 0, "indices": [0]}],
                    }
                ],
            },
            {
                "base": 1,
                "ext_indices": [{"id": 0, "range": 1}],
                "terms": [
                    {
                        "coeff": {"numer": 1, "denom": 1},
                        "sum_indices": [],
                        "factors": [{"tensor": 1, "indices": [0]}],
                    }
                ],
            },
        ],
    }

    tokens, _ = tokenize_state_snapshot(snapshot)
    factor_ranges_by_def = {
        tokens["def_index"][i].item(): tokens["range_id"][i].item()
        for i, kind in enumerate(tokens["token_kind"].tolist())
        if kind == TOKEN_KIND.FACTOR_INDEX
    }

    assert factor_ranges_by_def == {0: 0, 1: 1}


def test_state_tokenization_matches_compact_golden_row_sequence():
    snapshot = {
        "ranges": [{"id": 0, "size": 4}, {"id": 1, "size": 6}],
        "tensors": [{"id": 0, "symmetry": []}, {"id": 1, "symmetry": []}],
        "definitions": [
            {
                "base": 1,
                "ext_indices": [{"id": 0, "range": 0}],
                "terms": [
                    {
                        "coeff": [2, 3],
                        "sum_indices": [{"id": 1, "range": 1}],
                        "factors": [{"tensor": 0, "indices": [0, 1]}],
                    }
                ],
            }
        ],
    }

    tokens, _ = tokenize_state_snapshot(snapshot)
    fields = (
        "token_kind",
        "term_index",
        "factor_index",
        "tensor_id",
        "range_id",
        "coeff_num",
        "coeff_den",
    )

    assert _selected_row_values(tokens, fields) == [
        (TOKEN_KIND.RANGE, SENTINEL, SENTINEL, SENTINEL, 0, 4, 1),
        (TOKEN_KIND.RANGE, SENTINEL, SENTINEL, SENTINEL, 1, 6, 1),
        (TOKEN_KIND.TENSOR, SENTINEL, SENTINEL, 0, SENTINEL, SENTINEL, SENTINEL),
        (TOKEN_KIND.TENSOR, SENTINEL, SENTINEL, 1, SENTINEL, SENTINEL, SENTINEL),
        (TOKEN_KIND.DEF_START, SENTINEL, SENTINEL, 1, SENTINEL, SENTINEL, SENTINEL),
        (TOKEN_KIND.EXT_INDEX, SENTINEL, SENTINEL, SENTINEL, 0, SENTINEL, SENTINEL),
        (TOKEN_KIND.TERM_START, 0, SENTINEL, SENTINEL, SENTINEL, SENTINEL, SENTINEL),
        (TOKEN_KIND.COEFF, 0, SENTINEL, SENTINEL, SENTINEL, 2, 3),
        (TOKEN_KIND.SUM_INDEX, 0, SENTINEL, SENTINEL, 1, SENTINEL, SENTINEL),
        (TOKEN_KIND.FACTOR_START, 0, 0, 0, SENTINEL, SENTINEL, SENTINEL),
        (TOKEN_KIND.FACTOR_INDEX, 0, 0, SENTINEL, 0, SENTINEL, SENTINEL),
        (TOKEN_KIND.FACTOR_INDEX, 0, 0, SENTINEL, 1, SENTINEL, SENTINEL),
        (TOKEN_KIND.FACTOR_END, 0, 0, SENTINEL, SENTINEL, SENTINEL, SENTINEL),
        (TOKEN_KIND.TERM_END, 0, SENTINEL, SENTINEL, SENTINEL, SENTINEL, SENTINEL),
        (TOKEN_KIND.DEF_END, SENTINEL, SENTINEL, SENTINEL, SENTINEL, SENTINEL, SENTINEL),
    ]
