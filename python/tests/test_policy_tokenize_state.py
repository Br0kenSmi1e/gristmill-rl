import jax.numpy as jnp

from gristmill_symbolics.policy import (
    SENTINEL,
    STATE_TOKEN_FIELDS,
    tokenize_state_snapshot,
)
from gristmill_symbolics.policy.constants import SEGMENT, TOKEN_KIND
from tests.policy_fixtures import actionable_state_snapshot


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
