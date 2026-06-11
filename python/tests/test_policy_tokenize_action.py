import jax.numpy as jnp

from gristmill_symbolics.policy import ACTION_TOKEN_FIELDS, tokenize_action_space_snapshot
from gristmill_symbolics.policy.constants import SIDE, TOKEN_KIND
from gristmill_symbolics.policy.tokenize import candidate_count, side_term_counts
from tests.policy_fixtures import actionable_action_space_snapshot


def test_action_space_tokenization_returns_action_fields():
    tokens, mask = tokenize_action_space_snapshot(actionable_action_space_snapshot())

    assert set(tokens) == set(ACTION_TOKEN_FIELDS)
    assert all(leaf.dtype == jnp.int32 for leaf in tokens.values())
    assert mask.dtype == jnp.bool_
    assert mask.shape == tokens["token_kind"].shape
    assert mask.all()


def test_action_space_tokenization_marks_candidates_sides_and_terms():
    tokens, _ = tokenize_action_space_snapshot(actionable_action_space_snapshot())

    kinds = tokens["token_kind"].tolist()
    assert kinds[0] == TOKEN_KIND.ACTION_SPACE_START
    assert TOKEN_KIND.CANDIDATE_START in kinds
    assert TOKEN_KIND.SIDE_START in kinds
    assert TOKEN_KIND.TERM_START in kinds
    assert TOKEN_KIND.ACTION_SPACE_END in kinds
    assert 0 in tokens["candidate_index"].tolist()
    assert SIDE.LEFT in tokens["side"].tolist()
    assert SIDE.RIGHT in tokens["side"].tolist()
    assert SIDE.REWRITTEN in tokens["side"].tolist()


def test_action_space_structural_counts_match_snapshot():
    snapshot = actionable_action_space_snapshot()
    tokens, mask = tokenize_action_space_snapshot(snapshot)
    first = snapshot["candidate_templates"][0]

    assert candidate_count(tokens, mask) == len(snapshot["candidate_templates"])
    assert side_term_counts(tokens, mask, candidate_index=0, side=SIDE.LEFT) == len(first["left_definition"]["terms"])
    assert side_term_counts(tokens, mask, candidate_index=0, side=SIDE.RIGHT) == len(first["right_definition"]["terms"])


def test_action_space_tokenization_is_deterministic():
    snapshot = actionable_action_space_snapshot()

    left_tokens, left_mask = tokenize_action_space_snapshot(snapshot)
    right_tokens, right_mask = tokenize_action_space_snapshot(snapshot)

    assert left_mask.tolist() == right_mask.tolist()
    assert {k: v.tolist() for k, v in left_tokens.items()} == {k: v.tolist() for k, v in right_tokens.items()}
