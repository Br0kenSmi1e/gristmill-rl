import pytest

from transformer_policy.tokenize import (
    build_action_space_context,
    build_state_context,
    tokenize_tensor_def,
)
from transformer_policy.types import T

from .transformer_policy_fixtures import actionable_space, actionable_state


def tensor_definition_snapshot():
    state = actionable_state()
    return state.snapshot()["definitions"][0]


def test_tokenize_tensor_def_preserves_field_order_and_raw_ids():
    tokens = tokenize_tensor_def(tensor_definition_snapshot())

    assert tokens[:15] == (
        T("DEF_START"),
        T("BASE", tensor=3),
        T("EXT_INDEX", position=0, id=0, range=0),
        T("EXT_INDEX", position=1, id=1, range=0),
        T("TERM_START", position=0),
        T("COEFF_NUM", value=1),
        T("COEFF_DEN", value=1),
        T("SUM_INDEX", position=0, id=2, range=0),
        T("FACTOR", position=0, tensor=0, arity=2),
        T("INDEX", position=0, id=0),
        T("INDEX", position=1, id=2),
        T("FACTOR", position=1, tensor=1, arity=2),
        T("INDEX", position=0, id=2),
        T("INDEX", position=1, id=1),
        T("TERM_END", position=0),
    )
    assert tokens[-1] == T("DEF_END")


@pytest.mark.parametrize("coeff", [{}, {"numer": 1}, {"denom": 1}])
def test_tokenize_tensor_def_rejects_malformed_coeff_dicts(coeff):
    definition = tensor_definition_snapshot()
    definition["terms"][0]["coeff"] = coeff

    with pytest.raises(TypeError, match="unsupported coeff shape"):
        tokenize_tensor_def(definition)


def test_build_state_context_wraps_definitions():
    state = actionable_state()

    tokens = build_state_context(state.snapshot())

    assert tokens[0] == T("STATE_START")
    assert tokens[1] == T("STATE_DEF", def_index=0)
    assert T("BASE", tensor=3) in tokens
    assert tokens[-1] == T("STATE_END")


def test_build_action_space_context_wraps_candidates_and_nested_defs():
    _, space = actionable_space()

    tokens = build_action_space_context(space.snapshot())

    assert tokens[0] == T("ACTION_SPACE_START", def_index=0)
    assert T("CAND_START", candidate_index=0) in tokens
    assert T("LEFT_DEF_START", candidate_index=0) in tokens
    assert T("RIGHT_DEF_START", candidate_index=0) in tokens
    assert T("REWRITTEN_DEF_START", candidate_index=0) in tokens
    assert tokens[-1] == T("ACTION_SPACE_END")
