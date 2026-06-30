import jax.numpy as jnp
import pytest

from gristmill_symbolics.model.tokenizer import (
    SIDE,
    SYM_ACTION,
    TOKEN_KIND,
    decode_action_space_snapshot,
    decode_computation_snapshot,
    tokenize_action_space_snapshot,
    tokenize_computation_snapshot,
)


def _computation_snapshot():
    return {
        "ranges": [{"id": 0, "size": 4}, {"id": 1, "size": 6}],
        "tensors": [
            {
                "id": 0,
                "symmetry": [
                    {"perm": [1, 0], "action": "Negate"},
                    {"perm": [0, 1], "action": "Identity"},
                ],
            },
            {"id": 1, "symmetry": []},
        ],
        "definitions": [
            {
                "base": 1,
                "ext_indices": [{"id": 0, "range": 0}],
                "terms": [
                    {
                        "coeff": {"numer": 2, "denom": 3},
                        "sum_indices": [{"id": 1, "range": 1}],
                        "factors": [{"tensor": 0, "indices": [0, 1]}],
                    }
                ],
            }
        ],
    }


def _action_space_snapshot():
    definition = _computation_snapshot()["definitions"][0]
    return {
        "def_index": 0,
        "candidate_templates": [
            {
                "left_definition": definition,
                "right_definition": {
                    "base": 1,
                    "ext_indices": [{"id": 0, "range": 0}],
                    "terms": [
                        {
                            "coeff": {"numer": 1, "denom": 1},
                            "sum_indices": [],
                            "factors": [{"tensor": 0, "indices": [0]}],
                        }
                    ],
                },
            }
        ],
    }


def _kinds(tokens):
    return tokens["token_kind"].tolist()


def test_computation_tokenization_round_trips_symmetry_snapshot():
    snapshot = _computation_snapshot()

    tokens, mask = tokenize_computation_snapshot(snapshot)

    assert all(leaf.dtype == jnp.int32 for leaf in tokens.values())
    assert mask.dtype == jnp.bool_
    assert mask.all()
    assert decode_computation_snapshot(tokens, mask) == snapshot


def test_computation_tokens_include_tensor_symmetry_structure():
    tokens, _ = tokenize_computation_snapshot(_computation_snapshot())
    kinds = _kinds(tokens)

    assert kinds[:2] == [TOKEN_KIND.RANGE, TOKEN_KIND.RANGE]
    assert TOKEN_KIND.TENSOR_START in kinds
    assert TOKEN_KIND.SYMMETRY_START in kinds
    assert TOKEN_KIND.SYMMETRY_PERM in kinds
    assert TOKEN_KIND.SYMMETRY_END in kinds
    assert TOKEN_KIND.TENSOR_END in kinds
    assert SYM_ACTION.NEGATE in tokens["symmetry_action"].tolist()


def test_action_space_tokenization_round_trips_without_rewritten_side():
    snapshot = _action_space_snapshot()

    tokens, mask = tokenize_action_space_snapshot(snapshot)

    assert decode_action_space_snapshot(tokens, mask) == snapshot
    assert SIDE.LEFT in tokens["side"].tolist()
    assert SIDE.RIGHT in tokens["side"].tolist()
    assert "rewritten_definition" not in snapshot["candidate_templates"][0]


def test_decode_rejects_truncated_definition():
    snapshot = _computation_snapshot()
    tokens, mask = tokenize_computation_snapshot(snapshot)
    tokens = {field: values[:-1] for field, values in tokens.items()}
    mask = mask[:-1]

    with pytest.raises(ValueError, match="expected"):
        decode_computation_snapshot(tokens, mask)


def test_decode_rejects_mismatched_tensor_end_scope():
    tokens, mask = tokenize_computation_snapshot(_computation_snapshot())
    position = tokens["token_kind"].tolist().index(TOKEN_KIND.TENSOR_END)
    tokens = dict(tokens)
    tokens["tensor_id"] = tokens["tensor_id"].at[position].set(99)

    with pytest.raises(ValueError, match="tensor_id"):
        decode_computation_snapshot(tokens, mask)


def test_decode_rejects_mismatched_action_space_end_scope():
    tokens, mask = tokenize_action_space_snapshot(_action_space_snapshot())
    tokens = dict(tokens)
    tokens["def_index"] = tokens["def_index"].at[-1].set(99)

    with pytest.raises(ValueError, match="def_index"):
        decode_action_space_snapshot(tokens, mask)


def test_decode_rejects_wrong_definition_index():
    tokens, mask = tokenize_computation_snapshot(_computation_snapshot())
    tokens = dict(tokens)
    tokens["def_index"] = _replace_value(tokens["def_index"], 0, 99)

    with pytest.raises(ValueError, match="def_index"):
        decode_computation_snapshot(tokens, mask)


def test_decode_rejects_wrong_term_index():
    tokens, mask = tokenize_computation_snapshot(_computation_snapshot())
    tokens = dict(tokens)
    tokens["term_index"] = _replace_value(tokens["term_index"], 0, 99)

    with pytest.raises(ValueError, match="term_index"):
        decode_computation_snapshot(tokens, mask)


def test_decode_rejects_wrong_factor_index():
    tokens, mask = tokenize_computation_snapshot(_computation_snapshot())
    tokens = dict(tokens)
    tokens["factor_index"] = _replace_value(tokens["factor_index"], 0, 99)

    with pytest.raises(ValueError, match="factor_index"):
        decode_computation_snapshot(tokens, mask)


def test_decode_rejects_wrong_candidate_index():
    tokens, mask = tokenize_action_space_snapshot(_action_space_snapshot())
    tokens = dict(tokens)
    tokens["candidate_index"] = _replace_value(tokens["candidate_index"], 0, 99)

    with pytest.raises(ValueError, match="candidate_index"):
        decode_action_space_snapshot(tokens, mask)


def test_decode_rejects_wrong_segment():
    tokens, mask = tokenize_computation_snapshot(_computation_snapshot())
    tokens = dict(tokens)
    tokens["segment"] = tokens["segment"].at[0].set(3)

    with pytest.raises(ValueError, match="segment"):
        decode_computation_snapshot(tokens, mask)


def test_decode_rejects_wrong_position():
    tokens, mask = tokenize_computation_snapshot(_computation_snapshot())
    tokens = dict(tokens)
    tokens["position"] = tokens["position"].at[0].set(99)

    with pytest.raises(ValueError, match="position"):
        decode_computation_snapshot(tokens, mask)


def test_decode_rejects_irrelevant_fields():
    tokens, mask = tokenize_computation_snapshot(_computation_snapshot())
    tokens = dict(tokens)
    tokens["candidate_index"] = tokens["candidate_index"].at[0].set(123)

    with pytest.raises(ValueError, match="candidate_index"):
        decode_computation_snapshot(tokens, mask)


def test_decode_rejects_non_pad_masked_out_rows():
    snapshot = _computation_snapshot()
    tokens, mask = tokenize_computation_snapshot(snapshot)
    tokens = {
        field: values.at[-1].set(values[-2])
        for field, values in tokens.items()
    }
    mask = mask.at[-1].set(False)

    with pytest.raises(ValueError, match="PAD"):
        decode_computation_snapshot(tokens, mask)


def test_decode_rejects_non_empty_pad_rows():
    snapshot = _computation_snapshot()
    tokens, mask = tokenize_computation_snapshot(snapshot)
    tokens = dict(tokens)
    tokens["token_kind"] = tokens["token_kind"].at[-1].set(TOKEN_KIND.PAD)
    tokens["range_id"] = tokens["range_id"].at[-1].set(7)
    mask = mask.at[-1].set(False)

    with pytest.raises(ValueError, match="PAD"):
        decode_computation_snapshot(tokens, mask)


def test_decode_rejects_unknown_token_kind():
    tokens, mask = tokenize_computation_snapshot(_computation_snapshot())
    tokens = dict(tokens)
    tokens["token_kind"] = tokens["token_kind"].at[0].set(999)

    with pytest.raises(ValueError, match="unknown token kind"):
        decode_computation_snapshot(tokens, mask)


def _replace_value(values, old: int, new: int):
    return jnp.where(values == old, jnp.asarray(new, values.dtype), values)
