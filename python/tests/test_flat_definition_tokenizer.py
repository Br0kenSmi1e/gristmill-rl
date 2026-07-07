import numpy as np
import pytest

from gristmill_symbolics.tokenizer import (
    FlatDefinitionTokenizer,
    TokenizerError,
)


def _tokenizer() -> FlatDefinitionTokenizer:
    return FlatDefinitionTokenizer(
        max_range_id=3,
        max_tensor_id=4,
        max_index_id=5,
        coeff_nums=(-1, 1, 2),
        coeff_dens=(1, 2, 3),
    )


def _definition() -> dict[str, object]:
    return {
        "base": 3,
        "ext_indices": [
            {"id": 0, "range": 0},
            {"id": 1, "range": 0},
        ],
        "terms": [
            {
                "coeff": {"numer": 1, "denom": 1},
                "sum_indices": [{"id": 2, "range": 0}],
                "factors": [
                    {"tensor": 0, "indices": [0, 2]},
                    {"tensor": 1, "indices": [2, 1]},
                ],
            },
            {
                "coeff": {"numer": -1, "denom": 2},
                "sum_indices": [],
                "factors": [{"tensor": 2, "indices": [0]}],
            },
        ],
    }


def test_token_names_are_inspectable_for_configured_vocabulary():
    tokenizer = _tokenizer()

    ids = tokenizer.encode_definition(_definition())

    assert tokenizer.pad_token_id == 0
    assert [tokenizer.token_name(token_id) for token_id in ids] == [
        "def_start",
        "tensorid3",
        "indexid0",
        "rangeid0",
        "indexid1",
        "rangeid0",
        "coeff_num1",
        "coeff_den1",
        "indexid2",
        "rangeid0",
        "tensorid0",
        "indexid0",
        "indexid2",
        "tensorid1",
        "indexid2",
        "indexid1",
        "coeff_num-1",
        "coeff_den2",
        "tensorid2",
        "indexid0",
        "def_end",
    ]


def test_definition_round_trips_through_raw_integer_tokens():
    tokenizer = _tokenizer()
    definition = _definition()

    ids = tokenizer.encode_definition(definition)
    decoded = tokenizer.decode_definition(ids)

    assert all(type(token_id) is int for token_id in ids)
    assert decoded == definition


def test_encode_rejects_unsupported_values_and_malformed_definitions():
    tokenizer = _tokenizer()
    definition = _definition()

    unsupported_tensor = dict(definition, base=9)
    with pytest.raises(TokenizerError, match="base.*outside supported range"):
        tokenizer.encode_definition(unsupported_tensor)

    unsupported_coeff = {
        **definition,
        "terms": [
            {
                **definition["terms"][0],
                "coeff": {"numer": 7, "denom": 1},
            }
        ],
    }
    with pytest.raises(TokenizerError, match="coeff_num7"):
        tokenizer.encode_definition(unsupported_coeff)

    missing_terms = {
        "base": 3,
        "ext_indices": [],
    }
    with pytest.raises(TokenizerError, match="missing key.*terms"):
        tokenizer.encode_definition(missing_terms)

    malformed_coeff = {
        **definition,
        "terms": [
            {
                **definition["terms"][0],
                "coeff": [1, 1],
            }
        ],
    }
    with pytest.raises(TokenizerError, match="coeff.*dict"):
        tokenizer.encode_definition(malformed_coeff)

    unsupported_key = {**definition, 1: "extra"}
    with pytest.raises(TokenizerError, match="unsupported key"):
        tokenizer.encode_definition(unsupported_key)


def test_encode_rejects_non_list_snapshot_fields():
    tokenizer = _tokenizer()
    definition = _definition()

    tuple_external_indices = {
        **definition,
        "ext_indices": tuple(definition["ext_indices"]),
    }
    with pytest.raises(TokenizerError, match="must be a list"):
        tokenizer.encode_definition(tuple_external_indices)

    range_factor_indices = {
        **definition,
        "terms": [
            {
                **definition["terms"][0],
                "factors": [
                    {
                        **definition["terms"][0]["factors"][0],
                        "indices": range(2),
                    }
                ],
            }
        ],
    }
    with pytest.raises(TokenizerError, match="must be a list"):
        tokenizer.encode_definition(range_factor_indices)


def test_constructor_rejects_non_iterable_coefficient_vocabulary():
    with pytest.raises(TokenizerError, match="coeff_nums.*sequence"):
        FlatDefinitionTokenizer(
            max_range_id=3,
            max_tensor_id=4,
            max_index_id=5,
            coeff_nums=1,
            coeff_dens=(1,),
        )


def test_decode_rejects_malformed_raw_streams():
    tokenizer = _tokenizer()
    valid = tokenizer.encode_definition(_definition())

    with pytest.raises(TokenizerError, match="sequence of integer IDs"):
        tokenizer.decode_definition({valid[0]: None})

    with pytest.raises(TokenizerError, match="def_start"):
        tokenizer.decode_definition(valid[1:])

    with pytest.raises(TokenizerError, match="raw token stream cannot contain pad"):
        tokenizer.decode_definition([tokenizer.pad_token_id, *valid])

    with pytest.raises(TokenizerError, match="unknown token id"):
        tokenizer.decode_definition([10_000])

    missing_external_range = valid.copy()
    del missing_external_range[3]
    with pytest.raises(TokenizerError, match="external index rangeid"):
        tokenizer.decode_definition(missing_external_range)

    denominator_where_term_should_start = valid.copy()
    denominator_where_term_should_start[6] = valid[7]
    with pytest.raises(TokenizerError, match="coeff_num or def_end"):
        tokenizer.decode_definition(denominator_where_term_should_start)


def test_padded_encode_uses_right_padding_and_round_trips():
    tokenizer = _tokenizer()
    definition = _definition()
    raw = tokenizer.encode_definition(definition)

    ids, mask = tokenizer.encode_definition_padded(definition, max_len=len(raw) + 3)

    assert ids.dtype == np.int32
    assert mask.dtype == np.bool_
    np.testing.assert_array_equal(ids[: len(raw)], np.asarray(raw, dtype=np.int32))
    np.testing.assert_array_equal(
        ids[len(raw) :],
        np.full((3,), tokenizer.pad_token_id, dtype=np.int32),
    )
    np.testing.assert_array_equal(mask[: len(raw)], np.ones((len(raw),), dtype=np.bool_))
    np.testing.assert_array_equal(mask[len(raw) :], np.zeros((3,), dtype=np.bool_))
    assert tokenizer.decode_definition_padded(ids, mask) == definition


def test_padded_encode_rejects_overlong_definition():
    tokenizer = _tokenizer()
    definition = _definition()
    raw = tokenizer.encode_definition(definition)

    with pytest.raises(TokenizerError, match="exceeds max_len"):
        tokenizer.encode_definition_padded(definition, max_len=len(raw) - 1)


def test_padded_decode_rejects_invalid_padding_and_mask():
    tokenizer = _tokenizer()
    ids, mask = tokenizer.encode_definition_padded(_definition(), max_len=32)

    non_right_padded_mask = mask.copy()
    non_right_padded_mask[0] = False
    non_right_padded_mask[1] = True
    with pytest.raises(TokenizerError, match="right-padding"):
        tokenizer.decode_definition_padded(ids, non_right_padded_mask)

    non_pad_tail = ids.copy()
    non_pad_tail[-1] = ids[0]
    with pytest.raises(TokenizerError, match="pad_token_id"):
        tokenizer.decode_definition_padded(non_pad_tail, mask)

    with pytest.raises(TokenizerError, match="same shape"):
        tokenizer.decode_definition_padded(ids, mask[:-1])

    with pytest.raises(TokenizerError, match="boolean"):
        tokenizer.decode_definition_padded(ids, mask.astype(np.int32))
