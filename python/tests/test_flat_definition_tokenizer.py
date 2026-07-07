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


def test_decode_rejects_malformed_raw_streams():
    tokenizer = _tokenizer()
    valid = tokenizer.encode_definition(_definition())

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
