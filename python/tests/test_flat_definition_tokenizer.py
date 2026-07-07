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


def _second_definition() -> dict[str, object]:
    return {
        "base": 2,
        "ext_indices": [{"id": 0, "range": 0}],
        "terms": [
            {
                "coeff": {"numer": 2, "denom": 3},
                "sum_indices": [],
                "factors": [{"tensor": 4, "indices": [0]}],
            },
        ],
    }


class _IntLike:
    def __init__(self, value: int):
        self.value = value

    def __index__(self) -> int:
        return self.value

    def __add__(self, value: int) -> int:
        return self.value + value


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


def test_constructor_preserves_configuration_inputs():
    max_range_id = _IntLike(3)
    max_tensor_id = _IntLike(4)
    max_index_id = _IntLike(5)
    coeff_nums = [-1, 1, 2]
    coeff_dens = [1, 2, 3]

    tokenizer = FlatDefinitionTokenizer(
        max_range_id=max_range_id,
        max_tensor_id=max_tensor_id,
        max_index_id=max_index_id,
        coeff_nums=coeff_nums,
        coeff_dens=coeff_dens,
    )

    assert tokenizer.max_range_id is max_range_id
    assert tokenizer.max_tensor_id is max_tensor_id
    assert tokenizer.max_index_id is max_index_id
    assert tokenizer.coeff_nums is coeff_nums
    assert tokenizer.coeff_dens is coeff_dens


def test_tokenizer_exposes_raw_definition_api_only():
    tokenizer = _tokenizer()

    assert not hasattr(tokenizer, "encode_definition_padded")
    assert not hasattr(tokenizer, "decode_definition_padded")


def test_definition_round_trips_through_raw_integer_tokens():
    tokenizer = _tokenizer()
    definition = _definition()

    ids = tokenizer.encode_definition(definition)
    decoded = tokenizer.decode_definition(ids)

    assert all(type(token_id) is int for token_id in ids)
    assert decoded == definition


def test_definition_sequence_round_trips_as_concatenated_raw_tokens():
    tokenizer = _tokenizer()
    definitions = [_definition(), _second_definition()]

    ids = tokenizer.encode_definitions(definitions)

    assert ids == [
        *tokenizer.encode_definition(definitions[0]),
        *tokenizer.encode_definition(definitions[1]),
    ]
    assert tokenizer.decode_definitions(ids) == definitions
    assert tokenizer.encode_definitions([]) == []
    assert tokenizer.decode_definitions([]) == []


def test_encode_rejects_values_outside_configured_vocabulary():
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


def test_encode_accepts_snapshot_like_iterables_and_ignores_extra_keys():
    tokenizer = _tokenizer()
    definition = {
        "base": 3,
        "ext_indices": (
            {"id": 0, "range": 0, "unused": "ignored"},
        ),
        "terms": (
            {
                "coeff": {"numer": 1, "denom": 1, "unused": "ignored"},
                "sum_indices": (
                    {"id": 2, "range": 0, "unused": "ignored"},
                ),
                "factors": (
                    {
                        "tensor": 0,
                        "indices": range(2),
                        "unused": "ignored",
                    },
                ),
                "unused": "ignored",
            },
        ),
        "unused": "ignored",
    }

    ids = tokenizer.encode_definition(definition)

    assert tokenizer.decode_definition(ids) == {
        "base": 3,
        "ext_indices": [{"id": 0, "range": 0}],
        "terms": [
            {
                "coeff": {"numer": 1, "denom": 1},
                "sum_indices": [{"id": 2, "range": 0}],
                "factors": [{"tensor": 0, "indices": [0, 1]}],
            },
        ],
    }


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


def test_decode_definitions_rejects_malformed_concatenated_streams():
    tokenizer = _tokenizer()
    valid = tokenizer.encode_definitions([_definition(), _second_definition()])

    with pytest.raises(TokenizerError, match="sequence of integer IDs"):
        tokenizer.decode_definitions({})

    with pytest.raises(TokenizerError, match="def_start"):
        tokenizer.decode_definitions(valid[1:])

    missing_final_end = valid[:-1]
    with pytest.raises(TokenizerError, match="def_end"):
        tokenizer.decode_definitions(missing_final_end)

    with pytest.raises(TokenizerError, match="raw token stream cannot contain pad"):
        tokenizer.decode_definitions([tokenizer.pad_token_id, *valid])

    with pytest.raises(TokenizerError, match="unknown token id"):
        tokenizer.decode_definitions([10_000])
