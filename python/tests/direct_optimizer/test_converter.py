import pytest

from gristmill_symbolics.direct_optimizer.converter import (
    _definitions_to_text,
    computation_to_source_text,
    computation_to_target_text,
    source_text_to_snapshot,
    target_text_to_definitions,
)
from tests.direct_optimizer.fixtures import source_comp


def test_source_text_round_trips_full_snapshot():
    comp = source_comp()

    text = computation_to_source_text(comp)

    assert source_text_to_snapshot(text) == comp.snapshot()
    assert text.splitlines() == [
        "range id range_id:0 size dim_size:8",
        "tensor id tensor_id:0",
        "symmetry action sym_action:Identity",
        "perm axis:0",
        "endsymmetry",
        "endtensor",
        "tensor id tensor_id:1",
        "endtensor",
        "def base tensor_id:1",
        "ext id index_id:0 range range_id:0",
        "term",
        "coeff numer coeff_num:1 denom coeff_den:1",
        "factor tensor tensor_id:0",
        "index index_id:0",
        "endfactor",
        "endterm",
        "enddef",
    ]


def test_target_text_round_trips_definitions_only():
    comp = source_comp()

    text = computation_to_target_text(comp)

    assert target_text_to_definitions(text) == comp.snapshot()["definitions"]
    assert text.splitlines()[0] == "def base tensor_id:1"
    assert all(not line.startswith("range ") for line in text.splitlines())
    assert all(not line.startswith("tensor ") for line in text.splitlines())


def test_definition_printing_accepts_coeff_list_and_dict_forms():
    definition = source_comp().snapshot()["definitions"][0]
    dict_definition = definition
    list_definition = {
        "base": definition["base"],
        "ext_indices": definition["ext_indices"],
        "terms": [
            {
                "coeff": [1, 1],
                "sum_indices": definition["terms"][0]["sum_indices"],
                "factors": definition["terms"][0]["factors"],
            }
        ],
    }

    for coeff_definition in (dict_definition, list_definition):
        text = _definitions_to_text([coeff_definition])

        assert "coeff numer coeff_num:1 denom coeff_den:1" in text.splitlines()
        assert target_text_to_definitions(text)[0]["terms"][0]["coeff"] == {
            "numer": 1,
            "denom": 1,
        }


def test_target_parser_accepts_signed_coeff_num():
    text = "\n".join(
        [
            "def base tensor_id:1",
            "term",
            "coeff numer coeff_num:-3 denom coeff_den:2",
            "endterm",
            "enddef",
        ]
    )

    assert target_text_to_definitions(text)[0]["terms"][0]["coeff"] == {
        "numer": -3,
        "denom": 2,
    }


@pytest.mark.parametrize(
    "bad_text, message",
    [
        ("foo id range_id:0", "unknown keyword"),
        ("def base range_id:1\nenddef", "expected tensor_id"),
        ("def base\nenddef", "missing fields"),
        ("def base tensor_id:1 extra\nenddef", "extra fields"),
        ("range id range_id:0 size dim_size:8", "unknown keyword"),
        ("def base tensor_id:-1\nenddef", "tensor_id"),
        (
            "\n".join(
                [
                    "def base tensor_id:1",
                    "term",
                    "coeff numer coeff_num:-1 denom coeff_den:0",
                    "endterm",
                    "enddef",
                ]
            ),
            "coeff_den",
        ),
        ("def base tensor_id:1\nterm\nenddef", "unclosed term"),
        ("endterm", "unexpected endterm"),
    ],
)
def test_parser_rejects_malformed_dsl(bad_text, message):
    with pytest.raises(ValueError, match=message):
        target_text_to_definitions(bad_text)


@pytest.mark.parametrize(
    "bad_text, message",
    [
        ("range id range_id:-1 size dim_size:8", "range_id"),
        ("range id range_id:0 size dim_size:-8", "dim_size"),
        ("tensor id tensor_id:-1\nendtensor", "tensor_id"),
        (
            "\n".join(
                [
                    "tensor id tensor_id:0",
                    "symmetry action sym_action:Identity",
                    "perm axis:-1",
                    "endsymmetry",
                    "endtensor",
                ]
            ),
            "axis",
        ),
    ],
)
def test_source_parser_rejects_negative_unsigned_scalars(bad_text, message):
    with pytest.raises(ValueError, match=message):
        source_text_to_snapshot(bad_text)


@pytest.mark.parametrize(
    "bad_text, message",
    [
        ("def base tensor_id:-1\nenddef", "tensor_id"),
        (
            "def base tensor_id:1\next id index_id:-1 range range_id:0\nenddef",
            "index_id",
        ),
        (
            "def base tensor_id:1\next id index_id:0 range range_id:-1\nenddef",
            "range_id",
        ),
        (
            "\n".join(
                [
                    "def base tensor_id:1",
                    "term",
                    "coeff numer coeff_num:1 denom coeff_den:1",
                    "sum id index_id:-1 range range_id:0",
                    "endterm",
                    "enddef",
                ]
            ),
            "index_id",
        ),
        (
            "\n".join(
                [
                    "def base tensor_id:1",
                    "term",
                    "coeff numer coeff_num:1 denom coeff_den:1",
                    "factor tensor tensor_id:-1",
                    "endfactor",
                    "endterm",
                    "enddef",
                ]
            ),
            "tensor_id",
        ),
        (
            "\n".join(
                [
                    "def base tensor_id:1",
                    "term",
                    "coeff numer coeff_num:1 denom coeff_den:1",
                    "factor tensor tensor_id:0",
                    "index index_id:-1",
                    "endfactor",
                    "endterm",
                    "enddef",
                ]
            ),
            "index_id",
        ),
    ],
)
def test_target_parser_rejects_negative_unsigned_scalars(bad_text, message):
    with pytest.raises(ValueError, match=message):
        target_text_to_definitions(bad_text)


def test_source_parser_rejects_unknown_symmetry_action():
    text = "\n".join(
        [
            "tensor id tensor_id:0",
            "symmetry action sym_action:Transpose",
            "perm axis:0",
            "endsymmetry",
            "endtensor",
        ]
    )

    with pytest.raises(ValueError, match="sym_action"):
        source_text_to_snapshot(text)


def test_source_parser_rejects_unclosed_tensor():
    with pytest.raises(ValueError, match="unclosed tensor"):
        source_text_to_snapshot("tensor id tensor_id:0")
