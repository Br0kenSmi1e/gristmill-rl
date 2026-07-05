import pytest

from gristmill_symbolics.direct_optimizer.converter import (
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


@pytest.mark.parametrize(
    "bad_text, message",
    [
        ("foo id range_id:0", "unknown keyword"),
        ("def base range_id:1\nenddef", "expected tensor_id"),
        ("def base tensor_id:1\nterm\nenddef", "unclosed term"),
        ("endterm", "unexpected endterm"),
    ],
)
def test_parser_rejects_malformed_dsl(bad_text, message):
    with pytest.raises(ValueError, match=message):
        target_text_to_definitions(bad_text)
