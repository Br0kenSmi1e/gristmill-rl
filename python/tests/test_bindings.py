import json
import math
from pathlib import Path

import pytest

from gristmill_symbolics import (
    ActionSpace,
    GristmillSymbolicsError,
    TensorComputation,
    action_space_for_def,
    action_spaces_for_batch,
    apply_decision,
    apply_decisions_for_batch,
    validate_decision,
    validate_decisions_for_batch,
)

ROOT = Path(__file__).resolve().parents[2]
BASIC_FIXTURE = ROOT / "tests" / "fixtures" / "repr" / "basic.json"


def test_module_exports_thin_rewrite_surface():
    import gristmill_symbolics

    assert gristmill_symbolics.__all__ == (
        "ActionSpace",
        "GristmillSymbolicsError",
        "TensorComputation",
        "action_space_for_def",
        "action_spaces_for_batch",
        "apply_decision",
        "apply_decisions_for_batch",
        "validate_decision",
        "validate_decisions_for_batch",
    )
    assert not hasattr(gristmill_symbolics, "RewriteState")
    assert not hasattr(gristmill_symbolics, "RewriteStateRow")
    assert not hasattr(gristmill_symbolics, "ActionSpaceRow")
    assert not hasattr(gristmill_symbolics, "ValidatedActionRow")


def test_load_json_validates_and_snapshots_basic_fixture():
    comp = TensorComputation.load_json(BASIC_FIXTURE)

    assert comp.snapshot() == {
        "ranges": [{"id": 0, "size": 3}],
        "tensors": [
            {
                "id": 0,
                "symmetry": [{"perm": [0], "action": "Identity"}],
            }
        ],
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
            }
        ],
    }


def test_from_json_string_validates_and_clones():
    comp = TensorComputation.from_json_string(BASIC_FIXTURE.read_text())

    clone = comp.clone()

    assert clone.snapshot() == comp.snapshot()
    assert clone is not comp


def test_invalid_json_string_raises_gristmill_error():
    with pytest.raises(GristmillSymbolicsError):
        TensorComputation.from_json_string("{")


def test_invalid_representation_raises_gristmill_error():
    text = """
    {
      "ranges": [{ "id": 7, "size": 3 }],
      "tensors": [],
      "definitions": []
    }
    """

    with pytest.raises(GristmillSymbolicsError):
        TensorComputation.from_json_string(text)


def test_log_total_flops_returns_python_float():
    comp = TensorComputation.load_json(BASIC_FIXTURE)

    value = comp.log_total_flops()

    assert isinstance(value, float)
    assert value == pytest.approx(math.log(6.0))


def actionable_json() -> str:
    return json.dumps(
        {
            "ranges": [{"id": 0, "size": 8}],
            "tensors": [
                {"id": 0, "symmetry": []},
                {"id": 1, "symmetry": []},
                {"id": 2, "symmetry": []},
                {"id": 3, "symmetry": []},
            ],
            "definitions": [
                {
                    "base": 3,
                    "ext_indices": [
                        {"id": 0, "range": 0},
                        {"id": 1, "range": 0},
                    ],
                    "terms": [
                        {
                            "coeff": [1, 1],
                            "sum_indices": [{"id": 2, "range": 0}],
                            "factors": [
                                {"tensor": 0, "indices": [0, 2]},
                                {"tensor": 1, "indices": [2, 1]},
                            ],
                        },
                        {
                            "coeff": [1, 1],
                            "sum_indices": [{"id": 3, "range": 0}],
                            "factors": [
                                {"tensor": 0, "indices": [0, 3]},
                                {"tensor": 2, "indices": [3, 1]},
                            ],
                        },
                    ],
                }
            ],
        }
    )


def exact_empty_json() -> str:
    return json.dumps(
        {
            "ranges": [{"id": 0, "size": 8}],
            "tensors": [
                {"id": 0, "symmetry": []},
                {"id": 1, "symmetry": []},
            ],
            "definitions": [
                {
                    "base": 1,
                    "ext_indices": [{"id": 0, "range": 0}],
                    "terms": [
                        {
                            "coeff": [1, 1],
                            "sum_indices": [],
                            "factors": [{"tensor": 0, "indices": [0]}],
                        },
                        {
                            "coeff": [1, 1],
                            "sum_indices": [],
                            "factors": [{"tensor": 0, "indices": [0]}],
                        },
                    ],
                }
            ],
        }
    )


def first_full_decision(space):
    template = space.snapshot()["candidate_templates"][0]
    return {
        "candidate_index": 0,
        "left_mask": [True] * len(template["left_definition"]["terms"]),
        "right_mask": [True] * len(template["right_definition"]["terms"]),
    }


def test_action_space_for_def_returns_visible_factorization_templates():
    comp = TensorComputation.from_json_string(actionable_json())

    space = action_space_for_def(comp, 0)

    assert isinstance(space, ActionSpace)
    assert space.def_index == 0
    assert space.candidate_count > 0
    candidate = space.snapshot()["candidate_templates"][0]
    assert set(candidate) == {"left_definition", "right_definition"}


def test_action_space_for_def_returns_none_for_unfactorable_definition():
    comp = TensorComputation.load_json(BASIC_FIXTURE)

    assert action_space_for_def(comp, 0) is None


def test_validate_and_apply_decision_call_rust_directly():
    comp = TensorComputation.from_json_string(actionable_json())
    before = comp.snapshot()
    space = action_space_for_def(comp, 0)
    decision = first_full_decision(space)

    validate_decision(space, decision)
    apply_decision(comp, space, decision)

    after = comp.snapshot()
    assert after != before
    assert len(after["definitions"]) == len(before["definitions"]) + 2


def test_batch_rewrite_functions_mirror_rust_batch_api():
    active = TensorComputation.from_json_string(actionable_json())
    skipped = TensorComputation.from_json_string(actionable_json())
    spaces = action_spaces_for_batch([active, skipped], [0, None])
    decision = first_full_decision(spaces[0])

    validate_decisions_for_batch(spaces, [decision, None])
    applied = apply_decisions_for_batch([active, skipped], spaces, [decision, None])

    assert applied == [True, False]
    assert len(active.snapshot()["definitions"]) == 3
    assert len(skipped.snapshot()["definitions"]) == 1


def test_to_json_string_round_trips_basic_fixture():
    comp = TensorComputation.load_json(BASIC_FIXTURE)

    encoded = comp.to_json_string()
    decoded = TensorComputation.from_json_string(encoded)

    assert decoded.snapshot() == comp.snapshot()


def test_write_json_round_trips_basic_fixture(tmp_path):
    comp = TensorComputation.load_json(BASIC_FIXTURE)
    path = tmp_path / "roundtrip.json"

    comp.write_json(path)

    assert TensorComputation.load_json(path).snapshot() == comp.snapshot()
