import json
import math
from pathlib import Path

import pytest

from gristmill_symbolics import ActionSpace, GristmillSymbolicsError, TensorComputation


ROOT = Path(__file__).resolve().parents[2]
BASIC_FIXTURE = ROOT / "tests" / "fixtures" / "repr" / "basic.json"


def test_module_exports_core_types():
    import gristmill_symbolics

    assert hasattr(gristmill_symbolics, "TensorComputation")
    assert hasattr(gristmill_symbolics, "ActionSpace")
    assert hasattr(gristmill_symbolics, "GristmillSymbolicsError")


def test_load_json_validates_and_snapshots_basic_fixture():
    comp = TensorComputation.load_json(BASIC_FIXTURE)

    snapshot = comp.snapshot()

    assert snapshot == {
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
    text = BASIC_FIXTURE.read_text()

    comp = TensorComputation.from_json_string(text)
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


def test_next_action_space_returns_none_for_basic_fixture():
    comp = TensorComputation.load_json(BASIC_FIXTURE)

    assert comp.next_action_space(0) is None


def test_next_action_space_returns_handle_and_public_snapshot():
    comp = TensorComputation.from_json_string(actionable_json())

    space = comp.next_action_space(0)
    snapshot = space.snapshot()

    assert isinstance(space, ActionSpace)
    assert space.def_index == 0
    assert space.candidate_count == len(snapshot["candidate_templates"])
    assert space.candidate_count > 0
    assert snapshot["def_index"] == 0
    first = snapshot["candidate_templates"][0]
    assert set(first) == {
        "left_definition",
        "right_definition",
        "rewritten_definition",
    }
    assert first["left_definition"]["terms"]
    assert first["right_definition"]["terms"]
    assert first["rewritten_definition"]["terms"]
