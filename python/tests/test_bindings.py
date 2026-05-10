def test_module_exports_core_types():
    import gristmill_symbolics

    assert hasattr(gristmill_symbolics, "TensorComputation")
    assert hasattr(gristmill_symbolics, "ActionSpace")
    assert hasattr(gristmill_symbolics, "GristmillSymbolicsError")


from pathlib import Path

import pytest

from gristmill_symbolics import GristmillSymbolicsError, TensorComputation


ROOT = Path(__file__).resolve().parents[2]
BASIC_FIXTURE = ROOT / "tests" / "fixtures" / "repr" / "basic.json"


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
    assert value > 0.0
