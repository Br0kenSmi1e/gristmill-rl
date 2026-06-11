import json
import math
from pathlib import Path

import pytest

from gristmill_symbolics import (
    ActionSpace,
    GristmillSymbolicsError,
    RewriteState,
    TensorComputation,
    validate_decision,
)


ROOT = Path(__file__).resolve().parents[2]
BASIC_FIXTURE = ROOT / "tests" / "fixtures" / "repr" / "basic.json"


def test_module_exports_core_types():
    import gristmill_symbolics

    assert hasattr(gristmill_symbolics, "TensorComputation")
    assert hasattr(gristmill_symbolics, "RewriteState")
    assert hasattr(gristmill_symbolics, "ActionSpace")
    assert hasattr(gristmill_symbolics, "RewriteStateRow")
    assert hasattr(gristmill_symbolics, "ActionSpaceRow")
    assert hasattr(gristmill_symbolics, "ValidatedActionRow")
    assert hasattr(gristmill_symbolics, "GristmillSymbolicsError")
    assert hasattr(gristmill_symbolics, "validate_decision")
    assert not hasattr(TensorComputation, "next_" "action_space")
    assert not hasattr(TensorComputation, "apply_decision_" "with_space")
    assert not hasattr(RewriteState, "step_" "with_space")


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


def test_rewrite_state_from_computation_clones_input_computation():
    comp = TensorComputation.from_json_string(actionable_json())
    before = comp.snapshot()
    state = RewriteState.from_computation(comp)
    space = state.action_space_for_def(0)
    decision = first_full_decision(space)

    validate_decision(space, decision)
    state.apply_validated_decision(space, decision)

    assert comp.snapshot() == before
    assert state.snapshot() != before


def test_rewrite_state_returns_none_for_basic_fixture():
    comp = TensorComputation.load_json(BASIC_FIXTURE)
    state = RewriteState.from_computation(comp)

    assert state.definition_mask() == [False]
    assert state.action_space_for_def(0) is None


def test_rewrite_state_definition_mask_returns_copy():
    comp = TensorComputation.from_json_string(actionable_json())
    state = RewriteState.from_computation(comp)
    mask = state.definition_mask()

    mask[0] = False

    assert state.definition_mask() == [True]


def test_rewrite_state_refines_exact_empty_mask_to_false():
    comp = TensorComputation.from_json_string(exact_empty_json())
    state = RewriteState.from_computation(comp)

    assert state.definition_mask() == [True]
    assert state.action_space_for_def(0) is None
    assert state.definition_mask() == [False]


def test_rewrite_state_action_space_handle_and_public_snapshot():
    comp = TensorComputation.from_json_string(actionable_json())
    state = RewriteState.from_computation(comp)

    space = state.action_space_for_def(0)
    snapshot = space.snapshot()

    assert isinstance(space, ActionSpace)
    assert space.def_index == 0
    assert space.candidate_count == len(snapshot["candidate_templates"])
    assert space.candidate_count > 0
    assert set(snapshot) == {"def_index", "candidate_templates"}
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


def test_rewrite_state_cost_and_json_delegate_to_inner_computation():
    comp = TensorComputation.from_json_string(actionable_json())
    state = RewriteState.from_computation(comp)

    value = state.log_total_flops()
    text = state.to_json_string()
    loaded = TensorComputation.from_json_string(text)

    assert isinstance(value, float)
    assert value == pytest.approx(comp.log_total_flops())
    assert loaded.snapshot() == state.snapshot()


def test_rewrite_state_apply_validated_decision_mutates_state_and_returns_none():
    comp = TensorComputation.from_json_string(actionable_json())
    state = RewriteState.from_computation(comp)
    space = state.action_space_for_def(0)
    before = state.snapshot()
    decision = first_full_decision(space)

    validate_decision(space, decision)
    result = state.apply_validated_decision(space, decision)
    after = state.snapshot()

    assert result is None
    assert len(after["tensors"]) == len(before["tensors"]) + 2
    assert len(after["definitions"]) == len(before["definitions"]) + 2
    assert after != before
    assert len(state.definition_mask()) == len(after["definitions"])


def test_validate_decision_raises_and_does_not_mutate():
    comp = TensorComputation.from_json_string(actionable_json())
    state = RewriteState.from_computation(comp)
    space = state.action_space_for_def(0)
    before = state.snapshot()
    bad_decision = {
        "candidate_index": 0,
        "left_mask": [],
        "right_mask": [True],
    }

    with pytest.raises(GristmillSymbolicsError):
        validate_decision(space, bad_decision)

    assert state.snapshot() == before


def test_validate_decision_rejects_malformed_decision_shape():
    comp = TensorComputation.from_json_string(actionable_json())
    state = RewriteState.from_computation(comp)
    space = state.action_space_for_def(0)

    with pytest.raises(TypeError):
        validate_decision(space, "not a dict")

    with pytest.raises(ValueError):
        validate_decision(
            space,
            {"candidate_index": 0, "left_mask": [True]},
        )

    with pytest.raises(TypeError):
        validate_decision(
            space,
            {"candidate_index": True, "left_mask": [True], "right_mask": [True]},
        )

    with pytest.raises(ValueError):
        validate_decision(
            space,
            {"candidate_index": -1, "left_mask": [True], "right_mask": [True]},
        )

    with pytest.raises(ValueError):
        validate_decision(
            space,
            {"candidate_index": 2**128, "left_mask": [True], "right_mask": [True]},
        )

    with pytest.raises(TypeError):
        validate_decision(
            space,
            {"candidate_index": 0, "left_mask": True, "right_mask": [True]},
        )

    with pytest.raises(TypeError):
        validate_decision(
            space,
            {"candidate_index": 0, "left_mask": [1], "right_mask": [True]},
        )


def test_action_space_handle_is_reusable_on_multiple_states():
    comp = TensorComputation.from_json_string(actionable_json())
    source_state = RewriteState.from_computation(comp)
    space = source_state.action_space_for_def(0)
    decision = first_full_decision(space)
    left = RewriteState.from_computation(comp)
    right = RewriteState.from_computation(comp)

    validate_decision(space, decision)
    left.apply_validated_decision(space, decision)
    validate_decision(space, decision)
    right.apply_validated_decision(space, decision)

    assert left.snapshot() == right.snapshot()


def test_to_json_string_round_trips_basic_fixture():
    comp = TensorComputation.load_json(BASIC_FIXTURE)

    text = comp.to_json_string()
    loaded = TensorComputation.from_json_string(text)

    assert loaded.snapshot() == comp.snapshot()


def test_write_json_round_trips_basic_fixture(tmp_path):
    comp = TensorComputation.load_json(BASIC_FIXTURE)
    output = tmp_path / "written.json"

    comp.write_json(output)
    loaded = TensorComputation.load_json(output)

    assert loaded.snapshot() == comp.snapshot()


def test_write_json_round_trips_rewritten_computation(tmp_path):
    comp = TensorComputation.from_json_string(actionable_json())
    state = RewriteState.from_computation(comp)
    space = state.action_space_for_def(0)
    assert space is not None
    template = space.snapshot()["candidate_templates"][0]
    decision = {
        "candidate_index": 0,
        "left_mask": [True] * len(template["left_definition"]["terms"]),
        "right_mask": [True] * len(template["right_definition"]["terms"]),
    }
    validate_decision(space, decision)
    state.apply_validated_decision(space, decision)
    output = tmp_path / "rewritten.json"

    state.write_json(output)
    loaded = TensorComputation.load_json(output)

    assert loaded.snapshot() == state.snapshot()
