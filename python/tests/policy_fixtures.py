from __future__ import annotations

import json

from gristmill_symbolics import TensorComputation, action_space_for_def
from gristmill_symbolics.model.transformer_action_selector import SelectorState


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
                    "ext_indices": [{"id": 0, "range": 0}, {"id": 1, "range": 0}],
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


def actionable_comp() -> TensorComputation:
    return TensorComputation.from_json_string(actionable_json())


def actionable_state() -> SelectorState:
    return SelectorState(comp=actionable_comp())


def actionable_state_snapshot():
    return actionable_comp().snapshot()


def actionable_action_space_snapshot():
    space = action_space_for_def(actionable_comp(), 0)
    assert space is not None
    return space.snapshot()
