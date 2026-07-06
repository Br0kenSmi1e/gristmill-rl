import json

from gristmill_symbolics import TensorComputation


def source_comp_json() -> str:
    return json.dumps(
        {
            "ranges": [{"id": 0, "size": 8}],
            "tensors": [
                {"id": 0, "symmetry": [{"perm": [0], "action": "Identity"}]},
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
                        }
                    ],
                }
            ],
        }
    )


def source_comp() -> TensorComputation:
    return TensorComputation.from_json_string(source_comp_json())


def two_candidate_jsons() -> tuple[str, str]:
    low = json.loads(source_comp_json())
    high = json.loads(source_comp_json())
    high["definitions"][0]["terms"][0]["coeff"] = [2, 1]
    return json.dumps(low), json.dumps(high)
