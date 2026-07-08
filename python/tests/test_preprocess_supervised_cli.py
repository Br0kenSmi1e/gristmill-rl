import argparse
import json

import numpy as np
import pytest

import gristmill_symbolics.preprocess_supervised as preprocess_supervised
from gristmill_symbolics.supervised_dataset import (
    load_preprocessed_supervised_dataset,
)


def _definition(*, base: int, coeff_numer: int = 1) -> dict[str, object]:
    return {
        "base": base,
        "ext_indices": [{"id": 0, "range": 0}],
        "terms": [
            {
                "coeff": [coeff_numer, 1],
                "sum_indices": [],
                "factors": [{"tensor": base, "indices": [0]}],
            }
        ],
    }


def _computation(definition: dict[str, object]) -> dict[str, object]:
    return {
        "ranges": [{"id": 0, "size": 3}],
        "tensors": [
            {"id": 0, "symmetry": []},
            {"id": 1, "symmetry": []},
            {"id": 2, "symmetry": []},
            {"id": 3, "symmetry": []},
        ],
        "definitions": [definition],
    }


def _row(*, weight: float = 1.0) -> dict[str, object]:
    return {
        "source": _computation(_definition(base=0)),
        "target": _computation(_definition(base=1)),
        "weight": weight,
    }


def test_main_preprocesses_one_jsonl_file_to_arrays_and_metadata(tmp_path):
    input_path = tmp_path / "train.jsonl"
    arrays_path = tmp_path / "train.npz"
    metadata_path = tmp_path / "train.json"
    rows = [_row(weight=0.25), _row(weight=1.75)]
    input_path.write_text("".join(f"{json.dumps(row)}\n" for row in rows))

    result = preprocess_supervised.main(
        [
            "--input",
            str(input_path),
            "--arrays-out",
            str(arrays_path),
            "--metadata-out",
            str(metadata_path),
            "--source-len",
            "12",
            "--target-len",
            "14",
            "--max-range-id",
            "1",
            "--max-tensor-id",
            "3",
            "--max-index-id",
            "2",
            "--coeff-nums",
            "-1,1,2",
            "--coeff-dens",
            "1",
        ]
    )

    dataset = load_preprocessed_supervised_dataset(arrays_path, metadata_path)

    assert result == 0
    assert arrays_path.exists()
    assert metadata_path.exists()
    assert dataset["source_ids"].shape == (2, 12)
    assert dataset["target_ids"].shape == (2, 14)
    np.testing.assert_allclose(dataset["example_weight"], np.asarray([0.25, 1.75]))
    assert dataset["metadata"]["tokenizer"]["coeff_nums"] == [-1, 1, 2]
    assert dataset["metadata"]["tokenizer"]["coeff_dens"] == [1]


def test_parse_int_csv_rejects_empty_items():
    with pytest.raises(argparse.ArgumentTypeError, match="comma-separated integers"):
        preprocess_supervised._parse_int_csv("1,,2")


def test_parse_int_csv_rejects_non_integer_items():
    with pytest.raises(argparse.ArgumentTypeError, match="comma-separated integers"):
        preprocess_supervised._parse_int_csv("1,two")
