import json
import math

import numpy as np
import pytest

from gristmill_symbolics import TensorComputation
from gristmill_symbolics.supervised_dataset import (
    SupervisedDatasetError,
    iter_supervised_batches,
    load_preprocessed_supervised_dataset,
    preprocess_supervised_jsonl,
    preprocess_supervised_rows,
    save_preprocessed_supervised_dataset,
)
from gristmill_symbolics.tokenizer import FlatDefinitionTokenizer


def _tokenizer() -> FlatDefinitionTokenizer:
    return FlatDefinitionTokenizer(
        max_range_id=1,
        max_tensor_id=3,
        max_index_id=2,
        coeff_nums=(1, 2),
        coeff_dens=(1,),
    )


def _definition(*, base: int, coeff_numer: int = 1) -> dict[str, object]:
    return {
        "base": base,
        "ext_indices": [{"id": 0, "range": 0}],
        "terms": [
            {
                "coeff": {"numer": coeff_numer, "denom": 1},
                "sum_indices": [],
                "factors": [{"tensor": base, "indices": [0]}],
            }
        ],
    }


def _json_definition(*, base: int, coeff_numer: int = 1) -> dict[str, object]:
    definition = _definition(base=base, coeff_numer=coeff_numer)
    term = definition["terms"][0]
    term["coeff"] = [coeff_numer, 1]
    return definition


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


def _row(
    *,
    source_definition: dict[str, object] | None = None,
    target_definition: dict[str, object] | None = None,
    weight: float = 1.0,
) -> dict[str, object]:
    if source_definition is None:
        source_definition = _json_definition(base=0)
    if target_definition is None:
        target_definition = _json_definition(base=1)
    return {
        "source": _computation(source_definition),
        "target": _computation(target_definition),
        "weight": weight,
    }


def _snapshot_definitions(comp: dict[str, object]) -> list[dict[str, object]]:
    return TensorComputation.from_json_string(json.dumps(comp)).snapshot()["definitions"]


def _assert_same_dataset(lhs, rhs) -> None:
    np.testing.assert_array_equal(lhs["source_ids"], rhs["source_ids"])
    np.testing.assert_array_equal(
        lhs["decoder_input_ids"],
        rhs["decoder_input_ids"],
    )
    np.testing.assert_array_equal(lhs["target_ids"], rhs["target_ids"])
    np.testing.assert_array_equal(lhs["target_mask"], rhs["target_mask"])
    np.testing.assert_array_equal(lhs["example_weight"], rhs["example_weight"])
    assert lhs["metadata"] == rhs["metadata"]


def _expected_metadata(
    tokenizer: FlatDefinitionTokenizer,
    *,
    source_len: int,
    target_len: int,
    num_examples: int,
) -> dict[str, object]:
    return {
        "source_len": source_len,
        "target_len": target_len,
        "vocab_size": tokenizer.vocab_size,
        "num_examples": num_examples,
        "tokenizer": {
            "max_range_id": tokenizer.max_range_id,
            "max_tensor_id": tokenizer.max_tensor_id,
            "max_index_id": tokenizer.max_index_id,
            "coeff_nums": [1, 2],
            "coeff_dens": [1],
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        },
    }


def test_preprocess_supervised_rows_builds_fixed_numpy_arrays_and_metadata():
    tokenizer = _tokenizer()
    rows = [
        _row(weight=0.25),
        _row(
            source_definition=_json_definition(base=2),
            target_definition=_json_definition(base=3, coeff_numer=2),
            weight=1.75,
        ),
    ]

    dataset = preprocess_supervised_rows(
        rows,
        tokenizer,
        source_len=12,
        target_len=14,
    )

    assert dataset["source_ids"].shape == (2, 12)
    assert dataset["decoder_input_ids"].shape == (2, 14)
    assert dataset["target_ids"].shape == (2, 14)
    assert dataset["target_mask"].shape == (2, 14)
    assert dataset["example_weight"].shape == (2,)
    assert dataset["source_ids"].dtype == np.int32
    assert dataset["decoder_input_ids"].dtype == np.int32
    assert dataset["target_ids"].dtype == np.int32
    assert dataset["target_mask"].dtype == np.bool_
    assert dataset["example_weight"].dtype == np.float32
    assert dataset["metadata"] == _expected_metadata(
        tokenizer,
        source_len=12,
        target_len=14,
        num_examples=2,
    )
    np.testing.assert_allclose(dataset["example_weight"], np.asarray([0.25, 1.75]))


def test_preprocess_supervised_rows_builds_framed_target_sequences_and_mask():
    tokenizer = _tokenizer()
    target_definition = _json_definition(base=1)
    row = _row(target_definition=target_definition)
    content_ids = tokenizer.encode_definitions(
        _snapshot_definitions(_computation(target_definition))
    )
    target_len = len(content_ids) + 4
    decoder_pad_count = target_len - len(content_ids) - 1
    target_pad_count = target_len - len(content_ids) - 1

    dataset = preprocess_supervised_rows(
        [row],
        tokenizer,
        source_len=12,
        target_len=target_len,
    )

    assert dataset["decoder_input_ids"].tolist() == [
        [
            tokenizer.bos_token_id,
            *content_ids,
            *([tokenizer.pad_token_id] * decoder_pad_count),
        ]
    ]
    assert dataset["target_ids"].tolist() == [
        [
            *content_ids,
            tokenizer.eos_token_id,
            *([tokenizer.pad_token_id] * target_pad_count),
        ]
    ]
    assert dataset["target_mask"].tolist() == [
        [*([True] * (len(content_ids) + 1)), *([False] * target_pad_count)]
    ]


def test_save_and_load_preprocessed_supervised_dataset_round_trips_arrays_and_metadata(
    tmp_path,
):
    tokenizer = _tokenizer()
    dataset = preprocess_supervised_rows(
        [
            _row(weight=0.5),
            _row(target_definition=_json_definition(base=2), weight=1.5),
        ],
        tokenizer,
        source_len=12,
        target_len=14,
    )
    arrays_path = tmp_path / "supervised_arrays.npz"
    metadata_path = tmp_path / "supervised_metadata.json"

    save_preprocessed_supervised_dataset(dataset, arrays_path, metadata_path)
    loaded = load_preprocessed_supervised_dataset(arrays_path, metadata_path)

    _assert_same_dataset(loaded, dataset)
    assert json.loads(metadata_path.read_text()) == dataset["metadata"]


def test_preprocess_supervised_jsonl_writes_artifacts_and_returns_loaded_dataset(
    tmp_path,
):
    tokenizer = _tokenizer()
    rows = [
        _row(weight=0.25),
        _row(
            source_definition=_json_definition(base=2),
            target_definition=_json_definition(base=3, coeff_numer=2),
            weight=1.75,
        ),
    ]
    input_path = tmp_path / "rows.jsonl"
    arrays_path = tmp_path / "arrays.npz"
    metadata_path = tmp_path / "metadata.json"
    input_path.write_text("".join(f"{json.dumps(row)}\n" for row in rows))

    dataset = preprocess_supervised_jsonl(
        input_path,
        arrays_path,
        metadata_path,
        tokenizer,
        source_len=12,
        target_len=14,
    )
    loaded = load_preprocessed_supervised_dataset(arrays_path, metadata_path)

    _assert_same_dataset(loaded, dataset)
    assert arrays_path.exists()
    assert metadata_path.read_text().endswith("\n")


def test_preprocess_supervised_jsonl_reports_invalid_json_with_1_based_row_number(
    tmp_path,
):
    tokenizer = _tokenizer()
    input_path = tmp_path / "rows.jsonl"
    input_path.write_text("{")

    with pytest.raises(SupervisedDatasetError, match="row 1: invalid JSON"):
        preprocess_supervised_jsonl(
            input_path,
            tmp_path / "arrays.npz",
            tmp_path / "metadata.json",
            tokenizer,
            source_len=12,
            target_len=14,
        )


def test_preprocess_supervised_jsonl_reports_empty_line_with_1_based_row_number(
    tmp_path,
):
    tokenizer = _tokenizer()
    input_path = tmp_path / "rows.jsonl"
    input_path.write_text("\n")

    with pytest.raises(SupervisedDatasetError, match="row 1: empty JSONL row"):
        preprocess_supervised_jsonl(
            input_path,
            tmp_path / "arrays.npz",
            tmp_path / "metadata.json",
            tokenizer,
            source_len=12,
            target_len=14,
        )


@pytest.mark.parametrize(
    ("missing_key", "message"),
    [
        ("source", "row 1: missing 'source'"),
        ("target", "row 1: missing 'target'"),
        ("weight", "row 1: missing 'weight'"),
    ],
)
def test_preprocess_supervised_jsonl_reports_missing_required_keys(
    tmp_path,
    missing_key,
    message,
):
    tokenizer = _tokenizer()
    row = _row()
    del row[missing_key]
    input_path = tmp_path / "rows.jsonl"
    input_path.write_text(f"{json.dumps(row)}\n")

    with pytest.raises(SupervisedDatasetError, match=message):
        preprocess_supervised_jsonl(
            input_path,
            tmp_path / "arrays.npz",
            tmp_path / "metadata.json",
            tokenizer,
            source_len=12,
            target_len=14,
        )


def test_preprocess_supervised_jsonl_reports_first_semantic_error_before_later_json_error(
    tmp_path,
):
    tokenizer = _tokenizer()
    row = _row()
    del row["source"]
    input_path = tmp_path / "rows.jsonl"
    input_path.write_text(f"{json.dumps(row)}\n{{\n")

    with pytest.raises(SupervisedDatasetError, match="row 1: missing 'source'"):
        preprocess_supervised_jsonl(
            input_path,
            tmp_path / "arrays.npz",
            tmp_path / "metadata.json",
            tokenizer,
            source_len=12,
            target_len=14,
        )


@pytest.mark.parametrize("weight", [-1.0, math.inf])
def test_preprocess_supervised_jsonl_reports_invalid_weight_with_row_number(
    tmp_path,
    weight,
):
    tokenizer = _tokenizer()
    input_path = tmp_path / "rows.jsonl"
    input_path.write_text(f"{json.dumps(_row(weight=weight))}\n")

    with pytest.raises(SupervisedDatasetError, match="row 1: weight must be"):
        preprocess_supervised_jsonl(
            input_path,
            tmp_path / "arrays.npz",
            tmp_path / "metadata.json",
            tokenizer,
            source_len=12,
            target_len=14,
        )


def test_preprocess_supervised_rows_reports_source_over_length_with_row_number():
    tokenizer = _tokenizer()

    with pytest.raises(SupervisedDatasetError, match="row 1: .*exceeds length"):
        preprocess_supervised_rows(
            [_row()],
            tokenizer,
            source_len=1,
            target_len=14,
        )


def test_preprocess_supervised_rows_reports_target_over_length_with_row_number():
    tokenizer = _tokenizer()

    with pytest.raises(
        SupervisedDatasetError,
        match="row 1: target content length .* exceeds target_len",
    ):
        preprocess_supervised_rows(
            [_row()],
            tokenizer,
            source_len=12,
            target_len=1,
        )


def test_preprocess_supervised_rows_reports_second_row_invalid_weight():
    tokenizer = _tokenizer()

    with pytest.raises(SupervisedDatasetError, match="row 2: weight must be"):
        preprocess_supervised_rows(
            [_row(weight=1.0), _row(weight=-1.0)],
            tokenizer,
            source_len=12,
            target_len=14,
        )


def test_iter_supervised_batches_drops_partial_batch_and_keeps_fixed_shapes():
    tokenizer = _tokenizer()
    dataset = preprocess_supervised_rows(
        [_row(weight=1.0), _row(weight=2.0), _row(weight=3.0)],
        tokenizer,
        source_len=12,
        target_len=14,
    )

    batches = list(iter_supervised_batches(dataset, batch_size=2))

    assert len(batches) == 1
    batch = batches[0]
    assert set(batch) == {
        "source_ids",
        "decoder_input_ids",
        "target_ids",
        "target_mask",
        "example_weight",
    }
    assert batch["source_ids"].shape == (2, 12)
    assert batch["decoder_input_ids"].shape == (2, 14)
    assert batch["target_ids"].shape == (2, 14)
    assert batch["target_mask"].shape == (2, 14)
    assert batch["example_weight"].shape == (2,)
    np.testing.assert_array_equal(batch["example_weight"], np.asarray([1.0, 2.0]))


def test_iter_supervised_batches_shuffles_with_provided_rng():
    tokenizer = _tokenizer()
    dataset = preprocess_supervised_rows(
        [_row(weight=1.0), _row(weight=2.0), _row(weight=3.0), _row(weight=4.0)],
        tokenizer,
        source_len=12,
        target_len=14,
    )

    batches = list(
        iter_supervised_batches(
            dataset,
            batch_size=2,
            shuffle=True,
            rng=np.random.default_rng(0),
        )
    )
    weights = np.concatenate([batch["example_weight"] for batch in batches])

    assert len(batches) == 2
    np.testing.assert_array_equal(np.sort(weights), np.asarray([1.0, 2.0, 3.0, 4.0]))
    assert weights.tolist() != [1.0, 2.0, 3.0, 4.0]


def test_iter_supervised_batches_rejects_non_positive_batch_size():
    tokenizer = _tokenizer()
    dataset = preprocess_supervised_rows(
        [_row()],
        tokenizer,
        source_len=12,
        target_len=14,
    )

    with pytest.raises(SupervisedDatasetError, match="batch_size must be positive"):
        iter_supervised_batches(dataset, batch_size=0)
