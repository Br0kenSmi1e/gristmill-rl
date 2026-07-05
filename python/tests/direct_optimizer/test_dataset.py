import json
import math

import pytest

from gristmill_symbolics.direct_optimizer.dataset import (
    BuildConfig,
    SplitConfig,
    build_processed_dataset,
    read_processed_jsonl,
    read_raw_candidates_jsonl,
    split_raw_candidates,
    write_processed_jsonl,
    write_raw_candidates_jsonl,
)
from tests.direct_optimizer.fixtures import source_comp_json, two_candidate_jsons


def _raw_record(
    *,
    candidate_json: str,
    outputs: list[int] | list[object] | None = None,
    candidate_log_flops: float = 1.0,
    input_computation: str | dict = source_comp_json(),
) -> dict:
    return {
        "input_computation": input_computation,
        "candidate_computation": candidate_json,
        "outputs": [1] if outputs is None else outputs,
        "initial_log_flops": 4.0,
        "candidate_log_flops": candidate_log_flops,
    }


def test_build_processed_dataset_preserves_outputs_and_weights_lower_cost_higher():
    low_json, high_json = two_candidate_jsons()
    rows = build_processed_dataset(
        [
            _raw_record(candidate_json=high_json, outputs=[1, 0], candidate_log_flops=3.0),
            _raw_record(candidate_json=low_json, outputs=[1, 0], candidate_log_flops=1.0),
        ],
        BuildConfig(beta=1.0),
    )

    assert [row["outputs"] for row in rows] == [[1, 0], [1, 0]]
    low_row = next(row for row in rows if row["candidate_log_flops"] == 1.0)
    high_row = next(row for row in rows if row["candidate_log_flops"] == 3.0)
    assert low_row["weight"] > high_row["weight"]
    assert math.isclose(low_row["weight"] + high_row["weight"], 1.0)


def test_different_output_orders_produce_different_input_groups():
    low_json, high_json = two_candidate_jsons()
    rows = build_processed_dataset(
        [
            _raw_record(candidate_json=low_json, outputs=[1, 0], candidate_log_flops=1.0),
            _raw_record(candidate_json=high_json, outputs=[0, 1], candidate_log_flops=1.0),
        ],
        BuildConfig(),
    )

    assert [row["outputs"] for row in rows] == [[1, 0], [0, 1]]
    assert rows[0]["input_key"] != rows[1]["input_key"]


@pytest.mark.parametrize("bad_outputs", [[1, 1], [True], [], ["1"]])
def test_build_processed_dataset_skips_malformed_outputs(bad_outputs):
    low_json, _ = two_candidate_jsons()

    assert (
        build_processed_dataset(
            [_raw_record(candidate_json=low_json, outputs=bad_outputs)],
            BuildConfig(),
        )
        == []
    )


def test_duplicate_candidates_keep_lowest_cost_and_single_weight():
    low_json, _ = two_candidate_jsons()
    rows = build_processed_dataset(
        [
            _raw_record(candidate_json=low_json, candidate_log_flops=3.0),
            _raw_record(candidate_json=low_json, candidate_log_flops=2.0),
        ],
        BuildConfig(),
    )

    assert len(rows) == 1
    assert rows[0]["candidate_log_flops"] == 2.0
    assert rows[0]["weight"] == 1.0


def test_split_raw_candidates_is_deterministic_and_preserves_records_unchanged():
    records = [{"id": i, "payload": {"value": i}} for i in range(10)]
    config = SplitConfig(train_fraction=0.6, valid_fraction=0.2, test_fraction=0.2, seed=7)

    first = split_raw_candidates(records, config)
    second = split_raw_candidates(records, config)

    assert first == second
    assert records == [{"id": i, "payload": {"value": i}} for i in range(10)]
    assert sorted(row["id"] for split in first for row in split) == list(range(10))
    assert [len(split) for split in first] == [6, 2, 2]


@pytest.mark.parametrize(
    "config",
    [
        SplitConfig(train_fraction=math.nan, valid_fraction=0.0, test_fraction=0.0),
        SplitConfig(train_fraction=-0.1, valid_fraction=0.6, test_fraction=0.5),
        SplitConfig(train_fraction=0.8, valid_fraction=0.1, test_fraction=0.2),
        SplitConfig(train_fraction=0.0, valid_fraction=0.5, test_fraction=0.5),
        SplitConfig(train_fraction=0.5, valid_fraction=0.5, test_fraction=0.0),
    ],
)
def test_split_raw_candidates_rejects_invalid_splits(config):
    with pytest.raises(ValueError):
        split_raw_candidates([{"id": 1}], config)


def test_jsonl_helpers_round_trip_raw_and_processed_rows(tmp_path):
    low_json, _ = two_candidate_jsons()
    source_object = json.loads(source_comp_json())
    raw_records = [
        _raw_record(
            candidate_json=low_json,
            input_computation=source_object,
            candidate_log_flops=2.0,
        )
    ]
    raw_path = tmp_path / "raw.jsonl"

    write_raw_candidates_jsonl(raw_records, raw_path)
    read_raw = read_raw_candidates_jsonl(raw_path)

    assert read_raw == raw_records
    processed_rows = build_processed_dataset(read_raw, BuildConfig())
    processed_path = tmp_path / "processed.jsonl"

    write_processed_jsonl(processed_rows, processed_path)

    assert read_processed_jsonl(processed_path) == processed_rows


@pytest.mark.parametrize("beta", [0.0, -1.0, math.inf, math.nan])
def test_build_processed_dataset_rejects_non_positive_or_non_finite_beta(beta):
    low_json, _ = two_candidate_jsons()

    with pytest.raises(ValueError):
        build_processed_dataset([_raw_record(candidate_json=low_json)], BuildConfig(beta=beta))
