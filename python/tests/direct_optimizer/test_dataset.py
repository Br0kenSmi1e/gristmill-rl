import json
import math
import subprocess
import sys

import pytest

from gristmill_symbolics import TensorComputation
from gristmill_symbolics.direct_optimizer import dataset
from gristmill_symbolics.direct_optimizer.dataset import (
    BuildConfig,
    GenerationConfig,
    SplitConfig,
    build_processed_dataset,
    generate_raw_candidates,
    main as dataset_main,
    read_processed_jsonl,
    read_raw_candidates_jsonl,
    split_raw_candidates,
    write_processed_jsonl,
    write_raw_candidates_jsonl,
)
from tests.direct_optimizer.fixtures import source_comp_json, two_candidate_jsons
from tests.test_bindings import actionable_json


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


def _equivalent_raw_records(count: int) -> list[dict]:
    return [
        _raw_record(
            candidate_json=source_comp_json(),
            candidate_log_flops=1.0 + index,
        )
        for index in range(count)
    ]


def test_generate_raw_candidates_is_deterministic_and_skips_initial_state():
    comp = TensorComputation.from_json_string(actionable_json())
    config = GenerationConfig(seed=3, trajectories_per_input=2, max_steps=1)

    first = generate_raw_candidates([(comp, [3])], config)
    second = generate_raw_candidates(
        [(TensorComputation.from_json_string(actionable_json()), [3])],
        config,
    )

    assert first == second
    assert first
    assert all(
        row["input_computation"] != row["candidate_computation"] for row in first
    )
    assert all(row["outputs"] == [3] for row in first)
    assert set(first[0]) >= {
        "input_computation",
        "candidate_computation",
        "outputs",
        "initial_log_flops",
        "candidate_log_flops",
    }


def test_dataset_build_splits_cli_processes_each_split_independently(tmp_path):
    raw_path = tmp_path / "raw.jsonl"
    train_path = tmp_path / "train.jsonl"
    valid_path = tmp_path / "valid.jsonl"
    test_path = tmp_path / "test.jsonl"
    write_raw_candidates_jsonl(_equivalent_raw_records(8), raw_path)

    exit_code = dataset_main(
        [
            "build-splits",
            "--raw-input",
            str(raw_path),
            "--train-output",
            str(train_path),
            "--valid-output",
            str(valid_path),
            "--test-output",
            str(test_path),
            "--train-fraction",
            "0.5",
            "--valid-fraction",
            "0.25",
            "--test-fraction",
            "0.25",
            "--split-seed",
            "9",
            "--beta",
            "1.0",
            "--verify",
        ]
    )

    assert exit_code == 0
    for path in (train_path, valid_path, test_path):
        rows = read_processed_jsonl(path)
        assert rows
        groups = {}
        for row in rows:
            groups.setdefault(row["input_key"], 0.0)
            groups[row["input_key"]] += row["weight"]
        assert all(math.isclose(total, 1.0) for total in groups.values())


def test_dataset_module_help_smoke():
    result = subprocess.run(
        [sys.executable, "-m", "gristmill_symbolics.direct_optimizer.dataset", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "build-splits" in result.stdout


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


def test_build_processed_dataset_skips_invalid_input_computation():
    low_json, _ = two_candidate_jsons()

    assert (
        build_processed_dataset(
            [_raw_record(candidate_json=low_json, input_computation="{")],
            BuildConfig(),
        )
        == []
    )


def test_build_processed_dataset_skips_invalid_candidate_computation():
    assert (
        build_processed_dataset(
            [_raw_record(candidate_json="{")],
            BuildConfig(),
        )
        == []
    )


def test_build_processed_dataset_uses_provided_costs_without_recomputing(monkeypatch):
    snapshot = json.loads(source_comp_json())

    class FakeTensorComputation:
        @staticmethod
        def from_json_string(_text):
            return FakeTensorComputation()

        def snapshot(self):
            return snapshot

        def log_total_flops(self):
            raise AssertionError("log_total_flops should not be called")

    monkeypatch.setattr(dataset, "TensorComputation", FakeTensorComputation)

    rows = build_processed_dataset(
        [_raw_record(candidate_json=source_comp_json(), candidate_log_flops=2.0)],
        BuildConfig(),
    )

    assert len(rows) == 1
    assert rows[0]["initial_log_flops"] == 4.0
    assert rows[0]["candidate_log_flops"] == 2.0


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
