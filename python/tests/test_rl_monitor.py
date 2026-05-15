import json

import pytest

from gristmill_rl.monitor import MonitorWriter
from gristmill_rl.monitor import load_baselines
from gristmill_rl.monitor import parse_baseline_arg

from .rl_fixtures import actionable_json


def test_parse_baseline_arg_accepts_name_and_path():
    name, path = parse_baseline_arg("greedy=outputs/greedy/final.json")

    assert name == "greedy"
    assert str(path) == "outputs/greedy/final.json"


@pytest.mark.parametrize("value", ["greedy", "=path.json", "   =path.json", "greedy="])
def test_parse_baseline_arg_rejects_malformed_values(value):
    with pytest.raises(ValueError):
        parse_baseline_arg(value)


def test_load_baselines_rejects_duplicate_names(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(actionable_json())

    with pytest.raises(ValueError, match="duplicate baseline name"):
        load_baselines(
            [
                ("greedy", baseline_path),
                ("greedy", baseline_path),
            ]
        )


def test_load_baselines_computes_log_flops(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(actionable_json())

    baselines = load_baselines([("greedy", baseline_path)])

    assert len(baselines) == 1
    assert baselines[0].name == "greedy"
    assert baselines[0].path == baseline_path
    assert baselines[0].log_flops > 0.0


def test_monitor_writer_writes_baselines_and_metrics_jsonl(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(actionable_json())
    baselines = load_baselines([("greedy", baseline_path)])

    writer = MonitorWriter(tmp_path / "run", baselines=baselines)
    writer.write_baselines()
    episode_metrics = {
        "episode": 1,
        "episodes": 2,
        "replay_size": 1,
        "episode_steps": 1,
        "episode_records": 1,
        "initial_log_flops": 13.0,
        "final_log_flops": 12.25,
        "last_policy_loss": 0.5,
        "last_value_loss": 0.25,
        "last_total_loss": 0.75,
        "params_changed": True,
    }
    writer.append_metrics(episode_metrics)

    baseline_doc = json.loads((tmp_path / "run" / "baselines.json").read_text())
    assert baseline_doc["baselines"][0]["name"] == "greedy"
    assert baseline_doc["baselines"][0]["path"] == str(baseline_path)
    assert baseline_doc["baselines"][0]["log_flops"] == baselines[0].log_flops

    lines = (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 1
    metrics = json.loads(lines[0])
    for key, value in episode_metrics.items():
        assert metrics[key] == value
    assert metrics["flops_improvement"] == pytest.approx(0.75)
