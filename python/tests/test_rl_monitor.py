import json
import urllib.error
import urllib.request

import pytest

from gristmill_rl.monitor import DASHBOARD_HTML
from gristmill_rl.monitor import MonitorServer
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


def test_monitor_writer_rejects_existing_metrics_jsonl(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.jsonl").write_text('{"episode": 99}\n')

    with pytest.raises(FileExistsError, match="metrics log already exists"):
        MonitorWriter(run_dir, baselines=[])


def test_monitor_server_serves_dashboard_and_json_api(tmp_path):
    writer = MonitorWriter(tmp_path / "run", baselines=[])
    writer.write_baselines()
    writer.append_metrics(
        {
            "episode": 1,
            "episodes": 1,
            "replay_size": 1,
            "episode_steps": 1,
            "episode_records": 1,
            "initial_log_flops": 13.0,
            "final_log_flops": 12.0,
            "last_policy_loss": 0.4,
            "last_value_loss": 0.2,
            "last_total_loss": 0.6,
            "params_changed": True,
        }
    )

    server = MonitorServer(tmp_path / "run")
    server.start()
    try:
        with urllib.request.urlopen(server.url, timeout=5) as response:
            html = response.read().decode("utf-8")
        assert "Gristmill RL Training Monitor" in html
        assert "/api/metrics" in html

        with urllib.request.urlopen(f"{server.url}/api/metrics", timeout=5) as response:
            metrics_doc = json.loads(response.read().decode("utf-8"))
        assert metrics_doc["metrics"][0]["episode"] == 1
        assert metrics_doc["metrics"][0]["flops_improvement"] == 1.0

        with urllib.request.urlopen(f"{server.url}/api/baselines", timeout=5) as response:
            baseline_doc = json.loads(response.read().decode("utf-8"))
        assert baseline_doc == {"baselines": []}
    finally:
        server.stop()


def test_monitor_server_metrics_api_reports_malformed_jsonl(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.jsonl").write_text("{bad json\n")
    (run_dir / "baselines.json").write_text('{"baselines": []}')

    server = MonitorServer(run_dir)
    server.start()
    try:
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"{server.url}/api/metrics", timeout=5)
        assert exc_info.value.code == 500
    finally:
        server.stop()


def test_monitor_server_url_requires_started_server(tmp_path):
    server = MonitorServer(tmp_path / "run")

    with pytest.raises(RuntimeError, match="monitor server has not started"):
        _ = server.url


def test_monitor_server_rejects_non_loopback_host(tmp_path):
    with pytest.raises(ValueError, match="monitor host must be loopback"):
        MonitorServer(tmp_path / "run", host="0.0.0.0")


def test_monitor_server_unknown_path_returns_not_found(tmp_path):
    server = MonitorServer(tmp_path / "run")
    server.start()
    try:
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"{server.url}/missing", timeout=5)
        assert exc_info.value.code == 404
    finally:
        server.stop()


def test_monitor_server_metrics_api_handles_missing_jsonl(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    server = MonitorServer(run_dir)
    server.start()
    try:
        with urllib.request.urlopen(f"{server.url}/api/metrics", timeout=5) as response:
            metrics_doc = json.loads(response.read().decode("utf-8"))
        assert metrics_doc["log_dir"] == str(run_dir)
        assert metrics_doc["metrics"] == []
    finally:
        server.stop()


def test_monitor_server_baselines_api_handles_missing_json(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    server = MonitorServer(run_dir)
    server.start()
    try:
        with urllib.request.urlopen(f"{server.url}/api/baselines", timeout=5) as response:
            baseline_doc = json.loads(response.read().decode("utf-8"))
        assert baseline_doc == {"baselines": []}
    finally:
        server.stop()


def test_dashboard_baseline_legend_uses_text_nodes_for_user_names():
    assert "key.innerHTML" not in DASHBOARD_HTML
    assert 'document.createTextNode(`${baseline.name} baseline`)' in DASHBOARD_HTML
