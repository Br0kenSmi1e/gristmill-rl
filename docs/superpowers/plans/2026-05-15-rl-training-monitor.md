# RL Training Monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in dependency-light graphical monitor for `python -m gristmill_rl.train`.

**Architecture:** Training remains the metrics producer. A new `gristmill_rl.monitor` module owns baseline loading, JSONL metric writing, a localhost HTTP server, and the browser dashboard. `train.py` only parses monitor flags, starts/stops the monitor, and appends already-built episode metrics.

**Tech Stack:** Python stdlib (`argparse`, `dataclasses`, `http.server`, `json`, `pathlib`, `socketserver`, `threading`), existing `gristmill_symbolics.TensorComputation`, pytest, browser inline SVG/JavaScript.

---

## File Structure

- Create `python/gristmill_rl/monitor.py`
  - `BaselineMetric` data class.
  - `parse_baseline_arg()`.
  - `load_baselines()`.
  - `MonitorWriter`.
  - `MonitorServer`.
  - Dashboard HTML/CSS/JavaScript constant.
  - JSON file readers used by the HTTP API.
- Create `python/tests/test_rl_monitor.py`
  - Unit tests for baseline parsing, duplicate handling, loading, writer output, and HTTP serving.
- Modify `python/gristmill_rl/train.py`
  - Add monitor fields to `RunnerConfig`.
  - Add CLI args and validation.
  - Start monitor when enabled.
  - Append per-episode metrics through `MonitorWriter`.
  - Preserve existing stdout JSON behavior.
- Modify `python/tests/test_rl_train.py`
  - CLI validation tests.
  - Tiny monitored training run test.

---

## Task 1: Baseline Parsing, Loading, And Metrics Writer

**Files:**
- Create: `python/tests/test_rl_monitor.py`
- Create: `python/gristmill_rl/monitor.py`

- [ ] **Step 1: Write failing monitor data tests**

Create `python/tests/test_rl_monitor.py` with:

```python
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


@pytest.mark.parametrize("value", ["greedy", "=path.json", "   =path.json"])
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
    writer.append_metrics(
        {
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
    )

    baseline_doc = json.loads((tmp_path / "run" / "baselines.json").read_text())
    assert baseline_doc["baselines"][0]["name"] == "greedy"
    assert baseline_doc["baselines"][0]["path"] == str(baseline_path)
    assert baseline_doc["baselines"][0]["log_flops"] == baselines[0].log_flops

    lines = (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 1
    metrics = json.loads(lines[0])
    assert metrics["episode"] == 1
    assert metrics["flops_improvement"] == pytest.approx(0.75)
```

- [ ] **Step 2: Run the monitor data tests and verify they fail**

Run:

```bash
cd python
uv run pytest tests/test_rl_monitor.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'gristmill_rl.monitor'`.

- [ ] **Step 3: Implement monitor data utilities and writer**

Create `python/gristmill_rl/monitor.py` with this initial content:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from gristmill_symbolics import TensorComputation


@dataclass(frozen=True)
class BaselineMetric:
    name: str
    path: Path
    log_flops: float

    def to_json(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "path": str(self.path),
            "log_flops": self.log_flops,
        }


def parse_baseline_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--baseline must use NAME=PATH")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError("--baseline name must not be empty")
    if not raw_path:
        raise ValueError("--baseline path must not be empty")
    return name, Path(raw_path)


def load_baselines(items: Sequence[tuple[str, Path]]) -> list[BaselineMetric]:
    seen: set[str] = set()
    baselines: list[BaselineMetric] = []
    for name, path in items:
        if name in seen:
            raise ValueError(f"duplicate baseline name: {name}")
        seen.add(name)
        comp = TensorComputation.load_json(path)
        baselines.append(
            BaselineMetric(name=name, path=path, log_flops=float(comp.log_total_flops()))
        )
    return baselines


class MonitorWriter:
    def __init__(self, log_dir: Path, *, baselines: Sequence[BaselineMetric]):
        self.log_dir = Path(log_dir)
        self.baselines = list(baselines)
        self.metrics_path = self.log_dir / "metrics.jsonl"
        self.baselines_path = self.log_dir / "baselines.json"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def write_baselines(self) -> None:
        payload = {"baselines": [baseline.to_json() for baseline in self.baselines]}
        self.baselines_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    def append_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(metrics)
        enriched["flops_improvement"] = float(
            enriched["initial_log_flops"] - enriched["final_log_flops"]
        )
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(enriched, sort_keys=True))
            handle.write("\n")
        return enriched
```

- [ ] **Step 4: Run the monitor data tests and verify they pass**

Run:

```bash
cd python
uv run pytest tests/test_rl_monitor.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add python/gristmill_rl/monitor.py python/tests/test_rl_monitor.py
git commit -m "feat: add rl monitor metrics writer"
```

---

## Task 2: HTTP Server And Dashboard

**Files:**
- Modify: `python/gristmill_rl/monitor.py`
- Modify: `python/tests/test_rl_monitor.py`

- [ ] **Step 1: Add failing HTTP server tests**

Append these imports to `python/tests/test_rl_monitor.py`:

```python
import urllib.error
import urllib.request

from gristmill_rl.monitor import MonitorServer
```

Append these tests to `python/tests/test_rl_monitor.py`:

```python
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
```

- [ ] **Step 2: Run the HTTP tests and verify they fail**

Run:

```bash
cd python
uv run pytest tests/test_rl_monitor.py::test_monitor_server_serves_dashboard_and_json_api tests/test_rl_monitor.py::test_monitor_server_metrics_api_reports_malformed_jsonl -v
```

Expected: FAIL with `ImportError` or `AttributeError` for `MonitorServer`.

- [ ] **Step 3: Add monitor server imports and JSON readers**

Add these imports near the top of `python/gristmill_rl/monitor.py`:

```python
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from urllib.parse import urlparse
```

Add these helper functions below `load_baselines()`:

```python
def read_metrics(log_dir: Path) -> list[dict[str, Any]]:
    metrics_path = Path(log_dir) / "metrics.jsonl"
    if not metrics_path.exists():
        return []
    metrics: list[dict[str, Any]] = []
    with metrics_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                metrics.append(json.loads(stripped))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON in {metrics_path} at line {line_number}: {error.msg}"
                ) from error
    return metrics


def read_baselines(log_dir: Path) -> dict[str, Any]:
    baselines_path = Path(log_dir) / "baselines.json"
    if not baselines_path.exists():
        return {"baselines": []}
    return json.loads(baselines_path.read_text())
```

- [ ] **Step 4: Add the dashboard HTML**

Add this constant near the bottom of `python/gristmill_rl/monitor.py`:

```python
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Gristmill RL Training Monitor</title>
  <style>
    :root {
      --bg: #f6f4ef;
      --panel: #ffffff;
      --ink: #1f2933;
      --muted: #68717d;
      --line: #d9d4c8;
      --green: #25865a;
      --blue: #2b6cb0;
      --red: #c75146;
      --gold: #b7791f;
      --purple: #6b46c1;
    }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      max-width: 1180px;
      margin: 0 auto;
      padding: 22px;
    }
    header, .baselines {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 20px;
    }
    h1 {
      font-size: 24px;
      margin: 0 0 4px;
      letter-spacing: 0;
    }
    .sub, .muted {
      color: var(--muted);
    }
    .sub {
      margin: 0;
    }
    .status {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 10px;
      background: #fffaf0;
      white-space: nowrap;
      color: #5f491c;
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin: 18px 0 14px;
    }
    .stat, .chart, .baselines {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 1px 0 rgba(31, 41, 51, 0.04);
    }
    .stat {
      padding: 12px;
      min-height: 88px;
    }
    .label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .04em;
      margin-bottom: 6px;
    }
    .value {
      font-size: 24px;
      font-weight: 700;
    }
    .delta {
      margin-top: 4px;
      color: var(--green);
      font-size: 12px;
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }
    .chart {
      min-height: 280px;
      padding: 14px 14px 10px;
    }
    .chart h2 {
      font-size: 15px;
      margin: 0 0 2px;
    }
    .chart p {
      color: var(--muted);
      margin: 0 0 10px;
      font-size: 12px;
    }
    svg {
      width: 100%;
      height: 210px;
      display: block;
      overflow: visible;
    }
    .axis {
      stroke: #c8c1b4;
      stroke-width: 1;
    }
    .grid-line {
      stroke: #eee8dd;
      stroke-width: 1;
    }
    .curve {
      fill: none;
      stroke-width: 2.5;
      stroke-linejoin: round;
      stroke-linecap: round;
    }
    .curve-thin {
      fill: none;
      stroke-width: 1.8;
      stroke-linejoin: round;
      stroke-linecap: round;
    }
    .baseline {
      stroke-width: 2;
      stroke-dasharray: 6 5;
    }
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      color: var(--muted);
      font-size: 12px;
      margin-top: 4px;
    }
    .key {
      display: inline-flex;
      align-items: center;
      gap: 5px;
    }
    .swatch {
      width: 12px;
      height: 3px;
      border-radius: 999px;
      display: inline-block;
      background: currentColor;
    }
    .baselines {
      margin-top: 14px;
      padding: 12px 14px;
      align-items: center;
      color: var(--muted);
    }
    .baseline-list {
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
    }
    code {
      background: #f1eee7;
      padding: 2px 5px;
      border-radius: 4px;
      color: #40464f;
    }
    @media (max-width: 800px) {
      header, .baselines {
        align-items: start;
        flex-direction: column;
      }
      .stats, .grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Gristmill RL Training Monitor</h1>
        <p class="sub" id="subtitle">Loading run metrics...</p>
      </div>
      <div class="status" id="status">connecting</div>
    </header>

    <section class="stats">
      <div class="stat">
        <div class="label">Episode</div>
        <div class="value" id="episode-value">-</div>
        <div class="delta" id="episode-detail">waiting for metrics</div>
      </div>
      <div class="stat">
        <div class="label">Final Log Flops</div>
        <div class="value" id="flops-value">-</div>
        <div class="delta" id="flops-detail">best: -</div>
      </div>
      <div class="stat">
        <div class="label">Improvement</div>
        <div class="value" id="improvement-value">-</div>
        <div class="delta">vs initial log flops</div>
      </div>
      <div class="stat">
        <div class="label">Total Loss</div>
        <div class="value" id="loss-value">-</div>
        <div class="delta" id="loss-detail">policy -, value -</div>
      </div>
    </section>

    <section class="grid">
      <div class="chart">
        <h2>Training Loss</h2>
        <p>Policy, value, and total loss per episode.</p>
        <svg id="loss-chart" viewBox="0 0 520 210" role="img" aria-label="Training loss curves"></svg>
        <div class="legend">
          <span class="key" style="color:#2b6cb0"><span class="swatch"></span>total</span>
          <span class="key" style="color:#25865a"><span class="swatch"></span>policy</span>
          <span class="key" style="color:#c75146"><span class="swatch"></span>value</span>
        </div>
      </div>

      <div class="chart">
        <h2>Flops Progress</h2>
        <p>RL final log flops compared with baseline final JSON outputs.</p>
        <svg id="flops-chart" viewBox="0 0 520 210" role="img" aria-label="Flops progress with baselines"></svg>
        <div class="legend" id="flops-legend">
          <span class="key" style="color:#8b949e"><span class="swatch"></span>initial</span>
          <span class="key" style="color:#2b6cb0"><span class="swatch"></span>RL final</span>
        </div>
      </div>

      <div class="chart">
        <h2>Improvement</h2>
        <p>Per-episode reduction in log flops from the starting computation.</p>
        <svg id="improvement-chart" viewBox="0 0 520 210" role="img" aria-label="Improvement curve"></svg>
        <div class="legend">
          <span class="key" style="color:#25865a"><span class="swatch"></span>initial_log_flops - final_log_flops</span>
        </div>
      </div>

      <div class="chart">
        <h2>Episode Steps</h2>
        <p>How many rewrite steps each episode used before stopping or hitting max steps.</p>
        <svg id="steps-chart" viewBox="0 0 520 210" role="img" aria-label="Episode steps bars"></svg>
        <div class="legend">
          <span class="key" style="color:#2b6cb0"><span class="swatch"></span>episode_steps</span>
        </div>
      </div>
    </section>

    <section class="baselines" id="baseline-footer">
      <div><strong>Baseline comparisons</strong> are loaded from <code>--baseline name=path.json</code> before training starts.</div>
      <div class="baseline-list" id="baseline-list">none</div>
    </section>
  </main>

  <script>
    const chart = {left: 36, right: 500, top: 20, bottom: 190, width: 464, height: 170};
    const baselineColors = ["#b7791f", "#6b46c1", "#805ad5", "#dd6b20", "#319795"];

    function fmt(value) {
      if (value === undefined || value === null || !Number.isFinite(Number(value))) {
        return "-";
      }
      return Number(value).toFixed(2);
    }

    function clearSvg(svg) {
      while (svg.firstChild) {
        svg.removeChild(svg.firstChild);
      }
    }

    function addLine(svg, className, x1, y1, x2, y2, stroke) {
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("class", className);
      line.setAttribute("x1", x1);
      line.setAttribute("y1", y1);
      line.setAttribute("x2", x2);
      line.setAttribute("y2", y2);
      if (stroke) {
        line.setAttribute("stroke", stroke);
      }
      svg.appendChild(line);
    }

    function addPolyline(svg, points, color, className) {
      const polyline = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
      polyline.setAttribute("class", className);
      polyline.setAttribute("stroke", color);
      polyline.setAttribute("points", points);
      svg.appendChild(polyline);
    }

    function addRect(svg, x, y, width, height, color) {
      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("x", x);
      rect.setAttribute("y", y);
      rect.setAttribute("width", width);
      rect.setAttribute("height", height);
      rect.setAttribute("fill", color);
      svg.appendChild(rect);
    }

    function drawFrame(svg) {
      [20, 64, 108, 152].forEach((y) => addLine(svg, "grid-line", chart.left, y, chart.right, y));
      addLine(svg, "axis", chart.left, chart.bottom, chart.right, chart.bottom);
      addLine(svg, "axis", chart.left, chart.top, chart.left, chart.bottom);
    }

    function scale(values) {
      const finite = values.filter((value) => Number.isFinite(value));
      if (finite.length === 0) {
        return {min: 0, max: 1};
      }
      let min = Math.min(...finite);
      let max = Math.max(...finite);
      if (min === max) {
        min -= 1;
        max += 1;
      }
      const pad = (max - min) * 0.08;
      return {min: min - pad, max: max + pad};
    }

    function pointSeries(values, yScale) {
      if (values.length === 1) {
        const x = chart.left;
        const y = chart.bottom - ((values[0] - yScale.min) / (yScale.max - yScale.min)) * chart.height;
        return `${x},${y}`;
      }
      return values.map((value, index) => {
        const x = chart.left + (index / Math.max(values.length - 1, 1)) * chart.width;
        const y = chart.bottom - ((value - yScale.min) / (yScale.max - yScale.min)) * chart.height;
        return `${x},${y}`;
      }).join(" ");
    }

    function drawSeriesChart(id, series) {
      const svg = document.getElementById(id);
      clearSvg(svg);
      drawFrame(svg);
      const values = series.flatMap((item) => item.values);
      if (values.length === 0) {
        return;
      }
      const yScale = scale(values);
      series.forEach((item) => {
        if (item.values.length > 0) {
          addPolyline(svg, pointSeries(item.values, yScale), item.color, item.className);
        }
      });
    }

    function drawFlopsChart(metrics, baselines) {
      const svg = document.getElementById("flops-chart");
      clearSvg(svg);
      drawFrame(svg);
      const initial = metrics.map((item) => Number(item.initial_log_flops));
      const final = metrics.map((item) => Number(item.final_log_flops));
      const baselineValues = baselines.map((item) => Number(item.log_flops));
      const values = initial.concat(final).concat(baselineValues);
      if (values.length === 0) {
        return;
      }
      const yScale = scale(values);
      if (initial.length > 0) {
        addPolyline(svg, pointSeries(initial, yScale), "#8b949e", "curve-thin");
      }
      if (final.length > 0) {
        addPolyline(svg, pointSeries(final, yScale), "#2b6cb0", "curve");
      }
      baselines.forEach((baseline, index) => {
        const y = chart.bottom - ((Number(baseline.log_flops) - yScale.min) / (yScale.max - yScale.min)) * chart.height;
        addLine(svg, "baseline", chart.left, y, chart.right, y, baselineColors[index % baselineColors.length]);
      });
    }

    function drawBarChart(id, values, color) {
      const svg = document.getElementById(id);
      clearSvg(svg);
      drawFrame(svg);
      if (values.length === 0) {
        return;
      }
      const yScale = scale([0].concat(values));
      const barWidth = Math.max(2, chart.width / values.length - 2);
      values.forEach((value, index) => {
        const x = chart.left + (index / values.length) * chart.width + 1;
        const y = chart.bottom - ((value - yScale.min) / (yScale.max - yScale.min)) * chart.height;
        addRect(svg, x, y, barWidth, chart.bottom - y, color);
      });
    }

    function updateSummary(metrics) {
      const latest = metrics[metrics.length - 1];
      if (!latest) {
        document.getElementById("status").textContent = "waiting for metrics";
        return;
      }
      const best = Math.min(...metrics.map((item) => Number(item.final_log_flops)));
      document.getElementById("episode-value").textContent = `${latest.episode} / ${latest.episodes}`;
      document.getElementById("episode-detail").textContent = "last update just now";
      document.getElementById("flops-value").textContent = fmt(latest.final_log_flops);
      document.getElementById("flops-detail").textContent = `best: ${fmt(best)}`;
      document.getElementById("improvement-value").textContent = fmt(latest.flops_improvement);
      document.getElementById("loss-value").textContent = fmt(latest.last_total_loss);
      document.getElementById("loss-detail").textContent = `policy ${fmt(latest.last_policy_loss)}, value ${fmt(latest.last_value_loss)}`;
      document.getElementById("status").textContent = `live - ${metrics.length} metric record${metrics.length === 1 ? "" : "s"}`;
    }

    function updateBaselines(baselines) {
      const list = document.getElementById("baseline-list");
      const legend = document.getElementById("flops-legend");
      list.textContent = "";
      document.querySelectorAll(".baseline-key").forEach((node) => node.remove());
      if (baselines.length === 0) {
        list.textContent = "none";
        return;
      }
      baselines.forEach((baseline, index) => {
        const color = baselineColors[index % baselineColors.length];
        const span = document.createElement("span");
        span.textContent = `${baseline.name}: ${fmt(baseline.log_flops)}`;
        list.appendChild(span);

        const key = document.createElement("span");
        key.className = "key baseline-key";
        key.style.color = color;
        key.innerHTML = `<span class="swatch"></span>${baseline.name} baseline`;
        legend.appendChild(key);
      });
    }

    async function refresh() {
      try {
        const [metricsResponse, baselinesResponse] = await Promise.all([
          fetch("/api/metrics"),
          fetch("/api/baselines"),
        ]);
        if (!metricsResponse.ok || !baselinesResponse.ok) {
          throw new Error("monitor API returned an error");
        }
        const metricsDoc = await metricsResponse.json();
        const baselinesDoc = await baselinesResponse.json();
        const metrics = metricsDoc.metrics || [];
        const baselines = baselinesDoc.baselines || [];
        document.getElementById("subtitle").textContent = metricsDoc.log_dir || "training run";
        updateSummary(metrics);
        updateBaselines(baselines);
        drawSeriesChart("loss-chart", [
          {values: metrics.map((item) => Number(item.last_total_loss)), color: "#2b6cb0", className: "curve"},
          {values: metrics.map((item) => Number(item.last_policy_loss)), color: "#25865a", className: "curve-thin"},
          {values: metrics.map((item) => Number(item.last_value_loss)), color: "#c75146", className: "curve-thin"},
        ]);
        drawFlopsChart(metrics, baselines);
        drawSeriesChart("improvement-chart", [
          {values: metrics.map((item) => Number(item.flops_improvement)), color: "#25865a", className: "curve"},
        ]);
        drawBarChart("steps-chart", metrics.map((item) => Number(item.episode_steps)), "#2b6cb0");
      } catch (error) {
        document.getElementById("status").textContent = "monitor error";
      }
    }

    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>
"""
```

- [ ] **Step 5: Add `MonitorServer`**

Add this class below `DASHBOARD_HTML` in `python/gristmill_rl/monitor.py`:

```python
class MonitorServer:
    def __init__(self, log_dir: Path, *, host: str = "127.0.0.1", port: int = 0):
        self.log_dir = Path(log_dir)
        self.host = host
        self.port = port
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: Thread | None = None

    @property
    def url(self) -> str:
        if self._httpd is None:
            raise RuntimeError("monitor server has not started")
        host, port = self._httpd.server_address
        display_host = "localhost" if host in {"127.0.0.1", "0.0.0.0"} else host
        return f"http://{display_host}:{port}"

    def start(self) -> None:
        if self._httpd is not None:
            return
        log_dir = self.log_dir

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                return

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                try:
                    if parsed.path == "/":
                        self._send_html(DASHBOARD_HTML)
                    elif parsed.path == "/api/metrics":
                        self._send_json(
                            {"log_dir": str(log_dir), "metrics": read_metrics(log_dir)}
                        )
                    elif parsed.path == "/api/baselines":
                        self._send_json(read_baselines(log_dir))
                    else:
                        self.send_error(HTTPStatus.NOT_FOUND, "not found")
                except ValueError as error:
                    self._send_json({"error": str(error)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

            def _send_html(self, html: str) -> None:
                body = html.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_json(
                self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK
            ) -> None:
                body = json.dumps(payload, sort_keys=True).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._httpd is None:
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._httpd = None
        self._thread = None
```

- [ ] **Step 6: Run the HTTP tests and verify they pass**

Run:

```bash
cd python
uv run pytest tests/test_rl_monitor.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit Task 2**

Run:

```bash
git add python/gristmill_rl/monitor.py python/tests/test_rl_monitor.py
git commit -m "feat: serve rl monitor dashboard"
```

---

## Task 3: Train CLI Integration

**Files:**
- Modify: `python/gristmill_rl/train.py`
- Modify: `python/tests/test_rl_train.py`

- [ ] **Step 1: Add failing CLI validation tests**

Append these tests to `python/tests/test_rl_train.py`:

```python
def test_train_parse_args_accepts_monitor_options(tmp_path):
    config = train.parse_args(
        [
            "--input",
            str(tmp_path / "input.json"),
            "--monitor",
            "--log-dir",
            str(tmp_path / "run"),
            "--baseline",
            "greedy=greedy.json",
            "--baseline",
            "random=random.json",
        ]
    )

    assert config.monitor
    assert config.log_dir == tmp_path / "run"
    assert config.baselines == (
        ("greedy", Path("greedy.json")),
        ("random", Path("random.json")),
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["--input", "input.json", "--monitor"],
        ["--input", "input.json", "--log-dir", "run"],
        ["--input", "input.json", "--baseline", "greedy=out.json"],
    ],
)
def test_train_parse_args_rejects_invalid_monitor_combinations(argv):
    with pytest.raises(SystemExit):
        train.parse_args(argv)
```

Add these imports to the top of `python/tests/test_rl_train.py`:

```python
from pathlib import Path

import pytest

import gristmill_rl.train as train
```

- [ ] **Step 2: Run the new CLI validation tests and verify they fail**

Run:

```bash
cd python
uv run pytest tests/test_rl_train.py::test_train_parse_args_accepts_monitor_options tests/test_rl_train.py::test_train_parse_args_rejects_invalid_monitor_combinations -v
```

Expected: FAIL because `RunnerConfig` and `parse_args()` do not support monitor options.

- [ ] **Step 3: Extend `RunnerConfig` and imports**

In `python/gristmill_rl/train.py`, add these imports:

```python
from gristmill_rl.monitor import MonitorServer, MonitorWriter
from gristmill_rl.monitor import load_baselines, parse_baseline_arg
```

Update `RunnerConfig` with:

```python
    monitor: bool = False
    log_dir: Path | None = None
    baselines: tuple[tuple[str, Path], ...] = ()
```

- [ ] **Step 4: Extend `parse_args()`**

In `python/gristmill_rl/train.py`, add parser arguments after checkpoint args:

```python
    parser.add_argument(
        "--monitor",
        action="store_true",
        default=RunnerConfig.monitor,
        help="Start a local browser monitor and write metrics under --log-dir.",
    )
    parser.add_argument("--log-dir", type=Path, default=RunnerConfig.log_dir)
    parser.add_argument(
        "--baseline",
        action="append",
        default=[],
        help="Named baseline final JSON in NAME=PATH form. May be repeated.",
    )
```

After `args = parser.parse_args(argv)`, insert these parsing checks:

```python
    if args.monitor and args.log_dir is None:
        parser.error("--monitor requires --log-dir")
    if args.log_dir is not None and not args.monitor:
        parser.error("--log-dir requires --monitor")
    if args.baseline and not args.monitor:
        parser.error("--baseline requires --monitor")
    try:
        baselines = tuple(parse_baseline_arg(value) for value in args.baseline)
    except ValueError as error:
        parser.error(str(error))
```

Add these fields to the returned `RunnerConfig`:

```python
        monitor=args.monitor,
        log_dir=args.log_dir,
        baselines=baselines,
```

- [ ] **Step 5: Run the CLI validation tests and verify they pass**

Run:

```bash
cd python
uv run pytest tests/test_rl_train.py::test_train_parse_args_accepts_monitor_options tests/test_rl_train.py::test_train_parse_args_rejects_invalid_monitor_combinations -v
```

Expected: PASS.

- [ ] **Step 6: Add failing monitored training run test**

Append this test to `python/tests/test_rl_train.py`:

```python
def test_train_cli_monitor_writes_run_artifacts(tmp_path):
    input_path = tmp_path / "input.json"
    baseline_path = tmp_path / "baseline.json"
    log_dir = tmp_path / "run"
    input_path.write_text(actionable_json())
    baseline_path.write_text(actionable_json())

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gristmill_rl.train",
            "--input",
            str(input_path),
            "--episodes",
            "1",
            "--max-steps",
            "1",
            "--simulations",
            "2",
            "--actions-per-node",
            "1",
            "--sample-attempts",
            "4",
            "--train-steps",
            "1",
            "--batch-size",
            "1",
            "--seed",
            "0",
            "--monitor",
            "--log-dir",
            str(log_dir),
            "--baseline",
            f"greedy={baseline_path}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    stdout_lines = result.stdout.strip().splitlines()
    assert any(line.startswith("monitor_url=") for line in stdout_lines)

    final_metrics = json.loads(stdout_lines[-1])
    assert final_metrics["episodes"] == 1

    baseline_doc = json.loads((log_dir / "baselines.json").read_text())
    assert baseline_doc["baselines"][0]["name"] == "greedy"
    assert baseline_doc["baselines"][0]["path"] == str(baseline_path)
    assert baseline_doc["baselines"][0]["log_flops"] > 0.0

    metrics_lines = (log_dir / "metrics.jsonl").read_text().splitlines()
    assert len(metrics_lines) == 1
    episode_metrics = json.loads(metrics_lines[0])
    assert episode_metrics["episode"] == 1
    assert episode_metrics["flops_improvement"] == pytest.approx(
        episode_metrics["initial_log_flops"] - episode_metrics["final_log_flops"]
    )
```

- [ ] **Step 7: Run the monitored training test and verify it fails**

Run:

```bash
cd python
uv run pytest tests/test_rl_train.py::test_train_cli_monitor_writes_run_artifacts -v
```

Expected: FAIL because `train.run()` does not start the monitor or write run artifacts.

- [ ] **Step 8: Integrate monitor setup into `run()`**

Replace the existing `run()` function in `python/gristmill_rl/train.py` with:

```python
def run(config: RunnerConfig) -> dict[str, float | int | bool | str | None]:
    rng = np.random.default_rng(config.seed)
    checkpoint_in: str | None = None
    if config.checkpoint_in is None:
        hidden_dim = config.hidden_dim if config.hidden_dim is not None else 32
        model = PolicyValueModel(hidden_dim=hidden_dim, rng_seed=config.seed)
        feature_config = FeatureConfig()
    else:
        loaded = load_checkpoint(config.checkpoint_in)
        hidden_dim = loaded.metadata.hidden_dim
        if config.hidden_dim is not None and config.hidden_dim != hidden_dim:
            raise ValueError(
                f"--hidden-dim {config.hidden_dim} does not match checkpoint "
                f"hidden_dim {hidden_dim}"
            )
        model = loaded.model
        feature_config = loaded.feature_config
        checkpoint_in = str(config.checkpoint_in)
    replay = ReplayBuffer(capacity=config.replay_capacity, seed=config.seed)
    train_config = TrainConfig()
    rollout_config = RolloutConfig(
        max_steps=config.max_steps,
        simulations=config.simulations,
        actions_per_node=config.actions_per_node,
        sample_attempts=config.sample_attempts,
        temperature=config.temperature,
        c_puct=config.c_puct,
    )

    monitor_writer: MonitorWriter | None = None
    monitor_server: MonitorServer | None = None
    if config.monitor:
        if config.log_dir is None:
            raise ValueError("--monitor requires --log-dir")
        baselines = load_baselines(config.baselines)
        monitor_writer = MonitorWriter(config.log_dir, baselines=baselines)
        monitor_writer.write_baselines()
        monitor_server = MonitorServer(config.log_dir)
        monitor_server.start()
        print(f"monitor_url={monitor_server.url}")

    last_total_loss = 0.0
    last_policy_loss = 0.0
    last_value_loss = 0.0
    params_changed = False
    last_initial_log_flops = 0.0
    last_final_log_flops = 0.0
    last_episode_steps = 0
    last_episode_records = 0
    checkpoint_out = str(config.checkpoint_out) if config.checkpoint_out else None

    try:
        for episode in range(config.episodes):
            rollout = run_policy_rollout(
                _load_comp(config.input),
                model=model,
                feature_config=feature_config,
                config=rollout_config,
                rng=rng,
            )
            last_initial_log_flops = rollout.initial_log_flops
            last_final_log_flops = rollout.final_log_flops
            last_episode_steps = rollout.steps
            completed_items = rollout.trace.complete(final_log_flops=last_final_log_flops)
            replay.extend(completed_items)
            last_episode_records = len(completed_items)

            for _ in range(config.train_steps):
                if len(replay) == 0:
                    break
                metrics = train_step(
                    model,
                    batch=_item_batch(
                        replay.sample(batch_size=config.batch_size), feature_config
                    ),
                    config=train_config,
                )
                last_policy_loss = float(metrics["policy_loss"])
                last_value_loss = float(metrics["value_loss"])
                last_total_loss = float(metrics["total_loss"])
                params_changed = bool(params_changed or metrics["params_changed"])

            episode_metrics: dict[str, float | int | bool] = {
                "episode": episode + 1,
                "episodes": config.episodes,
                "replay_size": len(replay),
                "episode_steps": last_episode_steps,
                "episode_records": last_episode_records,
                "initial_log_flops": last_initial_log_flops,
                "final_log_flops": last_final_log_flops,
                "last_policy_loss": last_policy_loss,
                "last_value_loss": last_value_loss,
                "last_total_loss": last_total_loss,
                "params_changed": params_changed,
            }
            print(json.dumps(episode_metrics, sort_keys=True))
            if monitor_writer is not None:
                monitor_writer.append_metrics(episode_metrics)

        if config.checkpoint_out is not None:
            save_checkpoint(
                config.checkpoint_out,
                model=model,
                feature_config=feature_config,
                hidden_dim=hidden_dim,
                metadata={"seed": config.seed, "episodes": config.episodes},
                overwrite=config.checkpoint_overwrite,
            )
    finally:
        if monitor_server is not None:
            monitor_server.stop()

    return {
        "episodes": config.episodes,
        "replay_size": len(replay),
        "last_episode_steps": last_episode_steps,
        "last_episode_records": last_episode_records,
        "initial_log_flops": last_initial_log_flops,
        "final_log_flops": last_final_log_flops,
        "last_policy_loss": last_policy_loss,
        "last_value_loss": last_value_loss,
        "last_total_loss": last_total_loss,
        "params_changed": params_changed,
        "checkpoint_in": checkpoint_in,
        "checkpoint_out": checkpoint_out,
    }
```

- [ ] **Step 9: Run the monitored training test and verify it passes**

Run:

```bash
cd python
uv run pytest tests/test_rl_train.py::test_train_cli_monitor_writes_run_artifacts -v
```

Expected: PASS.

- [ ] **Step 10: Run the full train test file**

Run:

```bash
cd python
uv run pytest tests/test_rl_train.py -v
```

Expected: PASS.

- [ ] **Step 11: Commit Task 3**

Run:

```bash
git add python/gristmill_rl/train.py python/tests/test_rl_train.py
git commit -m "feat: integrate rl training monitor cli"
```

---

## Task 4: Verification And Manual Monitor Check

**Files:**
- Modify only if verification finds a defect.

- [ ] **Step 1: Run focused monitor and train tests**

Run:

```bash
cd python
uv run pytest tests/test_rl_monitor.py tests/test_rl_train.py -v
```

Expected: PASS.

- [ ] **Step 2: Run broader RL tests**

Run:

```bash
cd python
uv run pytest tests/test_rl_model.py tests/test_rl_rollout.py tests/test_rl_search.py tests/test_rl_replay.py tests/test_rl_features.py tests/test_rl_actions.py tests/test_rl_sample.py tests/test_rl_checkpoint.py tests/test_rl_train.py -v
```

Expected: PASS.

- [ ] **Step 3: Run a manual monitored training command**

Run:

```bash
cd python
uv run python -m gristmill_rl.train \
  --input ../tests/fixtures/repr/basic.json \
  --episodes 2 \
  --max-steps 1 \
  --simulations 2 \
  --actions-per-node 1 \
  --sample-attempts 4 \
  --train-steps 1 \
  --batch-size 1 \
  --seed 0 \
  --monitor \
  --log-dir /tmp/gristmill-rl-monitor-demo \
  --baseline basic=../tests/fixtures/repr/basic.json
```

Expected:

- stdout includes a `monitor_url=http://localhost:<port>` line.
- stdout still ends with final JSON metrics.
- `/tmp/gristmill-rl-monitor-demo/metrics.jsonl` exists with two lines.
- `/tmp/gristmill-rl-monitor-demo/baselines.json` exists.

- [ ] **Step 4: Open the printed monitor URL**

Open the printed URL in a browser.

Expected:

- The dashboard title is `Gristmill RL Training Monitor`.
- Four summary cards are visible.
- The chart grid matches the approved demo layout.
- The baseline footer includes `basic`.
- The browser page does not require internet access.

- [ ] **Step 5: Final status check**

Run:

```bash
git status --short
```

Expected: only intentional tracked source/test changes are present.
If `.superpowers/` appears because the brainstorming visual companion was used,
leave it untracked and do not include it in implementation commits.

- [ ] **Step 6: Commit verification fixes if any were needed**

If Step 1 through Step 5 required fixes, commit them:

```bash
git add python/gristmill_rl/monitor.py python/gristmill_rl/train.py python/tests/test_rl_monitor.py python/tests/test_rl_train.py
git commit -m "fix: polish rl training monitor"
```

If no fixes were needed, do not create an empty commit.

---

## Spec Coverage Self-Review

- CLI coverage: Task 3 adds `--monitor`, `--log-dir`, repeated `--baseline`, and invalid-combination tests.
- Artifact coverage: Task 1 writes `metrics.jsonl` and `baselines.json`.
- Baseline coverage: Task 1 parses, deduplicates, loads, and computes `log_total_flops()`.
- Dashboard coverage: Task 2 serves HTML matching the approved layout and exposes `/api/metrics` plus `/api/baselines`.
- Error coverage: Task 1 and Task 3 cover malformed baselines and invalid CLI combinations; Task 2 covers malformed JSONL API errors.
- Existing behavior coverage: Task 3 checks final stdout JSON remains parseable; Task 4 runs existing train and RL tests.
- Dependency coverage: All implementation steps use stdlib plus existing project dependencies.
