from __future__ import annotations

import json
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any, Sequence
from urllib.parse import urlparse

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
    body { margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    main { max-width: 1180px; margin: 0 auto; padding: 22px; }
    header, .baselines { display: flex; align-items: end; justify-content: space-between; gap: 20px; }
    h1 { font-size: 24px; margin: 0 0 4px; letter-spacing: 0; }
    .sub, .muted { color: var(--muted); }
    .sub { margin: 0; }
    .status { border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; background: #fffaf0; white-space: nowrap; color: #5f491c; }
    .stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 18px 0 14px; }
    .stat, .chart, .baselines { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; box-shadow: 0 1px 0 rgba(31, 41, 51, 0.04); }
    .stat { padding: 12px; min-height: 88px; }
    .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; margin-bottom: 6px; }
    .value { font-size: 24px; font-weight: 700; }
    .delta { margin-top: 4px; color: var(--green); font-size: 12px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    .chart { min-height: 280px; padding: 14px 14px 10px; }
    .chart h2 { font-size: 15px; margin: 0 0 2px; }
    .chart p { color: var(--muted); margin: 0 0 10px; font-size: 12px; }
    svg { width: 100%; height: 210px; display: block; overflow: visible; }
    .axis { stroke: #c8c1b4; stroke-width: 1; }
    .grid-line { stroke: #eee8dd; stroke-width: 1; }
    .curve { fill: none; stroke-width: 2.5; stroke-linejoin: round; stroke-linecap: round; }
    .curve-thin { fill: none; stroke-width: 1.8; stroke-linejoin: round; stroke-linecap: round; }
    .baseline { stroke-width: 2; stroke-dasharray: 6 5; }
    .legend { display: flex; flex-wrap: wrap; gap: 10px; color: var(--muted); font-size: 12px; margin-top: 4px; }
    .key { display: inline-flex; align-items: center; gap: 5px; }
    .swatch { width: 12px; height: 3px; border-radius: 999px; display: inline-block; background: currentColor; }
    .baselines { margin-top: 14px; padding: 12px 14px; align-items: center; color: var(--muted); }
    .baseline-list { display: flex; flex-wrap: wrap; gap: 14px; }
    code { background: #f1eee7; padding: 2px 5px; border-radius: 4px; color: #40464f; }
    @media (max-width: 800px) { header, .baselines { align-items: start; flex-direction: column; } .stats, .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <main>
    <header>
      <div><h1>Gristmill RL Training Monitor</h1><p class="sub" id="subtitle">Loading run metrics...</p></div>
      <div class="status" id="status">connecting</div>
    </header>
    <section class="stats">
      <div class="stat"><div class="label">Episode</div><div class="value" id="episode-value">-</div><div class="delta" id="episode-detail">waiting for metrics</div></div>
      <div class="stat"><div class="label">Final Log Flops</div><div class="value" id="flops-value">-</div><div class="delta" id="flops-detail">best: -</div></div>
      <div class="stat"><div class="label">Improvement</div><div class="value" id="improvement-value">-</div><div class="delta">vs initial log flops</div></div>
      <div class="stat"><div class="label">Total Loss</div><div class="value" id="loss-value">-</div><div class="delta" id="loss-detail">policy -, value -</div></div>
    </section>
    <section class="grid">
      <div class="chart"><h2>Training Loss</h2><p>Policy, value, and total loss per episode.</p><svg id="loss-chart" viewBox="0 0 520 210" role="img" aria-label="Training loss curves"></svg><div class="legend"><span class="key" style="color:#2b6cb0"><span class="swatch"></span>total</span><span class="key" style="color:#25865a"><span class="swatch"></span>policy</span><span class="key" style="color:#c75146"><span class="swatch"></span>value</span></div></div>
      <div class="chart"><h2>Flops Progress</h2><p>RL final log flops compared with baseline final JSON outputs.</p><svg id="flops-chart" viewBox="0 0 520 210" role="img" aria-label="Flops progress with baselines"></svg><div class="legend" id="flops-legend"><span class="key" style="color:#8b949e"><span class="swatch"></span>initial</span><span class="key" style="color:#2b6cb0"><span class="swatch"></span>RL final</span></div></div>
      <div class="chart"><h2>Improvement</h2><p>Per-episode reduction in log flops from the starting computation.</p><svg id="improvement-chart" viewBox="0 0 520 210" role="img" aria-label="Improvement curve"></svg><div class="legend"><span class="key" style="color:#25865a"><span class="swatch"></span>initial_log_flops - final_log_flops</span></div></div>
      <div class="chart"><h2>Episode Steps</h2><p>How many rewrite steps each episode used before stopping or hitting max steps.</p><svg id="steps-chart" viewBox="0 0 520 210" role="img" aria-label="Episode steps bars"></svg><div class="legend"><span class="key" style="color:#2b6cb0"><span class="swatch"></span>episode_steps</span></div></div>
    </section>
    <section class="baselines" id="baseline-footer"><div><strong>Baseline comparisons</strong> are loaded from <code>--baseline name=path.json</code> before training starts.</div><div class="baseline-list" id="baseline-list">none</div></section>
  </main>
  <script>
    const chart = {left: 36, right: 500, top: 20, bottom: 190, width: 464, height: 170};
    const baselineColors = ["#b7791f", "#6b46c1", "#805ad5", "#dd6b20", "#319795"];
    function fmt(value) { if (value === undefined || value === null || !Number.isFinite(Number(value))) { return "-"; } return Number(value).toFixed(2); }
    function clearSvg(svg) { while (svg.firstChild) { svg.removeChild(svg.firstChild); } }
    function addLine(svg, className, x1, y1, x2, y2, stroke) { const line = document.createElementNS("http://www.w3.org/2000/svg", "line"); line.setAttribute("class", className); line.setAttribute("x1", x1); line.setAttribute("y1", y1); line.setAttribute("x2", x2); line.setAttribute("y2", y2); if (stroke) { line.setAttribute("stroke", stroke); } svg.appendChild(line); }
    function addPolyline(svg, points, color, className) { const polyline = document.createElementNS("http://www.w3.org/2000/svg", "polyline"); polyline.setAttribute("class", className); polyline.setAttribute("stroke", color); polyline.setAttribute("points", points); svg.appendChild(polyline); }
    function addRect(svg, x, y, width, height, color) { const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect"); rect.setAttribute("x", x); rect.setAttribute("y", y); rect.setAttribute("width", width); rect.setAttribute("height", height); rect.setAttribute("fill", color); svg.appendChild(rect); }
    function drawFrame(svg) { [20, 64, 108, 152].forEach((y) => addLine(svg, "grid-line", chart.left, y, chart.right, y)); addLine(svg, "axis", chart.left, chart.bottom, chart.right, chart.bottom); addLine(svg, "axis", chart.left, chart.top, chart.left, chart.bottom); }
    function scale(values) { const finite = values.filter((value) => Number.isFinite(value)); if (finite.length === 0) { return {min: 0, max: 1}; } let min = Math.min(...finite); let max = Math.max(...finite); if (min === max) { min -= 1; max += 1; } const pad = (max - min) * 0.08; return {min: min - pad, max: max + pad}; }
    function pointSeries(values, yScale) { if (values.length === 1) { const x = chart.left; const y = chart.bottom - ((values[0] - yScale.min) / (yScale.max - yScale.min)) * chart.height; return `${x},${y}`; } return values.map((value, index) => { const x = chart.left + (index / Math.max(values.length - 1, 1)) * chart.width; const y = chart.bottom - ((value - yScale.min) / (yScale.max - yScale.min)) * chart.height; return `${x},${y}`; }).join(" "); }
    function drawSeriesChart(id, series) { const svg = document.getElementById(id); clearSvg(svg); drawFrame(svg); const values = series.flatMap((item) => item.values); if (values.length === 0) { return; } const yScale = scale(values); series.forEach((item) => { if (item.values.length > 0) { addPolyline(svg, pointSeries(item.values, yScale), item.color, item.className); } }); }
    function drawFlopsChart(metrics, baselines) { const svg = document.getElementById("flops-chart"); clearSvg(svg); drawFrame(svg); const initial = metrics.map((item) => Number(item.initial_log_flops)); const final = metrics.map((item) => Number(item.final_log_flops)); const baselineValues = baselines.map((item) => Number(item.log_flops)); const values = initial.concat(final).concat(baselineValues); if (values.length === 0) { return; } const yScale = scale(values); if (initial.length > 0) { addPolyline(svg, pointSeries(initial, yScale), "#8b949e", "curve-thin"); } if (final.length > 0) { addPolyline(svg, pointSeries(final, yScale), "#2b6cb0", "curve"); } baselines.forEach((baseline, index) => { const y = chart.bottom - ((Number(baseline.log_flops) - yScale.min) / (yScale.max - yScale.min)) * chart.height; addLine(svg, "baseline", chart.left, y, chart.right, y, baselineColors[index % baselineColors.length]); }); }
    function drawBarChart(id, values, color) { const svg = document.getElementById(id); clearSvg(svg); drawFrame(svg); if (values.length === 0) { return; } const yScale = scale([0].concat(values)); const barWidth = Math.max(2, chart.width / values.length - 2); values.forEach((value, index) => { const x = chart.left + (index / values.length) * chart.width + 1; const y = chart.bottom - ((value - yScale.min) / (yScale.max - yScale.min)) * chart.height; addRect(svg, x, y, barWidth, chart.bottom - y, color); }); }
    function updateSummary(metrics) { const latest = metrics[metrics.length - 1]; if (!latest) { document.getElementById("status").textContent = "waiting for metrics"; return; } const best = Math.min(...metrics.map((item) => Number(item.final_log_flops))); document.getElementById("episode-value").textContent = `${latest.episode} / ${latest.episodes}`; document.getElementById("episode-detail").textContent = "last update just now"; document.getElementById("flops-value").textContent = fmt(latest.final_log_flops); document.getElementById("flops-detail").textContent = `best: ${fmt(best)}`; document.getElementById("improvement-value").textContent = fmt(latest.flops_improvement); document.getElementById("loss-value").textContent = fmt(latest.last_total_loss); document.getElementById("loss-detail").textContent = `policy ${fmt(latest.last_policy_loss)}, value ${fmt(latest.last_value_loss)}`; document.getElementById("status").textContent = `live - ${metrics.length} metric record${metrics.length === 1 ? "" : "s"}`; }
    function updateBaselines(baselines) { const list = document.getElementById("baseline-list"); const legend = document.getElementById("flops-legend"); list.textContent = ""; document.querySelectorAll(".baseline-key").forEach((node) => node.remove()); if (baselines.length === 0) { list.textContent = "none"; return; } baselines.forEach((baseline, index) => { const color = baselineColors[index % baselineColors.length]; const span = document.createElement("span"); span.textContent = `${baseline.name}: ${fmt(baseline.log_flops)}`; list.appendChild(span); const key = document.createElement("span"); key.className = "key baseline-key"; key.style.color = color; const swatch = document.createElement("span"); swatch.className = "swatch"; key.appendChild(swatch); key.appendChild(document.createTextNode(`${baseline.name} baseline`)); legend.appendChild(key); }); }
    async function refresh() { try { const [metricsResponse, baselinesResponse] = await Promise.all([fetch("/api/metrics"), fetch("/api/baselines")]); if (!metricsResponse.ok || !baselinesResponse.ok) { throw new Error("monitor API returned an error"); } const metricsDoc = await metricsResponse.json(); const baselinesDoc = await baselinesResponse.json(); const metrics = metricsDoc.metrics || []; const baselines = baselinesDoc.baselines || []; document.getElementById("subtitle").textContent = metricsDoc.log_dir || "training run"; updateSummary(metrics); updateBaselines(baselines); drawSeriesChart("loss-chart", [{values: metrics.map((item) => Number(item.last_total_loss)), color: "#2b6cb0", className: "curve"}, {values: metrics.map((item) => Number(item.last_policy_loss)), color: "#25865a", className: "curve-thin"}, {values: metrics.map((item) => Number(item.last_value_loss)), color: "#c75146", className: "curve-thin"}]); drawFlopsChart(metrics, baselines); drawSeriesChart("improvement-chart", [{values: metrics.map((item) => Number(item.flops_improvement)), color: "#25865a", className: "curve"}]); drawBarChart("steps-chart", metrics.map((item) => Number(item.episode_steps)), "#2b6cb0"); } catch (error) { document.getElementById("status").textContent = "monitor error"; } }
    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>
"""


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
        return f"http://{host}:{port}"

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
                    self._send_json(
                        {"error": str(error)}, status=HTTPStatus.INTERNAL_SERVER_ERROR
                    )

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
