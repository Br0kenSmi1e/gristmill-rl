# RL Training Monitor Design

## Summary

Add an opt-in graphical monitor for `python/gristmill_rl/train.py`.

The trainer remains the source of truth for training and metrics. When
monitoring is enabled, it writes durable per-episode records to a JSONL file,
starts a small local HTTP server, and prints a browser URL. The browser
dashboard renders live charts from those records with inline SVG and
JavaScript.

This design is dependency-light. It does not add TensorBoard, matplotlib,
Streamlit, Dash, or another plotting package.

## Goals

- Keep existing stdout JSON metrics stable for scripts and tests.
- Add `--monitor` for a live graphical view of a training run.
- Require explicit `--log-dir` when `--monitor` is used.
- Store run metrics as append-only JSONL under the log directory.
- Compare RL training progress with named baseline JSON computations.
- Match the approved dashboard mockup:
  - four summary cards,
  - a 2x2 chart grid,
  - a baseline footer,
  - restrained light visual design,
  - one-second browser refresh.

## Non-Goals

- TensorBoard integration.
- Matplotlib-based live plotting.
- Browser auto-open behavior.
- Multi-run comparison.
- Remote authentication or public network serving.
- Persisting replay data, optimizer state, or model checkpoints beyond the
  existing checkpoint feature.
- Pixel-perfect browser screenshot tests for v1.

## CLI

Extend `python -m gristmill_rl.train` with:

```text
--monitor
--log-dir PATH
--baseline NAME=PATH
```

Example:

```bash
uv run python -m gristmill_rl.train \
  --input input.json \
  --episodes 100 \
  --monitor \
  --log-dir runs/exp-001 \
  --baseline greedy=outputs/greedy/final.json \
  --baseline random-best=outputs/random/final.json
```

Rules:

- `--monitor` requires `--log-dir`.
- `--log-dir` without `--monitor` fails, because v1 does not add standalone
  metrics logging.
- `--baseline` without `--monitor` fails, because baselines are monitor-only
  in v1.
- `--baseline NAME=PATH` may be repeated.
- Baselines are optional.
- Baseline names must be non-empty and unique.
- Baseline values must use `NAME=PATH`; malformed values fail during argument
  parsing or monitor setup.
- Baseline paths must load as `TensorComputation` JSON files.
- Monitor startup prints a local URL and does not open a browser.
- Existing stdout JSON metrics remain parseable and keep their current keys.
- If monitor setup fails before training starts, the command fails clearly.
- If metrics writing fails during training, training fails clearly instead of
  silently hiding corrupt logs.

## Run Artifacts

When monitoring is enabled, `--log-dir` contains:

```text
metrics.jsonl
baselines.json
```

`metrics.jsonl` stores one JSON object per episode. Each line includes the
current per-episode metrics that `train.py` already prints, plus derived fields
needed by the dashboard:

```json
{
  "episode": 1,
  "episodes": 100,
  "replay_size": 4,
  "episode_steps": 4,
  "episode_records": 4,
  "initial_log_flops": 13.34,
  "final_log_flops": 12.82,
  "flops_improvement": 0.52,
  "last_policy_loss": 0.71,
  "last_value_loss": 0.39,
  "last_total_loss": 1.10,
  "params_changed": true
}
```

`baselines.json` stores baseline metadata computed before training starts:

```json
{
  "baselines": [
    {
      "name": "greedy",
      "path": "outputs/greedy/final.json",
      "log_flops": 12.50
    }
  ]
}
```

The monitor server may serve static HTML/CSS/JS from package code rather than
copying generated files into `--log-dir`. The durable run artifacts are the
JSON files above.

## Components

Add `python/gristmill_rl/monitor.py`.

Public responsibilities:

```text
parse_baseline_arg(value: str) -> tuple[str, Path]
load_baselines(items: Sequence[tuple[str, Path]]) -> list[BaselineMetric]

MonitorWriter
  creates the log directory
  writes baselines.json
  appends metrics.jsonl
  computes flops_improvement

MonitorServer
  serves the approved dashboard HTML
  exposes /api/metrics
  exposes /api/baselines
  runs on localhost in a background thread
  shuts down when training exits
```

`train.py` should stay focused on training. It only coordinates monitor setup
and sends already-built episode metrics to `MonitorWriter`.

## Data Flow

Training with monitoring enabled follows this flow:

```text
parse args
validate --monitor requires --log-dir
parse repeated --baseline values
load each baseline JSON as TensorComputation
compute each baseline log_total_flops()
create MonitorWriter(log_dir)
write baselines.json
start MonitorServer(log_dir)
print monitor URL

for each episode:
  run rollout
  train model
  build episode_metrics
  print episode_metrics JSON to stdout
  append episode_metrics plus derived fields to metrics.jsonl

on exit:
  stop monitor server
  optionally write checkpoint through existing checkpoint flow
  print final summary JSON to stdout
```

The monitor browser polls `/api/metrics` and `/api/baselines` every second. The
server reads the JSON files from `--log-dir` and returns compact JSON responses.
The browser owns chart rendering.

## Dashboard

The final monitor should match the approved random-data demo.

Header:

- `Gristmill RL Training Monitor`
- run/log-dir subtitle
- live status

Summary cards:

- current episode / total episodes
- latest final log flops and best final log flops
- latest improvement, computed as `initial_log_flops - final_log_flops`
- latest total loss with policy and value loss detail

Charts:

- training loss: total, policy, value
- flops progress: initial and final log flops plus baseline horizontal lines
- improvement per episode
- episode steps

Footer:

- loaded baselines from `--baseline NAME=PATH`

Behavior:

- Poll every second.
- If no metrics exist yet, show the same layout with empty states.
- If no baselines are provided, keep the flops chart layout and hide or soften
  the baseline footer and baseline legend items.
- Draw charts with inline SVG/JavaScript and no external browser dependencies.
- Serve only from localhost.

## Error Handling

Baseline errors:

- Missing `=` in `--baseline` fails with a clear message.
- Empty baseline names fail with a clear message.
- Duplicate baseline names fail before training starts.
- Missing or invalid baseline JSON paths fail before training starts.

Monitor errors:

- `--monitor` without `--log-dir` fails before training starts.
- `--log-dir` without `--monitor` fails before training starts.
- `--baseline` without `--monitor` fails before training starts.
- Failure to create `--log-dir` fails before training starts.
- Failure to bind the localhost server fails before training starts.
- Failure to append `metrics.jsonl` during training raises an exception and
  stops the command.

Dashboard errors:

- Empty metrics should render as an empty dashboard, not a JavaScript crash.
- Malformed existing JSONL lines should produce a JSON API error instead of a
  partial misleading chart.

## Testing

Focused tests should cover:

- CLI parsing accepts repeated `--baseline NAME=PATH`.
- `--monitor` without `--log-dir` fails before training.
- `--log-dir` and `--baseline` fail when used without `--monitor`.
- Baseline parsing rejects malformed values and duplicate names.
- Baseline loading computes `log_total_flops()` from JSON files.
- A tiny monitored training run writes `metrics.jsonl` and `baselines.json`.
- Existing stdout final JSON remains parseable and stable.
- The monitor API serves dashboard HTML plus JSON metrics and baselines from a
  temporary log directory.

Manual verification should run a short monitored training command, open the
printed URL, and check that the dashboard matches the approved layout.

## Open Implementation Notes

- Prefer a random available localhost port unless the implementation exposes a
  future port override.
- Keep the HTTP server lifecycle simple: background daemon thread plus explicit
  shutdown from `train.py`.
- Use atomic-enough append behavior for one training process; concurrent
  writers to the same log directory are out of scope.
- Do not add dashboard artifacts to git-generated run directories.
