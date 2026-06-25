# CCSD REINFORCE GPU Profiling

Use this on the RTX 4060 Ti machine after checking out the profiling branch.
The goal is to collect evidence for issue #20 before choosing the next
optimization.

This branch is for profiling the reusable batched policy jit wrappers:

```text
profile-ccsd-reinforce-jit-policy-wrappers
```

For the earlier static-shape-only comparison, use:

```text
profile-ccsd-reinforce-phase-timers
```

## Setup

```bash
cd /path/to/gristmill-symbolics
git fetch origin
git switch profile-ccsd-reinforce-jit-policy-wrappers
git rev-parse --short HEAD

cd python
INPUT=../tmp/ccsd/working_eqn.json
python - <<'PY'
import jax
print(jax.__version__)
print(jax.devices())
PY
```

Use `uv run --no-sync` or `.venv/bin/python`; plain `uv run` may replace the
CUDA JAX install.

## Completed Small-Batch Runtime Profile

Full dynamic batch-1 profile:

```bash
RUN=/tmp/ccsd-profile/batch1-steps64
mkdir -p "$RUN"

nvidia-smi \
  --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.free,power.draw \
  --format=csv \
  -lms 500 > "$RUN/nvidia-smi.csv" &
SMI_PID=$!

XLA_PYTHON_CLIENT_PREALLOCATE=false \
JAX_LOG_COMPILES=1 \
GRISTMILL_PROFILE_ROLLOUT=1 \
GRISTMILL_PROFILE_ROLLOUT_SYNC=1 \
/usr/bin/time -v uv run --no-sync python -m cProfile -o "$RUN/profile.prof" \
  -m gristmill_symbolics.reinforce.train \
  --input "$INPUT" \
  --updates 1 \
  --batch-size 1 \
  --max-steps 64 \
  --seed 42 \
  > "$RUN/stdout.jsonl" \
  2> "$RUN/stderr.log"

kill "$SMI_PID"
```

Repeat the same profile with:

```bash
RUN=/tmp/ccsd-profile/batch2-steps64
```

and change `--batch-size 1` to `--batch-size 2`.

## Summarize Any Profile Run

Set `RUN` to a completed dynamic or static run directory and run:

```bash
RUN=/tmp/ccsd-profile/batch1-steps64
uv run --no-sync python - "$RUN" <<'PY'
import collections
import csv
import json
import os
import pstats
import re
import sys

run = sys.argv[1]
stderr_path = os.path.join(run, "stderr.log")
stdout_path = os.path.join(run, "stdout.jsonl")
profile_path = os.path.join(run, "profile.prof")
gpu_path = os.path.join(run, "nvidia-smi.csv")

print(f"RUN={run}")

print("\n== final training metrics ==")
try:
    with open(stdout_path) as f:
        lines = [line.rstrip() for line in f if line.strip()]
    print(lines[-1] if lines else "(no stdout metrics)")
except FileNotFoundError:
    print("(missing stdout.jsonl)")

print("\n== /usr/bin/time summary ==")
time_keys = (
    "Command being timed",
    "User time",
    "System time",
    "Percent of CPU",
    "Elapsed",
    "Maximum resident set size",
    "Major",
    "Minor",
    "Voluntary context switches",
    "Involuntary context switches",
    "Swaps",
    "Exit status",
)
try:
    with open(stderr_path) as f:
        for line in f:
            stripped = line.rstrip()
            if any(key in stripped for key in time_keys):
                print(stripped)
except FileNotFoundError:
    print("(missing stderr.log)")

print("\n== cProfile top cumulative entries ==")
if os.path.exists(profile_path):
    pstats.Stats(profile_path).strip_dirs().sort_stats("cumtime").print_stats(60)
else:
    print("(missing profile.prof)")

print("\n== rollout phase totals ==")
totals = collections.defaultdict(float)
counts = collections.Counter()
max_action_l = 0
max_state_l = 0
max_definition_count = 0
events = 0

try:
    with open(stderr_path) as f:
        for line in f:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") != "rollout_phase":
                continue
            events += 1
            phase = event["phase"]
            totals[phase] += float(event["elapsed_ms"])
            counts[phase] += 1
            max_action_l = max(
                max_action_l, int(event.get("action_token_len_max") or 0)
            )
            max_state_l = max(
                max_state_l, int(event.get("state_token_len_max") or 0)
            )
            max_definition_count = max(
                max_definition_count, int(event.get("definition_count_max") or 0)
            )
except FileNotFoundError:
    pass

if events:
    for phase, elapsed_ms in sorted(
        totals.items(), key=lambda item: item[1], reverse=True
    ):
        print(f"{phase:28s} {elapsed_ms:12.3f} ms  count={counts[phase]}")
    print(f"max_state_token_len={max_state_l}")
    print(f"max_action_token_len={max_action_l}")
    print(f"max_definition_count={max_definition_count}")
else:
    print("(no rollout_phase JSON events)")

print("\n== JAX compile log summary ==")
compile_count = 0
finished_count = 0
compile_lines = []
try:
    with open(stderr_path) as f:
        for line in f:
            if "Compiling" in line:
                compile_count += 1
                if len(compile_lines) < 80:
                    compile_lines.append(line.rstrip())
            if "Finished XLA compilation" in line:
                finished_count += 1
except FileNotFoundError:
    pass
print(f"Compiling lines={compile_count}")
print(f"Finished XLA compilation lines={finished_count}")
if compile_lines:
    print("-- first compile lines --")
    print("\n".join(compile_lines))

print("\n== nvidia-smi telemetry ==")
def number(value):
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value or "")
    return float(match.group(0)) if match else None

def summarize_numeric(name, values):
    values = [value for value in values if value is not None]
    if not values:
        return
    print(
        f"{name}: min={min(values):.1f} avg={sum(values) / len(values):.1f} "
        f"max={max(values):.1f} samples={len(values)}"
    )

try:
    with open(gpu_path, newline="") as f:
        rows = [
            {key.strip(): value.strip() for key, value in row.items()}
            for row in csv.DictReader(f)
        ]
except FileNotFoundError:
    rows = []

if rows:
    gpu_util = []
    mem_util = []
    mem_used = []
    power = []
    for row in rows:
        for key, value in row.items():
            lowered = key.lower()
            if lowered.startswith("utilization.gpu"):
                gpu_util.append(number(value))
            elif lowered.startswith("utilization.memory"):
                mem_util.append(number(value))
            elif lowered.startswith("memory.used"):
                mem_used.append(number(value))
            elif lowered.startswith("power.draw"):
                power.append(number(value))
    summarize_numeric("gpu_util_percent", gpu_util)
    summarize_numeric("memory_util_percent", mem_util)
    summarize_numeric("memory_used_mib", mem_used)
    summarize_numeric("power_draw_w", power)
else:
    print("(missing nvidia-smi.csv)")
PY
```

## Static-Shape Rollout Profile

First run the dynamic profile above and use the timer summary to choose pads.
The reported `max_state_token_len` and `max_action_token_len` are lower bounds;
choose rounded-up values so static mode does not fail partway through the run.
Definition count is reported by the `definition_count_max` field in
`stack_state_tokens` events.

Full static batch-1 rerun for the same workload:

```bash
RUN=/tmp/ccsd-profile/batch1-steps64-static
mkdir -p "$RUN"

nvidia-smi \
  --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.free,power.draw \
  --format=csv \
  -lms 500 > "$RUN/nvidia-smi.csv" &
SMI_PID=$!

XLA_PYTHON_CLIENT_PREALLOCATE=false \
JAX_LOG_COMPILES=1 \
GRISTMILL_PROFILE_ROLLOUT=1 \
GRISTMILL_PROFILE_ROLLOUT_SYNC=1 \
/usr/bin/time -v uv run --no-sync python -m cProfile -o "$RUN/profile.prof" \
  -m gristmill_symbolics.reinforce.train \
  --input "$INPUT" \
  --updates 1 \
  --batch-size 1 \
  --max-steps 64 \
  --seed 42 \
  --static-policy-batch \
  --state-token-pad-to 4096 \
  --action-token-pad-to 4096 \
  --definition-pad-to 128 \
  > "$RUN/stdout.jsonl" \
  2> "$RUN/stderr.log"

kill "$SMI_PID"
```

Then run the `Summarize Any Profile Run` command with:

```bash
RUN=/tmp/ccsd-profile/batch1-steps64-static
```

If a pad is too small, the run fails fast with `TrainingError`; increase the
specific `--state-token-pad-to`, `--action-token-pad-to`, or
`--definition-pad-to` value named in the error and rerun. Keep the dynamic and
static runs otherwise identical so `stdout.jsonl`, `stderr.log`, cProfile, and
GPU telemetry are directly comparable.

For batch-size comparisons, repeat the static rerun with:

```bash
RUN=/tmp/ccsd-profile/batch2-steps64-static
```

and change `--batch-size 1` to `--batch-size 2`. Keep the same pads unless the
static run reports that a larger batch observes a larger token or definition
dimension.

## Jit Policy Wrapper Compile Check

Use this branch to check whether the reusable batched policy jit wrappers remove
the repeated same-shape policy compiles seen before this branch. Keep batch size,
pads, input, and seed fixed; vary only `--updates`.

```bash
for U in 1 2 3; do
  RUN=/tmp/ccsd-profile/batch2-steps64-static-jit-u$U
  mkdir -p "$RUN"

  nvidia-smi \
    --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.free,power.draw \
    --format=csv \
    -lms 500 > "$RUN/nvidia-smi.csv" &
  SMI_PID=$!

  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  JAX_LOG_COMPILES=1 \
  GRISTMILL_PROFILE_ROLLOUT=1 \
  GRISTMILL_PROFILE_ROLLOUT_SYNC=1 \
  /usr/bin/time -v uv run --no-sync python -m cProfile -o "$RUN/profile.prof" \
    -m gristmill_symbolics.reinforce.train \
    --input "$INPUT" \
    --updates "$U" \
    --batch-size 2 \
    --max-steps 64 \
    --seed 42 \
    --static-policy-batch \
    --state-token-pad-to 4096 \
    --action-token-pad-to 4096 \
    --definition-pad-to 128 \
    > "$RUN/stdout.jsonl" \
    2> "$RUN/stderr.log"

  kill "$SMI_PID"
  COMPILES=$(grep -c '^Compiling' "$RUN/stderr.log")
  echo "updates=$U compiles=$COMPILES"
done
```

For each run, summarize with the `Summarize Any Profile Run` script above.
Then group compile signatures:

```bash
RUN=/tmp/ccsd-profile/batch2-steps64-static-jit-u3
grep '^Compiling' "$RUN/stderr.log" \
  | sed -E 's/ Argument mapping:.*//' \
  | sort | uniq -c | sort -nr | head -80
```

Compare against the pre-jit-wrapper static profile, where the grouped logs
showed repeated same-shape action side scans:

```text
updates=1: 128 repeated jit(scan) compiles per scan signature
updates=2: 256 repeated jit(scan) compiles per scan signature
updates=3: 384 repeated jit(scan) compiles per scan signature
```

Good evidence for the jit-wrapper change:

- `grep -c '^Compiling'` drops materially for `updates=1`, `2`, and `3`.
- Grouped compile signatures no longer show `128 * updates` repeated
  `jit(scan)` compiles for the action side.
- Final JSON metrics are comparable for fixed input, seed, batch size, max
  steps, and pads.
- Peak GPU memory does not increase.

## Memory Boundary And HLO Profile

Try `batch_size=4` first, then `8`. For an OOM run, keep full stderr and HLO.

```bash
RUN=/tmp/ccsd-profile/batch8-steps64-oom
mkdir -p "$RUN/xla"

XLA_PYTHON_CLIENT_PREALLOCATE=false \
JAX_LOG_COMPILES=1 \
GRISTMILL_PROFILE_ROLLOUT=1 \
GRISTMILL_PROFILE_ROLLOUT_SYNC=1 \
XLA_FLAGS="--xla_dump_to=$RUN/xla --xla_dump_hlo_as_text" \
/usr/bin/time -v uv run --no-sync python -m gristmill_symbolics.reinforce.train \
  --input "$INPUT" \
  --updates 1 \
  --batch-size 8 \
  --max-steps 64 \
  --seed 42 \
  > "$RUN/stdout.jsonl" \
  2> "$RUN/stderr.log"
```

If the run completes and JAX supports device memory profiles in that environment,
save one near process exit with a short one-off wrapper or breakpoint:

```python
import jax
jax.profiler.save_device_memory_profile('/tmp/ccsd-profile/jax-device-memory.prof')
```

## Optional Nsight Profile

Run this only after phase timers show that JAX device work is a major part of
wall time. It is not the first profile to collect.

```bash
nsys profile \
  --trace=cuda,nvtx,cudnn,cublas,osrt \
  --sample=cpu \
  --force-overwrite=true \
  --output=/tmp/ccsd-profile/nsys-batch1 \
  uv run --no-sync python -m gristmill_symbolics.reinforce.train \
    --input "$INPUT" \
    --updates 1 \
    --batch-size 1 \
    --max-steps 64 \
    --seed 42
```

## Artifacts For Issue #20

For each run, paste:

- exact command and git SHA
- final JSON metrics line from `stdout.jsonl`, if completed
- `/usr/bin/time -v` summary from `stderr.log`
- max and typical GPU utilization and memory from `nvidia-smi.csv`
- top 60 cumulative `cProfile` entries
- rollout phase totals from `GRISTMILL_PROFILE_ROLLOUT` JSON lines in `stderr.log`
- `JAX_LOG_COMPILES` summary: compile count and repeated shape signatures
- for OOM: full XLA allocator/autotune message and the HLO shape

## Decision Criteria

- Rust/PyO3 row phases dominate: optimize or batch row snapshot/query/apply.
- Tokenization or stacking phases dominate: cache or restructure token construction.
- JAX compile logs dominate: stabilize shapes or precompile common shapes.
- Many small JAX phases dominate with low GPU utilization: batch larger policy work
  units and reduce Python/Rust/JAX crossings.
- `B x L x L` or `B x L x L x d_model` HLO/device memory dominates: target
  action attention/scoring memory next.
- Only autotuning fails while lower-batch steady memory is modest: investigate
  allocator/autotune settings separately before changing model behavior.
