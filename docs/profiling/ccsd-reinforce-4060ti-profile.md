# CCSD REINFORCE 4060 Ti Profiling

This runbook is for the profiling-only branch based on
`refactor/model-trainer-protocols`.

Use it on the RTX 4060 Ti machine with the CUDA JAX environment already
installed. Prefer `uv run --no-sync` so `uv` does not replace the CUDA-enabled
JAX install.

## Setup

```bash
cd /path/to/gristmill-symbolics
git fetch origin
git switch profile-model-trainer-protocols-4060ti
git rev-parse --short HEAD

cd python
INPUT=../tmp/ccsd/working_eqn.json
python - <<'PY'
import jax
print(jax.__version__)
print(jax.devices())
PY
```

## Single Profile Run

This records:

- training JSON metrics in `stdout.jsonl`;
- JAX compile logs, rollout phase JSON events, and `/usr/bin/time -v` in
  `stderr.log`;
- Python cProfile data in `profile.prof`;
- GPU utilization and memory samples in `nvidia-smi.csv`.

```bash
cd python
RUN=/tmp/ccsd-profile/model-trainer-protocols-4060ti/updates1
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
  -m gristmill_symbolics.cli.train \
  --input "$INPUT" \
  --updates 1 \
  --batch-size 2 \
  --max-steps 64 \
  --seed 42 \
  --state-token-pad-to 3072 \
  --action-token-pad-to 4096 \
  --definition-pad-to 128 \
  > "$RUN/stdout.jsonl" \
  2> "$RUN/stderr.log"

STATUS=$?
kill "$SMI_PID"
test "$STATUS" -eq 0
```

## Updates 1/2/3 Compile Profile

```bash
cd python
for UPDATES in 1 2 3; do
  RUN="/tmp/ccsd-profile/model-trainer-protocols-4060ti/updates${UPDATES}"
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
    -m gristmill_symbolics.cli.train \
    --input "$INPUT" \
    --updates "$UPDATES" \
    --batch-size 2 \
    --max-steps 64 \
    --seed 42 \
    --state-token-pad-to 3072 \
    --action-token-pad-to 4096 \
    --definition-pad-to 128 \
    > "$RUN/stdout.jsonl" \
    2> "$RUN/stderr.log"

  STATUS=$?
  kill "$SMI_PID"
  if [ "$STATUS" -ne 0 ]; then
    echo "profile failed for updates=${UPDATES}; see $RUN/stderr.log" >&2
    break
  fi
done
```

## Quick Summaries

```bash
for UPDATES in 1 2 3; do
  RUN="/tmp/ccsd-profile/model-trainer-protocols-4060ti/updates${UPDATES}"
  printf 'updates=%s compiles=%s final=%s\n' \
    "$UPDATES" \
    "$(grep -c '^Compiling' "$RUN/stderr.log")" \
    "$(tail -n 1 "$RUN/stdout.jsonl")"
done
```

Rollout phase totals:

```bash
uv run --no-sync python - /tmp/ccsd-profile/model-trainer-protocols-4060ti/updates1/stderr.log <<'PY'
import collections
import json
import sys

totals = collections.defaultdict(float)
counts = collections.Counter()
max_state = 0
max_action = 0
max_defs = 0

with open(sys.argv[1]) as handle:
    for line in handle:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") != "rollout_phase":
            continue
        phase = event["phase"]
        totals[phase] += float(event["elapsed_ms"])
        counts[phase] += 1
        max_state = max(max_state, int(event.get("state_token_len_max") or 0))
        max_action = max(max_action, int(event.get("action_token_len_max") or 0))
        max_defs = max(max_defs, int(event.get("definition_count_max") or 0))

for phase, elapsed_ms in sorted(totals.items(), key=lambda item: item[1], reverse=True):
    print(f"{phase:28s} {elapsed_ms:12.3f} ms  count={counts[phase]}")
print(f"max_state_token_len={max_state}")
print(f"max_action_token_len={max_action}")
print(f"max_definition_count={max_defs}")
PY
```

`/usr/bin/time -v` summary:

```bash
rg 'Elapsed|Maximum resident set size|User time|System time|Exit status' \
  /tmp/ccsd-profile/model-trainer-protocols-4060ti/updates1/stderr.log
```

## Environment Flags

`GRISTMILL_PROFILE_ROLLOUT=1` enables rollout phase JSON lines on `stderr`.

`GRISTMILL_PROFILE_ROLLOUT_SYNC=1` blocks JAX values inside timed phases. This
makes phase timings more useful, but can increase total wall time. Set it to
`0` when collecting lower-overhead compile-count-only runs.
