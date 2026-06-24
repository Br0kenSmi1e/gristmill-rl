# CCSD REINFORCE GPU Profiling

Use this on the RTX 4060 Ti machine after checking out the profiling branch.
The goal is to collect evidence for issue #20 before choosing any optimization.

## Setup

```bash
cd /path/to/gristmill-symbolics
git fetch origin
git switch profile-ccsd-reinforce-phase-timers
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

Summarize the Python profile:

```bash
uv run --no-sync python - <<'PY'
import pstats
p = pstats.Stats('/tmp/ccsd-profile/batch1-steps64/profile.prof')
p.strip_dirs().sort_stats('cumtime').print_stats(60)
PY
```

Summarize rollout phase timers:

```bash
uv run --no-sync python - <<'PY'
import collections
import json

path = '/tmp/ccsd-profile/batch1-steps64/stderr.log'
totals = collections.defaultdict(float)
counts = collections.Counter()
max_action_l = 0
max_state_l = 0

with open(path) as f:
    for line in f:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get('event') != 'rollout_phase':
            continue
        phase = event['phase']
        totals[phase] += event['elapsed_ms']
        counts[phase] += 1
        max_action_l = max(max_action_l, int(event.get('action_token_len_max') or 0))
        max_state_l = max(max_state_l, int(event.get('state_token_len_max') or 0))

for phase, elapsed_ms in sorted(totals.items(), key=lambda item: item[1], reverse=True):
    print(f'{phase:28s} {elapsed_ms:12.3f} ms  count={counts[phase]}')
print(f'max_state_token_len={max_state_l}')
print(f'max_action_token_len={max_action_l}')
PY
```

Repeat the same profile with:

```bash
RUN=/tmp/ccsd-profile/batch2-steps64
```

and change `--batch-size 1` to `--batch-size 2`.

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
