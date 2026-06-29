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

## Baseline Memory Profile

This records:

- training JSON metrics in `stdout.jsonl`;
- JAX compile logs, rollout phase JSON events, and `/usr/bin/time -v` in
  `stderr.log`;
- Python cProfile data in `profile.prof`;
- GPU utilization and memory samples in `nvidia-smi.csv`;
- before/after `nvidia-smi -q -d MEMORY` snapshots;
- git SHA/status and JAX device metadata.

```bash
cd python
INPUT=../tmp/ccsd/working_eqn.json
RUN_ROOT=/tmp/ccsd-profile/model-trainer-protocols-4060ti

uv run --no-sync python tools/profile_ccsd_memory.py \
  --input "$INPUT" \
  --run-root "$RUN_ROOT" \
  --run-name updates1-bs2 \
  --updates 1 \
  --batch-size 2 \
  --max-steps 64 \
  --seed 42 \
  --state-token-pad-to 3072 \
  --action-token-pad-to 4096 \
  --definition-pad-to 128

RUN=$RUN_ROOT/updates1-bs2
uv run --no-sync python tools/summarize_profile_run.py "$RUN"
```

## Updates 1/2/3 Baseline Series

```bash
cd python
INPUT=../tmp/ccsd/working_eqn.json
RUN_ROOT=/tmp/ccsd-profile/model-trainer-protocols-4060ti

for UPDATES in 1 2 3; do
  uv run --no-sync python tools/profile_ccsd_memory.py \
    --input "$INPUT" \
    --run-root "$RUN_ROOT" \
    --run-name "updates${UPDATES}-bs2" \
    --updates "$UPDATES" \
    --batch-size 2 \
    --max-steps 64 \
    --seed 42 \
    --state-token-pad-to 3072 \
    --action-token-pad-to 4096 \
    --definition-pad-to 128

  STATUS=$(cat "$RUN_ROOT/updates${UPDATES}-bs2/status.txt")
  if [ "$STATUS" -ne 0 ]; then
    echo "profile failed for updates=${UPDATES}; see $RUN_ROOT/updates${UPDATES}-bs2/stderr.log" >&2
    break
  fi
done
```

## Quick Summaries

```bash
for UPDATES in 1 2 3; do
  RUN="/tmp/ccsd-profile/model-trainer-protocols-4060ti/updates${UPDATES}-bs2"
  printf 'updates=%s compiles=%s final=%s\n' \
    "$UPDATES" \
    "$(grep -c '^Compiling' "$RUN/stderr.log")" \
    "$(tail -n 1 "$RUN/stdout.jsonl")"
done
```

Full summary for one run:

```bash
RUN=/tmp/ccsd-profile/model-trainer-protocols-4060ti/updates1-bs2
uv run --no-sync python tools/summarize_profile_run.py "$RUN"
```

Machine-readable summary:

```bash
RUN=/tmp/ccsd-profile/model-trainer-protocols-4060ti/updates1-bs2
uv run --no-sync python tools/summarize_profile_run.py "$RUN" --json
```

## OOM And HLO Boundary Profile

Use this after the completed baseline to capture the peak/failure shape. It
skips cProfile overhead and writes HLO text into `$RUN/xla`.

```bash
cd python
INPUT=../tmp/ccsd/working_eqn.json
RUN_ROOT=/tmp/ccsd-profile/model-trainer-protocols-4060ti

uv run --no-sync python tools/profile_ccsd_memory.py \
  --input "$INPUT" \
  --run-root "$RUN_ROOT" \
  --run-name batch8-oom-xla \
  --updates 1 \
  --batch-size 8 \
  --max-steps 64 \
  --seed 42 \
  --state-token-pad-to 3072 \
  --action-token-pad-to 4096 \
  --definition-pad-to 128 \
  --no-cprofile \
  --xla-dump

RUN=$RUN_ROOT/batch8-oom-xla
uv run --no-sync python tools/summarize_profile_run.py "$RUN"
```

## Environment Flags

The profile runner sets these for the training subprocess:

- `XLA_PYTHON_CLIENT_PREALLOCATE=false`
- `JAX_LOG_COMPILES=1`
- `GRISTMILL_PROFILE_ROLLOUT=1`
- `GRISTMILL_PROFILE_ROLLOUT_SYNC=1`, unless `--no-rollout-sync` is passed

`GRISTMILL_PROFILE_ROLLOUT_SYNC=1` blocks JAX values inside timed phases. This
makes phase timings more useful, but can increase total wall time. Use
`--no-rollout-sync` when collecting lower-overhead compile-count-only runs.

`--xla-dump` appends `--xla_dump_to=$RUN/xla --xla_dump_hlo_as_text` to
`XLA_FLAGS`.

## Artifacts To Keep

Keep the whole run directory. The most important files are:

- `command.txt`
- `git-sha.txt`
- `git-status.txt`
- `jax-env.txt`
- `stdout.jsonl`
- `stderr.log`
- `profile.prof`, when cProfile is enabled
- `nvidia-smi.csv`
- `nvidia-before.txt`
- `nvidia-after.txt`
- `status.txt`
- `xla/`, for `--xla-dump` runs

Before and after a memory optimization, compare:

- `peak_gpu_memory_mib` from `tools/summarize_profile_run.py`;
- `/usr/bin/time -v` max RSS, reported as `max_rss_kbytes`;
- JAX compile count;
- rollout phase totals and max token dimensions;
- OOM HLO shapes from `stderr.log` and `xla/`.
