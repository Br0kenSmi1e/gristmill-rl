# Deep-Cleanup CCSD Profiling On RTX 4060 Ti

This runbook profiles the current deep-cleanup training path on the actual CCSD
input. It records wall time, peak host RSS, GPU memory, GPU utilization, JAX
compile logs, and optional XLA HLO dumps for cuDNN attention verification.

The older measurements in
[issue #20](https://github.com/Br0kenSmi1e/gristmill-rl/issues/20) are useful
baseline context:

- dynamic `batch_size=1`, `max_steps=64`: about 17m34s wall time, about
  6.7 GB max RSS, low GPU utilization;
- dynamic `batch_size=2`, `max_steps=64`: about 27m34s wall time, about
  7.1 GB max RSS;
- dynamic `batch_size=8`: OOM from large materialized attention intermediates;
- static-shape batch 1 prototype: about 2m15s wall time, about 4.1 GB max RSS.

Do not compare those numbers directly against this branch without noting that
the model/trainer API has been refactored. Use them as sanity-check scale only.

## Environment Setup

Target machine:

- NVIDIA RTX 4060 Ti;
- recent NVIDIA Linux driver with working `nvidia-smi`;
- Python 3.11 or newer;
- `uv` installed.

Create the Python environment from the repo root:

```bash
cd "$(git rev-parse --show-toplevel)"
git fetch origin
git switch profile-deep-cleanup-ccsd-4060ti
git pull --ff-only

cd python
uv sync --group dev
```

Install a CUDA-enabled JAX wheel into the same uv environment. Use the CUDA
extra that matches the official JAX install guide for the machine:

```bash
uv pip install -U "jax[cuda13]"
```

If the driver or site CUDA setup requires CUDA 12 wheels, use
`"jax[cuda12]"` instead. Verify that JAX sees the GPU:

```bash
uv run --no-sync python - <<'PY'
import jax
print(jax.__version__)
print(jax.default_backend())
print(jax.devices())
PY
```

The default backend must print `gpu`. The profiling tool refuses to run on CPU
unless `--allow-cpu` is passed for local smoke tests.

## Input

Provide the CCSD input JSON on the profiling machine. The expected example path
is:

```bash
INPUT=../tmp/ccsd/working_eqn.json
```

The file is not committed in this worktree. Any equivalent CCSD
`TensorComputation` JSON can be passed through `--input`.

## Flash Attention / cuDNN Attention

The transformer encoder uses:

```python
jax.nn.dot_product_attention(..., implementation="cudnn")
```

when the JAX backend is GPU and `prefer_cudnn=True`. The encoder runs attention
in `bfloat16` by default while parameters remain `float32`.

The profiling defaults use `d_model=32` and `num_attention_heads=4`, so the
attention head dimension is `8`. cuDNN requires the head dimension to be no
larger than 128 and a multiple of 8.

Verify the cuDNN path once with an XLA dump:

```bash
cd "$(git rev-parse --show-toplevel)/python"
RUN_ROOT=/tmp/ccsd-profile/deep-cleanup-4060ti

uv run --no-sync python tools/profile_deep_cleanup_training.py \
  --input "$INPUT" \
  --run-root "$RUN_ROOT" \
  --run-prefix cudnn-check \
  --updates 1 \
  --batch-sizes 1 \
  --max-steps 1 \
  --xla-dump
```

Check the summary:

```bash
cat "$RUN_ROOT"/matrix.csv
cat "$RUN_ROOT"/cudnn-check-updates1-batch1/summary.json
```

`cudnn_hlo_count` should be greater than zero. If it is zero on a GPU run, the
cuDNN attention path was not confirmed; inspect the run's `xla/` directory and
`stderr.log`.

## Run The Scaling Profiles

The profiling command runs the real training CLI and wraps it with
`/usr/bin/time -v` plus an `nvidia-smi` sampler. The tool sets
`XLA_PYTHON_CLIENT_PREALLOCATE=false` so peak GPU memory samples are easier to
interpret.

Start with update scaling at a fixed batch size:

```bash
cd "$(git rev-parse --show-toplevel)/python"
RUN_ROOT=/tmp/ccsd-profile/deep-cleanup-4060ti/updates

uv run --no-sync python tools/profile_deep_cleanup_training.py \
  --input "$INPUT" \
  --run-root "$RUN_ROOT" \
  --run-prefix update-sweep \
  --updates 1,2,4,8 \
  --batch-sizes 1 \
  --max-steps 64
```

Then profile batch-size scaling at a fixed update count:

```bash
cd "$(git rev-parse --show-toplevel)/python"
RUN_ROOT=/tmp/ccsd-profile/deep-cleanup-4060ti/batch

uv run --no-sync python tools/profile_deep_cleanup_training.py \
  --input "$INPUT" \
  --run-root "$RUN_ROOT" \
  --run-prefix batch-sweep \
  --updates 1 \
  --batch-sizes 1,2,4,8 \
  --max-steps 64
```

If `batch_size=8` OOMs, keep the failed run. The `matrix.csv`, `stderr.log`,
and `nvidia-smi.csv` files still capture the failure point.

The default static pads are:

```text
state_token_pad_to=5000
action_token_pad_to=5000
definition_pad_to=128
candidate_pad_to=2048
side_term_pad_to=256
d_model=32
num_attention_layers=1
num_attention_heads=4
```

Override them only when the CCSD input exceeds a pad or when testing a specific
shape hypothesis. If you change the model width or head count, keep
`d_model / num_attention_heads` no larger than 128 and divisible by 8.

## Output Layout

Each run directory contains:

```text
case.json
command.txt
git-sha.txt
git-status.txt
jax-env.json
nvidia-smi.csv
stderr.log
stdout.jsonl
status.txt
summary.json
xla/              # only with --xla-dump
profile.prof      # only with --cprofile
```

Each sweep root also contains `matrix.csv` with one row per
`(updates, batch_size)` case:

```text
run_name
updates
batch_size
max_steps
status
elapsed_seconds
max_rss_kbytes
peak_gpu_memory_mib
avg_gpu_util_percent
peak_gpu_util_percent
compile_count
cudnn_hlo_count
final_metrics
```

Use `elapsed_seconds` for total time scaling and `peak_gpu_memory_mib` for GPU
memory scaling. `max_rss_kbytes` is host memory from `/usr/bin/time -v`.

## Optional Full Matrix

To collect a cross product in one command:

```bash
cd "$(git rev-parse --show-toplevel)/python"
RUN_ROOT=/tmp/ccsd-profile/deep-cleanup-4060ti/full-matrix

uv run --no-sync python tools/profile_deep_cleanup_training.py \
  --input "$INPUT" \
  --run-root "$RUN_ROOT" \
  --run-prefix full \
  --updates 1,2,4 \
  --batch-sizes 1,2,4 \
  --max-steps 64
```

This is more expensive and mixes compile amortization with batch-size memory
effects, so run the separate sweeps first.

## Reading Failures

If a case has nonzero `status`, inspect:

```bash
RUN=/tmp/ccsd-profile/deep-cleanup-4060ti/batch/batch-sweep-updates1-batch8
cat "$RUN"/summary.json
tail -200 "$RUN"/stderr.log
tail -20 "$RUN"/stdout.jsonl
```

For attention backend issues, rerun only the smallest failing case with
`--xla-dump` and check whether `cudnn_hlo_count` is nonzero.
