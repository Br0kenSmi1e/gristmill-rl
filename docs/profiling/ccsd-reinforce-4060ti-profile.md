# CCSD REINFORCE 4060 Ti Memory Profile

Use this on the RTX 4060 Ti machine with the CUDA JAX environment already
installed. Run commands from the repository checkout that contains
`tmp/ccsd/working_eqn.json`.

## Run The Current Peak Profile

```bash
cd "$(git rev-parse --show-toplevel)"
git fetch origin
git switch profile-model-trainer-protocols-4060ti
git pull --ff-only

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

The summary should include these sections:

- `== failure ==`
- `== memory ==`
- `== rollout phases ==`
- `== largest xla shapes ==`
- `== largest xla allocation lines ==`

Large allocation lines include the captured allocation block context when XLA
prints one. For the current failure, inspect the `jit_score_target`
`preallocated-temp` entry and its `context_shapes`.

## Rerun Only If Allocation Lines Are Missing

Use this only when the first summary has `(none found)` under
`== largest xla allocation lines ==`.

```bash
cd "$(git rev-parse --show-toplevel)"
cd python
INPUT=../tmp/ccsd/working_eqn.json
RUN_ROOT=/tmp/ccsd-profile/model-trainer-protocols-4060ti

uv run --no-sync python tools/profile_ccsd_memory.py \
  --input "$INPUT" \
  --run-root "$RUN_ROOT" \
  --run-name batch8-oom-xla-all-passes \
  --updates 1 \
  --batch-size 8 \
  --max-steps 64 \
  --seed 42 \
  --state-token-pad-to 3072 \
  --action-token-pad-to 4096 \
  --definition-pad-to 128 \
  --no-cprofile \
  --xla-dump-all-passes

RUN=$RUN_ROOT/batch8-oom-xla-all-passes
uv run --no-sync python tools/summarize_profile_run.py "$RUN"
```

`--xla-dump-all-passes` can create a large `$RUN/xla` directory. Do not use it
when the standard `--xla-dump` run already reports allocation lines.

## Optional Machine-Readable Summary

```bash
cd "$(git rev-parse --show-toplevel)/python"
RUN=/tmp/ccsd-profile/model-trainer-protocols-4060ti/batch8-oom-xla
uv run --no-sync python tools/summarize_profile_run.py "$RUN" --json
```
