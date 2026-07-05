# Direct Optimizer Dataset Design

## Role

The dataset module owns the path from symbolic candidate generation to
trainer-ready supervised rows.

It has two internal responsibilities:

1. Raw candidate generation: use the Rust symbolic rewrite engine to generate
   equivalent candidate computations from an input computation.
2. Dataset building and cleaning: convert raw records into processed training
   examples with source DSL, target DSL, grouping keys, finite costs, and
   cost-softmax weights.

The main intended path is generated data from the symbolic engine. The builder
also accepts raw JSONL so future generators or external candidate sources can
share the same processed dataset contract.

The trainer consumes only processed rows. It never consumes raw candidates.

## Validation Philosophy

Internal generation uses valid Rust rewrites, so equivalence is expected by
construction. The builder should still support optional verification for
debugging and external raw JSONL.

When verification is enabled, records that fail
`equivalent_computations(input_computation, candidate_computation, outputs)` are
rejected. When verification is disabled, no equivalence check is performed.

The builder should never emit a processed row unless it has valid source DSL,
target DSL, ordered outputs, finite costs, stable keys, and a normalized group
weight.

## Raw Candidate Generator

The generator starts from one or more seed computations:

```text
input_computation: TensorComputation
outputs: list[int]
```

It runs randomized rewrite trajectories through the existing symbolic rewrite
surface:

```text
action_space_for_def
validate_decision
apply_decision
```

Generation config:

```python
@dataclass(frozen=True)
class GenerationConfig:
    seed: int
    trajectories_per_input: int
    max_steps: int
    random_subsets: bool = False
    collect_intermediates: bool = True
```

For each trajectory:

1. Clone `input_computation`.
2. Repeatedly choose an actionable definition/action space.
3. Choose a candidate/action.
4. Validate and apply the decision.
5. Emit candidate records.

Candidate collection mode:

- If `collect_intermediates=True`, emit each post-rewrite state.
- If `collect_intermediates=False`, emit only the final trajectory state.
- Always skip the unchanged initial computation.

Raw candidate record shape:

```json
{
  "input_computation": "<JSON string or object>",
  "candidate_computation": "<JSON string or object>",
  "outputs": [1],
  "initial_log_flops": 4.2,
  "candidate_log_flops": 3.7
}
```

Generator internals may track seed, trajectory, or step information for logs, but
that metadata is not part of the required raw record schema.

## Dataset Builder And Cleaner

The builder consumes raw candidate records from either the internal generator or
raw JSONL.

Input record fields:

```text
input_computation
candidate_computation
outputs
initial_log_flops optional
candidate_log_flops optional
```

Processing steps:

1. Parse `input_computation` and `candidate_computation` as
   `TensorComputation`.
2. Validate `outputs`:
   - non-empty list;
   - integer tensor ids, not bool;
   - no duplicates;
   - order preserved.
3. If costs are missing, compute:

   ```text
   input_computation.log_total_flops()
   candidate_computation.log_total_flops()
   ```

4. Optionally verify:

   ```text
   equivalent_computations(input_computation, candidate_computation, outputs)
   ```

5. Convert to DSL:

   ```text
   source_text = computation_to_source_text(input_computation)
   target_text = computation_to_target_text(candidate_computation)
   ```

6. Group by:

   ```text
   input_key = stable hash(source_text + ordered outputs)
   ```

7. Deduplicate within each input group by:

   ```text
   candidate_key = stable hash(target_text)
   ```

8. Compute cost-softmax weights from `candidate_log_flops`.
9. Write processed rows.

Processed row schema:

```json
{
  "input_key": "sha256:...",
  "candidate_key": "sha256:...",
  "outputs": [1, 3],
  "source_text": "...",
  "target_text": "...",
  "initial_log_flops": 4.2,
  "candidate_log_flops": 3.7,
  "weight": 0.83
}
```

## Keys And Ordered Outputs

`outputs` are ordered. The builder validates and preserves the order exactly as
provided.

Rules:

- `outputs` must be a non-empty list.
- Each item must be an integer tensor id, not bool.
- No duplicates are allowed.
- Order is preserved.
- Grouping treats `[1, 3]` and `[3, 1]` as different output contracts.

Keys are stable identifiers, not replacements for the full text fields.
Processed rows still store `source_text` and `target_text`.

Key definitions:

```text
input_key = stable hash(source_text + ordered outputs)
candidate_key = stable hash(target_text)
```

The implementation should use a deterministic SHA-256 based representation, for
example a `sha256:`-prefixed hex digest over stable JSON payloads.

## Deduplication And Weighting

Deduplication happens per `input_key`.

If multiple raw records produce the same `candidate_key` for an input group:

- keep one processed candidate;
- use the lowest `candidate_log_flops` among duplicates;
- keep the matching `target_text`;
- recompute weights after deduplication.

This prevents duplicate trajectories from inflating probability mass for the
same target definitions.

Weights are computed per input group:

```text
shifted_i = candidate_log_flops_i - min_j(candidate_log_flops_j)
raw_i = exp(-beta * shifted_i)
weight_i = raw_i / sum_j(raw_j)
```

Rules:

- `beta > 0` for cost preference.
- Group weights sum to `1.0` within numerical tolerance.
- Lower `candidate_log_flops` gets higher weight.
- If a group has one candidate, its weight is `1.0`.
- Non-finite costs reject the record before weighting.
- Empty groups are not written.

## Public API And CLI

Public API should separate generation from building.

Configuration:

```python
@dataclass(frozen=True)
class GenerationConfig:
    seed: int
    trajectories_per_input: int
    max_steps: int
    random_subsets: bool = False
    collect_intermediates: bool = True


@dataclass(frozen=True)
class BuildConfig:
    beta: float = 1.0
    verify: bool = False
```

Core functions:

```python
generate_raw_candidates(
    inputs: Sequence[tuple[TensorComputation, Sequence[int]]],
    config: GenerationConfig,
) -> list[dict]

write_raw_candidates_jsonl(records: Sequence[dict], path: Path) -> None
read_raw_candidates_jsonl(path: Path) -> list[dict]

build_processed_dataset(
    raw_records: Sequence[dict],
    config: BuildConfig,
) -> list[dict]

write_processed_jsonl(rows: Sequence[dict], path: Path) -> None
read_processed_jsonl(path: Path) -> list[dict]
```

CLI shape:

```bash
python -m gristmill_symbolics.direct_optimizer.dataset generate \
  --input seed.json \
  --outputs 1 3 \
  --raw-output raw_candidates.jsonl \
  --seed 0 \
  --trajectories 64 \
  --max-steps 8 \
  --collect-intermediates

python -m gristmill_symbolics.direct_optimizer.dataset build \
  --raw-input raw_candidates.jsonl \
  --output processed_direct_dataset.jsonl \
  --beta 1.0 \
  --verify
```

A convenience `generate-build` subcommand can be added if trivial, but generation
and build should remain separate functions.

## Error Handling And Boundaries

Generation failures should be local to a trajectory:

- If a trajectory reaches no action space, it stops normally.
- If a rewrite validation or application error occurs, that trajectory stops and
  no invalid candidate is emitted.
- The generator should not emit unchanged initial computations.
- The generator should not call the verifier unless a debug or verification
  option is added later.

Builder failures:

- Malformed raw records are skipped.
- Invalid `outputs` are skipped.
- Non-finite costs are skipped.
- Converter failures are skipped.
- Verification failures are skipped only when `verify=True`; when
  `verify=False`, no equivalence check is performed.
- The builder should never emit a processed row without valid `source_text`,
  `target_text`, finite costs, ordered outputs, keys, and weight.

Boundaries:

- Dataset module may import converter, `TensorComputation`, rewrite functions,
  `equivalent_computations`, JSON/path utilities, NumPy, and math.
- Dataset module must not import model, trainer, sampler, action-selector model,
  REINFORCE trainer, or CLI checkpoint modules.
- Processed JSONL is the only contract with the trainer.

## Tests

Required focused tests:

- Raw generator emits candidate records with:

  ```text
  input_computation
  candidate_computation
  outputs
  initial_log_flops
  candidate_log_flops
  ```

  and does not emit the unchanged initial computation.

- Generator is deterministic for a fixed seed, config, and input.
- `collect_intermediates=True` emits post-rewrite intermediate states;
  `collect_intermediates=False` emits only final trajectory states.
- Builder validates ordered outputs:
  - preserves `[1, 3]`;
  - treats `[3, 1]` as a different input group;
  - rejects duplicates and bools.
- Builder converts records to `source_text` and `target_text`.
- Builder deduplicates the same candidate per input group.
- Builder weights candidates per input group:
  - weights sum to `1.0`;
  - lower `candidate_log_flops` has larger weight when `beta > 0`.
- Builder skips malformed records and non-finite costs.
- Optional verification mode rejects non-equivalent external raw records.
- JSONL read/write round-trips raw and processed rows.

Verification command:

```bash
uv run pytest python/tests/direct_optimizer/test_dataset.py -q
```

