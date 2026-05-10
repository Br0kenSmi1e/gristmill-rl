# Random Rewrite CLI Design

## Goal

Add a small command-line tool for correctness testing of the rewrite kernel.

The CLI should read a `TensorComputation` JSON file, apply up to a configured
number of seeded random rewrites through the existing `rewrite` module, and
write the resulting computation to a JSON file. Optional snapshots should make
failed runs easier to debug by preserving the full computation state after each
successful rewrite.

This is not an optimizer, replay system, or stable candidate-ordering layer.

## Scope

This design includes:

- command-line parsing for a random rewrite test run
- JSON input and output through the existing `io` module
- a seeded random policy over the existing `next_action_space` pipeline
- optional random nonempty subset masks
- optional per-step computation snapshots
- concise status and error reporting

This design excludes:

- cost or profitability ranking
- MCTS, greedy optimization, or search policy
- stable candidate IDs, candidate fingerprints, or action-space tracing
- replay from a trace file
- changes to the internal rewrite kernel APIs
- validation policy beyond errors already returned by the called library APIs

## CLI Contract

Command shape:

```text
gristmill-symbolics [--seed <U64>] [--steps <N>] [--random-subsets] [--snapshot-dir <DIR>] <input.json> <output.json>
```

Arguments and defaults:

- `<input.json>` is required.
- `<output.json>` is required.
- `--seed <U64>` is optional and defaults to `42`.
- `--steps <N>` is optional and defaults to `1`.
- `--random-subsets` is optional and defaults to disabled.
- `--snapshot-dir <DIR>` is optional.

`--steps` means "apply up to this many rewrites." If no action space exists
before the limit is reached, the run stops successfully and writes the state
reached so far.

The simplest smoke-test command is:

```text
gristmill-symbolics input.json output.json
```

which is equivalent to:

```text
gristmill-symbolics --seed 42 --steps 1 input.json output.json
```

## Rewrite Loop

The CLI should follow the intended `rewrite` module usage:

```text
start_from = 0

for step in 0..steps:
  action_space = next_action_space(comp, start_from)?
  if action_space is None:
    stop successfully

  decision = random decision inside action_space
  rewrite = build_rewrite(comp, action_space, decision)?
  apply_rewrite(comp, rewrite)?
  write snapshot if enabled
  start_from = action_space.def_index

write final output JSON
```

The loop should not scan all definitions itself. `next_action_space` owns the
definition scan and returns the first actionable definition from `start_from`.

When `--steps 0` is provided, the CLI applies no rewrites. It still writes the
input computation to `<output.json>` and writes `step-000.json` if snapshots are
enabled.

## Random Policy

The implementation should use `rand::rngs::StdRng` seeded with
`SeedableRng::seed_from_u64(seed)`.

For each returned `ActionSpace`:

- choose one `candidate_index` uniformly from
  `0..action_space.candidate_templates.len()`
- build a `Decision` for that candidate
- by default, use full masks:
  - every left term is kept
  - every right term is kept
- with `--random-subsets`, generate random side masks and force each side to be
  nonempty

The v1 reproducibility contract is intentionally modest. The same input, seed,
binary, and platform should produce the same run as long as internal unordered
iteration produces the same candidate order. The CLI does not sort candidates,
record candidate hashes, or try to repair `HashMap` ordering instability.
Snapshots are the primary debugging artifact.

## Snapshots

Snapshots are optional. If `--snapshot-dir DIR` is provided, the CLI should
create the directory if needed and write full computation JSON files into it.

Snapshot filenames:

```text
DIR/step-000.json
DIR/step-001.json
DIR/step-002.json
...
```

`step-000.json` is the input computation before any rewrite. `step-N.json` for
`N > 0` is the computation after `N` successful rewrites.

If a run stops early because no action space remains, no extra sentinel file is
written. The latest snapshot is the final reached state. If an error occurs
before a rewrite is applied at a step, the latest snapshot remains the last
successful state.

## Output And Status

The final computation is always written to `<output.json>` with the existing
`io::write_json` helper.

The CLI should print concise status information to stderr, including:

- seed
- requested step limit
- number of rewrites actually applied
- stop reason:
  - reached step limit
  - no action space remained

The CLI should not print JSON to stdout in v1. Both input and output paths are
required.

## Errors

Invalid command-line arguments are handled by `clap`.

Runtime errors should exit nonzero and include enough context to identify where
the failure occurred:

- input read errors include the input path
- output write errors include the output path
- snapshot write errors include the snapshot path
- rewrite errors include the step number

If an error occurs after one or more snapshots have been written, the CLI should
leave those snapshots in place for debugging.

## Dependencies

Add small CLI dependencies:

- `clap` for command-line parsing
- `rand` for seeded random choice

No dependency is needed for snapshot file naming or temporary paths in tests.

## Testing

Add integration coverage around the binary behavior.

Argument parsing and defaults:

- default seed is `42`
- default steps is `1`
- input and output paths are required

Rewrite behavior:

- a smoke run on a fixture writes valid output JSON
- `--steps 0` writes an unchanged computation
- early stop when no action space exists succeeds and writes output

Snapshot behavior:

- `step-000.json` is written before rewrites
- one snapshot is written after each successful rewrite
- no extra snapshot is written for early stop

Random subset behavior:

- `--random-subsets` creates nonempty masks for both sides
- the resulting rewrite still validates through the existing rewrite APIs

## Acceptance Criteria

The CLI is complete when:

- `src/main.rs` implements the command contract above
- `Cargo.toml` includes `clap` and `rand`
- the binary reads input JSON and writes output JSON through `io`
- random rewrites are applied through `next_action_space`, `build_rewrite`, and
  `apply_rewrite`
- `--seed`, `--steps`, `--random-subsets`, and `--snapshot-dir` behave as
  specified
- snapshots use `step-000.json`, `step-001.json`, and so on
- the run stops successfully when no action space remains before the step limit
- tests cover defaults, zero steps, early stop, snapshots, and random subsets
