# Direct Optimizer Sampler Design

## Purpose

The sampler module is the inference boundary for the direct optimizer. It turns
a trained direct optimizer checkpoint into verified optimization proposals for a
single input `TensorComputation`.

The sampler does not prove correctness by trusting the neural model. The model
is only a proposal distribution. Every generated candidate must pass symbolic
reconstruction and `equivalent_computations` before it can be returned.

The core inference flow is:

```text
checkpoint + input computation + ordered outputs
  -> source DSL tokens
  -> batched model samples of target DSL tokens
  -> target DSL text
  -> reconstructed candidate computations
  -> verifier filter
  -> lowest-log-flops valid candidate
```

## Boundary

The sampler owns:

- loading direct optimizer checkpoints for inference;
- converting an input computation into source DSL tokens;
- calling the model sampling helper with static padded shapes;
- decoding generated target tokens into target DSL text;
- reconstructing full candidate computations from the input computation and
  generated definitions;
- running symbolic equivalence checks;
- computing candidate log flops;
- selecting the best valid candidate;
- reporting inference metrics.

The sampler does not own:

- dataset generation, splitting, deduplication, or weighting;
- weighted teacher-forced loss;
- optimizer state updates;
- validation or test loss evaluation;
- checkpoint writing during training;
- action-selector or rewrite-trajectory logic;
- grammar-constrained decoding in the MVP.

## Dependencies

The sampler may import:

```text
gristmill_symbolics.TensorComputation
gristmill_symbolics.equivalent_computations
gristmill_symbolics.direct_optimizer.converter
gristmill_symbolics.direct_optimizer.model
```

It uses a direct optimizer checkpoint helper shared with the trainer for loading
checkpoint payloads. That helper belongs under the direct optimizer package, not
under the existing generic CLI checkpoint module.

The sampler must not import:

```text
gristmill_symbolics.dataset generation internals
gristmill_symbolics.trainer.reinforce
gristmill_symbolics.model.transformer_action_selector
gristmill_symbolics.cli.checkpoint
```

The sampler does not import the direct optimizer trainer module. Shared
checkpoint read/write mechanics belong in `direct_optimizer.checkpoint` so
trainer and sampler can depend on the same serialization contract without
depending on each other.

## Public API

The main programmatic API is:

```python
optimize_with_model(
    model,
    params,
    input_computation: TensorComputation,
    outputs: list[int],
    *,
    num_samples: int,
    sample_batch_size: int,
    source_len: int,
    target_len: int,
    temperature: float,
    seed: int,
) -> tuple[TensorComputation | None, dict]
```

`outputs` is ordered and is passed to `equivalent_computations` exactly as
provided.

The function returns:

- the lowest-cost verified candidate, or `None` if no valid candidate is found;
- a metrics dictionary describing what happened during sampling.

The model and params arguments are explicit so tests can pass fake or tiny
models without depending on checkpoint files.

## Checkpoint Loading API

The sampler also exposes a convenience API for checkpoint-backed inference:

```python
optimize_from_checkpoint(
    checkpoint_path: str,
    input_computation: TensorComputation,
    outputs: list[int],
    *,
    num_samples: int,
    sample_batch_size: int,
    source_len: int | None = None,
    target_len: int | None = None,
    temperature: float = 1.0,
    seed: int = 0,
) -> tuple[TensorComputation | None, dict]
```

`optimize_from_checkpoint` loads:

- converter schema version;
- model constructor kwargs;
- model state.

It ignores:

- optimizer state;
- epoch;
- update count;
- last train/valid loss.

If `source_len` or `target_len` is omitted, the sampler uses the checkpoint model
kwargs. If the caller provides a value that disagrees with the checkpoint, the
sampler raises `ValueError`.

## Static Shape Policy

Sampling uses fixed JAX shapes:

```text
source_tokens: [sample_batch_size, source_len]
generated_tokens: [sample_batch_size, target_len]
```

The input source tokens are repeated across the sample batch.

`num_samples` does not need to be divisible by `sample_batch_size`. The sampler
must run enough fixed-size batches to cover `num_samples`, then ignore padded
extra generated rows from the final batch.

Example:

```text
num_samples = 10
sample_batch_size = 4

run 3 model batches = 12 generated rows
evaluate first 10 rows
ignore final 2 rows
```

This avoids recompiling a smaller final batch and keeps the user-facing API
simple.

## Sampling Flow

For each sampling run:

1. Validate `num_samples > 0`, `sample_batch_size > 0`, `source_len > 0`,
   `target_len > 0`, and `temperature > 0`.
2. Convert `input_computation` to source DSL text with the converter.
3. Encode source DSL text into structured source tokens.
4. Verify the encoded source fits `source_len`; otherwise raise `ValueError`.
5. Reuse the padded source token row across each sample batch.
6. Split the PRNG key per sample batch.
7. Call the model sampling helper to produce target token arrays.
8. Decode each generated target token row to target DSL text.
9. Parse target DSL text with `target_text_to_definitions` to classify syntax
   failures separately from reconstruction failures.
10. Reconstruct a candidate with
   `target_text_to_computation(input_computation, target_text)`.
11. Run `equivalent_computations(input_computation, candidate, outputs)`.
12. For verified candidates, compute `candidate.log_total_flops()`.
13. Keep the valid candidate with the lowest log flops.

If no candidate is valid, return `(None, metrics)`.

## Failure Accounting

The metrics dictionary includes:

```python
{
    "total_samples": int,
    "decode_failures": int,
    "parse_failures": int,
    "reconstruction_failures": int,
    "verifier_failures": int,
    "valid_samples": int,
    "best_log_flops": float | None,
}
```

Definitions:

- `total_samples`: generated rows evaluated, excluding padded extra rows.
- `decode_failures`: token arrays could not be converted to target text.
- `parse_failures`: target DSL text was syntactically invalid.
- `reconstruction_failures`: target DSL parsed, but could not be combined with
  the input computation into a valid `TensorComputation`.
- `verifier_failures`: reconstruction succeeded, but symbolic equivalence failed.
- `valid_samples`: reconstructed and verified candidates.
- `best_log_flops`: cost of the selected candidate, or `None`.

The sampler separates target failures by phase:

1. Structured token arrays to target DSL text. A `ValueError` here increments
   `decode_failures`.
2. Target DSL text to definitions through `target_text_to_definitions`. A
   `ValueError` here increments `parse_failures`.
3. Target DSL text plus the input computation to a full candidate through
   `target_text_to_computation`. A `ValueError` here increments
   `reconstruction_failures`.

This explicit sequence lets the sampler produce stable metrics even though the
converter raises the same exception type for parser and reconstruction errors.

Unexpected internal exceptions from the model or checkpoint loader must not be
silently converted into sample failures. They should propagate, because they
indicate infrastructure bugs rather than bad generated candidates.

## CLI

The sampler CLI is:

```bash
python -m gristmill_symbolics.direct_optimizer.sample \
  --checkpoint direct_optimizer_ckpt \
  --input input_computation.json \
  --outputs 1,3 \
  --samples 64 \
  --sample-batch-size 8 \
  --temperature 1.0 \
  --output optimized.json
```

`--outputs` preserves user order. Comma-separated values are sufficient for the
MVP.

CLI behavior:

- read the input computation from JSON;
- load the checkpoint;
- sample and verify candidates;
- write the selected candidate as computation JSON when a valid candidate exists;
- exit nonzero when no valid candidate exists unless an explicit
  `--allow-no-result` flag is added later;
- print the metrics dictionary as JSON to stdout.

The CLI must not invoke dataset generation or training.

## Testing

Required tests:

1. Fake generated rows:
   - invalid target text;
   - valid target DSL that reconstructs a non-equivalent computation;
   - valid equivalent target DSL.
   The sampler returns only the equivalent candidate.
2. Metrics:
   - counters reflect decode/parse/reconstruction/verifier failures and valid
     samples;
   - `best_log_flops` is set only when there is at least one valid sample.
3. Best candidate selection:
   - two verified candidates with different `log_total_flops`;
   - sampler returns the lower-cost candidate.
4. Static batch padding:
   - `num_samples` not divisible by `sample_batch_size`;
   - padded extra rows are ignored and do not affect metrics.
5. Checkpoint-backed loading:
   - sampler loads model kwargs and model state from a direct optimizer
     checkpoint;
   - optimizer state is not required for inference.
6. CLI smoke test:
   - loads a tiny checkpoint or fake checkpoint fixture;
   - writes optimized JSON when a valid sample is produced.

## Non-Goals

The sampler MVP will not implement:

- beam search;
- reranking beyond `log_total_flops`;
- grammar-constrained masks;
- integration with rewrite action trajectories;
- repair of invalid generated DSL;
- multi-input batched optimization API;
- stochastic acceptance of higher-cost candidates;
- checkpoint registration in `gristmill_symbolics.cli.checkpoint`.

The sampler remains a narrow inference module: generate many candidates,
filter them through symbolic correctness, and return the cheapest verified one.
