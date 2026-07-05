# Direct Optimizer Trainer Design

## Role

The trainer module owns supervised weighted maximum-likelihood training for the
direct optimizer model.

It consumes processed JSONL files produced by the dataset module, collates them
into static padded batches, runs batched JAX/Flax NNX training steps, evaluates
validation/test loss, and saves/restores direct optimizer checkpoints.

It does not generate candidates, verify equivalence, reconstruct
`TensorComputation`, sample optimized computations, or use the existing
action-selector/REINFORCE training stack.

## Data Contract

Trainer consumes processed JSONL files:

```text
train_dataset: required
valid_dataset: optional
test_dataset: optional
```

The trainer does not split datasets internally. Train/valid/test files are
created before training by the dataset module or an external workflow.

Each processed row must contain:

```text
source_text
target_text
weight
input_key
candidate_key
candidate_log_flops
```

The trainer trusts processed dataset semantics. It does not:

- re-verify equivalence;
- recompute costs;
- validate outputs;
- regroup rows;
- deduplicate rows;
- recompute weights.

Trainer validation is limited to model-shape compatibility:

1. Required processed fields exist.
2. `source_text` and `target_text` encode through converter structured-token
   encoding.
3. Encoded source length is `<= source_len`.
4. Encoded target/label length is `<= target_len`.
5. Scalar values fit `scalar_value_min` and `scalar_value_max`.

## Static Batch Collation

Static shape is a trainer contract.

Constructor or CLI settings fix:

```text
batch_size
source_len
target_len
scalar_value_min
scalar_value_max
```

Collation steps:

1. Encode source and target text through the converter structured-token layer.
2. Build decoder inputs and labels with `make_decoder_inputs`.
3. Pad each token field to fixed `source_len` or `target_len`.
4. Batch with fixed `batch_size`.
5. Drop final partial batches for train, valid, and test.
6. If an enabled dataset has fewer compatible rows than `batch_size`, raise
   `ValueError`.

Batch fields:

```python
{
    "source_tokens": TokenBatch[batch_size, source_len],
    "decoder_input_tokens": TokenBatch[batch_size, target_len],
    "target_tokens": TokenBatch[batch_size, target_len],
    "target_mask": bool[batch_size, target_len],
    "example_weight": float32[batch_size],
}
```

No dummy rows or `real_example_mask` are part of the MVP. Final partial batches
are dropped for all datasets. CLI logs may report incompatible-row counts and
dropped-remainder counts, but those are not core metrics.

## Training And Evaluation Objective

The trainer uses the model's sequence log-probability:

```text
sequence_log_prob_i = sequence_log_prob(logits_i, target_i, target_mask_i)
nll_i = -sequence_log_prob_i
loss = sum(example_weight_i * nll_i) / max(sum(example_weight_i), epsilon)
```

The same weighted loss computation is used for training, validation, and test.

Rules:

- Training uses only the train dataset.
- Validation uses only the valid dataset.
- Test uses only the test dataset and only when requested.
- Validation and test never update model or optimizer state.
- Evaluation calls use `deterministic=True`.
- Train calls use `deterministic=False` only if dropout is greater than zero.
- No unweighted NLL metric is part of the MVP.

Core metrics:

```text
train_loss
valid_loss
test_loss
updates
epoch
```

## NNX Train Step

Use the official Flax NNX training style.

The trainer should use:

```text
nnx.Optimizer
nnx.value_and_grad
nnx.jit
```

The compiled train step receives one static batch and updates model/optimizer
state:

```python
train_step(model, optimizer, batch) -> dict[str, float]
```

The compiled eval step receives one static batch and does not mutate
model/optimizer state:

```python
eval_step(model, batch) -> dict[str, float]
```

Performance rules:

- No Python loop over examples inside train/eval steps.
- Batch size and sequence lengths are fixed per compiled step.
- Dataset shuffling and collation happen outside JIT.
- Metric aggregation across batches happens outside JIT.

## Train Loop And CLI

Training loop:

1. Load train dataset.
2. Optionally load valid dataset.
3. Optionally load test dataset for final evaluation.
4. Collate compatible rows into fixed-size batches.
5. Initialize or restore model/optimizer from checkpoint.
6. For each epoch:
   - shuffle train batches with epoch RNG;
   - run compiled train step for each train batch;
   - compute average train loss;
   - if a valid dataset is provided, compute valid loss;
   - save checkpoint.
7. If a test dataset is provided and test evaluation is requested, compute test
   loss after training.

CLI shape:

```bash
python -m gristmill_symbolics.direct_optimizer.train \
  --train-dataset train.jsonl \
  --valid-dataset valid.jsonl \
  --test-dataset test.jsonl \
  --checkpoint-out direct_optimizer_ckpt \
  --checkpoint-in direct_optimizer_ckpt \
  --epochs 10 \
  --batch-size 8 \
  --learning-rate 1e-3 \
  --source-len 2048 \
  --target-len 2048 \
  --scalar-value-min -4096 \
  --scalar-value-max 4096 \
  --d-model 128 \
  --num-layers 2 \
  --num-heads 4 \
  --seed 0
```

`--checkpoint-in` is optional. If present, it restores model, optimizer, epoch,
and update count. Constructor/static-shape arguments must match the checkpoint,
or loading raises `ValueError`.

## Checkpointing

Use the standard Flax NNX + Orbax checkpoint path.

Checkpoint content:

```text
schema_version
converter_schema_version
model_kwargs
model_state
optimizer_state
epoch
updates
last_train_loss
last_valid_loss optional
```

Rules:

- Save after each epoch by default.
- `checkpoint_in` restores model state, optimizer state, epoch, and update
  count.
- `checkpoint_out` writes the latest training state.
- Static model kwargs must match on restore:

  ```text
  source_len
  target_len
  scalar_value_min
  scalar_value_max
  d_model
  num_layers
  num_heads
  ```

- Validation/test rows are not stored in the checkpoint.
- Existing `gristmill_symbolics.cli.checkpoint` is not used.
- A checkpoint can be loaded by the sampler for inference, even if optimizer
  state is ignored there.

## Boundaries

Trainer may import:

```text
dataset JSONL readers
converter token encoding
model module
JAX
Flax NNX
Optax
Orbax
NumPy/math/path utilities
```

Trainer must not import:

```text
TensorComputation
equivalent_computations
sampler
gristmill_symbolics.model.transformer_action_selector
gristmill_symbolics.trainer.reinforce
gristmill_symbolics.cli.checkpoint
```

The trainer trusts processed dataset semantics and only checks model-shape
compatibility. It does not generate candidates or sample optimized computations.

## Tests

Required focused tests:

- Loads processed dataset files and collates fixed-size batches.
- Drops final partial batch for train, valid, and test.
- Raises if an enabled dataset has fewer compatible rows than `batch_size`.
- Rejects incompatible rows exceeding source/target length or scalar bounds.
- Train step changes model state and returns finite `train_loss`.
- Validation step returns finite `valid_loss` and does not change model or
  optimizer state.
- Checkpoint save/load restores model state, optimizer state, epoch, and
  updates.
- Restore rejects incompatible static model kwargs.
- CLI can run one tiny epoch and write a checkpoint.
- Trainer module does not import forbidden symbolic, verifier, action-selector,
  REINFORCE, or CLI checkpoint modules.

Verification command:

```bash
uv run pytest python/tests/direct_optimizer/test_trainer.py -q
```

