# Direct Optimizer Model Design

## Role

The model module owns the neural proposal distribution over direct optimizer
structured DSL tokens:

```text
p(target_definitions | source_computation)
```

It is a pure conditional sequence model. It does not parse DSL text,
reconstruct `TensorComputation`, call the verifier, compute costs, group
examples, read processed JSONL, own optimizer state, or define dataset-weighted
training loss.

The model receives already-padded structured token batches from the converter
and trainer. It returns structured-token logits, log-probability helpers, and
batched sampled token sequences.

## Architecture Choice

Use a structured-token encoder-decoder Transformer.

Rejected alternatives:

- Flat vocabulary encoder-decoder: simpler, but loses the semantic distinction
  between scalar roles such as `tensor_id:3` and `index_id:3`.
- Decoder-only prefix model: workable, but source/target masking and trainer
  loss boundaries are less explicit than an encoder-decoder model.

The chosen factorization is:

```text
p(y | x) = product_t p(y_t | y_<t, x)
```

where `x` is source DSL structured tokens and `y` is target definitions
structured tokens.

## Flax NNX Structure

Use the official Flax NNX style for the neural network.

Core network:

```python
class DirectOptimizerTransformer(nnx.Module):
    def __init__(
        self,
        *,
        source_len: int,
        target_len: int,
        scalar_value_min: int,
        scalar_value_max: int,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
        init_scale: float = 0.02,
        rngs: nnx.Rngs,
    ):
        ...

    def __call__(
        self,
        source_tokens,
        decoder_input_tokens,
        *,
        deterministic: bool = True,
    ):
        ...
```

Model module public surface:

```python
DirectOptimizerTransformer
make_decoder_inputs(target_tokens) -> tuple[decoder_input_tokens, labels, mask]
token_log_probs(logits, target_tokens) -> jax.Array
sequence_log_prob(logits, target_tokens, target_mask) -> jax.Array
sample_tokens(
    model,
    rng,
    source_tokens,
    *,
    max_length,
    temperature,
    mask_provider=None,
) -> tuple[object, jax.Array]
```

Boundaries:

- The NNX module owns parameters and forward computation.
- Pure helper functions own scoring and sampling logic.
- Trainer owns `nnx.Optimizer`, weighted objective, and checkpoint packaging.
- No separate public config dataclass; constructor kwargs are the configuration.
- Model module does not import `TensorComputation`, verifier, costs, dataset,
  trainer, sampler, existing action-selector model, REINFORCE trainer, or CLI
  checkpoint modules.

## Static Shape And Batching Requirements

Static shape is a first-class contract.

Constructor settings fix:

```text
source_len
target_len
scalar_value_min
scalar_value_max
d_model
num_layers
num_heads
```

Every compiled training or scoring call should receive arrays shaped:

```text
source_tokens:        [batch, source_len]
decoder_input_tokens: [batch, target_len]
target_tokens:        [batch, target_len]
target_mask:          [batch, target_len]
```

For structured token batches, each token field has the same `[batch, length]`
shape:

```text
kind
keyword
scalar_type
scalar_value
mask
```

Rules:

- The model never infers pad length dynamically inside compiled code.
- Overlong source or target examples are rejected by trainer/data collation.
- Batch size is static per compiled trainer step.
- Sampling returns `[num_samples, target_len]` even when sequences stop early.
- Use JAX/NNX transforms for batched execution, not Python loops over samples.
- A `jax.lax.scan` loop over target positions is acceptable for autoregressive
  sampling.

These rules let the trainer and sampler compile stable kernels and avoid shape
churn.

## Structured Inputs

The model consumes structured token arrays from the converter:

```python
{
    "kind": int32[batch, length],
    "keyword": int32[batch, length],
    "scalar_type": int32[batch, length],
    "scalar_value": int32[batch, length],
    "mask": bool[batch, length],
}
```

Training batches use:

```python
source_tokens: TokenBatch[batch, source_len]
decoder_input_tokens: TokenBatch[batch, target_len]
target_tokens: TokenBatch[batch, target_len]
target_mask: bool[batch, target_len]
example_weight: float32[batch]
```

`example_weight` is passed through trainer batches, but the model module does not
apply it.

Special token kinds:

```text
PAD
BOS
EOS
KEYWORD
SCALAR
```

Teacher forcing layout:

```text
target logical tokens:    t0 t1 ... tn
decoder input tokens:     BOS t0 ... t(n-1)
target label tokens:      t0 t1 ... tn EOS
```

Padding fills unused positions with `PAD`, and masks decide which target
positions contribute to scoring.

## Embedding Strategy

The model embeds structured tokens by summing small components:

```text
token_embedding =
    kind_embedding[kind]
  + keyword_embedding[keyword]
  + scalar_type_embedding[scalar_type]
  + scalar_value_projection(scalar_value)
  + position_embedding[position]
```

Rules:

- `KEYWORD` tokens use `kind`, `keyword`, and position.
- `SCALAR` tokens use `kind`, `scalar_type`, projected numeric
  `scalar_value`, and position.
- `PAD`, `BOS`, and `EOS` use `kind` and position as appropriate; `PAD` is
  masked to zero.
- `scalar_value` is a numeric feature, not a categorical token vocabulary, so
  unseen ids and sizes can still produce meaningful input embeddings.
- `sym_action` scalar values are encoded as small integers by the converter.

The MVP scalar projection can be a linear projection over a normalized numeric
feature. A later model can replace it with a richer numeric embedding without
changing the converter DSL.

## Encoder-Decoder Network

Use a compact encoder-decoder Transformer.

Architecture:

```text
source tokens
  -> structured token embedder
  -> encoder self-attention blocks
  -> source memory

decoder input tokens
  -> structured token embedder
  -> causal decoder self-attention blocks
  -> cross-attention to source memory
  -> output heads
```

Each encoder layer:

```text
self-attention over source tokens
MLP/feed-forward
residual + layer norm
```

Each decoder layer:

```text
causal self-attention over generated target prefix
cross-attention over source memory
MLP/feed-forward
residual + layer norm
```

The implementation can use Flax NNX modules wrapping JAX/Flax attention, dense,
dropout, and layer-norm primitives. No custom attention kernel is required for
the MVP.

## Output Heads And Scoring

The decoder emits structured-token logits at every target position:

```text
kind_logits:         [batch, target_len, num_kinds]
keyword_logits:      [batch, target_len, num_keywords]
scalar_type_logits:  [batch, target_len, num_scalar_types]
scalar_value_logits: [batch, target_len, num_scalar_values]
```

`num_scalar_values` is fixed by constructor bounds:

```text
scalar_value_min <= scalar_value <= scalar_value_max
```

Values outside that range cannot be generated or scored and should be rejected
during trainer collation.

Scoring helpers compute log-probabilities:

```python
token_log_probs(logits, target_tokens) -> float32[batch, target_len]
sequence_log_prob(logits, target_tokens, target_mask) -> float32[batch]
```

Scoring rules:

- Always score `kind`.
- If label kind is `KEYWORD`, also score `keyword`.
- If label kind is `SCALAR`, also score `scalar_type` and `scalar_value`.
- Ignore irrelevant heads for `PAD`, `BOS`, and `EOS`.
- Ignore positions where `target_mask=False`.

The model returns log-probabilities. Dataset weights and optimizer loss belong
to the trainer.

## Sampling

Sampling is batched and autoregressive.

Inputs:

```python
model: DirectOptimizerTransformer
rng: jax.Array
source_tokens: TokenBatch[num_samples, source_len]
max_length: int  # <= target_len
temperature: float = 1.0
mask_provider = None
```

Output:

```python
sampled_tokens: TokenBatch[num_samples, target_len]
sample_mask: bool[num_samples, target_len]
```

Behavior:

1. Initialize decoder prefix with `BOS`.
2. For each position up to `max_length`, run the model on the static padded
   prefix.
3. Sample structured token components from output heads.
4. Stop each sample after `EOS`, but keep arrays padded to `target_len`.
5. Return full padded token arrays and mask.

Future grammar mask hook:

```python
mask_provider(prefix_tokens, step, source_context) -> StructuredTokenMask
```

It can restrict allowed ids for:

```text
kind
keyword
scalar_type
scalar_value
```

MVP behavior uses `mask_provider=None`, so invalid syntax is filtered later by
the converter and sampler.

Implementation requirements:

- no Python loop over samples;
- loop over positions should use `jax.lax.scan` or an NNX-compatible transform;
- output shapes are static.

## Tests

Required focused tests:

- NNX module initializes and returns logits with expected static shapes.
- Structured embedder distinguishes same scalar value with different scalar
  types: `tensor_id:3 != index_id:3`.
- Forward pass respects source and target masks: padded positions do not affect
  encoded output or log-probability.
- `make_decoder_inputs` creates `BOS + target prefix` and
  `target labels + EOS` with correct masks.
- `token_log_probs` scores the relevant heads only.
- `sequence_log_prob` ignores padded positions.
- Sampling returns static padded `[num_samples, target_len]` token batches.
- Sampling with fixed RNG/model state is deterministic.
- Scalar values outside configured bounds are rejected by collation or scoring
  helpers.
- Model module does not import `TensorComputation`, verifier, dataset, trainer,
  sampler, action-selector, REINFORCE, or CLI checkpoint modules.

Verification command:

```bash
uv run pytest python/tests/direct_optimizer/test_model.py -q
```

