# Story Spec: NNX Transformer Core

## Product Owner Inputs

Outcome:

Build a reusable Flax NNX Transformer encoder-decoder core that operates on
dense vectors and masks, so later model stories can wrap it with token
embeddings, vocabulary logits, losses, sampling, and tokenizer collation.

Deliverables:
- Public reusable Transformer core modules for:
  - encoder stack
  - decoder stack
  - encoder block
  - decoder block
- Dense-vector inputs only:
  - encoder input shape `[batch, source_len, d_model]`
  - decoder input shape `[batch, target_len, d_model]`
- Mask support for:
  - source padding mask
  - target padding mask
  - decoder causal self-attention
  - decoder cross-attention to encoder memory
- Configurable JAX attention backend:
  - `attention_implementation=None | "xla" | "cudnn"`
  - passed through to `jax.nn.dot_product_attention`
- Deterministic dropout control.
- Focused tests for output shapes, causal masking, source masking, and
  deterministic behavior.

Non-goals:
- Token IDs, tokenizer imports, BOS/EOS/PAD token semantics, vocabulary size, or
  definition grammar.
- Token embeddings, positional embeddings, tied output embeddings, or logits
  heads.
- Decoder-input, label, target-mask, loss, log-probability, sampling, grammar
  mask, dataset collation, trainer, CLI, checkpoint, or TensorComputation work.
- Forcing the `"cudnn"` attention implementation in tests or on CPU-only
  environments.

Acceptance criteria:
- Encoder stack returns `[batch, source_len, d_model]`.
- Decoder stack returns `[batch, target_len, d_model]`.
- Decoder output at an earlier target position is unchanged when later target
  positions change, with dropout disabled.
- Masked source positions do not affect decoder cross-attention output, with
  dropout disabled.
- The modules are standard Flax NNX modules and can be constructed with
  `nnx.Rngs`.
- The core uses `jax.nn.dot_product_attention` and passes through the configured
  `attention_implementation`.
- Tests run on the local CPU-only environment using the default attention
  implementation.
- The core imports no tokenizer code and exposes no model/training semantics.

Constraints:
- Use the existing Python package dependencies: JAX and Flax NNX.
- Keep the story scoped to reusable vector Transformer behavior.
- Keep public configuration on module constructors; do not add a separate
  public config dataclass.

## Role Boundary

The user is acting as Product Owner / Stakeholder.
Low-level implementation decisions are agent-owned.
User-provided implementation details are binding only when they express product,
compatibility, safety, performance, or operational constraints.

## Agent-Owned Notes

Assumptions:
- A small conventional Transformer block is enough for this story:
  pre-norm attention, feed-forward MLP, residual connections, and dropout.
- Attention mask shapes and internal head projection layout are implementation
  details as long as the public vector/mask behavior is correct.
- The future flat token model will own token embeddings, position embeddings,
  logits, and tokenizer integration.

Risk level: medium

Story size: medium

Escalation: none

Implementation hints:
- Prefer a focused package under `gristmill_symbolics/nn/`.
- Keep the attention helper private to the Transformer core.
- Use CPU-safe tests with `attention_implementation=None`; only assert that
  constructor configuration is stored/passed, not that `"cudnn"` runs locally.

Follow-up stories:
- Flat token seq2seq wrapper using tokenizer vocabulary IDs.
- Decoder-input/label/mask and sequence log-probability helpers.
- Dataset collation and generated-token sampling.
- Trainer/CLI/checkpoint integration.

## Definition of Done

- Acceptance criteria met
- Deliverables completed
- Non-goals avoided
- Verification run
- No unjustified scope, dependency, abstraction, public API, or broad refactor
  growth
