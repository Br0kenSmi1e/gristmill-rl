# Story Spec: Preprocessed Supervised Dataset

## Product Owner Inputs

Outcome:

Build the offline dataset boundary for definitions-only supervised training from
JSONL rows of weighted `TensorComputation` pairs.

Deliverables:
- Read raw JSONL rows shaped as
  `{"source": {...}, "target": {...}, "weight": 1.0}`.
- Validate each raw row strictly through `TensorComputation`.
- Tokenize only `snapshot()["definitions"]` for source and target
  computations.
- Write fixed-shape preprocessed arrays:
  - `source_ids: int32[N, source_len]`
  - `decoder_input_ids: int32[N, target_len]`
  - `target_ids: int32[N, target_len]`
  - `target_mask: bool[N, target_len]`
  - `example_weight: float32[N]`
- Write metadata with tokenizer config, lengths, vocabulary size, and example
  count.
- Load the preprocessed dataset back from disk.
- Provide a simple fixed-shape batch iterator for training.

Non-goals:
- No training CLI.
- No checkpointing.
- No validation loop.
- No sampling, reward, or REINFORCE integration.
- No full-computation tokenizer.
- No inferred sequence lengths.
- No skip-invalid mode.
- No dataset sharding.

Acceptance criteria:
- Tests cover JSONL to arrays to load round trip.
- Tests cover BOS/EOS/pad/mask target construction.
- Tests cover strict failures for invalid JSONL rows.
- Tests cover strict failures for over-length source and target rows.
- Tests cover batching preserves fixed array shapes.
- Existing tokenizer, scoring, supervised, and flat model tests remain green.

Constraints:
- Definitions-only: ignore ranges, tensor symmetry, and other global computation
  metadata.
- `source_len` and `target_len` are explicit inputs.
- Target content is valid only when `len(content_ids) + 1 <= target_len`.
- Decoder-input construction is private to preprocessing for now.
- First invalid row fails with row number and reason.
- Output arrays are NumPy arrays suitable for fixed-shape JAX training.

## Role Boundary

The user is acting as Product Owner / Stakeholder.
Low-level implementation decisions are agent-owned.
User-provided implementation details are binding only when they express product,
compatibility, safety, performance, or operational constraints.

## Agent-Owned Notes

Assumptions:
- The raw JSONL `source` and `target` values use the same JSON structure accepted
  by `TensorComputation.from_json_string`.
- The preprocessed output can be a compact `.npz` array file plus a JSON metadata
  sidecar.
- A small Python module is enough; CLI commands can wrap it in a later story.

Risk level: low

Story size: medium

Escalation: none

Implementation hints:
- Use `TensorComputation.from_json_string(json.dumps(value))` to validate row
  computations.
- Use `FlatDefinitionTokenizer.encode_definitions_padded` for source arrays.
- Build target arrays by padding `bos + content` as decoder input and
  `content + eos` as labels, so the EOS label is conditioned on the final
  content token.
- Keep row parsing, preprocessing, loading, and batching separate enough to test
  without adding training CLI code.

Follow-up stories:
- Minimal supervised training CLI over the preprocessed dataset.
- Checkpointable train state and resume.
- Validation loop and best-checkpoint selection.
- Optional dataset sharding or skip-invalid preprocessing mode.

## Definition of Done

- Acceptance criteria met
- Deliverables completed
- Non-goals avoided
- Verification run
- No unjustified scope, dependency, abstraction, public API, or broad refactor
  growth
