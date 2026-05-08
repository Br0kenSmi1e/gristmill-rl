# `rewrite` Module Design

## Goal

Define the thin action-facing wrapper for the rewrite kernel.

`rewrite` orchestrates the already-designed stages:

```text
split -> canon -> graph -> biclique
```

It exposes candidate templates, validates a receiver decision, builds a
`FactorizationRewrite`, and applies that rewrite to a `TensorComputation`.

It should not contain new algebraic reasoning. In particular, it should trust
`SplitInterface` through `ConstrGraph::interface` as the source of truth for
external and contracted indices.

## Scope

This module includes:

- action-space generation
- visible factorization templates
- hidden candidate sidecars
- decision validation
- selected sub-biclique construction
- factorization construction
- rewrite application to `TensorComputation`

This module excludes:

- split enumeration internals
- canonicalization internals
- graph construction internals
- biclique search internals
- cost/profitability filtering
- stale target-definition equality checks
- JSON I/O

## Public API

```rust
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Factorization {
    pub left_definition: TensorDef,
    pub right_definition: TensorDef,
    pub rewritten_definition: TensorDef,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActionSpace {
    pub def_index: usize,
    pub candidate_templates: Vec<Factorization>,
    candidate_graphs: Vec<ConstrGraph>,
    candidate_bicliques: Vec<Biclique>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decision {
    pub candidate_index: usize,
    pub left_mask: Vec<bool>,
    pub right_mask: Vec<bool>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FactorizationRewrite {
    pub def_index: usize,
    pub factorization: Factorization,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RewriteError {
    Canon(CanonError),
    Graph(GraphError),
    CandidateIndexOutOfRange { index: usize, len: usize },
    LeftMaskLengthMismatch { expected: usize, got: usize },
    RightMaskLengthMismatch { expected: usize, got: usize },
    EmptyLeftMask,
    EmptyRightMask,
    DefinitionIndexOutOfRange { index: usize, len: usize },
}

pub fn next_action_space(
    comp: &TensorComputation,
    start_from: usize,
) -> Result<Option<ActionSpace>, RewriteError>;

pub fn validate_decision(
    space: &ActionSpace,
    decision: &Decision,
) -> Result<(), RewriteError>;

pub fn build_rewrite(
    comp: &TensorComputation,
    space: &ActionSpace,
    decision: &Decision,
) -> Result<FactorizationRewrite, RewriteError>;

pub fn apply_rewrite(
    comp: &mut TensorComputation,
    rewrite: FactorizationRewrite,
) -> Result<(), RewriteError>;
```

`ActionSpace::candidate_graphs` and `ActionSpace::candidate_bicliques` are
private. They are aligned with `candidate_templates` by index:

```text
candidate_templates[i] <-> candidate_graphs[i]
candidate_templates[i] <-> candidate_bicliques[i]
```

The template is for receiver inspection. The hidden graph and biclique vectors
are for deterministic rewrite construction.

## Action-Space Generation

```rust
pub fn next_action_space(
    comp: &TensorComputation,
    start_from: usize,
) -> Result<Option<ActionSpace>, RewriteError>;
```

Behavior:

- scan `comp.definitions()` from `start_from`
- skip definitions with fewer than two terms
- generate candidate graphs and bicliques for the first actionable definition
- return `Ok(Some(ActionSpace))` if any candidates exist
- return `Ok(None)` if no actionable definition exists
- propagate `CanonError` and `GraphError` as `RewriteError`

Internal flow:

```text
(left_tid, right_tid) = fresh_rewrite_tensor_ids(comp)

for each (def_index, def) from start_from:
  if def.terms.len() < 2:
    continue

  (candidate_graphs, candidate_bicliques) = enumerate_candidates(comp, def)?
  if candidate_bicliques is empty:
    continue

  templates = []
  for each index i:
    templates.push(build_factorization(
      def,
      candidate_graphs[i],
      candidate_bicliques[i],
      left_tid,
      right_tid,
    ))

  return Ok(Some(ActionSpace {
    def_index,
    candidate_templates: templates,
    candidate_graphs,
    candidate_bicliques,
  }))

return Ok(None)
```

## Candidate Enumeration Pipeline

```rust
fn enumerate_candidates(
    comp: &TensorComputation,
    def: &TensorDef,
) -> Result<(Vec<ConstrGraph>, Vec<Biclique>), RewriteError>;
```

Pipeline:

```text
symmetry = canon::build_tensor_symmetry_map(comp.tensors())
pool = canon::build_index_pool(def)

left_owner_splits_by_term = vec![vec![]; def.terms.len()]
right_owner_splits_by_term = vec![vec![]; def.terms.len()]

for each term_idx, term in def.terms:
  for raw_split in split::enumerate_splits(term, def):
    (left_owner, right_owner) = canon::canon_split(raw_split, &symmetry, &pool)?
    left_owner_splits_by_term[term_idx].push(left_owner)
    right_owner_splits_by_term[term_idx].push(right_owner)

graphs = []
graphs.extend(graph::build_graphs_from_splits(def, &left_owner_splits_by_term)?)
graphs.extend(graph::build_graphs_from_splits(def, &right_owner_splits_by_term)?)

candidate_graphs = []
candidate_bicliques = []
for graph in graphs:
  for biclique in biclique::enumerate_bicliques(&graph):
    candidate_graphs.push(graph.clone())
    candidate_bicliques.push(biclique)

return (candidate_graphs, candidate_bicliques)
```

This is orchestration only. `rewrite` should not perform split, canon, graph,
or biclique logic itself.

## Decision Validation

```rust
pub fn validate_decision(
    space: &ActionSpace,
    decision: &Decision,
) -> Result<(), RewriteError>;
```

Rules:

- `candidate_index` must be in range for `space.candidate_templates`
- `left_mask.len()` must equal
  `candidate_templates[candidate_index].left_definition.terms.len()`
- `right_mask.len()` must equal
  `candidate_templates[candidate_index].right_definition.terms.len()`
- `left_mask` must contain at least one `true`
- `right_mask` must contain at least one `true`

Mask semantics:

- `true` means keep the corresponding side term
- mask order is the side term order in the visible candidate template
- strict subsets of a maximal biclique are legal

## Rewrite Construction

```rust
pub fn build_rewrite(
    comp: &TensorComputation,
    space: &ActionSpace,
    decision: &Decision,
) -> Result<FactorizationRewrite, RewriteError>;
```

Pipeline:

```text
def = comp.definitions()[space.def_index] or DefinitionIndexOutOfRange
validate_decision(space, decision)?
graph = space.candidate_graphs[decision.candidate_index] or CandidateIndexOutOfRange
biclique = space.candidate_bicliques[decision.candidate_index] or CandidateIndexOutOfRange
(left_tid, right_tid) = fresh_rewrite_tensor_ids(comp)
sub_biclique = sub_biclique_from_decision(graph, biclique, decision)
factorization = build_factorization(def, graph, sub_biclique, left_tid, right_tid)
return FactorizationRewrite { def_index: space.def_index, factorization }
```

No target-definition equality check is performed in the first design.

## Sub-Biclique Construction

```rust
fn sub_biclique_from_decision(
    graph: &ConstrGraph,
    biclique: &Biclique,
    decision: &Decision,
) -> Biclique;
```

Behavior:

- keep selected left node IDs and left coefficients by zipping
  `biclique.left_node_ids`, `biclique.left_coeffs`, and
  `decision.left_mask`
- keep selected right node IDs and right coefficients analogously
- compute `terms_used` by scanning `graph.edges` and OR-ing edges whose
  `left_id` and `right_id` are both selected

The masks have already been validated, so this helper should be mechanical.

## Factorization Construction

```rust
fn build_factorization(
    def: &TensorDef,
    graph: &ConstrGraph,
    biclique: &Biclique,
    left_tid: TensorId,
    right_tid: TensorId,
) -> Factorization;
```

This is the only place that turns graph-local records into tensor definitions.

Critical rule:

`SplitInterface` is the source of truth. Do not reconstruct external or
contracted indices from factor overlap.

Helpers:

```rust
fn contracted_indices(graph: &ConstrGraph) -> Vec<Index>;

fn side_external_indices(graph: &ConstrGraph) -> (Vec<Index>, Vec<Index>);

fn consumed_term_indices(biclique: &Biclique) -> Vec<usize>;

fn build_side_definition(
    source_nodes: &[Term],
    node_ids: &[usize],
    coeffs: &[Rational],
    side_external: &[Index],
    contracted: &[Index],
    tensor: TensorId,
) -> TensorDef;

fn build_side_term(
    source_nodes: &[Term],
    node_id: usize,
    coeff: &Rational,
) -> Term;

fn build_rewritten_definition(
    def: &TensorDef,
    left_def: &TensorDef,
    right_def: &TensorDef,
    contracted: &[Index],
    consumed: &[usize],
) -> TensorDef;
```

Index-source rules:

- `contracted_indices(graph)` returns `graph.interface.contracted.clone()`
- `side_external_indices(graph)` returns
  `(graph.interface.left_external.clone(), graph.interface.right_external.clone())`
- side definition `ext_indices` are `side_external + contracted`
- replacement term `sum_indices` are `contracted`
- replacement factors use `left_def.ext_indices` and `right_def.ext_indices`
- no helper should infer external or contracted ids from node terms

Side-term rules:

- `build_side_term` clones the source node term
- multiply its coefficient by the biclique side coefficient
- do not filter contracted ids from `sum_indices`; graph nodes already come
  from canonicalized split side terms, whose `sum_indices` contain only private
  dummy ids

Rewritten-definition rules:

- remove all consumed source terms from the original target definition
- append one replacement term:

```rust
Term {
    coeff: 1,
    sum_indices: contracted.to_vec(),
    factors: vec![
        Factor {
            tensor: left_def.base,
            indices: left_def.ext_indices.iter().map(|index| index.id).collect(),
        },
        Factor {
            tensor: right_def.base,
            indices: right_def.ext_indices.iter().map(|index| index.id).collect(),
        },
    ],
}
```

- keep the original target `base`
- keep the original target `ext_indices`

## Rewrite Application

```rust
pub fn apply_rewrite(
    comp: &mut TensorComputation,
    rewrite: FactorizationRewrite,
) -> Result<(), RewriteError>;
```

Validation before mutation:

- `rewrite.def_index` must exist in `comp.definitions()`

Mutation:

1. register two tensors with empty symmetry for the rewrite intermediates
2. remove the target definition at `def_index`
3. insert definitions at that same position in this order:
   - `left_definition`
   - `right_definition`
   - `rewritten_definition`

This mirrors the existing action-layer behavior.

The first design intentionally allows target definition drift between action
generation and rewrite application. Once a `FactorizationRewrite` is built,
`apply_rewrite` trusts it after checking only that `def_index` is in range.

## Small Helpers

```rust
fn fresh_rewrite_tensor_ids(comp: &TensorComputation) -> (TensorId, TensorId);

fn verify_rewrite_def_index(
    comp: &TensorComputation,
    rewrite: &FactorizationRewrite,
) -> Result<(), RewriteError>;

fn register_rewrite_tensors(comp: &mut TensorComputation);

fn replace_definition_with_factorization(
    comp: &mut TensorComputation,
    rewrite: FactorizationRewrite,
);

fn bits_to_vec(mask: u64) -> Vec<usize>;
```

These helpers should remain mechanical.

## Testing

Initial tests should cover:

- no action space when no definition is actionable
- `next_action_space` returns the first actionable definition
- candidate templates, hidden graphs, and hidden bicliques stay index-aligned
- `canon` and `graph` errors propagate through `next_action_space`
- decision rejects out-of-range candidate index
- decision rejects left and right mask length mismatches
- decision rejects empty left and right masks
- full-biclique decision builds the visible template exactly
- strict side subset decisions shrink the corresponding side definition
- `sub_biclique_from_decision` recomputes `terms_used` from selected edges
- factorization uses `graph.interface` for external and contracted indices
- side terms do not filter sum indices during rewrite construction
- apply registers two tensors and inserts three definitions
- apply rejects out-of-range definition index
- apply allows target definition drift after rewrite construction

## Acceptance Criteria

The `rewrite` module is complete when:

- public action API is fallible where upstream stages can fail
- `next_action_space` orchestrates split/canon/graph/biclique without embedding
  their internals
- `ActionSpace` exposes templates and keeps hidden graph/biclique vectors
  private
- decisions validate index, mask length, and nonempty-side requirements
- selected sub-bicliques are built mechanically from masks
- factorization construction uses `SplitInterface` for all external and
  contracted index truth
- `apply_rewrite` mutates `TensorComputation` only after the `def_index`
  boundary check
- no target-definition equality/staleness check is added in the first design
