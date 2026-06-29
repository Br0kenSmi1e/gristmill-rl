# Agent Guide

This repository uses an explicit model/trainer/CLI layout for Python training
code. The goal is to let agents add a new model or trainer without reading the
whole training stack first.

## Start Here

Read these files before changing training code:

- `python/gristmill_symbolics/model/protocols.py`
- `python/gristmill_symbolics/trainer/protocols.py`
- `python/gristmill_symbolics/cli/train_state.py`

The protocol files are the source of truth. Concrete implementations may add
constructor arguments and internal helpers, but they should conform to those
protocol shapes.

## Package Pattern

The supported training flow is:

```text
trainer.update(...) -> model.sample_with_logp_grad(...)
```

The current layout is:

```text
python/gristmill_symbolics/
  model/
    protocols.py
    transformer_action_selector/
      model.py
      rollout.py
      ...

  trainer/
    protocols.py
    reinforce/
      trainer.py
      objective.py

  cli/
    train.py
    train_state.py
    checkpoint.py
```

`model/` owns expression-model protocols and concrete model implementations.
`trainer/` owns trainer protocols and concrete optimizer/objective
implementations. `cli/` composes concrete models and trainers for command-line
training, checkpointing, and train-state progression.

Do not reintroduce supported top-level `policy` or `reinforce` training
packages. Those names were removed as public training paths.

## Model Boundary

Models implement `ExpressionModel` from
`python/gristmill_symbolics/model/protocols.py`.

A model is responsible for:

- owning model-specific constructor settings;
- initializing model parameters with `init_params(rng)`;
- converting a row-like batch into model inputs;
- sampling rewrite actions from the input action space;
- returning trajectory log probabilities and per-sample gradients with
  `sample_with_logp_grad(params, rng, row)`;
- returning model-local metrics only.

A model is not responsible for:

- rewards;
- advantages;
- optimizer state;
- applying optimizer updates;
- CLI parsing;
- checkpoint orchestration.

The current concrete model is
`python/gristmill_symbolics/model/transformer_action_selector/model.py`, exported
as `TransformerActionSelectorModel`.

When adding a new model:

- create a new subpackage under `python/gristmill_symbolics/model/`;
- export one concrete model class from that subpackage's `__init__.py`;
- keep public configuration in the model constructor;
- provide `constructor_kwargs()` if the model must be checkpointed;
- add protocol tests similar to
  `python/tests/model/transformer_action_selector/test_model_protocol.py`.

## Trainer Boundary

Trainers implement `Trainer` from
`python/gristmill_symbolics/trainer/protocols.py`.

A trainer is responsible for:

- owning optimizer/objective constructor settings;
- initializing optimizer state with `init_opt_state(params)`;
- validating the incoming batch and model protocol outputs;
- calling only `model.sample_with_logp_grad(params, rng, row)`;
- computing rewards, advantages, gradients, optimizer updates, and trainer
  metrics;
- returning `(new_params, new_opt_state, metrics)`.

A trainer is not responsible for:

- knowing a concrete model class;
- tokenizing model inputs;
- sampling actions directly;
- using model-specific rollout helpers;
- CLI parsing;
- checkpoint orchestration.

The current concrete trainer is
`python/gristmill_symbolics/trainer/reinforce/trainer.py`, exported as
`ReinforceTrainer`.

When adding a new trainer:

- create a new subpackage under `python/gristmill_symbolics/trainer/`;
- export one concrete trainer class from that subpackage's `__init__.py`;
- keep public configuration in the trainer constructor;
- depend on the `ExpressionModel` protocol, not on a concrete model package;
- provide `constructor_kwargs()` if the trainer must be checkpointed;
- add protocol tests similar to
  `python/tests/trainer/reinforce/test_trainer_protocol.py`.

## CLI And Checkpoints

The CLI layer is the composition boundary.

- `python/gristmill_symbolics/cli/train.py` chooses concrete classes from CLI
  arguments.
- `python/gristmill_symbolics/cli/train_state.py` owns generic train-state
  progression and should remain composition-only.
- `python/gristmill_symbolics/cli/checkpoint.py` serializes and restores
  concrete model/trainer constructors.

When adding a checkpointable model or trainer, update
`python/gristmill_symbolics/cli/checkpoint.py` with a new `kind` and
constructor-kwargs round trip. Keep checkpoint payloads explicit; do not pickle
arbitrary concrete instances as the schema.

## Dependency Rules

- Concrete model implementations must not import concrete trainers.
- Concrete trainer implementations must not import concrete models.
- CLI/checkpoint code may import concrete model and trainer classes.
- Shared protocol imports should point at `model/protocols.py` and
  `trainer/protocols.py`.
- Keep constructor settings inside concrete classes. Do not add public
  `...Config` dataclasses unless there is a separate accepted design.
- Preserve the supported call direction:
  `trainer.update(...) -> model.sample_with_logp_grad(...)`.

## Focused Verification

Use focused tests while editing:

```bash
uv run pytest python/tests/test_model_trainer_cli_layout.py -q
uv run pytest python/tests/model/transformer_action_selector/test_model_protocol.py -q
uv run pytest python/tests/trainer/reinforce/test_trainer_protocol.py -q
uv run pytest python/tests/cli/test_checkpoint.py python/tests/cli/test_checkpoint_schema.py -q
```

For broad verification before finishing Python training changes:

```bash
uv run pytest -q
cargo test
```
