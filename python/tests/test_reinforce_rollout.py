import jax
import jax.numpy as jnp

from gristmill_symbolics.reinforce.rollout import make_rng_grid
from gristmill_symbolics.reinforce.types import DECISION_ACTION, DECISION_TARGET


def test_make_rng_grid_uses_step_sample_decision_kind_axes():
    root = jax.random.PRNGKey(123)
    grid = make_rng_grid(root, update_index=7, max_steps=3, batch_size=2)
    expected = jax.random.split(jax.random.fold_in(root, 7), 3 * 2 * 2).reshape(
        (3, 2, 2, 2)
    )

    assert grid.shape == (3, 2, 2, 2)
    assert jnp.array_equal(grid, expected)
    assert jnp.array_equal(grid[0, 0, DECISION_TARGET], expected[0, 0, 0])
    assert jnp.array_equal(grid[0, 0, DECISION_ACTION], expected[0, 0, 1])
