"""Pure fixed-step TWAP scheduling helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def fixed_step_twap_execution_intervals(
    max_steps_in_episode: jax.Array,
) -> jax.Array:
    """Return normal decision intervals, excluding the terminal sentinel step."""
    max_steps = jnp.asarray(max_steps_in_episode, dtype=jnp.int32)
    return jnp.maximum(max_steps - 1, 0)


def fixed_step_twap_cumulative_quantity(
    task_size: jax.Array,
    completed_intervals: jax.Array,
    max_steps_in_episode: jax.Array,
) -> jax.Array:
    """Return the floor-apportioned quantity after completed decision intervals."""
    task = jnp.maximum(jnp.asarray(task_size, dtype=jnp.int32), 0)
    interval_count = fixed_step_twap_execution_intervals(max_steps_in_episode)
    safe_interval_count = jnp.maximum(interval_count, 1)
    completed = jnp.clip(
        jnp.asarray(completed_intervals, dtype=jnp.int32),
        0,
        interval_count,
    )

    base_quantity = jnp.floor_divide(task, safe_interval_count)
    remainder = jnp.mod(task, safe_interval_count)
    cumulative = (
        base_quantity * completed
        + jnp.floor_divide(remainder * completed, safe_interval_count)
    )
    return jnp.where(interval_count > 0, cumulative, 0).astype(jnp.int32)


def fixed_step_twap_child_quantity(
    task_size: jax.Array,
    step_counter: jax.Array,
    max_steps_in_episode: jax.Array,
) -> jax.Array:
    """Return one deterministic integer child for the current decision state."""
    step = jnp.asarray(step_counter, dtype=jnp.int32)
    interval_count = fixed_step_twap_execution_intervals(max_steps_in_episode)
    valid_step = (step >= 0) & (step < interval_count)
    clipped_step = jnp.clip(step, 0, jnp.maximum(interval_count - 1, 0))
    before = fixed_step_twap_cumulative_quantity(
        task_size,
        clipped_step,
        max_steps_in_episode,
    )
    after = fixed_step_twap_cumulative_quantity(
        task_size,
        clipped_step + 1,
        max_steps_in_episode,
    )
    return jnp.where(valid_step, after - before, 0).astype(jnp.int32)
