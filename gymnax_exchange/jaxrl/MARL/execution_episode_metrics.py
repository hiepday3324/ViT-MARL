"""Execution episode metrics accumulated across PPO rollout boundaries."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class ExecutionEpisodeMetrics(NamedTuple):
    episode_count: jax.Array
    episode_return_mean: jax.Array
    terminal_quant_left_mean: jax.Array
    terminal_fill_ratio_mean: jax.Array


def empty_execution_episode_metrics() -> ExecutionEpisodeMetrics:
    return ExecutionEpisodeMetrics(
        episode_count=jnp.asarray(0, dtype=jnp.int32),
        episode_return_mean=jnp.asarray(0.0, dtype=jnp.float32),
        terminal_quant_left_mean=jnp.asarray(0.0, dtype=jnp.float32),
        terminal_fill_ratio_mean=jnp.asarray(0.0, dtype=jnp.float32),
    )


def accumulate_execution_episode_metrics(
    running_episode_return: jax.Array,
    rewards: jax.Array,
    terminals: jax.Array,
    quant_left: jax.Array,
    task_size: jax.Array,
) -> tuple[jax.Array, ExecutionEpisodeMetrics]:
    """Accumulate returns and summarize episodes ending in this rollout."""
    running_episode_return = jnp.asarray(running_episode_return, dtype=jnp.float32)
    rewards = jnp.asarray(rewards, dtype=jnp.float32)
    terminals = jnp.asarray(terminals, dtype=jnp.bool_)
    quant_left = jnp.asarray(quant_left, dtype=jnp.float32)
    task_size = jnp.asarray(task_size, dtype=jnp.float32)

    if rewards.ndim != 2:
        raise ValueError(f"rewards must have shape (time, actors), got {rewards.shape}.")
    expected_shape = rewards.shape
    for name, value in (
        ("terminals", terminals),
        ("quant_left", quant_left),
        ("task_size", task_size),
    ):
        if value.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {value.shape}."
            )
    if running_episode_return.shape != expected_shape[1:]:
        raise ValueError(
            "running_episode_return must have shape "
            f"{expected_shape[1:]}, got {running_episode_return.shape}."
        )

    def _step(running_return, transition):
        reward, terminal, terminal_quant_left, transition_task_size = transition
        updated_return = running_return + reward
        safe_task_size = jnp.where(transition_task_size > 0, transition_task_size, 1.0)
        fill_ratio = jnp.clip(
            1.0 - terminal_quant_left / safe_task_size,
            0.0,
            1.0,
        )
        completed_return = jnp.where(terminal, updated_return, 0.0)
        completed_quant_left = jnp.where(terminal, terminal_quant_left, 0.0)
        completed_fill_ratio = jnp.where(terminal, fill_ratio, 0.0)
        next_running_return = jnp.where(terminal, 0.0, updated_return)
        return next_running_return, (
            terminal.astype(jnp.int32),
            completed_return,
            completed_quant_left,
            completed_fill_ratio,
        )

    next_running_return, completed = jax.lax.scan(
        _step,
        running_episode_return,
        (rewards, terminals, quant_left, task_size),
    )
    terminal_count, completed_return, completed_quant_left, completed_fill_ratio = completed
    episode_count = jnp.sum(terminal_count)
    safe_count = jnp.maximum(episode_count, 1).astype(jnp.float32)
    metrics = ExecutionEpisodeMetrics(
        episode_count=episode_count,
        episode_return_mean=jnp.sum(completed_return) / safe_count,
        terminal_quant_left_mean=jnp.sum(completed_quant_left) / safe_count,
        terminal_fill_ratio_mean=jnp.sum(completed_fill_ratio) / safe_count,
    )
    return next_running_return, metrics
