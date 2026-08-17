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
    full_completion_rate: jax.Array
    realized_is_bps_mean: jax.Array
    forced_liquidation_is_bps_mean: jax.Array
    twap_forced_liquidation_is_bps_mean: jax.Array
    twap_advantage_bps_mean: jax.Array
    twap_win_rate: jax.Array


def empty_execution_episode_metrics() -> ExecutionEpisodeMetrics:
    return ExecutionEpisodeMetrics(
        episode_count=jnp.asarray(0, dtype=jnp.int32),
        episode_return_mean=jnp.asarray(0.0, dtype=jnp.float32),
        terminal_quant_left_mean=jnp.asarray(0.0, dtype=jnp.float32),
        terminal_fill_ratio_mean=jnp.asarray(0.0, dtype=jnp.float32),
        full_completion_rate=jnp.asarray(0.0, dtype=jnp.float32),
        realized_is_bps_mean=jnp.asarray(0.0, dtype=jnp.float32),
        forced_liquidation_is_bps_mean=jnp.asarray(0.0, dtype=jnp.float32),
        twap_forced_liquidation_is_bps_mean=jnp.asarray(0.0, dtype=jnp.float32),
        twap_advantage_bps_mean=jnp.asarray(0.0, dtype=jnp.float32),
        twap_win_rate=jnp.asarray(0.0, dtype=jnp.float32),
    )


def accumulate_execution_episode_metrics(
    running_episode_return: jax.Array,
    rewards: jax.Array,
    terminals: jax.Array,
    quant_left: jax.Array,
    task_size: jax.Array,
    *,
    full_completion: jax.Array,
    realized_is_bps: jax.Array,
    realized_is_valid: jax.Array,
    forced_liquidation_is_bps: jax.Array,
    forced_liquidation_is_valid: jax.Array,
    twap_forced_liquidation_is_bps: jax.Array,
    twap_forced_liquidation_is_valid: jax.Array,
    twap_advantage_bps: jax.Array,
    twap_comparison_valid: jax.Array,
    twap_win: jax.Array,
    axis_name: str | None = None,
) -> tuple[jax.Array, ExecutionEpisodeMetrics]:
    """Accumulate returns and summarize episodes ending in this rollout."""
    running_episode_return = jnp.asarray(running_episode_return, dtype=jnp.float32)
    rewards = jnp.asarray(rewards, dtype=jnp.float32)
    terminals = jnp.asarray(terminals, dtype=jnp.bool_)
    quant_left = jnp.asarray(quant_left, dtype=jnp.float32)
    task_size = jnp.asarray(task_size, dtype=jnp.float32)
    full_completion = jnp.asarray(full_completion, dtype=jnp.bool_)
    realized_is_bps = jnp.asarray(realized_is_bps, dtype=jnp.float32)
    realized_is_valid = jnp.asarray(realized_is_valid, dtype=jnp.bool_)
    forced_liquidation_is_bps = jnp.asarray(
        forced_liquidation_is_bps,
        dtype=jnp.float32,
    )
    forced_liquidation_is_valid = jnp.asarray(
        forced_liquidation_is_valid,
        dtype=jnp.bool_,
    )
    twap_forced_liquidation_is_bps = jnp.asarray(
        twap_forced_liquidation_is_bps,
        dtype=jnp.float32,
    )
    twap_forced_liquidation_is_valid = jnp.asarray(
        twap_forced_liquidation_is_valid,
        dtype=jnp.bool_,
    )
    twap_advantage_bps = jnp.asarray(twap_advantage_bps, dtype=jnp.float32)
    twap_comparison_valid = jnp.asarray(twap_comparison_valid, dtype=jnp.bool_)
    twap_win = jnp.asarray(twap_win, dtype=jnp.float32)

    if rewards.ndim != 2:
        raise ValueError(f"rewards must have shape (time, actors), got {rewards.shape}.")
    expected_shape = rewards.shape
    for name, value in (
        ("terminals", terminals),
        ("quant_left", quant_left),
        ("task_size", task_size),
        ("full_completion", full_completion),
        ("realized_is_bps", realized_is_bps),
        ("realized_is_valid", realized_is_valid),
        ("forced_liquidation_is_bps", forced_liquidation_is_bps),
        ("forced_liquidation_is_valid", forced_liquidation_is_valid),
        (
            "twap_forced_liquidation_is_bps",
            twap_forced_liquidation_is_bps,
        ),
        (
            "twap_forced_liquidation_is_valid",
            twap_forced_liquidation_is_valid,
        ),
        ("twap_advantage_bps", twap_advantage_bps),
        ("twap_comparison_valid", twap_comparison_valid),
        ("twap_win", twap_win),
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
    def _aggregate_sum(value):
        if axis_name is None:
            return value
        return jax.lax.psum(value, axis_name)

    episode_count = _aggregate_sum(jnp.sum(terminal_count))
    safe_count = jnp.maximum(episode_count, 1).astype(jnp.float32)
    completed_return_sum = _aggregate_sum(jnp.sum(completed_return))
    completed_quant_left_sum = _aggregate_sum(jnp.sum(completed_quant_left))
    completed_fill_ratio_sum = _aggregate_sum(jnp.sum(completed_fill_ratio))
    full_completion_sum = _aggregate_sum(
        jnp.sum((terminals & full_completion).astype(jnp.float32))
    )

    def _safe_terminal_mean(values, validity):
        validity = terminals & validity & jnp.isfinite(values)
        valid_count = _aggregate_sum(jnp.sum(validity.astype(jnp.int32)))
        value_sum = _aggregate_sum(jnp.sum(jnp.where(validity, values, 0.0)))
        denominator = jnp.maximum(valid_count, 1).astype(jnp.float32)
        return value_sum / denominator

    metrics = ExecutionEpisodeMetrics(
        episode_count=episode_count,
        episode_return_mean=completed_return_sum / safe_count,
        terminal_quant_left_mean=completed_quant_left_sum / safe_count,
        terminal_fill_ratio_mean=completed_fill_ratio_sum / safe_count,
        full_completion_rate=full_completion_sum / safe_count,
        realized_is_bps_mean=_safe_terminal_mean(
            realized_is_bps,
            realized_is_valid,
        ),
        forced_liquidation_is_bps_mean=_safe_terminal_mean(
            forced_liquidation_is_bps,
            forced_liquidation_is_valid,
        ),
        twap_forced_liquidation_is_bps_mean=_safe_terminal_mean(
            twap_forced_liquidation_is_bps,
            twap_forced_liquidation_is_valid,
        ),
        twap_advantage_bps_mean=_safe_terminal_mean(
            twap_advantage_bps,
            twap_comparison_valid,
        ),
        twap_win_rate=_safe_terminal_mean(
            twap_win,
            twap_comparison_valid,
        ),
    )
    return next_running_return, metrics
