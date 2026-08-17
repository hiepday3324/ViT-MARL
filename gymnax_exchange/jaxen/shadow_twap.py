"""Read-only shadow TWAP execution against a raw LOB snapshot."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from gymnax_exchange.jaxob import jaxob_constants as cst


class ShadowTWAPStep(NamedTuple):
    scheduled_child_quantity: jax.Array
    child_quantity: jax.Array
    filled_quantity: jax.Array
    execution_cost: jax.Array
    unexecuted_quantity: jax.Array
    cumulative_filled_quantity: jax.Array
    remaining_task_quantity: jax.Array
    cumulative_execution_cost: jax.Array


class TerminalExecutionBenchmark(NamedTuple):
    """Per-episode execution benchmarks evaluated on a terminal LOB snapshot."""

    full_completion: jax.Array
    realized_is_bps: jax.Array
    realized_is_valid: jax.Array
    forced_liquidation_is_bps: jax.Array
    forced_liquidation_is_valid: jax.Array
    twap_forced_liquidation_is_bps: jax.Array
    twap_forced_liquidation_is_valid: jax.Array
    twap_advantage_bps: jax.Array
    twap_comparison_valid: jax.Array
    twap_win: jax.Array


def empty_terminal_execution_benchmark() -> TerminalExecutionBenchmark:
    return TerminalExecutionBenchmark(
        full_completion=jnp.asarray(False),
        realized_is_bps=jnp.asarray(0.0, dtype=jnp.float32),
        realized_is_valid=jnp.asarray(False),
        forced_liquidation_is_bps=jnp.asarray(0.0, dtype=jnp.float32),
        forced_liquidation_is_valid=jnp.asarray(False),
        twap_forced_liquidation_is_bps=jnp.asarray(0.0, dtype=jnp.float32),
        twap_forced_liquidation_is_valid=jnp.asarray(False),
        twap_advantage_bps=jnp.asarray(0.0, dtype=jnp.float32),
        twap_comparison_valid=jnp.asarray(False),
        twap_win=jnp.asarray(0.0, dtype=jnp.float32),
    )


def shadow_market_order_sweep(
    side_orders: jax.Array,
    child_quantity: jax.Array,
    *,
    sweep_ascending: jax.Array,
    tick_size: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return actual fill and tick-price execution value without mutating the book."""
    side_orders = jnp.asarray(side_orders)
    if side_orders.ndim != 2 or side_orders.shape[-1] <= cst.OrderSideFeat.Q.value:
        raise ValueError(
            "side_orders must have shape (orders, features) with price and quantity."
        )

    raw_prices = side_orders[:, cst.OrderSideFeat.P.value]
    raw_quantities = side_orders[:, cst.OrderSideFeat.Q.value]
    finite = jnp.isfinite(raw_prices) & jnp.isfinite(raw_quantities)
    prices = jnp.where(finite, raw_prices, cst.EMPTY_SLOT).astype(jnp.int32)
    quantities = jnp.where(finite, raw_quantities, 0).astype(jnp.float32)
    valid = (prices > 0) & (quantities > 0)

    invalid_priority = jnp.asarray(jnp.iinfo(jnp.int32).max, dtype=jnp.int32)
    directional_price = jnp.where(sweep_ascending, prices, -prices)
    priority = jnp.where(valid, directional_price, invalid_priority)
    order = jnp.argsort(priority)

    sorted_prices = prices[order]
    sorted_quantities = jnp.where(valid[order], quantities[order], 0.0)
    child_quantity = jnp.maximum(jnp.asarray(child_quantity, dtype=jnp.float32), 0.0)
    quantity_before = jnp.cumsum(sorted_quantities) - sorted_quantities
    fills = jnp.minimum(
        sorted_quantities,
        jnp.maximum(child_quantity - quantity_before, 0.0),
    )

    safe_tick_size = jnp.maximum(jnp.asarray(tick_size, dtype=jnp.int32), 1)
    prices_in_ticks = jnp.floor_divide(sorted_prices, safe_tick_size).astype(jnp.float32)
    filled_quantity = jnp.sum(fills, dtype=jnp.float32)
    execution_cost = jnp.sum(fills * prices_in_ticks, dtype=jnp.float32)
    return filled_quantity, execution_cost


def simulate_shadow_twap_interval(
    ask_raw_orders: jax.Array,
    bid_raw_orders: jax.Array,
    scheduled_child_quantity: jax.Array,
    shadow_remaining_task_quantity: jax.Array,
    shadow_cumulative_filled_quantity: jax.Array,
    shadow_cumulative_execution_cost: jax.Array,
    is_sell_task: jax.Array,
    tick_size: jax.Array,
) -> ShadowTWAPStep:
    """Execute one fixed TWAP child order against a read-only LOB snapshot."""
    scheduled_child_quantity = jnp.maximum(
        jnp.asarray(scheduled_child_quantity, dtype=jnp.float32),
        0.0,
    )
    shadow_remaining_task_quantity = jnp.maximum(
        jnp.asarray(shadow_remaining_task_quantity, dtype=jnp.float32),
        0.0,
    )
    child_quantity = jnp.minimum(
        scheduled_child_quantity,
        shadow_remaining_task_quantity,
    )

    side_orders = jax.lax.cond(
        jnp.asarray(is_sell_task, dtype=jnp.bool_),
        lambda: bid_raw_orders,
        lambda: ask_raw_orders,
    )
    filled_quantity, execution_cost = shadow_market_order_sweep(
        side_orders,
        child_quantity,
        sweep_ascending=jnp.logical_not(jnp.asarray(is_sell_task, dtype=jnp.bool_)),
        tick_size=tick_size,
    )
    unexecuted_quantity = jnp.maximum(child_quantity - filled_quantity, 0.0)
    cumulative_filled_quantity = (
        jnp.asarray(shadow_cumulative_filled_quantity, dtype=jnp.float32)
        + filled_quantity
    )
    remaining_task_quantity = jnp.maximum(
        shadow_remaining_task_quantity - filled_quantity,
        0.0,
    )
    cumulative_execution_cost = (
        jnp.asarray(shadow_cumulative_execution_cost, dtype=jnp.float32)
        + execution_cost
    )
    return ShadowTWAPStep(
        scheduled_child_quantity=scheduled_child_quantity,
        child_quantity=child_quantity,
        filled_quantity=filled_quantity,
        execution_cost=execution_cost,
        unexecuted_quantity=unexecuted_quantity,
        cumulative_filled_quantity=cumulative_filled_quantity,
        remaining_task_quantity=remaining_task_quantity,
        cumulative_execution_cost=cumulative_execution_cost,
    )


def signed_implementation_shortfall_bps(
    execution_cost: jax.Array,
    executed_quantity: jax.Array,
    arrival_price: jax.Array,
    is_sell_task: jax.Array,
    tick_size: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return signed IS in bps and whether the realized price is well-defined.

    ``execution_cost`` uses tick-price times quantity, while ``arrival_price``
    uses raw-price units. Positive IS is worse for both buy and sell tasks.
    """
    execution_cost = jnp.asarray(execution_cost, dtype=jnp.float32)
    executed_quantity = jnp.asarray(executed_quantity, dtype=jnp.float32)
    arrival_price = jnp.asarray(arrival_price, dtype=jnp.float32)
    tick_size = jnp.asarray(tick_size, dtype=jnp.float32)
    safe_tick_size = jnp.where(tick_size > 0.0, tick_size, 1.0)
    arrival_price_ticks = arrival_price / safe_tick_size

    valid = (
        jnp.isfinite(execution_cost)
        & jnp.isfinite(executed_quantity)
        & jnp.isfinite(arrival_price_ticks)
        & jnp.isfinite(tick_size)
        & (execution_cost > 0.0)
        & (executed_quantity > 0.0)
        & (arrival_price_ticks > 0.0)
        & (tick_size > 0.0)
    )
    safe_quantity = jnp.where(executed_quantity > 0.0, executed_quantity, 1.0)
    safe_arrival = jnp.where(arrival_price_ticks > 0.0, arrival_price_ticks, 1.0)
    realized_price = execution_cost / safe_quantity
    direction = jnp.where(
        jnp.asarray(is_sell_task, dtype=jnp.bool_),
        -1.0,
        1.0,
    )
    is_bps = direction * (realized_price - safe_arrival) / safe_arrival * 10_000.0
    is_bps = jnp.where(valid & jnp.isfinite(is_bps), is_bps, 0.0)
    return is_bps.astype(jnp.float32), valid


def compute_terminal_execution_benchmark(
    ask_raw_orders: jax.Array,
    bid_raw_orders: jax.Array,
    *,
    task_quantity: jax.Array,
    terminal_quant_left: jax.Array,
    rl_cumulative_filled_quantity: jax.Array,
    rl_cumulative_execution_cost: jax.Array,
    shadow_cumulative_filled_quantity: jax.Array,
    shadow_remaining_task_quantity: jax.Array,
    shadow_cumulative_execution_cost: jax.Array,
    arrival_price: jax.Array,
    is_sell_task: jax.Array,
    tick_size: jax.Array,
    quantity_tolerance: float = 1e-4,
) -> TerminalExecutionBenchmark:
    """Evaluate RL and Shadow TWAP execution without mutating the terminal LOB."""
    task_quantity = jnp.asarray(task_quantity, dtype=jnp.float32)
    terminal_quant_left = jnp.asarray(terminal_quant_left, dtype=jnp.float32)
    rl_filled = jnp.asarray(rl_cumulative_filled_quantity, dtype=jnp.float32)
    rl_cost = jnp.asarray(rl_cumulative_execution_cost, dtype=jnp.float32)
    shadow_filled = jnp.asarray(
        shadow_cumulative_filled_quantity,
        dtype=jnp.float32,
    )
    shadow_remaining = jnp.asarray(
        shadow_remaining_task_quantity,
        dtype=jnp.float32,
    )
    shadow_cost = jnp.asarray(
        shadow_cumulative_execution_cost,
        dtype=jnp.float32,
    )
    tolerance = jnp.asarray(quantity_tolerance, dtype=jnp.float32)
    is_sell_task = jnp.asarray(is_sell_task, dtype=jnp.bool_)

    task_valid = jnp.isfinite(task_quantity) & (task_quantity > 0.0)
    rl_finite = (
        task_valid
        & jnp.isfinite(terminal_quant_left)
        & jnp.isfinite(rl_filled)
        & jnp.isfinite(rl_cost)
    )
    twap_finite = (
        task_valid
        & jnp.isfinite(shadow_filled)
        & jnp.isfinite(shadow_remaining)
        & jnp.isfinite(shadow_cost)
    )
    full_completion = rl_finite & (jnp.abs(terminal_quant_left) <= tolerance)

    realized_is_bps, realized_is_valid = signed_implementation_shortfall_bps(
        rl_cost,
        rl_filled,
        arrival_price,
        is_sell_task,
        tick_size,
    )
    realized_is_valid = realized_is_valid & task_valid
    realized_is_bps = jnp.where(realized_is_valid, realized_is_bps, 0.0)

    terminal_side_orders = jax.lax.cond(
        is_sell_task,
        lambda: bid_raw_orders,
        lambda: ask_raw_orders,
    )
    sweep_ascending = jnp.logical_not(is_sell_task)

    rl_residual = jnp.clip(terminal_quant_left, 0.0, jnp.maximum(task_quantity, 0.0))
    rl_liquidation_fill, rl_liquidation_cost = shadow_market_order_sweep(
        terminal_side_orders,
        rl_residual,
        sweep_ascending=sweep_ascending,
        tick_size=tick_size,
    )
    rl_forced_quantity = rl_filled + rl_liquidation_fill
    rl_forced_cost = rl_cost + rl_liquidation_cost
    rl_quantity_consistent = (
        jnp.abs(task_quantity - rl_filled - terminal_quant_left) <= tolerance
    )
    rl_fully_priced = jnp.abs(rl_forced_quantity - task_quantity) <= tolerance
    rl_forced_is_bps, rl_forced_price_valid = signed_implementation_shortfall_bps(
        rl_forced_cost,
        task_quantity,
        arrival_price,
        is_sell_task,
        tick_size,
    )
    rl_forced_valid = (
        rl_finite
        & rl_quantity_consistent
        & rl_fully_priced
        & rl_forced_price_valid
        & (terminal_quant_left >= -tolerance)
    )
    rl_forced_is_bps = jnp.where(rl_forced_valid, rl_forced_is_bps, 0.0)

    twap_residual = jnp.clip(
        shadow_remaining,
        0.0,
        jnp.maximum(task_quantity, 0.0),
    )
    twap_liquidation_fill, twap_liquidation_cost = shadow_market_order_sweep(
        terminal_side_orders,
        twap_residual,
        sweep_ascending=sweep_ascending,
        tick_size=tick_size,
    )
    twap_forced_quantity = shadow_filled + twap_liquidation_fill
    twap_forced_cost = shadow_cost + twap_liquidation_cost
    twap_quantity_consistent = (
        jnp.abs(task_quantity - shadow_filled - shadow_remaining) <= tolerance
    )
    twap_fully_priced = jnp.abs(twap_forced_quantity - task_quantity) <= tolerance
    (
        twap_forced_is_bps,
        twap_forced_price_valid,
    ) = signed_implementation_shortfall_bps(
        twap_forced_cost,
        task_quantity,
        arrival_price,
        is_sell_task,
        tick_size,
    )
    twap_forced_valid = (
        twap_finite
        & twap_quantity_consistent
        & twap_fully_priced
        & twap_forced_price_valid
        & (shadow_remaining >= -tolerance)
    )
    twap_forced_is_bps = jnp.where(
        twap_forced_valid,
        twap_forced_is_bps,
        0.0,
    )

    comparison_valid = rl_forced_valid & twap_forced_valid
    twap_advantage_bps = jnp.where(
        comparison_valid,
        twap_forced_is_bps - rl_forced_is_bps,
        0.0,
    )
    twap_win = jnp.where(
        comparison_valid & (rl_forced_is_bps < twap_forced_is_bps),
        1.0,
        0.0,
    )
    return TerminalExecutionBenchmark(
        full_completion=full_completion,
        realized_is_bps=realized_is_bps,
        realized_is_valid=realized_is_valid,
        forced_liquidation_is_bps=rl_forced_is_bps,
        forced_liquidation_is_valid=rl_forced_valid,
        twap_forced_liquidation_is_bps=twap_forced_is_bps,
        twap_forced_liquidation_is_valid=twap_forced_valid,
        twap_advantage_bps=twap_advantage_bps,
        twap_comparison_valid=comparison_valid,
        twap_win=twap_win.astype(jnp.float32),
    )
