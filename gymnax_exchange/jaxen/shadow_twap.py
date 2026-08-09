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
