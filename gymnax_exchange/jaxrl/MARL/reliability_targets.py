"""Side-aware reliability targets and losses for execution-policy training."""

from __future__ import annotations

from typing import Any, Mapping

import jax.numpy as jnp
import optax

from gymnax_exchange.jaxob import jaxob_constants as cst


def _align_reliability_tensor(values, labels, *, name):
    if values.shape == labels.shape + (1,):
        return jnp.squeeze(values, axis=-1)
    if values.shape != labels.shape:
        raise ValueError(
            f"{name} must match labels exactly or have one trailing singleton "
            f"dimension; got {name}={values.shape}, labels={labels.shape}."
        )
    return values


def masked_reliability_loss(
    reliability_scores,
    labels,
    mask,
    loss_type="bce",
    eps=1e-8,
    *,
    reliability_logits=None,
):
    """Compute a masked reliability loss for binary or soft survival targets.

    BCE uses logits when supplied so saturated sigmoid probabilities do not lose
    their gradient through probability clipping. MSE and legacy callers without
    logits continue to operate on probabilities.
    """
    reliability_scores = _align_reliability_tensor(
        reliability_scores,
        labels,
        name="reliability_scores",
    )
    if mask.shape != labels.shape:
        raise ValueError(
            f"mask must match labels shape; got mask={mask.shape}, labels={labels.shape}."
        )

    r = jnp.clip(reliability_scores, eps, 1.0 - eps)
    y = labels.astype(jnp.float32)
    m = mask.astype(jnp.float32)

    if loss_type == "bce":
        if reliability_logits is not None:
            logits = _align_reliability_tensor(
                reliability_logits,
                labels,
                name="reliability_logits",
            )
            loss = optax.sigmoid_binary_cross_entropy(logits, y)
        else:
            loss = -(y * jnp.log(r) + (1.0 - y) * jnp.log(1.0 - r))
    elif loss_type == "mse":
        loss = jnp.square(r - y)
    else:
        raise ValueError(f"Unknown reliability_loss_type: {loss_type}")

    return jnp.sum(loss * m) / jnp.maximum(jnp.sum(m), eps)


def resolve_rollout_is_sell_task(
    agent_info: Mapping[str, Any] | None,
    *,
    task: str | None,
    num_steps: int,
    batch_size: int,
):
    """Resolve a ``(time, batch)`` task-side tensor from rollout info or config.

    A random execution task must carry the sampled task direction in rollout
    info. Falling back to a guessed side would corrupt task-side diagnostics.
    """
    if agent_info is not None and "is_sell_task" in agent_info:
        is_sell_task = jnp.asarray(agent_info["is_sell_task"], dtype=jnp.float32)
    elif task == "buy":
        is_sell_task = jnp.zeros((num_steps, batch_size), dtype=jnp.float32)
    elif task == "sell":
        is_sell_task = jnp.ones((num_steps, batch_size), dtype=jnp.float32)
    elif task == "random":
        raise ValueError(
            "Execution task='random' requires rollout info['agent']['is_sell_task'] "
            "to build task-side reliability diagnostics."
        )
    else:
        raise ValueError(
            "Cannot resolve is_sell_task: rollout info is missing the field and "
            f"Execution task={task!r} is not a supported fixed task."
        )

    if is_sell_task.ndim == 3:
        if is_sell_task.shape[-1] != 1:
            raise ValueError(
                "is_sell_task must have shape (time, batch) or (time, batch, 1); "
                f"got {is_sell_task.shape}."
            )
        is_sell_task = jnp.squeeze(is_sell_task, axis=-1)

    if is_sell_task.ndim == 0:
        is_sell_task = jnp.full((num_steps, batch_size), is_sell_task)
    elif is_sell_task.ndim == 1:
        if is_sell_task.shape[0] >= num_steps and batch_size == 1:
            is_sell_task = is_sell_task[:num_steps, None]
        elif is_sell_task.shape[0] == batch_size:
            is_sell_task = jnp.broadcast_to(is_sell_task[None, :], (num_steps, batch_size))
        elif is_sell_task.shape[0] == num_steps and batch_size == 1:
            is_sell_task = is_sell_task[:, None]
        else:
            raise ValueError(
                "Cannot broadcast is_sell_task with shape "
                f"{is_sell_task.shape} to ({num_steps}, {batch_size})."
            )
    elif is_sell_task.ndim == 2:
        if is_sell_task.shape[0] >= num_steps and is_sell_task.shape[1] == batch_size:
            is_sell_task = is_sell_task[:num_steps]
        elif is_sell_task.shape == (num_steps, batch_size):
            pass
        elif is_sell_task.shape == (1, batch_size):
            is_sell_task = jnp.broadcast_to(is_sell_task, (num_steps, batch_size))
        elif is_sell_task.shape == (batch_size, 1):
            is_sell_task = jnp.broadcast_to(is_sell_task[:, 0][None, :], (num_steps, batch_size))
        else:
            raise ValueError(
                "Cannot broadcast is_sell_task with shape "
                f"{is_sell_task.shape} to ({num_steps}, {batch_size})."
            )
    else:
        raise ValueError(
            "is_sell_task must have at most three dimensions; "
            f"got {is_sell_task.shape}."
        )

    return (is_sell_task > 0).astype(jnp.float32)


def _broadcast_raw_orders(raw_orders, *, required_steps, batch_size, name):
    raw_orders = jnp.asarray(raw_orders, dtype=jnp.float32)
    if raw_orders.ndim != 4:
        raise ValueError(
            f"{name} must have shape (time, batch, nOrders, 6), got {raw_orders.shape}."
        )
    if raw_orders.shape[0] < required_steps:
        raise ValueError(
            f"{name} must include current and future horizon frames: "
            f"need at least {required_steps}, got {raw_orders.shape[0]}."
        )
    if raw_orders.shape[1] != batch_size:
        if batch_size % raw_orders.shape[1] != 0:
            raise ValueError(
                f"Cannot broadcast {name} to actor vision observations: "
                f"{name} batch={raw_orders.shape[1]}, vision batch={batch_size}."
            )
        raw_orders = jnp.repeat(raw_orders, batch_size // raw_orders.shape[1], axis=1)
    return raw_orders


def _fullbook_volume_at_key(raw_orders, price_key, tick_size):
    raw_orders = jnp.asarray(raw_orders, dtype=jnp.float32)
    price_key = jnp.asarray(price_key, dtype=jnp.float32)
    tick_size = jnp.asarray(tick_size, dtype=jnp.float32)
    price = price_key * tick_size
    matches = raw_orders[..., None, :, 0] == price[..., :, None]
    qty = raw_orders[..., None, :, 1]
    return jnp.sum(jnp.where(matches, qty, 0.0), axis=-1).astype(jnp.float32)


def _broadcast_step_trades(new_trades, *, required_steps, batch_size):
    new_trades = jnp.asarray(new_trades, dtype=jnp.float32)
    if new_trades.ndim != 4 or new_trades.shape[-1] != cst.TRADE_FEAT:
        raise ValueError(
            "new_trades must have shape (time, batch, nTradesLogged, 8), "
            f"got {new_trades.shape}."
        )
    if new_trades.shape[0] < required_steps:
        raise ValueError(
            "new_trades must include the current rollout and future padding: "
            f"need at least {required_steps}, got {new_trades.shape[0]}."
        )
    if new_trades.shape[1] != batch_size:
        if batch_size % new_trades.shape[1] != 0:
            raise ValueError(
                "Cannot broadcast new_trades to actor observations: "
                f"trades batch={new_trades.shape[1]}, actor batch={batch_size}."
            )
        new_trades = jnp.repeat(new_trades, batch_size // new_trades.shape[1], axis=1)
    return new_trades


def _broadcast_trade_valid_mask(
    trade_valid_mask,
    *,
    required_steps,
    batch_size,
    n_trades_logged,
):
    trade_valid_mask = jnp.asarray(trade_valid_mask, dtype=jnp.bool_)
    if trade_valid_mask.ndim != 3:
        raise ValueError(
            "trade_valid_mask must have shape (time, batch, nTradesLogged), "
            f"got {trade_valid_mask.shape}."
        )
    if trade_valid_mask.shape[0] < required_steps:
        raise ValueError(
            "trade_valid_mask must include the current rollout and future padding: "
            f"need at least {required_steps}, got {trade_valid_mask.shape[0]}."
        )
    if trade_valid_mask.shape[2] != n_trades_logged:
        raise ValueError(
            "trade_valid_mask and new_trades disagree on trade capacity: "
            f"mask={trade_valid_mask.shape[2]}, trades={n_trades_logged}."
        )
    if trade_valid_mask.shape[1] != batch_size:
        if batch_size % trade_valid_mask.shape[1] != 0:
            raise ValueError(
                "Cannot broadcast trade_valid_mask to actor observations: "
                f"mask batch={trade_valid_mask.shape[1]}, actor batch={batch_size}."
            )
        trade_valid_mask = jnp.repeat(
            trade_valid_mask,
            batch_size // trade_valid_mask.shape[1],
            axis=1,
        )
    return trade_valid_mask


def _executed_passive_volume_at_key(
    new_trades,
    trade_valid_mask,
    price_key,
    tick_size,
    *,
    side,
):
    """Return passive execution volume at token prices for one transition."""
    trade_price = new_trades[..., cst.TradesFeat.P.value]
    trade_quantity = new_trades[..., cst.TradesFeat.Q.value]
    absolute_price = price_key * jnp.asarray(tick_size, dtype=jnp.float32)
    price_matches = trade_price[..., None, :] == absolute_price[..., :, None]
    if side == "ask":
        side_matches = trade_quantity < 0
    elif side == "bid":
        side_matches = trade_quantity > 0
    else:
        raise ValueError(f"Unknown passive trade side: {side}")
    matches = price_matches & side_matches[..., None, :] & trade_valid_mask[..., None, :]
    return jnp.sum(
        jnp.where(matches, jnp.abs(trade_quantity[..., None, :]), 0.0),
        axis=-1,
    ).astype(jnp.float32)


def _normalize_episode_done(episode_done, *, required_steps, batch_size):
    """Normalize post-step actor done flags to ``(time, batch)``."""
    if episode_done is None:
        return jnp.zeros((required_steps, batch_size), dtype=jnp.bool_)

    episode_done = jnp.asarray(episode_done, dtype=jnp.bool_)
    if episode_done.ndim == 3:
        if episode_done.shape[-1] != 1:
            raise ValueError(
                "episode_done with three dimensions must have trailing size one; "
                f"got {episode_done.shape}."
            )
        episode_done = jnp.squeeze(episode_done, axis=-1)

    if episode_done.ndim == 1:
        if episode_done.shape[0] < required_steps:
            raise ValueError(
                "episode_done must include the full current plus future horizon: "
                f"need at least {required_steps}, got {episode_done.shape[0]}."
            )
        return jnp.broadcast_to(
            episode_done[:required_steps, None],
            (required_steps, batch_size),
        )

    if episode_done.ndim == 2:
        if episode_done.shape[0] < required_steps:
            raise ValueError(
                "episode_done must include the full current plus future horizon: "
                f"need at least {required_steps}, got {episode_done.shape[0]}."
            )
        episode_done = episode_done[:required_steps]
        if episode_done.shape[1] == batch_size:
            return episode_done
        if episode_done.shape[1] == 1:
            return jnp.broadcast_to(episode_done, (required_steps, batch_size))
        raise ValueError(
            "episode_done batch dimension must match vision observations or be one; "
            f"got {episode_done.shape[1]}, expected {batch_size}."
        )

    raise ValueError(
        "episode_done must be shaped (time,), (time, batch), or "
        f"(time, batch, 1); got {episode_done.shape}."
    )


def _build_valid_horizon_mask(episode_done, *, num_steps, survival_delta_steps):
    """Return samples with no invalid transition in ``[t, t + Delta - 1]``."""
    invalid_windows = [
        episode_done[tau:tau + num_steps]
        for tau in range(survival_delta_steps)
    ]
    return ~jnp.any(jnp.stack(invalid_windows, axis=0), axis=0)


def _safe_masked_mean(values, mask, eps):
    values = jnp.asarray(values, dtype=jnp.float32)
    mask = jnp.asarray(mask, dtype=jnp.float32)
    return jnp.sum(jnp.where(mask > 0, values, 0.0)) / jnp.maximum(jnp.sum(mask), eps)


def _safe_masked_std(values, mask, eps):
    mean = _safe_masked_mean(values, mask, eps)
    return jnp.sqrt(_safe_masked_mean(jnp.square(values - mean), mask, eps))


def _safe_masked_extreme(values, mask, *, mode):
    values = jnp.asarray(values, dtype=jnp.float32)
    valid = jnp.asarray(mask, dtype=jnp.bool_)
    has_values = jnp.any(valid)
    if mode == "min":
        extreme = jnp.min(jnp.where(valid, values, jnp.inf))
    elif mode == "max":
        extreme = jnp.max(jnp.where(valid, values, -jnp.inf))
    else:
        raise ValueError(f"Unknown extreme mode: {mode}")
    return jnp.where(has_values, extreme, 0.0)


def build_liquidity_survival_targets(
    vision_obs,
    mid_prices,
    *,
    tick_size,
    survival_delta_steps,
    survival_min_volume,
    survival_ratio,
    num_steps,
    episode_done=None,
    survival_availability_temperature=0.15,
    ask_raw_orders=None,
    bid_raw_orders=None,
    new_trades=None,
    trade_valid_mask=None,
    trade_buffer_saturated=None,
    return_diagnostics=False,
    eps=1e-8,
):
    """Build execution-aware side/price liquidity reliability targets.

    Inputs use the existing LOB contract ``(time, batch, levels, features,
    sides)``. Outputs are ``(time, batch, levels, sides)`` with side order
    ``[Ask, Bid]``. For each horizon ``tau``, the ratio is
    ``clip((cumulative passive execution + same-side resting volume) / Q0, 0, 1)``.
    The final label is the mean ratio over ``tau=1..Delta``. New orders that
    refill the same side and absolute price count as reliable liquidity; order
    identity is deliberately ignored.

    ``episode_done`` and ``trade_buffer_saturated`` are post-step transition
    flags. Any flagged transition in ``[t, t + Delta - 1]`` masks the complete
    target because those transitions connect ``S_t`` through ``S_{t+Delta}``.
    """
    if survival_delta_steps < 1:
        raise ValueError("survival_delta_steps must be at least one.")

    vision_obs = jnp.asarray(vision_obs, dtype=jnp.float32)
    mid_prices = jnp.asarray(mid_prices, dtype=jnp.float32)
    required_steps = num_steps + survival_delta_steps
    if vision_obs.shape[0] < required_steps:
        raise ValueError(
            "vision_obs must include current and future horizon frames: "
            f"need at least {required_steps}, got {vision_obs.shape[0]}."
        )

    if mid_prices.ndim == 1:
        mid_prices = mid_prices[:, None]
    if mid_prices.shape[0] < required_steps:
        raise ValueError(
            "mid_prices must include current and future horizon frames: "
            f"need at least {required_steps}, got {mid_prices.shape[0]}."
        )
    if mid_prices.shape[1] != vision_obs.shape[1]:
        if vision_obs.shape[1] % mid_prices.shape[1] != 0:
            raise ValueError(
                "Cannot broadcast world mid_prices to actor vision observations: "
                f"mid_prices batch={mid_prices.shape[1]}, vision batch={vision_obs.shape[1]}."
            )
        mid_prices = jnp.repeat(mid_prices, vision_obs.shape[1] // mid_prices.shape[1], axis=1)

    current_obs = vision_obs[:num_steps]
    current_mid = mid_prices[:num_steps, :, None]
    ask_gap = current_obs[..., 0, 0]
    bid_gap = current_obs[..., 0, 1]
    token_ask_volume = jnp.expm1(current_obs[..., 1, 0])
    token_bid_volume = jnp.expm1(current_obs[..., 1, 1])

    tick_size = jnp.asarray(tick_size, dtype=jnp.float32)
    ask_key = jnp.rint((current_mid + ask_gap * tick_size) / tick_size)
    bid_key = jnp.rint((current_mid - bid_gap * tick_size) / tick_size)
    if ask_raw_orders is None or bid_raw_orders is None:
        raise ValueError(
            "ask_raw_orders and bid_raw_orders are required to build "
            "fullbook absolute-price reliability targets."
        )
    ask_raw_orders = _broadcast_raw_orders(
        ask_raw_orders,
        required_steps=required_steps,
        batch_size=vision_obs.shape[1],
        name="ask_raw_orders",
    )
    bid_raw_orders = _broadcast_raw_orders(
        bid_raw_orders,
        required_steps=required_steps,
        batch_size=vision_obs.shape[1],
        name="bid_raw_orders",
    )
    if new_trades is None or trade_valid_mask is None or trade_buffer_saturated is None:
        raise ValueError(
            "new_trades, trade_valid_mask, and trade_buffer_saturated are required "
            "for execution-aware reliability targets."
        )
    new_trades = _broadcast_step_trades(
        new_trades,
        required_steps=required_steps,
        batch_size=vision_obs.shape[1],
    )
    trade_valid_mask = _broadcast_trade_valid_mask(
        trade_valid_mask,
        required_steps=required_steps,
        batch_size=vision_obs.shape[1],
        n_trades_logged=new_trades.shape[2],
    )
    trade_buffer_saturated = _normalize_episode_done(
        trade_buffer_saturated,
        required_steps=required_steps,
        batch_size=vision_obs.shape[1],
    )

    current_ask_orders = ask_raw_orders[:num_steps]
    current_bid_orders = bid_raw_orders[:num_steps]
    q0_ask = _fullbook_volume_at_key(current_ask_orders, ask_key, tick_size)
    q0_bid = _fullbook_volume_at_key(current_bid_orders, bid_key, tick_size)
    q0 = jnp.stack([q0_ask, q0_bid], axis=-1)

    future_ratios = []
    future_resting_volumes = []
    cumulative_execution_volumes = []
    unexplained_missing_volumes = []
    cumulative_ask_execution = jnp.zeros_like(q0_ask)
    cumulative_bid_execution = jnp.zeros_like(q0_bid)
    finite_future_data = jnp.ones((num_steps, vision_obs.shape[1]), dtype=jnp.bool_)
    for tau in range(1, survival_delta_steps + 1):
        transition_index = tau - 1
        cumulative_ask_execution = cumulative_ask_execution + _executed_passive_volume_at_key(
            new_trades[transition_index:transition_index + num_steps],
            trade_valid_mask[transition_index:transition_index + num_steps],
            ask_key,
            tick_size,
            side="ask",
        )
        cumulative_bid_execution = cumulative_bid_execution + _executed_passive_volume_at_key(
            new_trades[transition_index:transition_index + num_steps],
            trade_valid_mask[transition_index:transition_index + num_steps],
            bid_key,
            tick_size,
            side="bid",
        )
        q_tau_ask = _fullbook_volume_at_key(
            ask_raw_orders[tau:tau + num_steps],
            ask_key,
            tick_size,
        )
        q_tau_bid = _fullbook_volume_at_key(
            bid_raw_orders[tau:tau + num_steps],
            bid_key,
            tick_size,
        )
        ask_ratio = jnp.clip(
            (cumulative_ask_execution + q_tau_ask) / (q0_ask + eps),
            0.0,
            1.0,
        )
        bid_ratio = jnp.clip(
            (cumulative_bid_execution + q_tau_bid) / (q0_bid + eps),
            0.0,
            1.0,
        )
        future_ratios.append(jnp.stack([ask_ratio, bid_ratio], axis=-1))
        q_tau = jnp.stack([q_tau_ask, q_tau_bid], axis=-1)
        cumulative_execution = jnp.stack(
            [cumulative_ask_execution, cumulative_bid_execution],
            axis=-1,
        )
        future_resting_volumes.append(q_tau)
        cumulative_execution_volumes.append(cumulative_execution)
        unexplained_missing_volumes.append(
            jnp.maximum(q0 - cumulative_execution - q_tau, 0.0)
        )
        finite_future_data = finite_future_data & jnp.all(
            jnp.isfinite(ask_raw_orders[tau:tau + num_steps]),
            axis=(-1, -2),
        )
        finite_future_data = finite_future_data & jnp.all(
            jnp.isfinite(bid_raw_orders[tau:tau + num_steps]),
            axis=(-1, -2),
        )

    episode_done = _normalize_episode_done(
        episode_done,
        required_steps=required_steps,
        batch_size=vision_obs.shape[1],
    )
    valid_horizon = _build_valid_horizon_mask(
        episode_done,
        num_steps=num_steps,
        survival_delta_steps=survival_delta_steps,
    )
    valid_trade_horizon = _build_valid_horizon_mask(
        trade_buffer_saturated,
        num_steps=num_steps,
        survival_delta_steps=survival_delta_steps,
    )
    invalid_trade_data = ~jnp.all(jnp.isfinite(new_trades), axis=(-1, -2))
    finite_trade_horizon = _build_valid_horizon_mask(
        invalid_trade_data,
        num_steps=num_steps,
        survival_delta_steps=survival_delta_steps,
    )
    ratios = jnp.stack(future_ratios, axis=0)
    q_tau_values = jnp.stack(future_resting_volumes, axis=0)
    cumulative_execution_values = jnp.stack(cumulative_execution_volumes, axis=0)
    cancel_star_values = jnp.stack(unexplained_missing_volumes, axis=0)
    target = jnp.mean(ratios, axis=0)

    current_raw_finite = jnp.all(jnp.isfinite(current_ask_orders), axis=(-1, -2))
    current_raw_finite = current_raw_finite & jnp.all(
        jnp.isfinite(current_bid_orders),
        axis=(-1, -2),
    )
    token_finite = jnp.all(jnp.isfinite(current_obs), axis=3)
    key_finite = jnp.stack([jnp.isfinite(ask_key), jnp.isfinite(bid_key)], axis=-1)
    key_valid = key_finite & (jnp.stack([ask_key, bid_key], axis=-1) > 0)
    q0_valid = jnp.isfinite(q0) & (q0 > survival_min_volume)
    token_volume = jnp.stack([token_ask_volume, token_bid_volume], axis=-1)
    token_volume_valid = jnp.isfinite(token_volume) & (token_volume > 0)
    target_finite = jnp.isfinite(target)
    sample_valid = (
        valid_horizon
        & valid_trade_horizon
        & finite_trade_horizon
        & finite_future_data
        & current_raw_finite
        & jnp.all(jnp.isfinite(current_mid), axis=-1)
    )
    side_mask = (
        q0_valid
        & token_finite
        & token_volume_valid
        & key_valid
        & target_finite
        & sample_valid[:, :, None, None]
    )
    target = jnp.where(target_finite, target, 0.0).astype(jnp.float32)
    side_mask = side_mask.astype(jnp.float32)

    # These legacy tuning arguments remain accepted so existing CLI/config files
    # do not break, but the execution-aware target does not use them.
    del survival_ratio, survival_availability_temperature

    if not return_diagnostics:
        return target, side_mask

    valid_mask = side_mask > 0
    ask_mask = valid_mask[..., 0]
    bid_mask = valid_mask[..., 1]
    target_diag = {
        "valid_target_count": jnp.sum(side_mask),
        "valid_target_rate": jnp.mean(side_mask),
        "done_masked_rate": jnp.mean((~valid_horizon).astype(jnp.float32)),
        "trade_buffer_saturated_rate": jnp.mean(
            trade_buffer_saturated[:num_steps + survival_delta_steps - 1].astype(jnp.float32)
        ),
        "q0_mean": _safe_masked_mean(q0, valid_mask, eps),
        "q0_min": _safe_masked_extreme(q0, valid_mask, mode="min"),
        "q0_max": _safe_masked_extreme(q0, valid_mask, mode="max"),
        "q_tau_mean": _safe_masked_mean(
            q_tau_values,
            jnp.broadcast_to(valid_mask[None, ...], q_tau_values.shape),
            eps,
        ),
        "cumulative_executed_mean": _safe_masked_mean(
            cumulative_execution_values,
            jnp.broadcast_to(valid_mask[None, ...], cumulative_execution_values.shape),
            eps,
        ),
        "cancel_star_mean": _safe_masked_mean(
            cancel_star_values,
            jnp.broadcast_to(valid_mask[None, ...], cancel_star_values.shape),
            eps,
        ),
        "net_missing_liquidity_mean": _safe_masked_mean(
            cancel_star_values,
            jnp.broadcast_to(valid_mask[None, ...], cancel_star_values.shape),
            eps,
        ),
        "target_mean": _safe_masked_mean(target, valid_mask, eps),
        "target_std": _safe_masked_std(target, valid_mask, eps),
        "target_min": _safe_masked_extreme(target, valid_mask, mode="min"),
        "target_max": _safe_masked_extreme(target, valid_mask, mode="max"),
        "ask_valid_count": jnp.sum(ask_mask.astype(jnp.float32)),
        "bid_valid_count": jnp.sum(bid_mask.astype(jnp.float32)),
        "ask_target_mean": _safe_masked_mean(target[..., 0], ask_mask, eps),
        "ask_target_min": _safe_masked_extreme(target[..., 0], ask_mask, mode="min"),
        "ask_target_max": _safe_masked_extreme(target[..., 0], ask_mask, mode="max"),
        "bid_target_mean": _safe_masked_mean(target[..., 1], bid_mask, eps),
        "bid_target_min": _safe_masked_extreme(target[..., 1], bid_mask, mode="min"),
        "bid_target_max": _safe_masked_extreme(target[..., 1], bid_mask, mode="max"),
        "ask_key_t0_b0": ask_key[0, 0],
        "bid_key_t0_b0": bid_key[0, 0],
    }
    return target, side_mask, target_diag
