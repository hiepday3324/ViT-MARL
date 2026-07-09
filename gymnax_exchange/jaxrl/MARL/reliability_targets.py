"""Side-aware reliability targets and losses for execution-policy training."""

from __future__ import annotations

from typing import Any, Mapping

import jax.numpy as jnp


def masked_reliability_loss(reliability_scores, labels, mask, loss_type="bce", eps=1e-8):
    """Compute a masked reliability loss for binary or soft survival targets."""
    if reliability_scores.shape == labels.shape + (1,):
        reliability_scores = jnp.squeeze(reliability_scores, axis=-1)
    elif reliability_scores.shape != labels.shape:
        raise ValueError(
            "reliability_scores must match labels exactly or have one trailing "
            f"singleton dimension; got scores={reliability_scores.shape}, "
            f"labels={labels.shape}."
        )
    if mask.shape != labels.shape:
        raise ValueError(
            f"mask must match labels shape; got mask={mask.shape}, labels={labels.shape}."
        )

    r = jnp.clip(reliability_scores, eps, 1.0 - eps)
    y = labels.astype(jnp.float32)
    m = mask.astype(jnp.float32)

    if loss_type == "bce":
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
    info. Falling back to a guessed side would corrupt the side-aware target.
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
            "to build actionability-weighted reliability targets."
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


def _matched_future_volumes(current_key, future_key, future_volume):
    matches = future_key[..., None, :] == current_key[..., :, None]
    return jnp.max(
        jnp.where(matches, future_volume[..., None, :], 0.0),
        axis=-1,
    )


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
    """Return samples whose inclusive interval ``[t, t + Delta]`` has no done."""
    invalid_windows = [
        episode_done[tau:tau + num_steps]
        for tau in range(survival_delta_steps + 1)
    ]
    return ~jnp.any(jnp.stack(invalid_windows, axis=0), axis=0)


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
    survival_target_mode="actionability_weighted_min_horizon",
    is_sell_task=None,
    actionability_mode="passive_limit",
    actionability_eta=0.1,
    actionability_depth=3,
    actionability_far_level_weight=0.25,
    survival_availability_temperature=0.15,
    ask_raw_orders=None,
    bid_raw_orders=None,
    survival_target_book_source="vision_topk",
    survival_fullbook_match_mode="same_side",
    eps=1e-8,
):
    """Build side-aware liquidity targets from normalized LOB vision frames.

    Inputs use the existing LOB contract ``(time, batch, levels, features,
    sides)``. Outputs are ``(time, batch, levels, sides)`` with side order
    ``[Ask, Bid]``. ``episode_done`` is an optional post-step, actor-aligned
    done flag; any done in the inclusive interval ``[t, t + Delta]`` masks
    that target from the reliability loss without changing its label value.
    The actionability-weighted mode uses an availability-weighted mean
    volume-survival target, attenuating less executable sides and far levels.
    """
    valid_modes = {
        "final_step_binary",
        "min_horizon_soft",
        "actionability_weighted_min_horizon",
    }
    if survival_target_mode not in valid_modes:
        raise ValueError(
            f"Unknown survival_target_mode: {survival_target_mode}. "
            f"Expected one of {sorted(valid_modes)}."
        )
    valid_book_sources = {"vision_topk", "fullbook"}
    if survival_target_book_source not in valid_book_sources:
        raise ValueError(
            f"Unknown survival_target_book_source: {survival_target_book_source}. "
            f"Expected one of {sorted(valid_book_sources)}."
        )
    valid_fullbook_match_modes = {"same_side", "absolute_price_sum"}
    if (
        survival_target_book_source == "fullbook"
        and survival_fullbook_match_mode not in valid_fullbook_match_modes
    ):
        raise ValueError(
            f"Unknown survival_fullbook_match_mode: {survival_fullbook_match_mode}. "
            f"Expected one of {sorted(valid_fullbook_match_modes)}."
        )
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
    ask_volume = jnp.expm1(current_obs[..., 1, 0])
    bid_volume = jnp.expm1(current_obs[..., 1, 1])

    tick_size = jnp.asarray(tick_size, dtype=jnp.float32)
    ask_key = jnp.rint((current_mid + ask_gap * tick_size) / tick_size)
    bid_key = jnp.rint((current_mid - bid_gap * tick_size) / tick_size)
    ask_mask = ask_volume >= survival_min_volume
    bid_mask = bid_volume >= survival_min_volume
    if survival_target_book_source == "fullbook":
        if ask_raw_orders is None or bid_raw_orders is None:
            raise ValueError(
                "ask_raw_orders and bid_raw_orders are required when "
                "survival_target_book_source='fullbook'."
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

    future_ratios = []
    final_matched_ask_volume = None
    final_matched_bid_volume = None
    for tau in range(1, survival_delta_steps + 1):
        if survival_target_book_source == "fullbook":
            future_ask_at_ask_key = _fullbook_volume_at_key(
                ask_raw_orders[tau:tau + num_steps],
                ask_key,
                tick_size,
            )
            future_bid_at_bid_key = _fullbook_volume_at_key(
                bid_raw_orders[tau:tau + num_steps],
                bid_key,
                tick_size,
            )
            if survival_fullbook_match_mode == "absolute_price_sum":
                future_bid_at_ask_key = _fullbook_volume_at_key(
                    bid_raw_orders[tau:tau + num_steps],
                    ask_key,
                    tick_size,
                )
                future_ask_at_bid_key = _fullbook_volume_at_key(
                    ask_raw_orders[tau:tau + num_steps],
                    bid_key,
                    tick_size,
                )
                matched_ask_volume = future_ask_at_ask_key + future_bid_at_ask_key
                matched_bid_volume = future_ask_at_bid_key + future_bid_at_bid_key
            else:
                matched_ask_volume = future_ask_at_ask_key
                matched_bid_volume = future_bid_at_bid_key
        else:
            future_obs = vision_obs[tau:tau + num_steps]
            future_mid = mid_prices[tau:tau + num_steps, :, None]
            future_ask_gap = future_obs[..., 0, 0]
            future_bid_gap = future_obs[..., 0, 1]
            future_ask_volume = jnp.expm1(future_obs[..., 1, 0])
            future_bid_volume = jnp.expm1(future_obs[..., 1, 1])
            future_ask_key = jnp.rint((future_mid + future_ask_gap * tick_size) / tick_size)
            future_bid_key = jnp.rint((future_mid - future_bid_gap * tick_size) / tick_size)

            matched_ask_volume = _matched_future_volumes(ask_key, future_ask_key, future_ask_volume)
            matched_bid_volume = _matched_future_volumes(bid_key, future_bid_key, future_bid_volume)
        ask_ratio = jnp.clip(matched_ask_volume / (ask_volume + eps), 0.0, 1.0)
        bid_ratio = jnp.clip(matched_bid_volume / (bid_volume + eps), 0.0, 1.0)
        future_ratios.append(jnp.stack([ask_ratio, bid_ratio], axis=-1))
        final_matched_ask_volume = matched_ask_volume
        final_matched_bid_volume = matched_bid_volume

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
    side_mask = jnp.stack([ask_mask, bid_mask], axis=-1)
    side_mask = side_mask & valid_horizon[:, :, None, None]
    if survival_target_mode == "final_step_binary":
        ask_label = final_matched_ask_volume >= survival_ratio * ask_volume
        bid_label = final_matched_bid_volume >= survival_ratio * bid_volume
        target = jnp.stack([ask_label & ask_mask, bid_label & bid_mask], axis=-1)
        return target.astype(jnp.float32), side_mask.astype(jnp.float32)

    ratios = jnp.stack(future_ratios, axis=0)
    min_survival = jnp.min(ratios, axis=0)
    if survival_target_mode == "min_horizon_soft":
        return min_survival.astype(jnp.float32), side_mask.astype(jnp.float32)

    if is_sell_task is None:
        raise ValueError(
            "is_sell_task is required for survival_target_mode="
            "'actionability_weighted_min_horizon'."
        )
    if actionability_mode != "passive_limit":
        raise ValueError(
            f"Unknown actionability_mode: {actionability_mode}. Expected 'passive_limit'."
        )

    is_sell_task = jnp.asarray(is_sell_task, dtype=jnp.bool_)
    expected_task_shape = (num_steps, vision_obs.shape[1])
    if is_sell_task.shape != expected_task_shape:
        raise ValueError(
            "is_sell_task must have shape "
            f"{expected_task_shape}, got {is_sell_task.shape}."
        )

    del actionability_eta, actionability_depth, actionability_far_level_weight
    mean_survival = jnp.mean(ratios, axis=0)
    availability_temperature = jnp.maximum(
        jnp.asarray(survival_availability_temperature, dtype=jnp.float32),
        eps,
    )
    availability = jnp.mean(
        1.0 / (1.0 + jnp.exp(-((ratios - survival_ratio) / availability_temperature))),
        axis=0,
    )
    robust_survival = mean_survival * availability
    target = robust_survival
    return target.astype(jnp.float32), side_mask.astype(jnp.float32)
