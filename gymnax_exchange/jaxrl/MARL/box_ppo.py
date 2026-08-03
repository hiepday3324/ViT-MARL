"""Numerically stable Box-policy sampling, replay, and PPO safety helpers."""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple

import distrax
import jax
import jax.numpy as jnp

from gymnax_exchange.jaxrl.MARL.gradient_diagnostics import (
    flatten_tree_with_paths,
    tree_l2_norm,
)


_LOG_TWO = jnp.log(jnp.asarray(2.0, dtype=jnp.float32))
_LOG_TWO_PI = jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=jnp.float32))

FIRST_NONFINITE_STAGE = {
    "none": 0,
    "total_loss": 1,
    "new_log_prob": 2,
    "logratio": 3,
    "ratio": 4,
    "gradients": 5,
    "candidate_params": 6,
    "candidate_optimizer_state": 7,
}
FIRST_NONFINITE_STAGE_NAME = {
    value: key for key, value in FIRST_NONFINITE_STAGE.items()
}


class PolicyActionSample(NamedTuple):
    action: jax.Array
    pre_tanh_action: jax.Array
    log_prob: jax.Array


class GuardedPPOUpdate(NamedTuple):
    train_state: Any
    accepted: jax.Array
    rejected_nonfinite: jax.Array
    first_nonfinite_stage: jax.Array
    gradients_finite: jax.Array
    candidate_params_finite: jax.Array
    candidate_optimizer_state_finite: jax.Array


def _box_shift_and_scale(action_low, action_high, dtype):
    action_low = jnp.asarray(action_low, dtype=dtype)
    action_high = jnp.asarray(action_high, dtype=dtype)
    return (action_high + action_low) / 2.0, (action_high - action_low) / 2.0


def box_action_from_pre_tanh(pre_tanh_action, action_low, action_high):
    """Apply the actor's Tanh then affine transform without an inverse path."""
    pre_tanh_action = jnp.asarray(pre_tanh_action)
    shift, scale = _box_shift_and_scale(
        action_low,
        action_high,
        pre_tanh_action.dtype,
    )
    return shift + scale * jnp.tanh(pre_tanh_action)


def _stable_tanh_log_abs_det_jacobian(pre_tanh_action):
    # log(1 - tanh(z)^2) = 2 * (log(2) - z - softplus(-2z)).
    z = jnp.asarray(pre_tanh_action)
    log_two = jnp.asarray(_LOG_TWO, dtype=z.dtype)
    return 2.0 * (log_two - z - jax.nn.softplus(-2.0 * z))


def box_log_prob_from_pre_tanh(
    pre_tanh_action,
    loc,
    log_std,
    action_low,
    action_high,
):
    """Evaluate transformed Box log-prob directly at a stored base sample.

    This is the same change-of-variables density as the Distrax transformed
    distribution, but it never reconstructs ``z`` with ``atanh(action)``.
    """
    z = jnp.asarray(pre_tanh_action)
    loc = jnp.asarray(loc, dtype=z.dtype)
    log_std = jnp.asarray(log_std, dtype=z.dtype)
    _, affine_scale = _box_shift_and_scale(action_low, action_high, z.dtype)

    inv_std = jnp.exp(-log_std)
    standardized = (z - loc) * inv_std
    log_two_pi = jnp.asarray(_LOG_TWO_PI, dtype=z.dtype)
    base_log_prob = -0.5 * (jnp.square(standardized) + log_two_pi) - log_std
    transform_log_det = (
        jnp.log(jnp.abs(affine_scale))
        + _stable_tanh_log_abs_det_jacobian(z)
    )
    return jnp.sum(base_log_prob - transform_log_det, axis=-1)


def sample_policy_action(
    pi,
    aux_info: Mapping[str, Any],
    seed,
    *,
    action_low=None,
    action_high=None,
) -> PolicyActionSample:
    """Sample a policy while preserving legacy Discrete semantics."""
    if "policy_loc" not in aux_info:
        action = pi.sample(seed=seed)
        return PolicyActionSample(
            action=action,
            pre_tanh_action=jnp.zeros_like(action),
            log_prob=pi.log_prob(action),
        )

    if action_low is None or action_high is None:
        raise ValueError("Box policy sampling requires action_low and action_high.")
    loc = jnp.asarray(aux_info["policy_loc"])
    log_std = jnp.asarray(aux_info["policy_log_std"], dtype=loc.dtype)
    base_distribution = distrax.Independent(
        distrax.Normal(loc, jnp.exp(log_std)),
        reinterpreted_batch_ndims=1,
    )
    pre_tanh_action = base_distribution.sample(seed=seed)
    action = box_action_from_pre_tanh(
        pre_tanh_action,
        action_low,
        action_high,
    )
    log_prob = box_log_prob_from_pre_tanh(
        pre_tanh_action,
        loc,
        log_std,
        action_low,
        action_high,
    )
    return PolicyActionSample(action, pre_tanh_action, log_prob)


def policy_log_prob_from_transition(
    pi,
    aux_info: Mapping[str, Any],
    action,
    pre_tanh_action,
    *,
    action_low=None,
    action_high=None,
):
    """Replay old actions under current parameters without inverting Box actions."""
    if "policy_loc" not in aux_info:
        return pi.log_prob(action)
    if action_low is None or action_high is None:
        raise ValueError("Box policy replay requires action_low and action_high.")
    return box_log_prob_from_pre_tanh(
        pre_tanh_action,
        aux_info["policy_loc"],
        aux_info["policy_log_std"],
        action_low,
        action_high,
    )


def tree_all_finite(tree):
    leaves = [
        jnp.asarray(value)
        for value in jax.tree_util.tree_leaves(tree)
        if value is not None
    ]
    if not leaves:
        return jnp.asarray(True)
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(value)) for value in leaves]))


def _global_bool(value, axis_name):
    value = jnp.asarray(value, dtype=jnp.bool_)
    if axis_name is None:
        return value
    return jax.lax.pmin(value.astype(jnp.int32), axis_name).astype(jnp.bool_)


def _select_pytree(condition, accepted_tree, rejected_tree):
    return jax.tree_util.tree_map(
        lambda accepted, rejected: jnp.where(condition, accepted, rejected),
        accepted_tree,
        rejected_tree,
    )


def guarded_ppo_apply_gradients(
    train_state,
    grads,
    *,
    total_loss,
    new_log_prob,
    logratio,
    ratio,
    axis_name=None,
) -> GuardedPPOUpdate:
    """Commit a PPO candidate only when loss, gradients, and state are finite."""
    candidate_state = train_state.apply_gradients(grads=grads)
    checks = (
        (
            FIRST_NONFINITE_STAGE["total_loss"],
            _global_bool(jnp.all(jnp.isfinite(total_loss)), axis_name),
        ),
        (
            FIRST_NONFINITE_STAGE["new_log_prob"],
            _global_bool(jnp.all(jnp.isfinite(new_log_prob)), axis_name),
        ),
        (
            FIRST_NONFINITE_STAGE["logratio"],
            _global_bool(jnp.all(jnp.isfinite(logratio)), axis_name),
        ),
        (
            FIRST_NONFINITE_STAGE["ratio"],
            _global_bool(jnp.all(jnp.isfinite(ratio)), axis_name),
        ),
        (
            FIRST_NONFINITE_STAGE["gradients"],
            _global_bool(tree_all_finite(grads), axis_name),
        ),
        (
            FIRST_NONFINITE_STAGE["candidate_params"],
            _global_bool(tree_all_finite(candidate_state.params), axis_name),
        ),
        (
            FIRST_NONFINITE_STAGE["candidate_optimizer_state"],
            _global_bool(tree_all_finite(candidate_state.opt_state), axis_name),
        ),
    )
    accepted = jnp.asarray(True)
    first_nonfinite_stage = jnp.asarray(
        FIRST_NONFINITE_STAGE["none"],
        dtype=jnp.int32,
    )
    for stage, check in checks:
        first_nonfinite_stage = jnp.where(
            accepted & ~check,
            jnp.asarray(stage, dtype=jnp.int32),
            first_nonfinite_stage,
        )
        accepted = accepted & check

    selected_state = _select_pytree(accepted, candidate_state, train_state)
    return GuardedPPOUpdate(
        train_state=selected_state,
        accepted=accepted,
        rejected_nonfinite=~accepted,
        first_nonfinite_stage=first_nonfinite_stage,
        gradients_finite=checks[4][1],
        candidate_params_finite=checks[5][1],
        candidate_optimizer_state_finite=checks[6][1],
    )


def empty_ppo_safety_state():
    return {
        "stopped": jnp.asarray(False),
        "accepted_minibatch_count": jnp.asarray(0, dtype=jnp.int32),
        "ppo_candidate_accepted": jnp.asarray(True),
        "ppo_candidate_rejected_nonfinite": jnp.asarray(False),
        "first_nonfinite_stage": jnp.asarray(0, dtype=jnp.int32),
        "rejected_epoch_index": jnp.asarray(-1, dtype=jnp.int32),
        "rejected_minibatch_index": jnp.asarray(-1, dtype=jnp.int32),
    }


def update_ppo_safety_state(
    safety_state,
    guarded_update: GuardedPPOUpdate,
    *,
    epoch_index,
    minibatch_index,
):
    """Record the first rejection and disable later commits for this agent/update."""
    attempt_active = ~safety_state["stopped"]
    rejected_now = attempt_active & guarded_update.rejected_nonfinite
    accepted_now = attempt_active & guarded_update.accepted
    return {
        "stopped": safety_state["stopped"] | rejected_now,
        "accepted_minibatch_count": (
            safety_state["accepted_minibatch_count"]
            + accepted_now.astype(jnp.int32)
        ),
        "ppo_candidate_accepted": (
            safety_state["ppo_candidate_accepted"] & ~rejected_now
        ),
        "ppo_candidate_rejected_nonfinite": (
            safety_state["ppo_candidate_rejected_nonfinite"] | rejected_now
        ),
        "first_nonfinite_stage": jnp.where(
            rejected_now,
            guarded_update.first_nonfinite_stage,
            safety_state["first_nonfinite_stage"],
        ),
        "rejected_epoch_index": jnp.where(
            rejected_now,
            jnp.asarray(epoch_index, dtype=jnp.int32),
            safety_state["rejected_epoch_index"],
        ),
        "rejected_minibatch_index": jnp.where(
            rejected_now,
            jnp.asarray(minibatch_index, dtype=jnp.int32),
            safety_state["rejected_minibatch_index"],
        ),
    }


def select_guarded_train_state(train_state, guarded_update, safety_state):
    """Do not commit candidates after this agent's first rejected minibatch."""
    may_commit = ~safety_state["stopped"]
    return _select_pytree(
        may_commit & guarded_update.accepted,
        guarded_update.train_state,
        train_state,
    )


def _per_dimension_stats(value):
    value = jnp.asarray(value)
    flattened = value.reshape((-1, value.shape[-1]))
    return {
        "mean": jnp.mean(flattened, axis=0),
        "std": jnp.std(flattened, axis=0),
        "min": jnp.min(flattened, axis=0),
        "max": jnp.max(flattened, axis=0),
    }


def _array_stats(value):
    value = jnp.asarray(value).reshape(-1)
    return {
        "mean": jnp.mean(value),
        "std": jnp.std(value),
        "min": jnp.min(value),
        "max": jnp.max(value),
    }


def _gradient_norm_for_paths(grads, predicate):
    matching = [
        jnp.asarray(value)
        for path, value in flatten_tree_with_paths(grads).items()
        if value is not None and predicate(path)
    ]
    if not matching:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return jnp.sqrt(sum(jnp.sum(jnp.square(value)) for value in matching))


def empty_box_ppo_numerics_diagnostics(action_dim=1):
    vector = jnp.zeros((action_dim,), dtype=jnp.float32)
    scalar = jnp.asarray(0.0, dtype=jnp.float32)
    boolean = jnp.asarray(False)
    return {
        "enabled": boolean,
        "active": boolean,
        "loc_mean": vector,
        "loc_std": vector,
        "loc_min": vector,
        "loc_max": vector,
        "log_std": vector,
        "std": vector,
        "pre_tanh_min": vector,
        "pre_tanh_max": vector,
        "action_min": vector,
        "action_max": vector,
        "exact_low_rate": vector,
        "exact_high_rate": vector,
        "exact_low_count": vector,
        "exact_high_count": vector,
        "action_sample_count": scalar,
        "near_low_rate": vector,
        "near_high_rate": vector,
        "old_log_prob_mean": scalar,
        "old_log_prob_min": scalar,
        "old_log_prob_max": scalar,
        "new_log_prob_mean": scalar,
        "new_log_prob_min": scalar,
        "new_log_prob_max": scalar,
        "logratio_mean": scalar,
        "logratio_std": scalar,
        "logratio_p95": scalar,
        "logratio_p99": scalar,
        "logratio_min": scalar,
        "logratio_max": scalar,
        "ratio_mean": scalar,
        "ratio_std": scalar,
        "ratio_p95": scalar,
        "ratio_p99": scalar,
        "ratio_min": scalar,
        "ratio_max": scalar,
        "advantage_mean": scalar,
        "advantage_std": scalar,
        "advantage_min": scalar,
        "advantage_max": scalar,
        "value_mean": scalar,
        "value_std": scalar,
        "value_min": scalar,
        "value_max": scalar,
        "actor_loc_grad_norm": scalar,
        "log_std_grad_norm": scalar,
        "total_grad_norm": scalar,
        "total_loss_finite": boolean,
        "loc_finite": boolean,
        "log_std_finite": boolean,
        "pre_tanh_finite": boolean,
        "action_finite": boolean,
        "old_log_prob_finite": boolean,
        "new_log_prob_finite": boolean,
        "logratio_finite": boolean,
        "ratio_finite": boolean,
        "advantage_finite": boolean,
        "value_finite": boolean,
        "gradients_finite": boolean,
        "candidate_params_finite": boolean,
        "candidate_optimizer_state_finite": boolean,
        "ppo_candidate_accepted": boolean,
        "ppo_candidate_rejected_nonfinite": boolean,
        "first_nonfinite_stage": jnp.asarray(0, dtype=jnp.int32),
        "epoch_index": jnp.asarray(-1, dtype=jnp.int32),
        "minibatch_index": jnp.asarray(-1, dtype=jnp.int32),
    }


def build_box_ppo_numerics_diagnostics(
    *,
    enabled,
    loc,
    log_std,
    pre_tanh_action,
    action,
    action_low,
    action_high,
    old_log_prob,
    new_log_prob,
    advantage,
    value,
    grads,
    total_loss,
    candidate_update: GuardedPPOUpdate | None = None,
    epoch_index=-1,
    minibatch_index=-1,
):
    """Build a static JAX scalar/vector tree without touching RNG or state."""
    action_dim = int(jnp.asarray(loc).shape[-1])
    if not bool(enabled):
        return empty_box_ppo_numerics_diagnostics(action_dim)

    loc = jnp.asarray(loc)
    log_std = jnp.broadcast_to(jnp.asarray(log_std, dtype=loc.dtype), loc.shape)
    pre_tanh_action = jnp.asarray(pre_tanh_action, dtype=loc.dtype)
    action = jnp.asarray(action, dtype=loc.dtype)
    action_low = jnp.asarray(action_low, dtype=loc.dtype)
    action_high = jnp.asarray(action_high, dtype=loc.dtype)
    old_log_prob = jnp.asarray(old_log_prob)
    new_log_prob = jnp.asarray(new_log_prob)
    logratio = new_log_prob - old_log_prob
    ratio = jnp.exp(logratio)
    loc_stats = _per_dimension_stats(loc)
    pre_tanh_stats = _per_dimension_stats(pre_tanh_action)
    action_stats = _per_dimension_stats(action)
    old_log_prob_stats = _array_stats(old_log_prob)
    new_log_prob_stats = _array_stats(new_log_prob)
    logratio_stats = _array_stats(logratio)
    ratio_stats = _array_stats(ratio)
    advantage_stats = _array_stats(advantage)
    value_stats = _array_stats(value)
    action_flat = action.reshape((-1, action_dim))
    near_tolerance = 1e-6 * jnp.maximum(action_high - action_low, 1.0)

    loc_grad_norm = _gradient_norm_for_paths(
        grads,
        lambda path: len(path) >= 2 and path[:2] == ("params", "Dense_1"),
    )
    log_std_grad_norm = _gradient_norm_for_paths(
        grads,
        lambda path: path == ("params", "log_std"),
    )
    if candidate_update is None:
        candidate_update = GuardedPPOUpdate(
            train_state=None,
            accepted=jnp.asarray(True),
            rejected_nonfinite=jnp.asarray(False),
            first_nonfinite_stage=jnp.asarray(0, dtype=jnp.int32),
            gradients_finite=tree_all_finite(grads),
            candidate_params_finite=jnp.asarray(True),
            candidate_optimizer_state_finite=jnp.asarray(True),
        )

    return {
        "enabled": jnp.asarray(True),
        "active": jnp.asarray(True),
        "loc_mean": loc_stats["mean"],
        "loc_std": loc_stats["std"],
        "loc_min": loc_stats["min"],
        "loc_max": loc_stats["max"],
        "log_std": log_std.reshape((-1, action_dim))[0],
        "std": jnp.exp(log_std.reshape((-1, action_dim))[0]),
        "pre_tanh_min": pre_tanh_stats["min"],
        "pre_tanh_max": pre_tanh_stats["max"],
        "action_min": action_stats["min"],
        "action_max": action_stats["max"],
        "exact_low_rate": jnp.mean(action_flat == action_low, axis=0),
        "exact_high_rate": jnp.mean(action_flat == action_high, axis=0),
        "exact_low_count": jnp.sum(action_flat == action_low, axis=0),
        "exact_high_count": jnp.sum(action_flat == action_high, axis=0),
        "action_sample_count": jnp.asarray(action_flat.shape[0], dtype=jnp.float32),
        "near_low_rate": jnp.mean(
            action_flat <= action_low + near_tolerance,
            axis=0,
        ),
        "near_high_rate": jnp.mean(
            action_flat >= action_high - near_tolerance,
            axis=0,
        ),
        "old_log_prob_mean": old_log_prob_stats["mean"],
        "old_log_prob_min": old_log_prob_stats["min"],
        "old_log_prob_max": old_log_prob_stats["max"],
        "new_log_prob_mean": new_log_prob_stats["mean"],
        "new_log_prob_min": new_log_prob_stats["min"],
        "new_log_prob_max": new_log_prob_stats["max"],
        "logratio_mean": logratio_stats["mean"],
        "logratio_std": logratio_stats["std"],
        "logratio_p95": jnp.percentile(logratio.reshape(-1), 95.0),
        "logratio_p99": jnp.percentile(logratio.reshape(-1), 99.0),
        "logratio_min": logratio_stats["min"],
        "logratio_max": logratio_stats["max"],
        "ratio_mean": ratio_stats["mean"],
        "ratio_std": ratio_stats["std"],
        "ratio_p95": jnp.percentile(ratio.reshape(-1), 95.0),
        "ratio_p99": jnp.percentile(ratio.reshape(-1), 99.0),
        "ratio_min": ratio_stats["min"],
        "ratio_max": ratio_stats["max"],
        "advantage_mean": advantage_stats["mean"],
        "advantage_std": advantage_stats["std"],
        "advantage_min": advantage_stats["min"],
        "advantage_max": advantage_stats["max"],
        "value_mean": value_stats["mean"],
        "value_std": value_stats["std"],
        "value_min": value_stats["min"],
        "value_max": value_stats["max"],
        "actor_loc_grad_norm": loc_grad_norm,
        "log_std_grad_norm": log_std_grad_norm,
        "total_grad_norm": tree_l2_norm(grads),
        "total_loss_finite": jnp.all(jnp.isfinite(total_loss)),
        "loc_finite": jnp.all(jnp.isfinite(loc)),
        "log_std_finite": jnp.all(jnp.isfinite(log_std)),
        "pre_tanh_finite": jnp.all(jnp.isfinite(pre_tanh_action)),
        "action_finite": jnp.all(jnp.isfinite(action)),
        "old_log_prob_finite": jnp.all(jnp.isfinite(old_log_prob)),
        "new_log_prob_finite": jnp.all(jnp.isfinite(new_log_prob)),
        "logratio_finite": jnp.all(jnp.isfinite(logratio)),
        "ratio_finite": jnp.all(jnp.isfinite(ratio)),
        "advantage_finite": jnp.all(jnp.isfinite(advantage)),
        "value_finite": jnp.all(jnp.isfinite(value)),
        "gradients_finite": candidate_update.gradients_finite,
        "candidate_params_finite": candidate_update.candidate_params_finite,
        "candidate_optimizer_state_finite": (
            candidate_update.candidate_optimizer_state_finite
        ),
        "ppo_candidate_accepted": candidate_update.accepted,
        "ppo_candidate_rejected_nonfinite": candidate_update.rejected_nonfinite,
        "first_nonfinite_stage": candidate_update.first_nonfinite_stage,
        "epoch_index": jnp.asarray(epoch_index, dtype=jnp.int32),
        "minibatch_index": jnp.asarray(minibatch_index, dtype=jnp.int32),
    }
