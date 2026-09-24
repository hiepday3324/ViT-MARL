"""Shared JAX utilities for PPO/survival gradient interaction diagnostics."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict

from gymnax_exchange.jaxrl.MARL.ppo_lifecycle import masked_mean


GRADIENT_GROUPS = (
    "total",
    "reliability_head",
    "vision_encoder",
    "fusion_shared_trunk",
    "actor_head",
    "critic_head",
)

GRADIENT_METRICS = (
    "param_leaf_count",
    "ppo_grad_norm",
    "survival_grad_norm_raw",
    "survival_grad_norm_weighted",
    "weighted_survival_to_ppo_grad_ratio",
    "ppo_survival_dot_raw",
    "ppo_survival_cosine_raw",
    "ppo_grad_nonzero",
    "survival_grad_nonzero",
    "cosine_valid",
    "joint_grad_norm",
    "decomposition_abs_error",
    "decomposition_rel_error",
)

PHASIC_GRADIENT_GROUPS = (
    "total",
    "reliability_head",
    "vision_encoder",
    "fusion_shared_trunk",
)

PHASIC_GRADIENT_METRICS = (
    "param_leaf_count",
    "ppo_grad_norm",
    "survival_grad_norm_raw",
    "ppo_survival_dot_raw",
    "ppo_survival_cosine_raw",
)

ACTOR_CRITIC_GRADIENT_GROUPS = (
    "total",
    "reliability_head",
    "vision_encoder",
    "fusion_shared_trunk",
    "critic_representation",
)

ACTOR_CRITIC_GRADIENT_METRICS = (
    "param_leaf_count",
    "policy_grad_norm",
    "value_grad_norm_raw",
    "value_grad_norm_weighted",
    "weighted_value_to_policy_grad_ratio",
    "policy_value_dot_raw",
    "policy_value_cosine_raw",
    "policy_grad_nonzero",
    "value_grad_nonzero",
    "cosine_valid",
    "policy_plus_value_grad_norm",
    "combined_to_policy_grad_ratio",
    "ppo_decomposition_abs_error",
    "ppo_decomposition_rel_error",
)

VALUE_CLIP_DIAGNOSTIC_METRICS = (
    "active_sample_count",
    "clip_eps",
    "target_mean",
    "target_std",
    "target_min",
    "target_max",
    "old_value_mean",
    "old_value_std",
    "old_value_min",
    "old_value_max",
    "new_value_mean",
    "new_value_std",
    "new_value_min",
    "new_value_max",
    "old_value_mae",
    "new_value_mae",
    "old_value_rmse",
    "new_value_rmse",
    "value_delta_mean",
    "value_delta_abs_mean",
    "value_delta_abs_max",
    "target_old_abs_error_mean",
    "target_old_abs_error_std",
    "target_distance_to_clip_ratio_mean",
    "target_distance_to_clip_ratio_p50",
    "target_distance_to_clip_ratio_p95",
    "clip_saturated_rate",
    "clipped_branch_selected_rate",
    "unclipped_branch_selected_rate",
    "branch_tie_rate",
    "value_gradient_locked_rate",
    "clip_saturated_and_unclipped_branch_rate",
    "old_value_explained_variance",
    "new_value_explained_variance",
    "explained_variance_valid",
    "target_std_to_clip_ratio",
    "target_old_error_to_clip_ratio",
    "new_value_std_to_target_std_ratio",
)

CRITIC_OPTIMIZER_DIAGNOSTIC_VARIANTS = (
    "current",
    "10x",
    "100x",
)

CRITIC_OPTIMIZER_DIAGNOSTIC_GROUPS = (
    "total_network",
    "critic_head",
    "fusion_shared_trunk",
    "vision_encoder",
    "reliability_head",
    "critic_representation",
)

CRITIC_OPTIMIZER_PARAMETER_GROUPS = (
    "total",
    "critic_head",
    "fusion_shared_trunk",
    "vision_encoder",
    "reliability_head",
    "critic_representation",
)

CRITIC_OPTIMIZER_VARIANT_METRICS = (
    "coefficient",
    "preclip_total_grad_norm",
    "estimated_global_clip_scale",
    "postclip_total_grad_norm",
    "global_clip_active",
)

CRITIC_OPTIMIZER_GROUP_METRICS = (
    "preclip_grad_norm",
    "postclip_grad_norm",
    "param_step_norm",
    "param_norm",
    "relative_param_step",
)

CRITIC_OPTIMIZER_SCALE_RESPONSE_METRICS = (
    "step_ratio_10x_to_current",
    "step_ratio_100x_to_current",
    "preclip_grad_ratio_10x_to_current",
    "preclip_grad_ratio_100x_to_current",
    "postclip_grad_ratio_10x_to_current",
    "postclip_grad_ratio_100x_to_current",
)

_CRITIC_OPTIMIZER_GROUP_RULE_NAMES = {
    "total_network": "total",
    "critic_head": "critic_head",
    "fusion_shared_trunk": "fusion_shared_trunk",
    "vision_encoder": "vision_encoder",
    "reliability_head": "reliability_head",
    "critic_representation": "critic_representation",
}

_PARAMS = ("params",)
_RELIABILITY_HEAD = _PARAMS + (
    "ReliabilityFusionRNN_0",
    "LevelWiseReliabilityHead_0",
)
_VISION_ENCODER = _PARAMS + ("VisionAgent_0",)
_RELIABILITY_FUSION = _PARAMS + ("ReliabilityFusionRNN_0",)
_CRITIC_REPRESENTATION = _PARAMS + ("CriticValueRNN_0",)
_ACTOR_DENSE_MODULES = frozenset(("Dense_0", "Dense_1"))
_CRITIC_DENSE_MODULES = frozenset(("Dense_2", "Dense_3"))


PARAMETER_GROUP_RULES = {
    "total": "all parameter leaves",
    "reliability_head": "prefix=params/ReliabilityFusionRNN_0/LevelWiseReliabilityHead_0",
    "vision_encoder": "prefix=params/VisionAgent_0",
    "fusion_shared_trunk": (
        "prefix=params/ReliabilityFusionRNN_0 excluding "
        "LevelWiseReliabilityHead_0; or actor-only ActorEmbedding, "
        "StableGatedCrossAttention_0, ScannedRNN_0"
    ),
    "actor_head": "prefix=params/Dense_0 or params/Dense_1; exact=params/log_std",
    "critic_head": "prefix=params/Dense_2 or params/Dense_3",
    "critic_representation": "prefix=params/CriticValueRNN_0",
}

AUXILIARY_TRAINABLE_GROUPS = (
    "reliability_head",
    "vision_encoder",
    "fusion_shared_trunk",
)

VALUE_PROBE_A_TRAINABLE_GROUPS = ("critic_head",)
VALUE_PROBE_B_FROZEN_GROUPS = ("actor_head",)
VALUE_PROBE_B_SEPARATE_TRAINABLE_GROUPS = (
    "critic_representation", "critic_head", "vision_encoder",
)


def _string_path(path: Sequence[Any]) -> tuple[str, ...]:
    return tuple(str(part) for part in path)


def _has_prefix(path: tuple[str, ...], prefix: tuple[str, ...]) -> bool:
    return path[: len(prefix)] == prefix


def parameter_path_in_group(path: Sequence[Any], group: str) -> bool:
    """Return whether a flattened Flax parameter path belongs to ``group``."""
    path = _string_path(path)
    if group == "total":
        return True
    if group == "reliability_head":
        return _has_prefix(path, _RELIABILITY_HEAD)
    if group == "vision_encoder":
        return _has_prefix(path, _VISION_ENCODER)
    if group == "fusion_shared_trunk":
        return (_has_prefix(path, _RELIABILITY_FUSION) and not _has_prefix(
            path,
            _RELIABILITY_HEAD,
        )) or any(
            _has_prefix(path, _PARAMS + (module,))
            for module in ("ActorEmbedding", "StableGatedCrossAttention_0", "ScannedRNN_0")
        )
    if group == "critic_representation":
        return _has_prefix(path, _CRITIC_REPRESENTATION)
    if group == "actor_head":
        return (
            len(path) >= 2
            and path[:1] == _PARAMS
            and path[1] in _ACTOR_DENSE_MODULES
        ) or path == _PARAMS + ("log_std",)
    if group == "critic_head":
        return (
            len(path) >= 2
            and path[:1] == _PARAMS
            and path[1] in _CRITIC_DENSE_MODULES
        )
    raise KeyError(f"Unknown gradient parameter group: {group!r}.")


def parameter_path_in_any_group(
    path: Sequence[Any],
    groups: Sequence[str],
) -> bool:
    """Return whether ``path`` belongs to at least one centralized group."""
    return any(parameter_path_in_group(path, group) for group in groups)


def mask_tree_to_groups(
    tree: Mapping[str, Any],
    groups: Sequence[str] = AUXILIARY_TRAINABLE_GROUPS,
) -> Mapping[str, Any]:
    """Zero every array leaf outside ``groups`` while preserving tree type."""
    was_frozen = isinstance(tree, FrozenDict)
    mutable_tree = unfreeze(tree) if was_frozen else tree
    flat_tree = flatten_dict(mutable_tree)
    masked_flat = {
        path: (
            value
            if value is None or parameter_path_in_any_group(path, groups)
            else jnp.zeros_like(value)
        )
        for path, value in flat_tree.items()
    }
    masked = unflatten_dict(masked_flat)
    return freeze(masked) if was_frozen else masked


def mask_tree_excluding_groups(
    tree: Mapping[str, Any],
    groups: Sequence[str],
) -> Mapping[str, Any]:
    """Zero array leaves inside ``groups`` while preserving tree type."""
    was_frozen = isinstance(tree, FrozenDict)
    mutable_tree = unfreeze(tree) if was_frozen else tree
    flat_tree = flatten_dict(mutable_tree)
    masked_flat = {
        path: (
            value
            if value is None or not parameter_path_in_any_group(path, groups)
            else jnp.zeros_like(value)
        )
        for path, value in flat_tree.items()
    }
    masked = unflatten_dict(masked_flat)
    return freeze(masked) if was_frozen else masked


def parameter_groups_mask(
    params: Mapping[str, Any],
    groups: Sequence[str],
    *,
    invert: bool = False,
) -> Mapping[str, Any]:
    """Build a static Optax mask from centralized parameter group rules."""
    was_frozen = isinstance(params, FrozenDict)
    mutable_params = unfreeze(params) if was_frozen else params
    flat_params = flatten_dict(mutable_params)
    mask = unflatten_dict(
        {
            path: (
                not parameter_path_in_any_group(path, groups)
                if invert
                else parameter_path_in_any_group(path, groups)
            )
            for path in flat_params
        }
    )
    return freeze(mask) if was_frozen else mask


def flatten_tree_with_paths(tree: Mapping[str, Any]) -> dict[tuple[str, ...], Any]:
    """Flatten a Flax/JAX mapping while preserving slash-style path components."""
    return {
        _string_path(path): value
        for path, value in flatten_dict(tree).items()
    }


def matching_parameter_paths(
    tree: Mapping[str, Any],
    group: str,
) -> tuple[tuple[str, ...], ...]:
    """Return deterministic sorted parameter paths assigned to ``group``."""
    paths = (
        path
        for path in flatten_tree_with_paths(tree)
        if parameter_path_in_group(path, group)
    )
    return tuple(sorted(paths))


def parameter_group_leaf_counts(tree: Mapping[str, Any]) -> dict[str, int]:
    return {
        group: len(matching_parameter_paths(tree, group))
        for group in GRADIENT_GROUPS + ("critic_representation",)
    }


def validate_required_parameter_groups(
    params: Mapping[str, Any],
    required_groups: Sequence[str] = GRADIENT_GROUPS,
) -> dict[str, int]:
    """Fail fast when the active model does not match a required path rule."""
    counts = parameter_group_leaf_counts(params)
    missing = [group for group in required_groups if counts[group] == 0]
    if missing:
        rules = "; ".join(
            f"{group}: {PARAMETER_GROUP_RULES[group]}" for group in missing
        )
        raise ValueError(
            "Gradient diagnostics could not match required parameter groups "
            f"{missing}. Active rules: {rules}."
        )
    return counts


def flatten_gradient_tree(
    gradients: Mapping[str, Any],
    group: str = "total",
) -> dict[tuple[str, ...], jax.Array]:
    """Select gradient leaves using the centralized parameter path rules."""
    return {
        path: jnp.asarray(value)
        for path, value in flatten_tree_with_paths(gradients).items()
        if value is not None and parameter_path_in_group(path, group)
    }


def scale_gradient_tree(tree: Any, scale: Any) -> Any:
    scale = jnp.asarray(scale)
    return jax.tree_util.tree_map(
        lambda value: None if value is None else jnp.asarray(value) * scale,
        tree,
        is_leaf=lambda value: value is None,
    )


def add_gradient_trees(left: Any, right: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda lhs, rhs: (
            None
            if lhs is None and rhs is None
            else jnp.asarray(0.0 if lhs is None else lhs)
            + jnp.asarray(0.0 if rhs is None else rhs)
        ),
        left,
        right,
        is_leaf=lambda value: value is None,
    )


def subtract_gradient_trees(left: Any, right: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda lhs, rhs: (
            None
            if lhs is None and rhs is None
            else jnp.asarray(0.0 if lhs is None else lhs)
            - jnp.asarray(0.0 if rhs is None else rhs)
        ),
        left,
        right,
        is_leaf=lambda value: value is None,
    )


def gradient_dot(left: Mapping[str, Any], right: Mapping[str, Any], group="total"):
    left_flat = flatten_gradient_tree(left, group)
    right_flat = flatten_gradient_tree(right, group)
    if left_flat.keys() != right_flat.keys():
        raise ValueError(f"Gradient trees differ for group {group!r}.")
    if not left_flat:
        return jnp.array(0.0, dtype=jnp.float32)
    return sum(
        jnp.sum(left_flat[path] * right_flat[path])
        for path in left_flat
    )


def tree_l2_norm(tree: Any):
    """Return the L2 norm of all non-None leaves in an arbitrary pytree."""
    leaves = [
        jnp.asarray(value)
        for value in jax.tree_util.tree_leaves(
            tree,
            is_leaf=lambda value: value is None,
        )
        if value is not None
    ]
    if not leaves:
        return jnp.array(0.0, dtype=jnp.float32)
    return jnp.sqrt(sum(jnp.sum(jnp.square(value)) for value in leaves))


def gradient_l2_norm(gradients: Mapping[str, Any], group="total"):
    flat = flatten_gradient_tree(gradients, group)
    return tree_l2_norm(flat)


def gradient_cosine(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    group="total",
    eps=1e-12,
):
    left_norm = gradient_l2_norm(left, group)
    right_norm = gradient_l2_norm(right, group)
    valid = (left_norm > eps) & (right_norm > eps)
    denominator = jnp.maximum(left_norm * right_norm, eps)
    cosine = jnp.where(valid, gradient_dot(left, right, group) / denominator, 0.0)
    return cosine, valid


def validate_value_representation_probe_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    updates = tuple(
        int(update)
        for update in config.get(
            "value_representation_probe_updates",
            (5, 10, 15),
        )
    )
    steps = int(config.get("value_representation_probe_steps", 150))
    learning_rate = float(
        config.get("value_representation_probe_lr", 1e-3)
    )
    train_fraction = float(
        config.get("value_representation_probe_train_fraction", 0.8)
    )
    split_seed = int(config.get("value_representation_probe_split_seed", 0))
    if not updates or any(update < 0 for update in updates):
        raise ValueError(
            "value_representation_probe_updates must contain non-negative "
            "update indices."
        )
    if len(set(updates)) != len(updates):
        raise ValueError("value_representation_probe_updates must be unique.")
    if steps < 1:
        raise ValueError("value_representation_probe_steps must be >= 1.")
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("value_representation_probe_lr must be finite and > 0.")
    if not np.isfinite(train_fraction) or not 0.0 < train_fraction < 1.0:
        raise ValueError(
            "value_representation_probe_train_fraction must be between 0 and 1."
        )
    if split_seed < 0:
        raise ValueError(
            "value_representation_probe_split_seed must be non-negative."
        )
    return {
        "updates": updates,
        "steps": steps,
        "learning_rate": learning_rate,
        "train_fraction": train_fraction,
        "split_seed": split_seed,
    }


def trajectory_train_holdout_masks(
    agent_active,
    *,
    num_environments: int,
    train_fraction: float,
    split_seed: int,
) -> tuple[jax.Array, jax.Array, int, int]:
    """Split complete recurrent trajectories by environment identity."""
    active = jnp.asarray(agent_active, dtype=jnp.bool_)
    if active.ndim != 2:
        raise ValueError("agent_active must have shape [time, actor].")
    actor_count = active.shape[1]
    num_environments = int(num_environments)
    if num_environments < 2 or actor_count % num_environments != 0:
        raise ValueError(
            "Probe split requires at least two environments and an actor "
            "count divisible by num_environments."
        )
    train_environment_count = int(np.floor(num_environments * train_fraction))
    train_environment_count = min(
        max(train_environment_count, 1),
        num_environments - 1,
    )
    actors_per_environment = actor_count // num_environments
    environment_ids = (
        jnp.arange(actor_count, dtype=jnp.int32) // actors_per_environment
    )
    environment_permutation = np.random.default_rng(int(split_seed)).permutation(
        num_environments
    )
    train_environment_membership = np.zeros(num_environments, dtype=np.bool_)
    train_environment_membership[
        environment_permutation[:train_environment_count]
    ] = True
    train_actor = jnp.asarray(train_environment_membership)[environment_ids]
    train_mask = active & train_actor[jnp.newaxis, :]
    holdout_mask = active & ~train_actor[jnp.newaxis, :]
    return (
        train_mask,
        holdout_mask,
        train_environment_count,
        num_environments - train_environment_count,
    )


def value_probe_statistics(prediction, target, mask, eps=1e-12):
    prediction = jnp.asarray(prediction, dtype=jnp.float32)
    target = jnp.asarray(target, dtype=jnp.float32)
    mask = jnp.broadcast_to(jnp.asarray(mask, dtype=jnp.bool_), target.shape)
    error = prediction - target
    mse = masked_mean(jnp.square(error), mask)
    mae = masked_mean(jnp.abs(error), mask)
    prediction_mean = masked_mean(prediction, mask)
    target_mean = masked_mean(target, mask)
    prediction_variance = masked_mean(
        jnp.square(prediction - prediction_mean),
        mask,
    )
    target_variance = masked_mean(jnp.square(target - target_mean), mask)
    residual = target - prediction
    residual_mean = masked_mean(residual, mask)
    residual_variance = masked_mean(
        jnp.square(residual - residual_mean),
        mask,
    )
    eps = jnp.asarray(eps, dtype=jnp.float32)
    active_count = jnp.sum(mask.astype(jnp.float32))
    explained_variance_valid = (active_count > 0) & (target_variance > eps)
    explained_variance = jnp.where(
        explained_variance_valid,
        1.0 - residual_variance / jnp.maximum(target_variance, eps),
        0.0,
    )
    prediction_std = jnp.sqrt(jnp.maximum(prediction_variance, 0.0))
    target_std = jnp.sqrt(jnp.maximum(target_variance, 0.0))
    return {
        "active_sample_count": active_count,
        "mse": mse,
        "rmse": jnp.sqrt(jnp.maximum(mse, 0.0)),
        "mae": mae,
        "explained_variance": explained_variance,
        "explained_variance_valid": explained_variance_valid,
        "prediction_mean": prediction_mean,
        "prediction_std": prediction_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "prediction_std_to_target_std_ratio": (
            prediction_std / jnp.maximum(target_std, eps)
        ),
    }


def _relative_parameter_change(initial_params, final_params, group, eps=1e-12):
    delta = subtract_gradient_trees(final_params, initial_params)
    return gradient_l2_norm(delta, group) / jnp.maximum(
        gradient_l2_norm(initial_params, group),
        jnp.asarray(eps, dtype=jnp.float32),
    )


def fit_value_representation_probe_variants(
    value_apply_fn,
    params,
    init_hstate,
    obs,
    rnn_reset,
    targets,
    train_mask,
    *,
    steps: int,
    learning_rate: float,
    use_separate_critic_representation: bool = False,
):
    """Fit isolated Probe A/B parameter copies and discard optimizer states."""
    params_a = jax.tree_util.tree_map(lambda value: value + 0, params)
    params_b = jax.tree_util.tree_map(lambda value: value + 0, params)
    mask_a = parameter_groups_mask(params, VALUE_PROBE_A_TRAINABLE_GROUPS)
    groups_b = (
        VALUE_PROBE_B_SEPARATE_TRAINABLE_GROUPS
        if use_separate_critic_representation else VALUE_PROBE_B_FROZEN_GROUPS
    )
    mask_b = parameter_groups_mask(
        params, groups_b, invert=not use_separate_critic_representation,
    )
    mask_b_tree = (
        mask_tree_to_groups
        if use_separate_critic_representation else mask_tree_excluding_groups
    )
    adam = optax.adam(learning_rate=learning_rate, eps=1e-5)
    optimizer_a = optax.masked(adam, mask_a)
    optimizer_b = optax.masked(adam, mask_b)
    optimizer_state_a = optimizer_a.init(params_a)
    optimizer_state_b = optimizer_b.init(params_b)

    def probe_loss(probe_params):
        prediction = value_apply_fn(
            probe_params,
            init_hstate,
            obs,
            rnn_reset,
        )
        return 0.5 * masked_mean(
            jnp.square(prediction - targets),
            train_mask,
        )

    def probe_step(carry, _):
        probe_params_a, state_a, probe_params_b, state_b = carry
        gradients_a = mask_tree_to_groups(
            jax.grad(probe_loss)(probe_params_a),
            VALUE_PROBE_A_TRAINABLE_GROUPS,
        )
        gradients_b = mask_b_tree(
            jax.grad(probe_loss)(probe_params_b),
            groups_b,
        )
        updates_a, state_a = optimizer_a.update(
            gradients_a,
            state_a,
            probe_params_a,
        )
        updates_b, state_b = optimizer_b.update(
            gradients_b,
            state_b,
            probe_params_b,
        )
        updates_a = mask_tree_to_groups(
            updates_a,
            VALUE_PROBE_A_TRAINABLE_GROUPS,
        )
        updates_b = mask_b_tree(
            updates_b,
            groups_b,
        )
        return (
            optax.apply_updates(probe_params_a, updates_a),
            state_a,
            optax.apply_updates(probe_params_b, updates_b),
            state_b,
        ), None

    (params_a, _, params_b, _), _ = jax.lax.scan(
        probe_step,
        (params_a, optimizer_state_a, params_b, optimizer_state_b),
        xs=None,
        length=int(steps),
    )
    return params_a, params_b


def _empty_value_probe_statistics():
    return {
        "active_sample_count": jnp.array(0.0, dtype=jnp.float32),
        "mse": jnp.array(0.0, dtype=jnp.float32),
        "rmse": jnp.array(0.0, dtype=jnp.float32),
        "mae": jnp.array(0.0, dtype=jnp.float32),
        "explained_variance": jnp.array(0.0, dtype=jnp.float32),
        "explained_variance_valid": jnp.array(False),
        "prediction_mean": jnp.array(0.0, dtype=jnp.float32),
        "prediction_std": jnp.array(0.0, dtype=jnp.float32),
        "target_mean": jnp.array(0.0, dtype=jnp.float32),
        "target_std": jnp.array(0.0, dtype=jnp.float32),
        "prediction_std_to_target_std_ratio": jnp.array(0.0, dtype=jnp.float32),
    }


def _empty_value_probe_split_metrics():
    return {
        "initial": _empty_value_probe_statistics(),
        "final": _empty_value_probe_statistics(),
    }


def empty_value_representation_probe_diagnostics(
    *,
    enabled=False,
    skipped_by_schedule=False,
    not_applicable=False,
    steps=0,
    learning_rate=0.0,
    train_fraction=0.0,
    split_seed=0,
):
    return {
        "enabled": jnp.asarray(enabled, dtype=jnp.bool_),
        "active": jnp.array(False),
        "skipped_by_schedule": jnp.asarray(
            skipped_by_schedule,
            dtype=jnp.bool_,
        ),
        "not_applicable": jnp.asarray(not_applicable, dtype=jnp.bool_),
        "steps": jnp.asarray(steps, dtype=jnp.int32),
        "learning_rate": jnp.asarray(learning_rate, dtype=jnp.float32),
        "train_fraction": jnp.asarray(train_fraction, dtype=jnp.float32),
        "split_seed": jnp.asarray(split_seed, dtype=jnp.int32),
        "train_environment_count": jnp.array(0, dtype=jnp.int32),
        "holdout_environment_count": jnp.array(0, dtype=jnp.int32),
        "probes_start_identical": jnp.array(False),
        "probe_a": {
            "train": _empty_value_probe_split_metrics(),
            "holdout": _empty_value_probe_split_metrics(),
            "parameter_change": {
                "critic_head": jnp.array(0.0, dtype=jnp.float32),
            },
        },
        "probe_b": {
            "train": _empty_value_probe_split_metrics(),
            "holdout": _empty_value_probe_split_metrics(),
            "parameter_change": {
                "critic_head": jnp.array(0.0, dtype=jnp.float32),
                "fusion_shared_trunk": jnp.array(0.0, dtype=jnp.float32),
                "vision_encoder": jnp.array(0.0, dtype=jnp.float32),
                "reliability_head": jnp.array(0.0, dtype=jnp.float32),
                "critic_representation": jnp.array(0.0, dtype=jnp.float32),
            },
        },
        "comparison": {
            "holdout_ev_gain_a": jnp.array(0.0, dtype=jnp.float32),
            "holdout_ev_gain_b": jnp.array(0.0, dtype=jnp.float32),
            "holdout_mae_reduction_a": jnp.array(0.0, dtype=jnp.float32),
            "holdout_mae_reduction_b": jnp.array(0.0, dtype=jnp.float32),
            "b_minus_a_final_holdout_ev": jnp.array(0.0, dtype=jnp.float32),
            "b_over_a_holdout_mae_reduction_ratio": jnp.array(
                0.0,
                dtype=jnp.float32,
            ),
        },
    }


def run_value_representation_probe(
    value_apply_fn,
    params,
    init_hstate,
    obs,
    rnn_reset,
    targets,
    agent_active,
    *,
    num_environments: int,
    steps: int,
    learning_rate: float,
    train_fraction: float,
    split_seed: int,
    use_separate_critic_representation: bool = False,
    eps=1e-12,
):
    """Run isolated head-only and shared-representation value probes."""
    train_mask, holdout_mask, train_count, holdout_count = (
        trajectory_train_holdout_masks(
            agent_active,
            num_environments=num_environments,
            train_fraction=train_fraction,
            split_seed=split_seed,
        )
    )
    initial_prediction = value_apply_fn(
        params,
        init_hstate,
        obs,
        rnn_reset,
    )
    probe_a_start = jax.tree_util.tree_map(lambda value: value + 0, params)
    probe_b_start = jax.tree_util.tree_map(lambda value: value + 0, params)
    probes_start_identical = tree_l2_norm(
        subtract_gradient_trees(probe_a_start, probe_b_start)
    ) == 0.0
    params_a, params_b = fit_value_representation_probe_variants(
        value_apply_fn,
        params,
        init_hstate,
        obs,
        rnn_reset,
        targets,
        train_mask,
        steps=steps,
        learning_rate=learning_rate,
        use_separate_critic_representation=use_separate_critic_representation,
    )
    prediction_a = value_apply_fn(
        params_a,
        init_hstate,
        obs,
        rnn_reset,
    )
    prediction_b = value_apply_fn(
        params_b,
        init_hstate,
        obs,
        rnn_reset,
    )
    initial_train = value_probe_statistics(
        initial_prediction,
        targets,
        train_mask,
        eps,
    )
    initial_holdout = value_probe_statistics(
        initial_prediction,
        targets,
        holdout_mask,
        eps,
    )
    final_a_train = value_probe_statistics(
        prediction_a,
        targets,
        train_mask,
        eps,
    )
    final_a_holdout = value_probe_statistics(
        prediction_a,
        targets,
        holdout_mask,
        eps,
    )
    final_b_train = value_probe_statistics(
        prediction_b,
        targets,
        train_mask,
        eps,
    )
    final_b_holdout = value_probe_statistics(
        prediction_b,
        targets,
        holdout_mask,
        eps,
    )
    holdout_ev_gain_a = (
        final_a_holdout["explained_variance"]
        - initial_holdout["explained_variance"]
    )
    holdout_ev_gain_b = (
        final_b_holdout["explained_variance"]
        - initial_holdout["explained_variance"]
    )
    holdout_mae_reduction_a = (
        initial_holdout["mae"] - final_a_holdout["mae"]
    )
    holdout_mae_reduction_b = (
        initial_holdout["mae"] - final_b_holdout["mae"]
    )
    eps = jnp.asarray(eps, dtype=jnp.float32)
    return {
        "enabled": jnp.array(True),
        "active": jnp.array(True),
        "skipped_by_schedule": jnp.array(False),
        "not_applicable": jnp.array(False),
        "steps": jnp.asarray(steps, dtype=jnp.int32),
        "learning_rate": jnp.asarray(learning_rate, dtype=jnp.float32),
        "train_fraction": jnp.asarray(train_fraction, dtype=jnp.float32),
        "split_seed": jnp.asarray(split_seed, dtype=jnp.int32),
        "train_environment_count": jnp.asarray(train_count, dtype=jnp.int32),
        "holdout_environment_count": jnp.asarray(
            holdout_count,
            dtype=jnp.int32,
        ),
        "probes_start_identical": probes_start_identical,
        "probe_a": {
            "train": {"initial": initial_train, "final": final_a_train},
            "holdout": {
                "initial": initial_holdout,
                "final": final_a_holdout,
            },
            "parameter_change": {
                "critic_head": _relative_parameter_change(
                    params,
                    params_a,
                    "critic_head",
                    eps,
                ),
            },
        },
        "probe_b": {
            "train": {"initial": initial_train, "final": final_b_train},
            "holdout": {
                "initial": initial_holdout,
                "final": final_b_holdout,
            },
            "parameter_change": {
                group: _relative_parameter_change(
                    params,
                    params_b,
                    group,
                    eps,
                )
                for group in (
                    "critic_head",
                    "fusion_shared_trunk",
                    "vision_encoder",
                    "reliability_head",
                    "critic_representation",
                )
            },
        },
        "comparison": {
            "holdout_ev_gain_a": holdout_ev_gain_a,
            "holdout_ev_gain_b": holdout_ev_gain_b,
            "holdout_mae_reduction_a": holdout_mae_reduction_a,
            "holdout_mae_reduction_b": holdout_mae_reduction_b,
            "b_minus_a_final_holdout_ev": (
                final_b_holdout["explained_variance"]
                - final_a_holdout["explained_variance"]
            ),
            "b_over_a_holdout_mae_reduction_ratio": (
                holdout_mae_reduction_b
                / jnp.maximum(jnp.abs(holdout_mae_reduction_a), eps)
            ),
        },
    }


def _empty_actor_critic_group_metrics(
    param_leaf_count: int,
) -> dict[str, jax.Array]:
    metrics = {
        metric: jnp.array(0.0, dtype=jnp.float32)
        for metric in ACTOR_CRITIC_GRADIENT_METRICS
        if metric
        not in {
            "param_leaf_count",
            "policy_grad_nonzero",
            "value_grad_nonzero",
            "cosine_valid",
        }
    }
    metrics.update(
        {
            "param_leaf_count": jnp.array(param_leaf_count, dtype=jnp.int32),
            "policy_grad_nonzero": jnp.array(False),
            "value_grad_nonzero": jnp.array(False),
            "cosine_valid": jnp.array(False),
        }
    )
    return metrics


def empty_actor_critic_gradient_diagnostics(
    params: Mapping[str, Any],
    *,
    enabled=False,
    skipped_by_cadence=False,
    not_applicable=False,
    reason_not_execution=False,
    max_grad_norm=0.0,
) -> dict[str, Any]:
    """Return the fixed actor/value diagnostic pytree for every branch."""
    counts = parameter_group_leaf_counts(params)
    return {
        "actor_critic_grad_diag_enabled": jnp.asarray(enabled, dtype=jnp.bool_),
        "actor_critic_grad_diag_active": jnp.array(False),
        "actor_critic_grad_diag_skipped_by_cadence": jnp.asarray(
            skipped_by_cadence,
            dtype=jnp.bool_,
        ),
        "actor_critic_grad_diag_not_applicable": jnp.asarray(
            not_applicable,
            dtype=jnp.bool_,
        ),
        "reason_not_execution": jnp.asarray(
            reason_not_execution,
            dtype=jnp.bool_,
        ),
        "pre_clip_total_grad_norm": jnp.array(0.0, dtype=jnp.float32),
        "max_grad_norm": jnp.asarray(max_grad_norm, dtype=jnp.float32),
        "estimated_global_clip_scale": jnp.array(0.0, dtype=jnp.float32),
        "groups": {
            group: _empty_actor_critic_group_metrics(counts[group])
            for group in ACTOR_CRITIC_GRADIENT_GROUPS
        },
    }


def summarize_actor_critic_gradient_interaction(
    params: Mapping[str, Any],
    actual_gradients: Mapping[str, Any],
    ppo_gradients: Mapping[str, Any],
    policy_gradients: Mapping[str, Any],
    value_gradients_raw: Mapping[str, Any],
    vf_coef: Any,
    max_grad_norm: Any,
    eps=1e-12,
) -> dict[str, Any]:
    """Summarize read-only PPO policy/value gradient interaction."""
    weighted_value = scale_gradient_tree(value_gradients_raw, vf_coef)
    policy_plus_value = add_gradient_trees(policy_gradients, weighted_value)
    decomposition_residual = subtract_gradient_trees(
        ppo_gradients,
        policy_plus_value,
    )
    counts = parameter_group_leaf_counts(params)
    groups = {}
    for group in ACTOR_CRITIC_GRADIENT_GROUPS:
        policy_norm = gradient_l2_norm(policy_gradients, group)
        value_norm_raw = gradient_l2_norm(value_gradients_raw, group)
        value_norm_weighted = gradient_l2_norm(weighted_value, group)
        combined_norm = gradient_l2_norm(policy_plus_value, group)
        ppo_norm = gradient_l2_norm(ppo_gradients, group)
        decomposition_abs_error = gradient_l2_norm(
            decomposition_residual,
            group,
        )
        cosine, cosine_valid = gradient_cosine(
            policy_gradients,
            value_gradients_raw,
            group,
            eps,
        )
        groups[group] = {
            "param_leaf_count": jnp.array(counts[group], dtype=jnp.int32),
            "policy_grad_norm": policy_norm,
            "value_grad_norm_raw": value_norm_raw,
            "value_grad_norm_weighted": value_norm_weighted,
            "weighted_value_to_policy_grad_ratio": (
                value_norm_weighted / jnp.maximum(policy_norm, eps)
            ),
            "policy_value_dot_raw": gradient_dot(
                policy_gradients,
                value_gradients_raw,
                group,
            ),
            "policy_value_cosine_raw": cosine,
            "policy_grad_nonzero": policy_norm > eps,
            "value_grad_nonzero": value_norm_raw > eps,
            "cosine_valid": cosine_valid,
            "policy_plus_value_grad_norm": combined_norm,
            "combined_to_policy_grad_ratio": (
                combined_norm / jnp.maximum(policy_norm, eps)
            ),
            "ppo_decomposition_abs_error": decomposition_abs_error,
            "ppo_decomposition_rel_error": (
                decomposition_abs_error / jnp.maximum(ppo_norm, eps)
            ),
        }

    pre_clip_total_grad_norm = tree_l2_norm(actual_gradients)
    max_grad_norm = jnp.asarray(max_grad_norm, dtype=jnp.float32)
    estimated_global_clip_scale = jnp.minimum(
        jnp.array(1.0, dtype=jnp.float32),
        max_grad_norm / jnp.maximum(pre_clip_total_grad_norm, eps),
    )
    return {
        "actor_critic_grad_diag_enabled": jnp.array(True),
        "actor_critic_grad_diag_active": jnp.array(True),
        "actor_critic_grad_diag_skipped_by_cadence": jnp.array(False),
        "actor_critic_grad_diag_not_applicable": jnp.array(False),
        "reason_not_execution": jnp.array(False),
        "pre_clip_total_grad_norm": pre_clip_total_grad_norm,
        "max_grad_norm": max_grad_norm,
        "estimated_global_clip_scale": estimated_global_clip_scale,
        "groups": groups,
    }


def empty_value_clip_diagnostics(
    *,
    enabled=False,
    skipped_by_cadence=False,
    not_applicable=False,
    reason_not_execution=False,
    clip_eps=0.0,
) -> dict[str, jax.Array]:
    """Return the fixed value-clipping diagnostic pytree for every branch."""
    diagnostics = {
        metric: jnp.array(0.0, dtype=jnp.float32)
        for metric in VALUE_CLIP_DIAGNOSTIC_METRICS
    }
    diagnostics.update(
        {
            "value_clip_diag_enabled": jnp.asarray(enabled, dtype=jnp.bool_),
            "value_clip_diag_active": jnp.array(False),
            "value_clip_diag_skipped_by_cadence": jnp.asarray(
                skipped_by_cadence,
                dtype=jnp.bool_,
            ),
            "value_clip_diag_not_applicable": jnp.asarray(
                not_applicable,
                dtype=jnp.bool_,
            ),
            "reason_not_execution": jnp.asarray(
                reason_not_execution,
                dtype=jnp.bool_,
            ),
            "explained_variance_valid": jnp.array(False),
            "clip_eps": jnp.asarray(clip_eps, dtype=jnp.float32),
        }
    )
    return diagnostics


def _masked_std(values, mask, eps=0.0):
    mean = masked_mean(values, mask)
    variance = masked_mean(jnp.square(values - mean), mask)
    return jnp.sqrt(jnp.maximum(variance, jnp.asarray(eps, values.dtype)))


def _masked_min(values, mask):
    has_active = jnp.any(mask)
    minimum = jnp.min(jnp.where(mask, values, jnp.inf))
    return jnp.where(has_active, minimum, jnp.zeros((), dtype=values.dtype))


def _masked_max(values, mask):
    has_active = jnp.any(mask)
    maximum = jnp.max(jnp.where(mask, values, -jnp.inf))
    return jnp.where(has_active, maximum, jnp.zeros((), dtype=values.dtype))


def _masked_quantile(values, mask, quantile):
    """Linear quantile over active entries using a static-shape JAX sort."""
    original_shape = jnp.shape(values)
    values = jnp.ravel(values)
    mask = jnp.ravel(jnp.broadcast_to(mask, original_shape)).astype(jnp.bool_)
    active_count = jnp.sum(mask.astype(jnp.int32))
    has_active = active_count > 0
    sorted_values = jnp.sort(jnp.where(mask, values, jnp.inf))
    sorted_values = jnp.where(
        has_active,
        sorted_values,
        jnp.zeros_like(sorted_values),
    )
    position = jnp.asarray(quantile, values.dtype) * jnp.maximum(
        active_count - 1,
        0,
    ).astype(values.dtype)
    lower = jnp.floor(position).astype(jnp.int32)
    upper = jnp.ceil(position).astype(jnp.int32)
    max_index = jnp.maximum(active_count - 1, 0)
    lower = jnp.clip(lower, 0, max_index)
    upper = jnp.clip(upper, 0, max_index)
    fraction = position - lower.astype(values.dtype)
    result = sorted_values[lower] + fraction * (
        sorted_values[upper] - sorted_values[lower]
    )
    return jnp.where(has_active, result, jnp.zeros((), dtype=values.dtype))


def summarize_value_clip_diagnostics(
    *,
    old_value,
    new_value,
    target,
    value_pred_clipped,
    value_losses,
    value_losses_clipped,
    agent_active,
    clip_eps,
    eps=1e-12,
) -> dict[str, jax.Array]:
    """Summarize the exact PPO value-clipping tensors without modifying them."""
    old_value = jnp.asarray(old_value)
    new_value = jnp.asarray(new_value)
    target = jnp.asarray(target)
    value_pred_clipped = jnp.asarray(value_pred_clipped)
    value_losses = jnp.asarray(value_losses)
    value_losses_clipped = jnp.asarray(value_losses_clipped)
    mask = jnp.broadcast_to(
        jnp.asarray(agent_active, dtype=jnp.bool_),
        target.shape,
    )
    dtype = target.dtype
    eps = jnp.asarray(eps, dtype=dtype)
    clip_eps = jnp.asarray(clip_eps, dtype=dtype)
    safe_clip_eps = jnp.maximum(jnp.abs(clip_eps), eps)

    target_mean = masked_mean(target, mask)
    target_std = _masked_std(target, mask)
    old_value_mean = masked_mean(old_value, mask)
    old_value_std = _masked_std(old_value, mask)
    new_value_mean = masked_mean(new_value, mask)
    new_value_std = _masked_std(new_value, mask)

    old_error = old_value - target
    new_error = new_value - target
    old_abs_error = jnp.abs(old_error)
    new_abs_error = jnp.abs(new_error)
    delta_v = new_value - old_value
    delta_abs = jnp.abs(delta_v)
    target_old_abs_error_std = _masked_std(old_abs_error, mask)
    target_distance_to_clip = old_abs_error / safe_clip_eps

    boundary_tolerance = jnp.maximum(eps, jnp.abs(clip_eps) * 1e-6)
    clip_saturated = delta_abs > jnp.abs(clip_eps) + boundary_tolerance
    branch_scale = jnp.maximum(
        jnp.maximum(jnp.abs(value_losses), jnp.abs(value_losses_clipped)),
        jnp.ones_like(value_losses),
    )
    branch_tolerance = jnp.maximum(eps, branch_scale * 1e-6)
    clipped_branch_dominates = (
        value_losses_clipped > value_losses + branch_tolerance
    )
    unclipped_branch_dominates = (
        value_losses > value_losses_clipped + branch_tolerance
    )
    branch_tie = ~(clipped_branch_dominates | unclipped_branch_dominates)
    value_gradient_locked = clip_saturated & clipped_branch_dominates
    saturated_unclipped = clip_saturated & unclipped_branch_dominates

    target_variance = masked_mean(jnp.square(target - target_mean), mask)
    old_residual_mean = masked_mean(target - old_value, mask)
    new_residual_mean = masked_mean(target - new_value, mask)
    old_residual_variance = masked_mean(
        jnp.square((target - old_value) - old_residual_mean),
        mask,
    )
    new_residual_variance = masked_mean(
        jnp.square((target - new_value) - new_residual_mean),
        mask,
    )
    explained_variance_valid = (jnp.sum(mask) > 0) & (target_variance > eps)
    safe_target_variance = jnp.maximum(target_variance, eps)
    old_explained_variance = jnp.where(
        explained_variance_valid,
        1.0 - old_residual_variance / safe_target_variance,
        0.0,
    )
    new_explained_variance = jnp.where(
        explained_variance_valid,
        1.0 - new_residual_variance / safe_target_variance,
        0.0,
    )

    as_rate = lambda condition: masked_mean(
        condition.astype(dtype),
        mask,
    )
    return {
        "value_clip_diag_enabled": jnp.array(True),
        "value_clip_diag_active": jnp.array(True),
        "value_clip_diag_skipped_by_cadence": jnp.array(False),
        "value_clip_diag_not_applicable": jnp.array(False),
        "reason_not_execution": jnp.array(False),
        "active_sample_count": jnp.sum(mask.astype(jnp.float32)),
        "clip_eps": clip_eps,
        "target_mean": target_mean,
        "target_std": target_std,
        "target_min": _masked_min(target, mask),
        "target_max": _masked_max(target, mask),
        "old_value_mean": old_value_mean,
        "old_value_std": old_value_std,
        "old_value_min": _masked_min(old_value, mask),
        "old_value_max": _masked_max(old_value, mask),
        "new_value_mean": new_value_mean,
        "new_value_std": new_value_std,
        "new_value_min": _masked_min(new_value, mask),
        "new_value_max": _masked_max(new_value, mask),
        "old_value_mae": masked_mean(old_abs_error, mask),
        "new_value_mae": masked_mean(new_abs_error, mask),
        "old_value_rmse": jnp.sqrt(masked_mean(jnp.square(old_error), mask)),
        "new_value_rmse": jnp.sqrt(masked_mean(jnp.square(new_error), mask)),
        "value_delta_mean": masked_mean(delta_v, mask),
        "value_delta_abs_mean": masked_mean(delta_abs, mask),
        "value_delta_abs_max": _masked_max(delta_abs, mask),
        "target_old_abs_error_mean": masked_mean(old_abs_error, mask),
        "target_old_abs_error_std": target_old_abs_error_std,
        "target_distance_to_clip_ratio_mean": masked_mean(
            target_distance_to_clip,
            mask,
        ),
        "target_distance_to_clip_ratio_p50": _masked_quantile(
            target_distance_to_clip,
            mask,
            0.50,
        ),
        "target_distance_to_clip_ratio_p95": _masked_quantile(
            target_distance_to_clip,
            mask,
            0.95,
        ),
        "clip_saturated_rate": as_rate(clip_saturated),
        "clipped_branch_selected_rate": as_rate(clipped_branch_dominates),
        "unclipped_branch_selected_rate": as_rate(
            unclipped_branch_dominates
        ),
        "branch_tie_rate": as_rate(branch_tie),
        "value_gradient_locked_rate": as_rate(value_gradient_locked),
        "clip_saturated_and_unclipped_branch_rate": as_rate(
            saturated_unclipped
        ),
        "old_value_explained_variance": old_explained_variance,
        "new_value_explained_variance": new_explained_variance,
        "explained_variance_valid": explained_variance_valid,
        "target_std_to_clip_ratio": target_std / safe_clip_eps,
        "target_old_error_to_clip_ratio": (
            masked_mean(old_abs_error, mask) / safe_clip_eps
        ),
        "new_value_std_to_target_std_ratio": (
            new_value_std / jnp.maximum(target_std, eps)
        ),
    }


def _empty_critic_optimizer_group_metrics() -> dict[str, jax.Array]:
    return {
        metric: jnp.array(0.0, dtype=jnp.float32)
        for metric in CRITIC_OPTIMIZER_GROUP_METRICS
    }


def empty_critic_optimizer_diagnostics(
    *,
    enabled=False,
    skipped_by_cadence=False,
    not_applicable=False,
    reason_not_execution=False,
    max_grad_norm=0.0,
) -> dict[str, Any]:
    """Return the fixed counterfactual optimizer diagnostic pytree."""
    variants = {}
    for variant in CRITIC_OPTIMIZER_DIAGNOSTIC_VARIANTS:
        variant_metrics = {
            metric: jnp.array(0.0, dtype=jnp.float32)
            for metric in CRITIC_OPTIMIZER_VARIANT_METRICS
        }
        variant_metrics["global_clip_active"] = jnp.array(False)
        variant_metrics["groups"] = {
            group: _empty_critic_optimizer_group_metrics()
            for group in CRITIC_OPTIMIZER_DIAGNOSTIC_GROUPS
        }
        variants[variant] = variant_metrics
    return {
        "critic_optimizer_diag_enabled": jnp.asarray(
            enabled,
            dtype=jnp.bool_,
        ),
        "critic_optimizer_diag_active": jnp.array(False),
        "critic_optimizer_diag_skipped_by_cadence": jnp.asarray(
            skipped_by_cadence,
            dtype=jnp.bool_,
        ),
        "critic_optimizer_diag_not_applicable": jnp.asarray(
            not_applicable,
            dtype=jnp.bool_,
        ),
        "reason_not_execution": jnp.asarray(
            reason_not_execution,
            dtype=jnp.bool_,
        ),
        "adam_internals_available": jnp.array(False),
        "max_grad_norm": jnp.asarray(max_grad_norm, dtype=jnp.float32),
        "variants": variants,
        "scale_response": {
            group: {
                metric: jnp.array(0.0, dtype=jnp.float32)
                for metric in CRITIC_OPTIMIZER_SCALE_RESPONSE_METRICS
            }
            for group in CRITIC_OPTIMIZER_DIAGNOSTIC_GROUPS
        },
    }


def summarize_critic_optimizer_diagnostics(
    *,
    params: Mapping[str, Any],
    optimizer_state: Any,
    optimizer: Any,
    policy_gradients: Mapping[str, Any],
    value_gradients_raw: Mapping[str, Any],
    current_vf_coef: Any,
    max_grad_norm: Any,
    eps=1e-12,
) -> dict[str, Any]:
    """Evaluate PPO critic-scale counterfactuals through the real optimizer."""
    current_vf_coef = jnp.asarray(current_vf_coef, dtype=jnp.float32)
    max_grad_norm = jnp.asarray(max_grad_norm, dtype=jnp.float32)
    eps = jnp.asarray(eps, dtype=jnp.float32)
    coefficients = {
        "current": current_vf_coef,
        "10x": current_vf_coef * 10.0,
        "100x": current_vf_coef * 100.0,
    }
    variants = {}
    for variant in CRITIC_OPTIMIZER_DIAGNOSTIC_VARIANTS:
        coefficient = coefficients[variant]
        counterfactual_gradients = add_gradient_trees(
            policy_gradients,
            scale_gradient_tree(value_gradients_raw, coefficient),
        )
        preclip_total_norm = tree_l2_norm(counterfactual_gradients)
        clip_scale = jnp.minimum(
            jnp.array(1.0, dtype=jnp.float32),
            max_grad_norm / jnp.maximum(preclip_total_norm, eps),
        )
        postclip_gradients = scale_gradient_tree(
            counterfactual_gradients,
            clip_scale,
        )
        updates, _counterfactual_optimizer_state = optimizer.update(
            counterfactual_gradients,
            optimizer_state,
            params,
        )
        group_metrics = {}
        for output_group in CRITIC_OPTIMIZER_DIAGNOSTIC_GROUPS:
            parameter_group = _CRITIC_OPTIMIZER_GROUP_RULE_NAMES[output_group]
            preclip_norm = gradient_l2_norm(
                counterfactual_gradients,
                parameter_group,
            )
            postclip_norm = gradient_l2_norm(
                postclip_gradients,
                parameter_group,
            )
            param_step_norm = gradient_l2_norm(updates, parameter_group)
            param_norm = gradient_l2_norm(params, parameter_group)
            group_metrics[output_group] = {
                "preclip_grad_norm": preclip_norm,
                "postclip_grad_norm": postclip_norm,
                "param_step_norm": param_step_norm,
                "param_norm": param_norm,
                "relative_param_step": (
                    param_step_norm / jnp.maximum(param_norm, eps)
                ),
            }
        variants[variant] = {
            "coefficient": coefficient,
            "preclip_total_grad_norm": preclip_total_norm,
            "estimated_global_clip_scale": clip_scale,
            "postclip_total_grad_norm": tree_l2_norm(postclip_gradients),
            "global_clip_active": clip_scale < 1.0,
            "groups": group_metrics,
        }

    current_groups = variants["current"]["groups"]
    groups_10x = variants["10x"]["groups"]
    groups_100x = variants["100x"]["groups"]
    scale_response = {}
    for group in CRITIC_OPTIMIZER_DIAGNOSTIC_GROUPS:
        current = current_groups[group]
        ten = groups_10x[group]
        hundred = groups_100x[group]
        scale_response[group] = {
            "step_ratio_10x_to_current": (
                ten["param_step_norm"]
                / jnp.maximum(current["param_step_norm"], eps)
            ),
            "step_ratio_100x_to_current": (
                hundred["param_step_norm"]
                / jnp.maximum(current["param_step_norm"], eps)
            ),
            "preclip_grad_ratio_10x_to_current": (
                ten["preclip_grad_norm"]
                / jnp.maximum(current["preclip_grad_norm"], eps)
            ),
            "preclip_grad_ratio_100x_to_current": (
                hundred["preclip_grad_norm"]
                / jnp.maximum(current["preclip_grad_norm"], eps)
            ),
            "postclip_grad_ratio_10x_to_current": (
                ten["postclip_grad_norm"]
                / jnp.maximum(current["postclip_grad_norm"], eps)
            ),
            "postclip_grad_ratio_100x_to_current": (
                hundred["postclip_grad_norm"]
                / jnp.maximum(current["postclip_grad_norm"], eps)
            ),
        }
    return {
        "critic_optimizer_diag_enabled": jnp.array(True),
        "critic_optimizer_diag_active": jnp.array(True),
        "critic_optimizer_diag_skipped_by_cadence": jnp.array(False),
        "critic_optimizer_diag_not_applicable": jnp.array(False),
        "reason_not_execution": jnp.array(False),
        "adam_internals_available": jnp.array(False),
        "max_grad_norm": max_grad_norm,
        "variants": variants,
        "scale_response": scale_response,
    }


def _empty_group_metrics(param_leaf_count: int) -> dict[str, jax.Array]:
    metrics = {
        metric: jnp.array(0.0, dtype=jnp.float32)
        for metric in GRADIENT_METRICS
        if metric not in {
            "param_leaf_count",
            "ppo_grad_nonzero",
            "survival_grad_nonzero",
            "cosine_valid",
        }
    }
    metrics.update(
        {
            "param_leaf_count": jnp.array(param_leaf_count, dtype=jnp.int32),
            "ppo_grad_nonzero": jnp.array(False),
            "survival_grad_nonzero": jnp.array(False),
            "cosine_valid": jnp.array(False),
        }
    )
    return metrics


def empty_gradient_interaction_diagnostics(
    params: Mapping[str, Any],
    *,
    enabled=False,
    skipped_by_cadence=False,
    not_applicable=False,
    reason_not_execution=False,
    reason_reliability_disabled=False,
    reason_survival_disabled=False,
    reason_phasic_ppo_only=False,
    survival_loss_pre_ppo=0.0,
) -> dict[str, Any]:
    """Return the fixed diagnostics pytree used by every control-flow branch."""
    counts = parameter_group_leaf_counts(params)
    return {
        "grad_diag_enabled": jnp.asarray(enabled, dtype=jnp.bool_),
        "grad_diag_active": jnp.array(False),
        "grad_diag_skipped_by_cadence": jnp.asarray(
            skipped_by_cadence,
            dtype=jnp.bool_,
        ),
        "grad_diag_not_applicable": jnp.asarray(not_applicable, dtype=jnp.bool_),
        "reason_not_execution": jnp.asarray(reason_not_execution, dtype=jnp.bool_),
        "reason_reliability_disabled": jnp.asarray(
            reason_reliability_disabled,
            dtype=jnp.bool_,
        ),
        "reason_survival_disabled": jnp.asarray(
            reason_survival_disabled,
            dtype=jnp.bool_,
        ),
        "reason_phasic_ppo_only": jnp.asarray(
            reason_phasic_ppo_only,
            dtype=jnp.bool_,
        ),
        "survival_loss_pre_ppo": jnp.asarray(
            survival_loss_pre_ppo,
            dtype=jnp.float32,
        ),
        "groups": {
            group: _empty_group_metrics(counts[group])
            for group in GRADIENT_GROUPS
        },
    }


def summarize_gradient_interaction(
    params: Mapping[str, Any],
    total_gradients: Mapping[str, Any],
    ppo_gradients: Mapping[str, Any],
    survival_gradients: Mapping[str, Any],
    lambda_surv: Any,
    survival_loss_pre_ppo=0.0,
    eps=1e-12,
) -> dict[str, Any]:
    """Summarize the numerical decomposition of the optimizer's total gradient."""
    weighted_survival = scale_gradient_tree(survival_gradients, lambda_surv)
    reconstructed = add_gradient_trees(ppo_gradients, weighted_survival)
    residual = subtract_gradient_trees(total_gradients, reconstructed)
    counts = parameter_group_leaf_counts(params)
    groups = {}
    for group in GRADIENT_GROUPS:
        ppo_norm = gradient_l2_norm(ppo_gradients, group)
        survival_norm = gradient_l2_norm(survival_gradients, group)
        weighted_survival_norm = gradient_l2_norm(weighted_survival, group)
        joint_norm = gradient_l2_norm(total_gradients, group)
        decomposition_abs_error = gradient_l2_norm(residual, group)
        cosine, cosine_valid = gradient_cosine(
            ppo_gradients,
            survival_gradients,
            group,
            eps,
        )
        groups[group] = {
            "param_leaf_count": jnp.array(counts[group], dtype=jnp.int32),
            "ppo_grad_norm": ppo_norm,
            "survival_grad_norm_raw": survival_norm,
            "survival_grad_norm_weighted": weighted_survival_norm,
            "weighted_survival_to_ppo_grad_ratio": (
                weighted_survival_norm / jnp.maximum(ppo_norm, eps)
            ),
            "ppo_survival_dot_raw": gradient_dot(
                ppo_gradients,
                survival_gradients,
                group,
            ),
            "ppo_survival_cosine_raw": cosine,
            "ppo_grad_nonzero": ppo_norm > eps,
            "survival_grad_nonzero": survival_norm > eps,
            "cosine_valid": cosine_valid,
            "joint_grad_norm": joint_norm,
            "decomposition_abs_error": decomposition_abs_error,
            "decomposition_rel_error": (
                decomposition_abs_error / jnp.maximum(joint_norm, eps)
            ),
        }
    return {
        "grad_diag_enabled": jnp.array(True),
        "grad_diag_active": jnp.array(True),
        "grad_diag_skipped_by_cadence": jnp.array(False),
        "grad_diag_not_applicable": jnp.array(False),
        "reason_not_execution": jnp.array(False),
        "reason_reliability_disabled": jnp.array(False),
        "reason_survival_disabled": jnp.array(False),
        "reason_phasic_ppo_only": jnp.array(False),
        "survival_loss_pre_ppo": jnp.asarray(
            survival_loss_pre_ppo,
            dtype=jnp.float32,
        ),
        "groups": groups,
    }


def summarize_phasic_gradient_interaction(
    params: Mapping[str, Any],
    ppo_gradients: Mapping[str, Any],
    survival_gradients: Mapping[str, Any],
    survival_loss_pre_ppo: Any,
    eps=1e-12,
) -> dict[str, Any]:
    """Summarize read-only PPO/survival gradients before phasic PPO updates."""
    diagnostics = empty_gradient_interaction_diagnostics(
        params,
        enabled=True,
        survival_loss_pre_ppo=survival_loss_pre_ppo,
    )
    diagnostics["grad_diag_active"] = jnp.array(True)
    counts = parameter_group_leaf_counts(params)
    for group in PHASIC_GRADIENT_GROUPS:
        ppo_norm = gradient_l2_norm(ppo_gradients, group)
        survival_norm = gradient_l2_norm(survival_gradients, group)
        cosine, cosine_valid = gradient_cosine(
            ppo_gradients,
            survival_gradients,
            group,
            eps,
        )
        diagnostics["groups"][group].update(
            {
                "param_leaf_count": jnp.array(counts[group], dtype=jnp.int32),
                "ppo_grad_norm": ppo_norm,
                "survival_grad_norm_raw": survival_norm,
                "ppo_survival_dot_raw": gradient_dot(
                    ppo_gradients,
                    survival_gradients,
                    group,
                ),
                "ppo_survival_cosine_raw": cosine,
                "ppo_grad_nonzero": ppo_norm > eps,
                "survival_grad_nonzero": survival_norm > eps,
                "cosine_valid": cosine_valid,
            }
        )
    return diagnostics


def gradient_diag_should_run(
    update_index: Any,
    cadence: int,
    epoch_index: Any,
    minibatch_index: Any,
):
    if int(cadence) < 1:
        raise ValueError("grad_interaction_diag_every_updates must be >= 1.")
    return (
        (jnp.asarray(update_index) % int(cadence) == 0)
        & (jnp.asarray(epoch_index) == 0)
        & (jnp.asarray(minibatch_index) == 0)
    )


def validate_gradient_diag_config(config: Mapping[str, Any]) -> int:
    cadence = int(config.get("grad_interaction_diag_every_updates", 1))
    if cadence < 1:
        raise ValueError("grad_interaction_diag_every_updates must be >= 1.")
    return cadence


def actor_critic_grad_diag_should_run(
    update_index: Any,
    cadence: int,
    epoch_index: Any,
    minibatch_index: Any,
):
    if int(cadence) < 1:
        raise ValueError(
            "actor_critic_grad_diag_every_updates must be >= 1."
        )
    return (
        (jnp.asarray(update_index) % int(cadence) == 0)
        & (jnp.asarray(epoch_index) == 0)
        & (jnp.asarray(minibatch_index) == 0)
    )


def validate_actor_critic_grad_diag_config(config: Mapping[str, Any]) -> int:
    cadence = int(config.get("actor_critic_grad_diag_every_updates", 1))
    if cadence < 1:
        raise ValueError(
            "actor_critic_grad_diag_every_updates must be >= 1."
        )
    return cadence


def value_clip_diag_should_run(
    update_index: Any,
    cadence: int,
    epoch_index: Any,
    minibatch_index: Any,
):
    if int(cadence) < 1:
        raise ValueError("value_clip_diag_every_updates must be >= 1.")
    return (
        (jnp.asarray(update_index) % int(cadence) == 0)
        & (jnp.asarray(epoch_index) == 0)
        & (jnp.asarray(minibatch_index) == 0)
    )


def validate_value_clip_diag_config(config: Mapping[str, Any]) -> int:
    cadence = int(config.get("value_clip_diag_every_updates", 1))
    if cadence < 1:
        raise ValueError("value_clip_diag_every_updates must be >= 1.")
    return cadence


def critic_optimizer_diag_should_run(
    update_index: Any,
    cadence: int,
    epoch_index: Any,
    minibatch_index: Any,
):
    if int(cadence) < 1:
        raise ValueError("critic_optimizer_diag_every_updates must be >= 1.")
    return (
        (jnp.asarray(update_index) % int(cadence) == 0)
        & (jnp.asarray(epoch_index) == 0)
        & (jnp.asarray(minibatch_index) == 0)
    )


def validate_critic_optimizer_diag_config(config: Mapping[str, Any]) -> int:
    cadence = int(config.get("critic_optimizer_diag_every_updates", 1))
    if cadence < 1:
        raise ValueError("critic_optimizer_diag_every_updates must be >= 1.")
    return cadence


def _host_scalar(value: Any) -> float:
    values = np.asarray(value)
    return float(np.mean(values))


def _host_bool(value: Any) -> bool:
    return _host_scalar(value) >= 0.5


def format_gradient_interaction_diagnostics(
    diagnostics: Mapping[str, Any],
    *,
    update: int,
    agent="EXE",
    optimization_mode="joint",
) -> tuple[list[str], dict[str, float]]:
    """Format scalar callback output for non-PMAP and replicated PMAP metrics."""
    enabled = _host_bool(diagnostics["grad_diag_enabled"])
    active = _host_bool(diagnostics["grad_diag_active"])
    skipped = _host_bool(diagnostics["grad_diag_skipped_by_cadence"])
    not_applicable = _host_bool(diagnostics["grad_diag_not_applicable"])
    status_metrics = {
        "enabled": float(enabled),
        "active": float(active),
        "skipped_by_cadence": float(skipped),
        "not_applicable": float(not_applicable),
    }
    if not enabled:
        return [f"GRAD_DIAG update={update} status=disabled"], status_metrics
    if not_applicable:
        reasons = []
        for key, label in (
            ("reason_not_execution", "not_execution_agent"),
            ("reason_reliability_disabled", "reliability_head_disabled"),
            ("reason_survival_disabled", "survival_loss_disabled"),
            ("reason_phasic_ppo_only", "phasic_ppo_only"),
        ):
            if _host_bool(diagnostics[key]):
                reasons.append(label)
        reason = ",".join(reasons) if reasons else "unknown"
        return [
            f"GRAD_DIAG update={update} status=not_applicable reason={reason}"
        ], status_metrics
    if skipped:
        return [
            f"GRAD_DIAG update={update} status=skipped_by_cadence"
        ], status_metrics
    if not active:
        return [
            f"GRAD_DIAG update={update} status=not_applicable "
            "reason=first_minibatch_not_observed"
        ], status_metrics

    if optimization_mode not in {"joint", "phasic"}:
        raise ValueError(
            "optimization_mode must be 'joint' or 'phasic'; "
            f"got {optimization_mode!r}."
        )
    groups = (
        PHASIC_GRADIENT_GROUPS
        if optimization_mode == "phasic"
        else GRADIENT_GROUPS
    )
    metric_names = (
        PHASIC_GRADIENT_METRICS
        if optimization_mode == "phasic"
        else GRADIENT_METRICS
    )

    lines = []
    wandb_metrics = dict(status_metrics)
    for group in groups:
        group_metrics = diagnostics["groups"][group]
        values = {
            key: _host_scalar(group_metrics[key])
            for key in metric_names
        }
        fields = [
            "GRAD_DIAG",
            f"update={update}",
            "status=active",
            f"agent={agent}",
            f"optimization_mode={optimization_mode}",
            "scope=first_minibatch_first_epoch",
            "params_state=pre_update",
            f"group={group}",
            f"param_leaf_count={int(round(values['param_leaf_count']))}",
            f"ppo_grad_norm={values['ppo_grad_norm']:.6g}",
            f"survival_grad_norm_raw={values['survival_grad_norm_raw']:.6g}",
            f"ppo_survival_dot_raw={values['ppo_survival_dot_raw']:.6g}",
            f"ppo_survival_cosine_raw={values['ppo_survival_cosine_raw']:.6g}",
        ]
        if optimization_mode == "joint":
            fields.extend(
                [
                    "survival_grad_norm_weighted="
                    f"{values['survival_grad_norm_weighted']:.6g}",
                    "weighted_survival_to_ppo_grad_ratio="
                    f"{values['weighted_survival_to_ppo_grad_ratio']:.6g}",
                    f"ppo_grad_nonzero={str(values['ppo_grad_nonzero'] >= 0.5).lower()}",
                    "survival_grad_nonzero="
                    f"{str(values['survival_grad_nonzero'] >= 0.5).lower()}",
                    f"cosine_valid={str(values['cosine_valid'] >= 0.5).lower()}",
                    f"joint_grad_norm={values['joint_grad_norm']:.6g}",
                    f"decomposition_abs_error={values['decomposition_abs_error']:.6g}",
                    f"decomposition_rel_error={values['decomposition_rel_error']:.6g}",
                ]
            )
        lines.append(" ".join(fields))
        for key, value in values.items():
            wandb_metrics[f"{group}/{key}"] = value
    return lines, wandb_metrics


def format_actor_critic_gradient_diagnostics(
    diagnostics: Mapping[str, Any],
    *,
    update: int,
    agent="EXE",
) -> tuple[list[str], dict[str, float]]:
    """Format scalar actor/value diagnostics outside the jitted update."""
    enabled = _host_bool(diagnostics["actor_critic_grad_diag_enabled"])
    active = _host_bool(diagnostics["actor_critic_grad_diag_active"])
    skipped = _host_bool(
        diagnostics["actor_critic_grad_diag_skipped_by_cadence"]
    )
    not_applicable = _host_bool(
        diagnostics["actor_critic_grad_diag_not_applicable"]
    )
    values = {
        "enabled": float(enabled),
        "active": float(active),
        "skipped_by_cadence": float(skipped),
        "not_applicable": float(not_applicable),
        "pre_clip_total_grad_norm": _host_scalar(
            diagnostics["pre_clip_total_grad_norm"]
        ),
        "max_grad_norm": _host_scalar(diagnostics["max_grad_norm"]),
        "estimated_global_clip_scale": _host_scalar(
            diagnostics["estimated_global_clip_scale"]
        ),
    }
    if not enabled:
        return [
            f"ACTOR_CRITIC_GRAD_DIAG update={update} status=disabled"
        ], values
    if not_applicable:
        reason = (
            "not_execution_agent"
            if _host_bool(diagnostics["reason_not_execution"])
            else "unknown"
        )
        return [
            f"ACTOR_CRITIC_GRAD_DIAG update={update} "
            f"status=not_applicable reason={reason}"
        ], values
    if skipped:
        return [
            f"ACTOR_CRITIC_GRAD_DIAG update={update} "
            "status=skipped_by_cadence"
        ], values
    if not active:
        return [
            f"ACTOR_CRITIC_GRAD_DIAG update={update} "
            "status=not_applicable reason=first_minibatch_not_observed"
        ], values

    lines = []
    for group in ACTOR_CRITIC_GRADIENT_GROUPS:
        group_values = {
            key: _host_scalar(diagnostics["groups"][group][key])
            for key in ACTOR_CRITIC_GRADIENT_METRICS
        }
        lines.append(
            " ".join(
                [
                    "ACTOR_CRITIC_GRAD_DIAG",
                    f"update={update}",
                    "status=active",
                    f"agent={agent}",
                    "scope=first_minibatch_first_epoch",
                    "params_state=pre_update",
                    f"group={group}",
                    "param_leaf_count="
                    f"{int(round(group_values['param_leaf_count']))}",
                    f"policy_grad_norm={group_values['policy_grad_norm']:.6g}",
                    "value_grad_norm_raw="
                    f"{group_values['value_grad_norm_raw']:.6g}",
                    "value_grad_norm_weighted="
                    f"{group_values['value_grad_norm_weighted']:.6g}",
                    "weighted_value_to_policy_grad_ratio="
                    f"{group_values['weighted_value_to_policy_grad_ratio']:.6g}",
                    "policy_value_dot_raw="
                    f"{group_values['policy_value_dot_raw']:.6g}",
                    "policy_value_cosine_raw="
                    f"{group_values['policy_value_cosine_raw']:.6g}",
                    "policy_grad_nonzero="
                    f"{str(group_values['policy_grad_nonzero'] >= 0.5).lower()}",
                    "value_grad_nonzero="
                    f"{str(group_values['value_grad_nonzero'] >= 0.5).lower()}",
                    "cosine_valid="
                    f"{str(group_values['cosine_valid'] >= 0.5).lower()}",
                    "policy_plus_value_grad_norm="
                    f"{group_values['policy_plus_value_grad_norm']:.6g}",
                    "combined_to_policy_grad_ratio="
                    f"{group_values['combined_to_policy_grad_ratio']:.6g}",
                    "ppo_decomposition_abs_error="
                    f"{group_values['ppo_decomposition_abs_error']:.6g}",
                    "ppo_decomposition_rel_error="
                    f"{group_values['ppo_decomposition_rel_error']:.6g}",
                    "pre_clip_total_grad_norm="
                    f"{values['pre_clip_total_grad_norm']:.6g}",
                    f"max_grad_norm={values['max_grad_norm']:.6g}",
                    "estimated_global_clip_scale="
                    f"{values['estimated_global_clip_scale']:.6g}",
                ]
            )
        )
        for key, value in group_values.items():
            values[f"{group}/{key}"] = value
    return lines, values


def format_value_clip_diagnostics(
    diagnostics: Mapping[str, Any],
    *,
    update: int,
    agent="EXE",
) -> tuple[list[str], dict[str, float]]:
    """Format scalar value-clipping diagnostics outside the jitted update."""
    enabled = _host_bool(diagnostics["value_clip_diag_enabled"])
    active = _host_bool(diagnostics["value_clip_diag_active"])
    skipped = _host_bool(
        diagnostics["value_clip_diag_skipped_by_cadence"]
    )
    not_applicable = _host_bool(
        diagnostics["value_clip_diag_not_applicable"]
    )
    values = {
        "enabled": float(enabled),
        "active": float(active),
        "skipped_by_cadence": float(skipped),
        "not_applicable": float(not_applicable),
    }
    if not enabled:
        return [f"VALUE_CLIP_DIAG update={update} status=disabled"], values
    if not_applicable:
        reason = (
            "not_execution_agent"
            if _host_bool(diagnostics["reason_not_execution"])
            else "unknown"
        )
        return [
            f"VALUE_CLIP_DIAG update={update} "
            f"status=not_applicable reason={reason}"
        ], values
    if skipped:
        return [
            f"VALUE_CLIP_DIAG update={update} status=skipped_by_cadence"
        ], values
    if not active:
        return [
            f"VALUE_CLIP_DIAG update={update} "
            "status=not_applicable reason=first_minibatch_not_observed"
        ], values

    metric_values = {
        key: _host_scalar(diagnostics[key])
        for key in VALUE_CLIP_DIAGNOSTIC_METRICS
    }
    fields = [
        "VALUE_CLIP_DIAG",
        f"update={update}",
        "status=active",
        f"agent={agent}",
        "scope=first_minibatch_first_epoch",
    ]
    fields.extend(f"{key}={value:.6g}" for key, value in metric_values.items())
    values.update(metric_values)
    return [" ".join(fields)], values


def format_critic_optimizer_diagnostics(
    diagnostics: Mapping[str, Any],
    *,
    update: int,
    agent="EXE",
) -> tuple[list[str], dict[str, float]]:
    """Format counterfactual optimizer diagnostics outside the JIT scan."""
    enabled = _host_bool(diagnostics["critic_optimizer_diag_enabled"])
    active = _host_bool(diagnostics["critic_optimizer_diag_active"])
    skipped = _host_bool(
        diagnostics["critic_optimizer_diag_skipped_by_cadence"]
    )
    not_applicable = _host_bool(
        diagnostics["critic_optimizer_diag_not_applicable"]
    )
    values = {
        "enabled": float(enabled),
        "active": float(active),
        "skipped_by_cadence": float(skipped),
        "not_applicable": float(not_applicable),
        "max_grad_norm": _host_scalar(diagnostics["max_grad_norm"]),
        "adam_internals_available": float(
            _host_bool(diagnostics["adam_internals_available"])
        ),
    }
    if not enabled:
        return [
            f"CRITIC_OPTIMIZER_DIAG update={update} status=disabled"
        ], values
    if not_applicable:
        reason = (
            "not_execution_agent"
            if _host_bool(diagnostics["reason_not_execution"])
            else "unknown"
        )
        return [
            f"CRITIC_OPTIMIZER_DIAG update={update} "
            f"status=not_applicable reason={reason}"
        ], values
    if skipped:
        return [
            f"CRITIC_OPTIMIZER_DIAG update={update} "
            "status=skipped_by_cadence"
        ], values
    if not active:
        return [
            f"CRITIC_OPTIMIZER_DIAG update={update} "
            "status=not_applicable reason=first_minibatch_not_observed"
        ], values

    lines = []
    for variant in CRITIC_OPTIMIZER_DIAGNOSTIC_VARIANTS:
        variant_diag = diagnostics["variants"][variant]
        variant_values = {
            metric: _host_scalar(variant_diag[metric])
            for metric in CRITIC_OPTIMIZER_VARIANT_METRICS
        }
        for metric, value in variant_values.items():
            values[f"{variant}/{metric}"] = value
        for group in CRITIC_OPTIMIZER_DIAGNOSTIC_GROUPS:
            group_values = {
                metric: _host_scalar(variant_diag["groups"][group][metric])
                for metric in CRITIC_OPTIMIZER_GROUP_METRICS
            }
            lines.append(
                " ".join(
                    [
                        "CRITIC_OPTIMIZER_DIAG",
                        f"update={update}",
                        "status=active",
                        f"agent={agent}",
                        "scope=first_minibatch_first_epoch",
                        "params_state=pre_update",
                        f"variant={variant}",
                        f"group={group}",
                        f"coefficient={variant_values['coefficient']:.6g}",
                        "preclip_grad_norm="
                        f"{group_values['preclip_grad_norm']:.6g}",
                        "postclip_grad_norm="
                        f"{group_values['postclip_grad_norm']:.6g}",
                        "param_step_norm="
                        f"{group_values['param_step_norm']:.6g}",
                        f"param_norm={group_values['param_norm']:.6g}",
                        "relative_param_step="
                        f"{group_values['relative_param_step']:.6g}",
                        "preclip_total_grad_norm="
                        f"{variant_values['preclip_total_grad_norm']:.6g}",
                        "estimated_global_clip_scale="
                        f"{variant_values['estimated_global_clip_scale']:.6g}",
                        "postclip_total_grad_norm="
                        f"{variant_values['postclip_total_grad_norm']:.6g}",
                        "global_clip_active="
                        f"{str(variant_values['global_clip_active'] >= 0.5).lower()}",
                    ]
                )
            )
            for metric, value in group_values.items():
                values[f"{variant}/{metric}/{group}"] = value

    for group in CRITIC_OPTIMIZER_DIAGNOSTIC_GROUPS:
        response = diagnostics["scale_response"][group]
        response_values = {
            metric: _host_scalar(response[metric])
            for metric in CRITIC_OPTIMIZER_SCALE_RESPONSE_METRICS
        }
        lines.append(
            " ".join(
                [
                    "CRITIC_OPTIMIZER_SCALE_RESPONSE",
                    f"update={update}",
                    "status=active",
                    f"agent={agent}",
                    f"group={group}",
                ]
                + [
                    f"{metric}={value:.6g}"
                    for metric, value in response_values.items()
                ]
            )
        )
        for metric, value in response_values.items():
            values[f"scale_response/{metric}/{group}"] = value
    return lines, values


def format_value_representation_probe_diagnostics(
    diagnostics: Mapping[str, Any],
    *,
    update: int,
    agent="EXE",
) -> tuple[list[str], dict[str, float]]:
    """Format probe scalars outside the jitted training update."""
    enabled = _host_bool(diagnostics["enabled"])
    active = _host_bool(diagnostics["active"])
    skipped = _host_bool(diagnostics["skipped_by_schedule"])
    not_applicable = _host_bool(diagnostics["not_applicable"])
    values = {
        "enabled": float(enabled),
        "active": float(active),
        "skipped_by_schedule": float(skipped),
        "not_applicable": float(not_applicable),
        "steps": _host_scalar(diagnostics["steps"]),
        "learning_rate": _host_scalar(diagnostics["learning_rate"]),
        "train_fraction": _host_scalar(diagnostics["train_fraction"]),
        "split_seed": _host_scalar(diagnostics["split_seed"]),
        "train_environment_count": _host_scalar(
            diagnostics["train_environment_count"]
        ),
        "holdout_environment_count": _host_scalar(
            diagnostics["holdout_environment_count"]
        ),
        "probes_start_identical": float(
            _host_bool(diagnostics["probes_start_identical"])
        ),
    }
    if not enabled:
        return [f"VALUE_REP_PROBE update={update} status=disabled"], values
    if not_applicable:
        return [
            f"VALUE_REP_PROBE update={update} status=not_applicable"
        ], values
    if skipped:
        return [
            f"VALUE_REP_PROBE update={update} status=skipped_by_schedule"
        ], values
    if not active:
        return [f"VALUE_REP_PROBE update={update} status=inactive"], values

    lines = []
    statistic_names = tuple(_empty_value_probe_statistics())
    for probe_name in ("probe_a", "probe_b"):
        for split_name in ("train", "holdout"):
            for stage_name in ("initial", "final"):
                statistics = diagnostics[probe_name][split_name][stage_name]
                statistic_values = {
                    name: _host_scalar(statistics[name])
                    for name in statistic_names
                }
                prefix = f"{probe_name}/{split_name}/{stage_name}"
                values.update(
                    {
                        f"{prefix}/{name}": value
                        for name, value in statistic_values.items()
                    }
                )
                lines.append(
                    " ".join(
                        [
                            "VALUE_REP_PROBE",
                            f"update={update}",
                            "status=active",
                            f"agent={agent}",
                            f"probe={probe_name}",
                            f"split={split_name}",
                            f"stage={stage_name}",
                        ]
                        + [
                            f"{name}={value:.6g}"
                            for name, value in statistic_values.items()
                        ]
                    )
                )
        for group, value in diagnostics[probe_name]["parameter_change"].items():
            values[f"{probe_name}/relative_param_change/{group}"] = (
                _host_scalar(value)
            )

    comparison_values = {
        name: _host_scalar(value)
        for name, value in diagnostics["comparison"].items()
    }
    values.update(
        {f"comparison/{name}": value for name, value in comparison_values.items()}
    )
    lines.append(
        " ".join(
            [
                "VALUE_REP_PROBE_COMPARISON",
                f"update={update}",
                "status=active",
                f"agent={agent}",
            ]
            + [f"{name}={value:.6g}" for name, value in comparison_values.items()]
        )
    )
    return lines, values
