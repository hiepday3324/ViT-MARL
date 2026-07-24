"""Shared JAX utilities for PPO/survival gradient interaction diagnostics."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict


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

_PARAMS = ("params",)
_RELIABILITY_HEAD = _PARAMS + (
    "ReliabilityFusionRNN_0",
    "LevelWiseReliabilityHead_0",
)
_VISION_ENCODER = _PARAMS + ("VisionAgent_0",)
_RELIABILITY_FUSION = _PARAMS + ("ReliabilityFusionRNN_0",)
_ACTOR_DENSE_MODULES = frozenset(("Dense_0", "Dense_1"))
_CRITIC_DENSE_MODULES = frozenset(("Dense_2", "Dense_3"))


PARAMETER_GROUP_RULES = {
    "total": "all parameter leaves",
    "reliability_head": "prefix=params/ReliabilityFusionRNN_0/LevelWiseReliabilityHead_0",
    "vision_encoder": "prefix=params/VisionAgent_0",
    "fusion_shared_trunk": (
        "prefix=params/ReliabilityFusionRNN_0 excluding "
        "LevelWiseReliabilityHead_0"
    ),
    "actor_head": "prefix=params/Dense_0 or params/Dense_1; exact=params/log_std",
    "critic_head": "prefix=params/Dense_2 or params/Dense_3",
}

AUXILIARY_TRAINABLE_GROUPS = (
    "reliability_head",
    "vision_encoder",
    "fusion_shared_trunk",
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
        return _has_prefix(path, _RELIABILITY_FUSION) and not _has_prefix(
            path,
            _RELIABILITY_HEAD,
        )
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
        for group in GRADIENT_GROUPS
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
        "groups": groups,
    }


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

    lines = []
    wandb_metrics = dict(status_metrics)
    for group in GRADIENT_GROUPS:
        group_metrics = diagnostics["groups"][group]
        values = {
            key: _host_scalar(group_metrics[key])
            for key in GRADIENT_METRICS
        }
        lines.append(
            " ".join(
                [
                    "GRAD_DIAG",
                    f"update={update}",
                    "status=active",
                    f"agent={agent}",
                    "scope=first_minibatch_first_epoch",
                    "params_state=pre_update",
                    f"group={group}",
                    f"param_leaf_count={int(round(values['param_leaf_count']))}",
                    f"ppo_grad_norm={values['ppo_grad_norm']:.6g}",
                    f"survival_grad_norm_raw={values['survival_grad_norm_raw']:.6g}",
                    f"survival_grad_norm_weighted={values['survival_grad_norm_weighted']:.6g}",
                    "weighted_survival_to_ppo_grad_ratio="
                    f"{values['weighted_survival_to_ppo_grad_ratio']:.6g}",
                    f"ppo_survival_dot_raw={values['ppo_survival_dot_raw']:.6g}",
                    f"ppo_survival_cosine_raw={values['ppo_survival_cosine_raw']:.6g}",
                    f"ppo_grad_nonzero={str(values['ppo_grad_nonzero'] >= 0.5).lower()}",
                    "survival_grad_nonzero="
                    f"{str(values['survival_grad_nonzero'] >= 0.5).lower()}",
                    f"cosine_valid={str(values['cosine_valid'] >= 0.5).lower()}",
                    f"joint_grad_norm={values['joint_grad_norm']:.6g}",
                    f"decomposition_abs_error={values['decomposition_abs_error']:.6g}",
                    f"decomposition_rel_error={values['decomposition_rel_error']:.6g}",
                ]
            )
        )
        for key, value in values.items():
            wandb_metrics[f"{group}/{key}"] = value
    return lines, wandb_metrics
