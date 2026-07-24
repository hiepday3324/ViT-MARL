"""Shared utilities for constrained phasic Reliability optimization.

The auxiliary phase updates the existing parameter tree with a separate
optimizer state. It never owns a second copy of model parameters and never
advances the PPO optimizer state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from gymnax_exchange.jaxrl.MARL.gradient_diagnostics import (
    AUXILIARY_TRAINABLE_GROUPS,
    GRADIENT_GROUPS,
    gradient_l2_norm,
    mask_tree_to_groups,
)
from gymnax_exchange.jaxrl.MARL.reliability_targets import (
    masked_reliability_loss,
)


@dataclass(frozen=True)
class PhasicReliabilitySettings:
    mode: str
    num_epochs: int
    num_minibatches: int
    survival_coef: float
    kl_coef: float
    target_kl: float
    early_stop_on_kl: bool
    reject_step_on_kl: bool
    learning_rate: float
    max_grad_norm: float
    anneal_lr: bool

    @property
    def enabled(self) -> bool:
        return self.mode == "phasic"


class PolicyStatistics(NamedTuple):
    logits: jax.Array
    loc: jax.Array
    log_std: jax.Array


class RolloutOutputs(NamedTuple):
    policy: PolicyStatistics
    value: jax.Array
    reliability_logits: jax.Array
    reliability_scores: jax.Array


class AuxiliaryBatch(NamedTuple):
    init_hstate: jax.Array
    obs: Any
    done: jax.Array
    labels: jax.Array
    mask: jax.Array
    reference_policy: PolicyStatistics
    reference_value: jax.Array


PHASIC_DIAGNOSTIC_FIELDS = (
    "phasic_aux_active",
    "policy_is_discrete",
    "policy_is_box",
    "aux_survival_coef",
    "aux_kl_coef",
    "aux_target_kl",
    "survival_loss_before_aux",
    "survival_loss_after_aux",
    "survival_mae_before_aux",
    "survival_mae_after_aux",
    "reliability_mean_before_aux",
    "reliability_mean_after_aux",
    "reliability_std_before_aux",
    "reliability_std_after_aux",
    "reliability_min_before_aux",
    "reliability_max_before_aux",
    "reliability_min_after_aux",
    "reliability_max_after_aux",
    "target_mean",
    "target_std",
    "policy_kl_before_aux",
    "full_rollout_policy_kl_after_aux",
    "last_candidate_policy_kl",
    "categorical_mean_abs_logit_change",
    "categorical_max_abs_logit_change",
    "box_mean_abs_loc_change",
    "box_max_abs_loc_change",
    "box_mean_abs_logstd_change",
    "box_max_abs_logstd_change",
    "value_prediction_mse_after_aux",
    "mean_abs_value_change",
    "max_abs_value_change",
    "aux_grad_norm_raw_total",
    "aux_grad_norm_applied_total",
    "aux_grad_norm_reliability_head",
    "aux_grad_norm_vision_encoder",
    "aux_grad_norm_fusion_shared_trunk",
    "aux_grad_norm_actor_head",
    "aux_grad_norm_critic_head",
    "aux_grad_norm_raw_reliability_head",
    "aux_grad_norm_raw_vision_encoder",
    "aux_grad_norm_raw_fusion_shared_trunk",
    "aux_grad_norm_raw_actor_head",
    "aux_grad_norm_raw_critic_head",
    "aux_steps_attempted",
    "aux_steps_accepted",
    "aux_steps_rejected",
    "aux_early_stopped",
)


def resolve_phasic_reliability_settings(
    config: Mapping[str, Any],
    *,
    execution_index: int | None,
    execution_actor_count: int | None,
) -> PhasicReliabilitySettings:
    """Validate and resolve host-side phasic settings."""
    mode = str(config.get("reliability_optimization_mode", "joint")).lower()
    if mode not in {"joint", "phasic"}:
        raise ValueError(
            "reliability_optimization_mode must be 'joint' or 'phasic'; "
            f"got {mode!r}."
        )

    epochs = int(config.get("aux_reliability_epochs", 1))
    if epochs < 1:
        raise ValueError("aux_reliability_epochs must be >= 1.")

    configured_minibatches = config.get("aux_reliability_num_minibatches", None)
    num_minibatches = int(
        config["NUM_MINIBATCHES"]
        if configured_minibatches is None
        else configured_minibatches
    )
    if num_minibatches < 1:
        raise ValueError("aux_reliability_num_minibatches must be >= 1.")

    survival_coef = float(config.get("aux_survival_coef", 1.0))
    kl_coef = float(config.get("aux_kl_coef", 1.0))
    if survival_coef < 0:
        raise ValueError("aux_survival_coef must be >= 0.")
    if kl_coef < 0:
        raise ValueError("aux_kl_coef must be >= 0.")

    target_kl = float(config.get("aux_target_kl", 0.005))
    reject_step = bool(config.get("aux_reject_step_on_kl", True))
    early_stop = bool(config.get("aux_early_stop_on_kl", True))
    if (reject_step or early_stop) and target_kl <= 0:
        raise ValueError("aux_target_kl must be > 0 when the KL guard is enabled.")

    if mode == "phasic":
        if execution_index is None or execution_actor_count is None:
            raise ValueError("Phasic Reliability optimization requires an Execution Agent.")
        if not bool(config.get("use_reliability_head", False)):
            raise ValueError("Phasic mode requires use_reliability_head=true.")
        if not bool(config.get("use_survival_loss", False)):
            raise ValueError("Phasic mode requires use_survival_loss=true.")
        if execution_actor_count % num_minibatches != 0:
            raise ValueError(
                "Execution actor count must be divisible by "
                "aux_reliability_num_minibatches: "
                f"{execution_actor_count} % {num_minibatches} != 0."
            )

    if execution_index is None:
        default_lr = 0.0
        default_max_grad_norm = 1.0
    else:
        default_lr = float(config["LR"][execution_index])
        default_max_grad_norm = float(config["MAX_GRAD_NORM"][execution_index])
    configured_lr = config.get("aux_learning_rate", None)
    learning_rate = default_lr if configured_lr is None else float(configured_lr)
    configured_max_norm = config.get("aux_max_grad_norm", None)
    max_grad_norm = (
        default_max_grad_norm
        if configured_max_norm is None
        else float(configured_max_norm)
    )
    if mode == "phasic" and learning_rate <= 0:
        raise ValueError("aux_learning_rate must resolve to a positive number.")
    if max_grad_norm <= 0:
        raise ValueError("aux_max_grad_norm must resolve to a positive number.")

    return PhasicReliabilitySettings(
        mode=mode,
        num_epochs=epochs,
        num_minibatches=num_minibatches,
        survival_coef=survival_coef,
        kl_coef=kl_coef,
        target_kl=target_kl,
        early_stop_on_kl=early_stop,
        reject_step_on_kl=reject_step,
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
        anneal_lr=bool(config.get("aux_anneal_lr", False)),
    )


def make_auxiliary_optimizer(
    settings: PhasicReliabilitySettings,
    *,
    total_updates: int,
):
    """Create an optimizer whose schedule/count is independent from PPO."""
    learning_rate: Any = settings.learning_rate
    if settings.anneal_lr:
        transition_steps = max(
            int(total_updates) * settings.num_epochs * settings.num_minibatches,
            1,
        )
        learning_rate = optax.linear_schedule(
            init_value=settings.learning_rate,
            end_value=0.0,
            transition_steps=transition_steps,
        )
    return optax.chain(
        optax.clip_by_global_norm(settings.max_grad_norm),
        optax.adam(learning_rate, eps=1e-5),
    )


def ppo_survival_loss_weight(
    settings: PhasicReliabilitySettings,
    configured_lambda: float,
) -> float:
    """Keep legacy joint weighting and isolate PPO in phasic mode."""
    return 0.0 if settings.enabled else float(configured_lambda)


def categorical_policy_kl(reference_logits, current_logits):
    """Mean ``KL(reference || current)`` for a categorical policy."""
    reference_logits = jnp.asarray(reference_logits, dtype=jnp.float32)
    current_logits = jnp.asarray(current_logits, dtype=jnp.float32)
    reference_log_prob = jax.nn.log_softmax(reference_logits, axis=-1)
    current_log_prob = jax.nn.log_softmax(current_logits, axis=-1)
    reference_prob = jnp.exp(reference_log_prob)
    per_sample = jnp.sum(
        reference_prob * (reference_log_prob - current_log_prob),
        axis=-1,
    )
    finite = jnp.all(jnp.isfinite(per_sample))
    mean_kl = jnp.maximum(jnp.mean(per_sample), 0.0)
    return jnp.where(finite, mean_kl, jnp.inf)


def diagonal_normal_policy_kl(
    reference_loc,
    reference_log_std,
    current_loc,
    current_log_std,
):
    """Analytic mean KL for diagonal Normal base distributions."""
    reference_loc = jnp.asarray(reference_loc, dtype=jnp.float32)
    current_loc = jnp.asarray(current_loc, dtype=jnp.float32)
    reference_log_std = jnp.broadcast_to(
        jnp.asarray(reference_log_std, dtype=jnp.float32),
        reference_loc.shape,
    )
    current_log_std = jnp.broadcast_to(
        jnp.asarray(current_log_std, dtype=jnp.float32),
        current_loc.shape,
    )
    reference_variance = jnp.exp(2.0 * reference_log_std)
    current_variance = jnp.exp(2.0 * current_log_std)
    per_dimension = 0.5 * (
        2.0 * (current_log_std - reference_log_std)
        + (
            reference_variance
            + jnp.square(reference_loc - current_loc)
        )
        / current_variance
        - 1.0
    )
    per_sample = jnp.sum(per_dimension, axis=-1)
    finite = jnp.all(jnp.isfinite(per_sample))
    mean_kl = jnp.maximum(jnp.mean(per_sample), 0.0)
    return jnp.where(finite, mean_kl, jnp.inf)


def policy_kl(
    reference: PolicyStatistics,
    current: PolicyStatistics,
    *,
    is_discrete: bool,
):
    if is_discrete:
        return categorical_policy_kl(reference.logits, current.logits)
    return diagonal_normal_policy_kl(
        reference.loc,
        reference.log_std,
        current.loc,
        current.log_std,
    )


def policy_statistics_from_aux(
    aux_info: Mapping[str, Any],
    *,
    is_discrete: bool,
) -> PolicyStatistics:
    """Normalize model policy arrays into a static policy-statistics pytree."""
    if is_discrete:
        logits = jnp.asarray(aux_info["policy_logits"], dtype=jnp.float32)
        placeholder = jnp.zeros(logits.shape[:-1] + (1,), dtype=logits.dtype)
        return PolicyStatistics(logits, placeholder, placeholder)
    loc = jnp.asarray(aux_info["policy_loc"], dtype=jnp.float32)
    log_std = jnp.broadcast_to(
        jnp.asarray(aux_info["policy_log_std"], dtype=jnp.float32),
        loc.shape,
    )
    placeholder = jnp.zeros(loc.shape[:-1] + (1,), dtype=loc.dtype)
    return PolicyStatistics(placeholder, loc, log_std)


def build_rollout_outputs(
    apply_fn,
    params,
    init_hstate,
    obs,
    done,
    *,
    is_discrete: bool,
) -> RolloutOutputs:
    """Run the canonical full recurrent rollout used by phasic diagnostics."""
    _hidden, _pi, value, _z_vision, aux_info = apply_fn(
        params,
        init_hstate,
        (obs, done),
    )
    return RolloutOutputs(
        policy=policy_statistics_from_aux(aux_info, is_discrete=is_discrete),
        value=jnp.asarray(value, dtype=jnp.float32),
        reliability_logits=jnp.asarray(
            aux_info["reliability_logits"],
            dtype=jnp.float32,
        ),
        reliability_scores=jnp.asarray(
            aux_info["reliability_scores"],
            dtype=jnp.float32,
        ),
    )


def stop_gradient_rollout_outputs(outputs: RolloutOutputs) -> RolloutOutputs:
    return jax.tree_util.tree_map(jax.lax.stop_gradient, outputs)


def _aligned_reliability_tensor(value, labels, name):
    value = jnp.asarray(value, dtype=jnp.float32)
    if value.shape == labels.shape + (1,):
        return jnp.squeeze(value, axis=-1)
    if value.shape == labels.shape:
        return value
    raise ValueError(
        f"{name} must match labels shape with an optional trailing singleton; "
        f"got {value.shape} versus {labels.shape}."
    )


def _masked_summary(
    scores,
    logits,
    labels,
    mask,
    *,
    loss_type: str,
    eps: float,
    axis_name: str | None,
):
    scores = _aligned_reliability_tensor(scores, labels, "reliability_scores")
    logits = _aligned_reliability_tensor(logits, labels, "reliability_logits")
    labels = jnp.asarray(labels, dtype=jnp.float32)
    mask = jnp.asarray(mask, dtype=jnp.float32)
    count = jnp.sum(mask)
    score_sum = jnp.sum(scores * mask)
    score_square_sum = jnp.sum(jnp.square(scores) * mask)
    target_sum = jnp.sum(labels * mask)
    target_square_sum = jnp.sum(jnp.square(labels) * mask)
    mae_sum = jnp.sum(jnp.abs(scores - labels) * mask)
    if loss_type == "bce":
        element_loss = optax.sigmoid_binary_cross_entropy(logits, labels)
    elif loss_type == "mse":
        element_loss = jnp.square(scores - labels)
    else:
        raise ValueError(f"Unknown reliability_loss_type: {loss_type}")
    loss_sum = jnp.sum(element_loss * mask)
    local_min = jnp.min(jnp.where(mask > 0, scores, jnp.inf))
    local_max = jnp.max(jnp.where(mask > 0, scores, -jnp.inf))

    if axis_name is not None:
        count = jax.lax.psum(count, axis_name)
        score_sum = jax.lax.psum(score_sum, axis_name)
        score_square_sum = jax.lax.psum(score_square_sum, axis_name)
        target_sum = jax.lax.psum(target_sum, axis_name)
        target_square_sum = jax.lax.psum(target_square_sum, axis_name)
        mae_sum = jax.lax.psum(mae_sum, axis_name)
        loss_sum = jax.lax.psum(loss_sum, axis_name)
        local_min = jax.lax.pmin(local_min, axis_name)
        local_max = jax.lax.pmax(local_max, axis_name)

    safe_count = jnp.maximum(count, eps)
    score_mean = score_sum / safe_count
    target_mean = target_sum / safe_count
    score_variance = jnp.maximum(score_square_sum / safe_count - score_mean**2, 0.0)
    target_variance = jnp.maximum(target_square_sum / safe_count - target_mean**2, 0.0)
    has_values = count > 0
    return {
        "loss": jnp.where(has_values, loss_sum / safe_count, 0.0),
        "mae": jnp.where(has_values, mae_sum / safe_count, 0.0),
        "score_mean": jnp.where(has_values, score_mean, 0.0),
        "score_std": jnp.where(has_values, jnp.sqrt(score_variance), 0.0),
        "score_min": jnp.where(has_values, local_min, 0.0),
        "score_max": jnp.where(has_values, local_max, 0.0),
        "target_mean": jnp.where(has_values, target_mean, 0.0),
        "target_std": jnp.where(has_values, jnp.sqrt(target_variance), 0.0),
        "valid_count": count,
    }


def policy_change_metrics(
    reference: PolicyStatistics,
    current: PolicyStatistics,
    *,
    is_discrete: bool,
    axis_name: str | None = None,
):
    if is_discrete:
        categorical_change = jnp.abs(current.logits - reference.logits)
        categorical_mean = jnp.mean(categorical_change)
        categorical_max = jnp.max(categorical_change)
        box_loc_mean = jnp.array(0.0, dtype=jnp.float32)
        box_loc_max = jnp.array(0.0, dtype=jnp.float32)
        box_logstd_mean = jnp.array(0.0, dtype=jnp.float32)
        box_logstd_max = jnp.array(0.0, dtype=jnp.float32)
    else:
        box_loc_change = jnp.abs(current.loc - reference.loc)
        box_logstd_change = jnp.abs(current.log_std - reference.log_std)
        categorical_mean = jnp.array(0.0, dtype=jnp.float32)
        categorical_max = jnp.array(0.0, dtype=jnp.float32)
        box_loc_mean = jnp.mean(box_loc_change)
        box_loc_max = jnp.max(box_loc_change)
        box_logstd_mean = jnp.mean(box_logstd_change)
        box_logstd_max = jnp.max(box_logstd_change)
    values = {
        "categorical_mean_abs_logit_change": categorical_mean,
        "categorical_max_abs_logit_change": categorical_max,
        "box_mean_abs_loc_change": box_loc_mean,
        "box_max_abs_loc_change": box_loc_max,
        "box_mean_abs_logstd_change": box_logstd_mean,
        "box_max_abs_logstd_change": box_logstd_max,
    }
    if axis_name is not None:
        for key in (
            "categorical_mean_abs_logit_change",
            "box_mean_abs_loc_change",
            "box_mean_abs_logstd_change",
        ):
            values[key] = jax.lax.pmean(values[key], axis_name)
        for key in (
            "categorical_max_abs_logit_change",
            "box_max_abs_loc_change",
            "box_max_abs_logstd_change",
        ):
            values[key] = jax.lax.pmax(values[key], axis_name)
    return values


def empty_phasic_aux_diagnostics(
    *,
    is_discrete: bool = False,
    settings: PhasicReliabilitySettings | None = None,
):
    diagnostics = {
        key: jnp.array(0.0, dtype=jnp.float32)
        for key in PHASIC_DIAGNOSTIC_FIELDS
    }
    diagnostics["policy_is_discrete"] = jnp.array(float(is_discrete), dtype=jnp.float32)
    diagnostics["policy_is_box"] = jnp.array(float(not is_discrete), dtype=jnp.float32)
    if settings is not None:
        diagnostics["aux_survival_coef"] = jnp.array(settings.survival_coef, dtype=jnp.float32)
        diagnostics["aux_kl_coef"] = jnp.array(settings.kl_coef, dtype=jnp.float32)
        diagnostics["aux_target_kl"] = jnp.array(settings.target_kl, dtype=jnp.float32)
    return diagnostics


def _tree_select(predicate, accepted, rejected):
    return jax.tree_util.tree_map(
        lambda accept_value, reject_value: jnp.where(
            predicate,
            accept_value,
            reject_value,
        ),
        accepted,
        rejected,
    )


def _split_actor_minibatches(batch, permutation, num_minibatches):
    shuffled = jax.tree_util.tree_map(
        lambda value: jnp.take(value, permutation, axis=1),
        batch,
    )

    def _split(value):
        reshaped = jnp.reshape(
            value,
            (value.shape[0], num_minibatches, -1) + value.shape[2:],
        )
        return jnp.swapaxes(reshaped, 0, 1)

    return jax.tree_util.tree_map(_split, shuffled)


def run_phasic_auxiliary_phase(
    *,
    apply_fn,
    params,
    aux_opt_state,
    aux_tx,
    init_hstate,
    obs,
    done,
    labels,
    mask,
    rng,
    settings: PhasicReliabilitySettings,
    is_discrete: bool,
    reliability_loss_type: str,
    survival_eps: float,
    axis_name: str | None = None,
):
    """Run finite phasic auxiliary epochs and return accepted parameters only."""
    if not settings.enabled:
        raise ValueError("run_phasic_auxiliary_phase requires mode='phasic'.")
    actor_count = int(labels.shape[1])
    if actor_count % settings.num_minibatches != 0:
        raise ValueError(
            f"Actor count {actor_count} is not divisible by "
            f"{settings.num_minibatches} auxiliary minibatches."
        )

    reference = stop_gradient_rollout_outputs(
        build_rollout_outputs(
            apply_fn,
            params,
            init_hstate,
            obs,
            done,
            is_discrete=is_discrete,
        )
    )
    before_summary = _masked_summary(
        reference.reliability_scores,
        reference.reliability_logits,
        labels,
        mask,
        loss_type=reliability_loss_type,
        eps=survival_eps,
        axis_name=axis_name,
    )

    full_batch = AuxiliaryBatch(
        init_hstate=init_hstate[jnp.newaxis, ...],
        obs=obs,
        done=done,
        labels=labels,
        mask=mask,
        reference_policy=reference.policy,
        reference_value=reference.value,
    )
    grad_sum_template = {
        f"raw_{group}": jnp.array(0.0, dtype=jnp.float32)
        for group in GRADIENT_GROUPS
    }
    grad_sum_template.update(
        {
            f"applied_{group}": jnp.array(0.0, dtype=jnp.float32)
            for group in GRADIENT_GROUPS
        }
    )
    initial_carry = (
        params,
        aux_opt_state,
        rng,
        jnp.array(False),
        jnp.array(0, dtype=jnp.int32),
        jnp.array(0, dtype=jnp.int32),
        jnp.array(0, dtype=jnp.int32),
        jnp.array(0.0, dtype=jnp.float32),
        grad_sum_template,
    )

    def _epoch_step(carry, _epoch_index):
        (
            current_params,
            current_aux_state,
            current_rng,
            early_stopped,
            attempted,
            accepted,
            rejected,
            last_candidate_kl,
            grad_sums,
        ) = carry
        current_rng, permutation_rng = jax.random.split(current_rng)
        permutation = jax.random.permutation(permutation_rng, actor_count)
        minibatches = _split_actor_minibatches(
            full_batch,
            permutation,
            settings.num_minibatches,
        )

        def _minibatch_step(minibatch_carry, minibatch):
            (
                step_params,
                step_aux_state,
                step_stopped,
                step_attempted,
                step_accepted,
                step_rejected,
                step_last_candidate_kl,
                step_grad_sums,
            ) = minibatch_carry

            def _attempt_step(_):
                minibatch_init_hstate = jnp.squeeze(
                    minibatch.init_hstate,
                    axis=0,
                )

                def _auxiliary_objective(candidate_params):
                    outputs = build_rollout_outputs(
                        apply_fn,
                        candidate_params,
                        minibatch_init_hstate,
                        minibatch.obs,
                        minibatch.done,
                        is_discrete=is_discrete,
                    )
                    survival_loss = masked_reliability_loss(
                        outputs.reliability_scores,
                        minibatch.labels,
                        minibatch.mask,
                        loss_type=reliability_loss_type,
                        eps=survival_eps,
                        reliability_logits=outputs.reliability_logits,
                    )
                    kl = policy_kl(
                        minibatch.reference_policy,
                        outputs.policy,
                        is_discrete=is_discrete,
                    )
                    total = (
                        settings.survival_coef * survival_loss
                        + settings.kl_coef * kl
                    )
                    return total, (survival_loss, kl)

                (_objective, _loss_parts), raw_gradients = jax.value_and_grad(
                    _auxiliary_objective,
                    has_aux=True,
                )(step_params)
                if axis_name is not None:
                    raw_gradients = jax.lax.pmean(raw_gradients, axis_name)
                applied_gradients = mask_tree_to_groups(
                    raw_gradients,
                    AUXILIARY_TRAINABLE_GROUPS,
                )
                candidate_updates, candidate_aux_state = aux_tx.update(
                    applied_gradients,
                    step_aux_state,
                    step_params,
                )
                candidate_params = optax.apply_updates(
                    step_params,
                    candidate_updates,
                )
                candidate_outputs = build_rollout_outputs(
                    apply_fn,
                    candidate_params,
                    minibatch_init_hstate,
                    minibatch.obs,
                    minibatch.done,
                    is_discrete=is_discrete,
                )
                candidate_kl = policy_kl(
                    minibatch.reference_policy,
                    candidate_outputs.policy,
                    is_discrete=is_discrete,
                )
                if axis_name is not None:
                    candidate_kl = jax.lax.pmean(candidate_kl, axis_name)
                reject_candidate = (
                    jnp.asarray(settings.reject_step_on_kl)
                    & (candidate_kl > settings.target_kl)
                )
                accept_candidate = ~reject_candidate
                next_params = _tree_select(
                    accept_candidate,
                    candidate_params,
                    step_params,
                )
                next_aux_state = _tree_select(
                    accept_candidate,
                    candidate_aux_state,
                    step_aux_state,
                )
                next_stopped = step_stopped | (
                    reject_candidate & jnp.asarray(settings.early_stop_on_kl)
                )
                step_norms = {}
                for group in GRADIENT_GROUPS:
                    step_norms[f"raw_{group}"] = gradient_l2_norm(
                        raw_gradients,
                        group,
                    )
                    step_norms[f"applied_{group}"] = gradient_l2_norm(
                        applied_gradients,
                        group,
                    )
                next_grad_sums = jax.tree_util.tree_map(
                    lambda accumulated, value: accumulated + value,
                    step_grad_sums,
                    step_norms,
                )
                return (
                    next_params,
                    next_aux_state,
                    next_stopped,
                    step_attempted + 1,
                    step_accepted + accept_candidate.astype(jnp.int32),
                    step_rejected + reject_candidate.astype(jnp.int32),
                    candidate_kl,
                    next_grad_sums,
                )

            return jax.lax.cond(
                ~step_stopped,
                _attempt_step,
                lambda _: minibatch_carry,
                operand=None,
            ), None

        minibatch_carry = (
            current_params,
            current_aux_state,
            early_stopped,
            attempted,
            accepted,
            rejected,
            last_candidate_kl,
            grad_sums,
        )
        minibatch_carry, _ = jax.lax.scan(
            _minibatch_step,
            minibatch_carry,
            minibatches,
        )
        return (
            minibatch_carry[0],
            minibatch_carry[1],
            current_rng,
            minibatch_carry[2],
            minibatch_carry[3],
            minibatch_carry[4],
            minibatch_carry[5],
            minibatch_carry[6],
            minibatch_carry[7],
        ), None

    final_carry, _ = jax.lax.scan(
        _epoch_step,
        initial_carry,
        jnp.arange(settings.num_epochs, dtype=jnp.int32),
    )
    (
        final_params,
        final_aux_state,
        final_rng,
        early_stopped,
        attempted,
        accepted,
        rejected,
        last_candidate_kl,
        grad_sums,
    ) = final_carry
    after = build_rollout_outputs(
        apply_fn,
        final_params,
        init_hstate,
        obs,
        done,
        is_discrete=is_discrete,
    )
    after_summary = _masked_summary(
        after.reliability_scores,
        after.reliability_logits,
        labels,
        mask,
        loss_type=reliability_loss_type,
        eps=survival_eps,
        axis_name=axis_name,
    )
    full_rollout_kl = policy_kl(
        reference.policy,
        after.policy,
        is_discrete=is_discrete,
    )
    policy_kl_before = policy_kl(
        reference.policy,
        reference.policy,
        is_discrete=is_discrete,
    )
    value_change = after.value - reference.value
    value_mse = jnp.mean(jnp.square(value_change))
    value_abs_mean = jnp.mean(jnp.abs(value_change))
    value_abs_max = jnp.max(jnp.abs(value_change))
    if axis_name is not None:
        full_rollout_kl = jax.lax.pmean(full_rollout_kl, axis_name)
        policy_kl_before = jax.lax.pmean(policy_kl_before, axis_name)
        value_mse = jax.lax.pmean(value_mse, axis_name)
        value_abs_mean = jax.lax.pmean(value_abs_mean, axis_name)
        value_abs_max = jax.lax.pmax(value_abs_max, axis_name)
    change_metrics = policy_change_metrics(
        reference.policy,
        after.policy,
        is_discrete=is_discrete,
        axis_name=axis_name,
    )
    safe_attempted = jnp.maximum(attempted.astype(jnp.float32), 1.0)
    diagnostics = empty_phasic_aux_diagnostics(
        is_discrete=is_discrete,
        settings=settings,
    )
    diagnostics.update(
        {
            "phasic_aux_active": jnp.array(1.0, dtype=jnp.float32),
            "survival_loss_before_aux": before_summary["loss"],
            "survival_loss_after_aux": after_summary["loss"],
            "survival_mae_before_aux": before_summary["mae"],
            "survival_mae_after_aux": after_summary["mae"],
            "reliability_mean_before_aux": before_summary["score_mean"],
            "reliability_mean_after_aux": after_summary["score_mean"],
            "reliability_std_before_aux": before_summary["score_std"],
            "reliability_std_after_aux": after_summary["score_std"],
            "reliability_min_before_aux": before_summary["score_min"],
            "reliability_max_before_aux": before_summary["score_max"],
            "reliability_min_after_aux": after_summary["score_min"],
            "reliability_max_after_aux": after_summary["score_max"],
            "target_mean": before_summary["target_mean"],
            "target_std": before_summary["target_std"],
            "policy_kl_before_aux": policy_kl_before,
            "full_rollout_policy_kl_after_aux": full_rollout_kl,
            "last_candidate_policy_kl": last_candidate_kl,
            "value_prediction_mse_after_aux": value_mse,
            "mean_abs_value_change": value_abs_mean,
            "max_abs_value_change": value_abs_max,
            "aux_grad_norm_raw_total": grad_sums["raw_total"] / safe_attempted,
            "aux_grad_norm_applied_total": grad_sums["applied_total"] / safe_attempted,
            "aux_grad_norm_reliability_head": (
                grad_sums["applied_reliability_head"] / safe_attempted
            ),
            "aux_grad_norm_vision_encoder": (
                grad_sums["applied_vision_encoder"] / safe_attempted
            ),
            "aux_grad_norm_fusion_shared_trunk": (
                grad_sums["applied_fusion_shared_trunk"] / safe_attempted
            ),
            "aux_grad_norm_actor_head": (
                grad_sums["applied_actor_head"] / safe_attempted
            ),
            "aux_grad_norm_critic_head": (
                grad_sums["applied_critic_head"] / safe_attempted
            ),
            "aux_grad_norm_raw_reliability_head": (
                grad_sums["raw_reliability_head"] / safe_attempted
            ),
            "aux_grad_norm_raw_vision_encoder": (
                grad_sums["raw_vision_encoder"] / safe_attempted
            ),
            "aux_grad_norm_raw_fusion_shared_trunk": (
                grad_sums["raw_fusion_shared_trunk"] / safe_attempted
            ),
            "aux_grad_norm_raw_actor_head": (
                grad_sums["raw_actor_head"] / safe_attempted
            ),
            "aux_grad_norm_raw_critic_head": (
                grad_sums["raw_critic_head"] / safe_attempted
            ),
            "aux_steps_attempted": attempted.astype(jnp.float32),
            "aux_steps_accepted": accepted.astype(jnp.float32),
            "aux_steps_rejected": rejected.astype(jnp.float32),
            "aux_early_stopped": early_stopped.astype(jnp.float32),
            **change_metrics,
        }
    )
    return final_params, final_aux_state, final_rng, diagnostics


def format_phasic_aux_diagnostics(
    diagnostics: Mapping[str, Any],
    *,
    update: int,
    mode: str,
):
    values = {
        key: float(np.mean(np.asarray(diagnostics[key])))
        for key in PHASIC_DIAGNOSTIC_FIELDS
    }
    if mode != "phasic" or values["phasic_aux_active"] < 0.5:
        return (
            f"PHASIC_AUX_DIAG update={update} status=inactive "
            f"optimization_mode={mode}",
            values,
        )
    fields = [
        "PHASIC_AUX_DIAG",
        f"update={update}",
        "status=active",
        "optimization_mode=phasic",
    ]
    for key in PHASIC_DIAGNOSTIC_FIELDS:
        if key == "phasic_aux_active":
            continue
        if key in {"policy_is_discrete", "policy_is_box", "aux_early_stopped"}:
            fields.append(f"{key}={str(values[key] >= 0.5).lower()}")
        elif key in {"aux_steps_attempted", "aux_steps_accepted", "aux_steps_rejected"}:
            fields.append(f"{key}={int(round(values[key]))}")
        else:
            fields.append(f"{key}={values[key]:.6g}")
    return " ".join(fields), values
