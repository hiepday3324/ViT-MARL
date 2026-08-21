"""JAX helpers for explicit per-agent PPO lifecycle semantics."""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class MaskedPPOTerms(NamedTuple):
    value_loss: jax.Array
    actor_loss: jax.Array
    entropy: jax.Array
    approx_kl: jax.Array
    clip_frac: jax.Array
    normalized_advantage: jax.Array


def masked_mean(values, mask, *, axis_name=None):
    """Mean over active samples, returning zero when the mask is empty."""
    values = jnp.asarray(values)
    mask = jnp.asarray(mask, dtype=values.dtype)
    count = jnp.sum(mask)
    numerator = jnp.sum(jnp.where(mask > 0, values, 0.0))
    if axis_name is not None:
        count = jax.lax.psum(count, axis_name)
        numerator = jax.lax.psum(numerator, axis_name)
    return numerator / jnp.maximum(
        count,
        jnp.asarray(1.0, dtype=values.dtype),
    )


def masked_normalize(values, mask, eps=1e-8, *, axis_name=None):
    """Normalize active entries only and force inactive entries to zero."""
    values = jnp.asarray(values)
    mask = jnp.asarray(mask, dtype=values.dtype)
    mean = masked_mean(values, mask, axis_name=axis_name)
    variance = masked_mean(
        jnp.square(values - mean),
        mask,
        axis_name=axis_name,
    )
    normalized = (values - mean) / jnp.sqrt(
        variance + jnp.asarray(eps, dtype=values.dtype)
    )
    return jnp.where(mask > 0, normalized, jnp.zeros_like(normalized))


def calculate_gae(
    gamma,
    gae_lambda,
    rewards,
    values,
    agent_done,
    last_value,
):
    """Compute GAE with the current transition's per-agent terminal signal."""
    rewards = jnp.asarray(rewards)
    values = jnp.asarray(values)
    agent_done = jnp.asarray(agent_done, dtype=jnp.bool_)

    def _step(carry, transition):
        gae, next_value = carry
        reward, value, done = transition
        not_done = 1.0 - done.astype(value.dtype)
        delta = reward + gamma * next_value * not_done - value
        gae = delta + gamma * gae_lambda * not_done * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        _step,
        (jnp.zeros_like(last_value), last_value),
        (rewards, values, agent_done),
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + values


def compute_masked_ppo_terms(
    *,
    ratio,
    logratio,
    advantage,
    value_loss_samples,
    entropy_samples,
    agent_active,
    clip_eps,
    axis_name=None,
):
    """Aggregate PPO terms over active transitions only."""
    active = jnp.asarray(agent_active, dtype=jnp.float32)
    normalized_advantage = masked_normalize(
        advantage,
        active,
        axis_name=axis_name,
    )
    actor_unclipped = ratio * normalized_advantage
    actor_clipped = (
        jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        * normalized_advantage
    )
    return MaskedPPOTerms(
        value_loss=masked_mean(
            value_loss_samples,
            active,
            axis_name=axis_name,
        ),
        actor_loss=-masked_mean(
            jnp.minimum(actor_unclipped, actor_clipped),
            active,
            axis_name=axis_name,
        ),
        entropy=masked_mean(
            entropy_samples,
            active,
            axis_name=axis_name,
        ),
        approx_kl=masked_mean(
            (ratio - 1.0) - logratio,
            active,
            axis_name=axis_name,
        ),
        clip_frac=masked_mean(
            (jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32),
            active,
            axis_name=axis_name,
        ),
        normalized_advantage=normalized_advantage,
    )


def next_rnn_reset(agent_done, global_done):
    """Reset the next observation after either an agent or world boundary."""
    return jnp.asarray(agent_done, dtype=jnp.bool_) | jnp.asarray(
        global_done,
        dtype=jnp.bool_,
    )
