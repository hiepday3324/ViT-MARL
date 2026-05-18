import jax
import jax.numpy as jnp
from flax import linen as nn


class EMASmoothing(nn.Module):
    """Exponential moving average over the leading time axis."""

    alpha: float = 0.5

    @nn.compact
    def __call__(self, o_t):
        """Smooth ``o_t`` with shape ``(time, batch, features)``."""
        o_t = jnp.asarray(o_t, dtype=jnp.float32)
        if o_t.ndim != 3:
            raise ValueError(f"EMASmoothing expects (time, batch, features), got {o_t.shape}")

        def ema_step(carry, x_t):
            next_carry = self.alpha * x_t + (1.0 - self.alpha) * carry
            return next_carry, next_carry

        init_carry = jnp.zeros_like(o_t[0])
        _, smoothed = jax.lax.scan(ema_step, init_carry, o_t)
        return smoothed


class StableGatedCrossAttention(nn.Module):
    """Cross-attend smoothed numeric state to per-level vision tokens."""

    d_model: int = 128

    @nn.compact
    def __call__(self, o_t_smoothed, z_t):
        """Fuse execution observations and vision tokens.

        ``o_t_smoothed`` is ``(time, batch, exec_features)``.
        ``z_t`` is either ``(time, batch, levels, vision_features)`` or a pooled
        ``(time, batch, vision_features)`` tensor. The pooled form is accepted as
        a compatibility fallback, but meaningful attention requires level tokens.
        """
        o_t_smoothed = jnp.asarray(o_t_smoothed, dtype=jnp.float32)
        z_t = jnp.asarray(z_t, dtype=jnp.float32)

        if z_t.ndim == o_t_smoothed.ndim:
            z_t = z_t[..., None, :]
        if z_t.ndim != o_t_smoothed.ndim + 1:
            raise ValueError(
                "StableGatedCrossAttention expects vision tokens shaped "
                f"{o_t_smoothed.shape[:-1]} + (levels, features), got {z_t.shape}"
            )

        query = nn.Dense(self.d_model, name="W_Q")(o_t_smoothed)
        keys = nn.Dense(self.d_model, name="W_K")(z_t)
        values = nn.Dense(self.d_model, name="W_V")(z_t)

        scores = jnp.sum(query[..., None, :] * keys, axis=-1) / jnp.sqrt(self.d_model)
        attention_weights = jax.nn.softmax(scores, axis=-1)
        attended_vision = jnp.sum(attention_weights[..., None] * values, axis=-2)

        gate_logits = nn.Dense(self.d_model, name="W_g")(o_t_smoothed)
        gate = nn.sigmoid(gate_logits)
        stable_features = gate * attended_vision

        x = nn.Dense(self.d_model, name="MLP_L1")(stable_features)
        x = nn.relu(x)
        x = nn.Dense(self.d_model // 2, name="MLP_L2")(x)
        return nn.relu(x)


if __name__ == "__main__":
    time_steps = 10
    batch_size = 32
    levels = 10
    d_o = 28
    d_z = 128

    rng = jax.random.PRNGKey(0)
    o_t_raw = jax.random.normal(rng, (time_steps, batch_size, d_o))
    z_t = jax.random.normal(rng, (time_steps, batch_size, levels, d_z))

    ema_module = EMASmoothing(alpha=0.5)
    variables_ema = ema_module.init(rng, o_t_raw)
    o_t_smoothed = ema_module.apply(variables_ema, o_t_raw)

    fusion_module = StableGatedCrossAttention(d_model=128)
    variables_fusion = fusion_module.init(rng, o_t_smoothed, z_t)
    h_compact = fusion_module.apply(variables_fusion, o_t_smoothed, z_t)

    print(f"exec input: {o_t_raw.shape}")
    print(f"vision tokens: {z_t.shape}")
    print(f"fused output: {h_compact.shape}")
