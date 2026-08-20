import jax
import jax.numpy as jnp
from flax import linen as nn


class StableGatedCrossAttention(nn.Module):
    """Cross-attend quantitative execution state to vision tokens."""

    d_model: int = 128

    @nn.compact
    def __call__(self, exec_obs, z_t):
        """Fuse execution observations and vision tokens.

        ``exec_obs`` is ``(time, batch, exec_features)``.
        ``z_t`` is either side-aware
        ``(time, batch, levels, sides, vision_features)``, legacy per-level
        ``(time, batch, levels, vision_features)``, or a pooled
        ``(time, batch, vision_features)`` tensor. Side-aware tokens are
        flattened internally into ``levels * sides`` attention tokens:
        L1-Ask, L1-Bid, L2-Ask, L2-Bid, ...
        """
        exec_obs = jnp.asarray(exec_obs, dtype=jnp.float32)
        z_t = jnp.asarray(z_t, dtype=jnp.float32)

        if z_t.ndim == exec_obs.ndim:
            z_t = z_t[..., None, :]
        elif z_t.ndim == exec_obs.ndim + 2:
            z_t = z_t.reshape(*z_t.shape[:-3], z_t.shape[-3] * z_t.shape[-2], z_t.shape[-1])
        if z_t.ndim != exec_obs.ndim + 1:
            raise ValueError(
                "StableGatedCrossAttention expects vision tokens shaped "
                f"{exec_obs.shape[:-1]} + (levels, features) or "
                f"{exec_obs.shape[:-1]} + (levels, sides, features), got {z_t.shape}"
            )

        query = nn.Dense(self.d_model, name="W_Q")(exec_obs)
        keys = nn.Dense(self.d_model, name="W_K")(z_t)
        values = nn.Dense(self.d_model, name="W_V")(z_t)

        scores = jnp.sum(query[..., None, :] * keys, axis=-1) / jnp.sqrt(self.d_model)
        attention_weights = jax.nn.softmax(scores, axis=-1)
        attended_vision = jnp.sum(attention_weights[..., None] * values, axis=-2)

        gate_logits = nn.Dense(self.d_model, name="W_g")(exec_obs)
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

    fusion_module = StableGatedCrossAttention(d_model=128)
    variables_fusion = fusion_module.init(rng, o_t_raw, z_t)
    h_compact = fusion_module.apply(variables_fusion, o_t_raw, z_t)

    print(f"exec input: {o_t_raw.shape}")
    print(f"vision tokens: {z_t.shape}")
    print(f"fused output: {h_compact.shape}")
