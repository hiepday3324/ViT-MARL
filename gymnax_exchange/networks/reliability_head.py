import jax.numpy as jnp
from flax import linen as nn


class LevelWiseReliabilityHead(nn.Module):
    """Estimate per-level token reliability before vision/execution fusion."""

    hidden_dim: int = 128
    level_embed_dim: int = 16
    max_levels: int = 64

    @nn.compact
    def __call__(self, *, z_tokens, obs_exec, h_prev, tick_shift=None):
        """Return reliability scores and reliability-filtered vision tokens.

        Args:
            z_tokens: Vision tokens shaped ``(time, batch, levels, features)``
                or single-step ``(batch, levels, features)``.
            obs_exec: Execution observations shaped ``(time, batch, features)``
                or single-step ``(batch, features)``.
            h_prev: Previous RNN hidden state shaped ``(batch, features)`` or
                ``(time, batch, features)``.
            tick_shift: Optional anchor shift shaped ``(time, batch)``,
                ``(time, batch, 1)``, or broadcastable to
                ``(time, batch, levels, 1)``.
        """
        z_tokens = jnp.asarray(z_tokens, dtype=jnp.float32)
        obs_exec = jnp.asarray(obs_exec, dtype=jnp.float32)
        h_prev = jnp.asarray(h_prev, dtype=jnp.float32)

        squeeze_time = False
        if z_tokens.ndim == 3:
            z_tokens = z_tokens[None, ...]
            squeeze_time = True
        elif z_tokens.ndim != 4:
            raise ValueError(
                "z_tokens must be (time, batch, levels, features) or "
                f"(batch, levels, features), got {z_tokens.shape}"
            )

        if obs_exec.ndim == 2:
            obs_exec = obs_exec[None, ...]
        elif obs_exec.ndim != 3:
            raise ValueError(
                "obs_exec must be (time, batch, features) or "
                f"(batch, features), got {obs_exec.shape}"
            )

        time_steps, batch_size, n_levels, _ = z_tokens.shape

        if h_prev.ndim == 2:
            h_prev = h_prev[None, :, :]
        elif h_prev.ndim != 3:
            raise ValueError(f"h_prev must be (batch, features) or (time, batch, features), got {h_prev.shape}")
        h_prev = jnp.broadcast_to(h_prev, (time_steps, batch_size, h_prev.shape[-1]))

        if tick_shift is None:
            tick_shift = jnp.zeros((time_steps, batch_size, n_levels, 1), dtype=jnp.float32)
        else:
            tick_shift = jnp.asarray(tick_shift, dtype=jnp.float32)
            if tick_shift.ndim == 1:
                tick_shift = tick_shift[None, :, None]
            elif tick_shift.ndim == 2:
                if tick_shift.shape == (time_steps, batch_size):
                    tick_shift = tick_shift[..., None]
                else:
                    tick_shift = tick_shift[None, ...]
            elif tick_shift.ndim == 3 and tick_shift.shape[:2] != (time_steps, batch_size):
                tick_shift = tick_shift[None, ...]
            if tick_shift.ndim == 3:
                tick_shift = tick_shift[..., None]
            elif tick_shift.ndim != 4:
                raise ValueError(
                    "tick_shift must be shaped (time, batch), (time, batch, 1), "
                    "(time, batch, levels, 1), (batch,), (batch, 1), or "
                    f"(batch, levels, 1), got {tick_shift.shape}"
                )
            tick_shift = jnp.broadcast_to(tick_shift, (time_steps, batch_size, n_levels, tick_shift.shape[-1]))

        obs_exec = jnp.broadcast_to(obs_exec[:, :, None, :], (time_steps, batch_size, n_levels, obs_exec.shape[-1]))
        h_prev = jnp.broadcast_to(h_prev[:, :, None, :], (time_steps, batch_size, n_levels, h_prev.shape[-1]))

        level_ids = jnp.arange(n_levels, dtype=jnp.int32)
        level_emb = nn.Embed(
            num_embeddings=self.max_levels,
            features=self.level_embed_dim,
            name="level_embed",
        )(level_ids)
        level_emb = jnp.broadcast_to(level_emb[None, None, :, :], (time_steps, batch_size, n_levels, self.level_embed_dim))

        z_proj = nn.relu(nn.Dense(self.hidden_dim, name="z_proj")(z_tokens))
        obs_proj = nn.relu(nn.Dense(self.hidden_dim, name="obs_proj")(obs_exec))
        h_proj = nn.relu(nn.Dense(self.hidden_dim, name="h_proj")(h_prev))
        shift_proj = nn.relu(nn.Dense(self.hidden_dim, name="shift_proj")(tick_shift))
        level_proj = nn.relu(nn.Dense(self.hidden_dim, name="level_proj")(level_emb))

        x = jnp.concatenate([z_proj, obs_proj, h_proj, shift_proj, level_proj], axis=-1)
        x = nn.relu(nn.Dense(self.hidden_dim, name="mlp_l1")(x))
        x = nn.relu(nn.Dense(self.hidden_dim, name="mlp_l2")(x))
        reliability_scores = nn.sigmoid(nn.Dense(1, name="score")(x))
        filtered_tokens = reliability_scores * z_tokens
        if squeeze_time:
            reliability_scores = jnp.squeeze(reliability_scores, axis=0)
            filtered_tokens = jnp.squeeze(filtered_tokens, axis=0)
        return reliability_scores, filtered_tokens
