import jax.numpy as jnp
from flax import linen as nn


class LevelWiseReliabilityHead(nn.Module):
    """Estimate side-aware liquidity reliability ``r_{t,k,s}``.

    Side-awareness comes from the token tensor's ``(levels, sides, features)``
    structure. The head does not use learned level-rank or side-ID embeddings.
    """

    hidden_dim: int = 128
    gate_epsilon: float = 0.1

    @nn.compact
    def __call__(self, *, z_tokens, obs_exec, h_prev, tick_shift=None):
        """Return reliability scores and reliability-filtered vision tokens.

        Args:
            z_tokens: Side-aware level-wise tokens shaped
                ``(time, batch, levels, sides, features)`` or single-step
                ``(batch, levels, sides, features)``. Legacy per-level tokens
                ``(time, batch, levels, features)`` and
                ``(batch, levels, features)`` are accepted as a compatibility
                fallback.
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
        squeeze_legacy_side = False
        if z_tokens.ndim == 3:
            z_tokens = z_tokens[None, ..., None, :]
            squeeze_time = True
            squeeze_legacy_side = True
        elif z_tokens.ndim == 4:
            if obs_exec.ndim == 2:
                z_tokens = z_tokens[None, ...]
                squeeze_time = True
            elif obs_exec.ndim == 3:
                z_tokens = z_tokens[..., None, :]
                squeeze_legacy_side = True
            else:
                raise ValueError(
                    "Cannot distinguish single-step side-aware tokens from legacy "
                    "time-major tokens without obs_exec shaped (batch, features) "
                    "or (time, batch, features); "
                    f"got {obs_exec.shape}."
                )
        elif z_tokens.ndim != 5:
            raise ValueError(
                "z_tokens must be side-aware (time, batch, levels, sides, features), "
                "single-step (batch, levels, sides, features), or legacy "
                f"per-level tokens; got {z_tokens.shape}"
            )

        if obs_exec.ndim == 2:
            obs_exec = obs_exec[None, ...]
        elif obs_exec.ndim != 3:
            raise ValueError(
                "obs_exec must be (time, batch, features) or "
                f"(batch, features), got {obs_exec.shape}"
            )

        time_steps, batch_size, n_levels, n_sides, _ = z_tokens.shape

        if h_prev.ndim == 2:
            h_prev = h_prev[None, :, :]
        elif h_prev.ndim != 3:
            raise ValueError(f"h_prev must be (batch, features) or (time, batch, features), got {h_prev.shape}")
        h_prev = jnp.broadcast_to(h_prev, (time_steps, batch_size, h_prev.shape[-1]))

        if tick_shift is None:
            tick_shift = jnp.zeros((time_steps, batch_size, n_levels, n_sides, 1), dtype=jnp.float32)
        else:
            tick_shift = jnp.asarray(tick_shift, dtype=jnp.float32)
            if tick_shift.ndim == 1:
                tick_shift = tick_shift[None, :, None, None, None]
            elif tick_shift.ndim == 2:
                if tick_shift.shape == (time_steps, batch_size):
                    tick_shift = tick_shift[:, :, None, None, None]
                else:
                    tick_shift = tick_shift[None, :, None, None, :]
            elif tick_shift.ndim == 3:
                if tick_shift.shape[:2] == (time_steps, batch_size):
                    tick_shift = tick_shift[:, :, None, None, :]
                else:
                    tick_shift = tick_shift[None, :, :, None, :]
            elif tick_shift.ndim == 4:
                if tick_shift.shape[:2] == (time_steps, batch_size):
                    tick_shift = tick_shift[:, :, :, None, :]
                else:
                    tick_shift = tick_shift[None, ...]
            elif tick_shift.ndim != 5:
                raise ValueError(
                    "tick_shift must be shaped (time, batch), (time, batch, 1), "
                    "(time, batch, levels, 1), (time, batch, levels, sides, 1), "
                    "(batch,), (batch, 1), (batch, levels, 1), or "
                    f"(batch, levels, sides, 1), got {tick_shift.shape}"
                )
            tick_shift = jnp.broadcast_to(tick_shift, (time_steps, batch_size, n_levels, n_sides, tick_shift.shape[-1]))

        obs_exec = jnp.broadcast_to(obs_exec[:, :, None, None, :], (time_steps, batch_size, n_levels, n_sides, obs_exec.shape[-1]))
        h_prev = jnp.broadcast_to(h_prev[:, :, None, None, :], (time_steps, batch_size, n_levels, n_sides, h_prev.shape[-1]))

        z_proj = nn.relu(nn.Dense(self.hidden_dim, name="z_proj")(z_tokens))
        obs_proj = nn.relu(nn.Dense(self.hidden_dim, name="obs_proj")(obs_exec))
        h_proj = nn.relu(nn.Dense(self.hidden_dim, name="h_proj")(h_prev))
        shift_proj = nn.relu(nn.Dense(self.hidden_dim, name="shift_proj")(tick_shift))

        x = jnp.concatenate([z_proj, obs_proj, h_proj, shift_proj], axis=-1)
        x = nn.relu(nn.Dense(self.hidden_dim, name="mlp_l1")(x))
        x = nn.relu(nn.Dense(self.hidden_dim, name="mlp_l2")(x))
        reliability_scores = nn.sigmoid(nn.Dense(1, name="score")(x))
        gate_epsilon = min(max(float(self.gate_epsilon), 0.0), 1.0)
        gate = gate_epsilon + (1.0 - gate_epsilon) * reliability_scores
        filtered_tokens = gate * z_tokens
        if squeeze_legacy_side:
            reliability_scores = jnp.squeeze(reliability_scores, axis=-2)
            filtered_tokens = jnp.squeeze(filtered_tokens, axis=-2)
        if squeeze_time:
            reliability_scores = jnp.squeeze(reliability_scores, axis=0)
            filtered_tokens = jnp.squeeze(filtered_tokens, axis=0)
        return reliability_scores, filtered_tokens
