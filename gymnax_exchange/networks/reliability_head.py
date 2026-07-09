import jax.numpy as jnp
from flax import linen as nn


def build_side_id_from_tokens(z_tokens):
    """Return side IDs broadcast to side-aware token shape.

    The project convention is side order ``[Ask, Bid]``:
    Ask receives ``+1`` and Bid receives ``-1``.
    """
    z_tokens = jnp.asarray(z_tokens)
    if z_tokens.ndim < 4:
        raise ValueError(
            "side_id derivation requires side-aware tokens shaped "
            "(batch, levels, sides, features) or "
            f"(time, batch, levels, sides, features), got {z_tokens.shape}."
        )
    n_sides = z_tokens.shape[-2]
    side_values = jnp.where(
        jnp.arange(n_sides) == 0,
        jnp.array(1.0, dtype=jnp.float32),
        jnp.array(-1.0, dtype=jnp.float32),
    )
    side_shape = (1,) * (z_tokens.ndim - 2) + (n_sides, 1)
    return jnp.broadcast_to(side_values.reshape(side_shape), z_tokens.shape[:-1] + (1,))


def select_h_prev_for_reliability(h_prev, use_h_prev_in_reliability=True):
    """Optionally zero previous RNN state before Reliability Head input."""
    h_prev = jnp.asarray(h_prev, dtype=jnp.float32)
    if bool(use_h_prev_in_reliability):
        return h_prev
    return jnp.zeros_like(h_prev)


class LevelWiseReliabilityHead(nn.Module):
    """Estimate side-aware liquidity reliability ``r_{t,k,s}``.

    Side-awareness comes from explicit numeric side IDs matching the token
    tensor's ``(levels, sides, features)`` structure, not from learned side-ID
    embeddings.
    """

    hidden_dim: int = 128
    gate_epsilon: float = 0.1

    @nn.compact
    def __call__(self, *, z_tokens, side_id, mid_context, h_prev):
        """Return reliability scores and reliability-filtered vision tokens.

        Args:
            z_tokens: Side-aware level-wise tokens shaped
                ``(time, batch, levels, sides, features)`` or single-step
                ``(batch, levels, sides, features)``. Legacy per-level tokens
                ``(time, batch, levels, features)`` and
                ``(batch, levels, features)`` are accepted as a compatibility
                fallback.
            side_id: Numeric side IDs broadcastable to
                ``(time, batch, levels, sides, 1)``. The convention is
                ``Ask=+1`` and ``Bid=-1``.
            mid_context: Mid-price context shaped ``(time, batch, 4)`` or
                single-step ``(batch, 4)``.
            h_prev: Previous RNN hidden state shaped ``(batch, features)`` or
                ``(time, batch, features)``.
        """
        z_tokens = jnp.asarray(z_tokens, dtype=jnp.float32)
        side_id = jnp.asarray(side_id, dtype=jnp.float32)
        mid_context = jnp.asarray(mid_context, dtype=jnp.float32)
        h_prev = jnp.asarray(h_prev, dtype=jnp.float32)

        squeeze_time = False
        squeeze_legacy_side = False
        if z_tokens.ndim == 3:
            z_tokens = z_tokens[None, ..., None, :]
            squeeze_time = True
            squeeze_legacy_side = True
        elif z_tokens.ndim == 4:
            if mid_context.ndim == 2:
                z_tokens = z_tokens[None, ...]
                squeeze_time = True
            elif mid_context.ndim == 3:
                z_tokens = z_tokens[..., None, :]
                squeeze_legacy_side = True
            else:
                raise ValueError(
                    "Cannot distinguish single-step side-aware tokens from legacy "
                    "time-major tokens without mid_context shaped (batch, features) "
                    "or (time, batch, features); "
                    f"got {mid_context.shape}."
                )
        elif z_tokens.ndim != 5:
            raise ValueError(
                "z_tokens must be side-aware (time, batch, levels, sides, features), "
                "single-step (batch, levels, sides, features), or legacy "
                f"per-level tokens; got {z_tokens.shape}"
            )

        if mid_context.ndim == 2:
            mid_context = mid_context[None, ...]
        elif mid_context.ndim != 3:
            raise ValueError(
                "mid_context must be (time, batch, features) or "
                f"(batch, features), got {mid_context.shape}"
            )

        time_steps, batch_size, n_levels, n_sides, _ = z_tokens.shape

        if h_prev.ndim == 2:
            h_prev = h_prev[None, :, :]
        elif h_prev.ndim != 3:
            raise ValueError(f"h_prev must be (batch, features) or (time, batch, features), got {h_prev.shape}")
        h_prev = jnp.broadcast_to(h_prev, (time_steps, batch_size, h_prev.shape[-1]))

        if side_id.ndim == 1:
            side_id = side_id[None, None, None, :, None]
        elif side_id.ndim == 2:
            side_id = side_id[None, None, :, :, None]
        elif side_id.ndim == 3:
            side_id = side_id[None, :, :, :, None]
        elif side_id.ndim == 4:
            if side_id.shape[:2] == (time_steps, batch_size):
                side_id = side_id[..., None]
            else:
                side_id = side_id[None, ...]
        elif side_id.ndim != 5:
            raise ValueError(
                "side_id must be broadcastable to "
                f"(time, batch, levels, sides, 1), got {side_id.shape}"
            )
        side_id = jnp.broadcast_to(side_id, (time_steps, batch_size, n_levels, n_sides, side_id.shape[-1]))

        mid_context = jnp.broadcast_to(mid_context[:, :, None, None, :], (time_steps, batch_size, n_levels, n_sides, mid_context.shape[-1]))
        h_prev = jnp.broadcast_to(h_prev[:, :, None, None, :], (time_steps, batch_size, n_levels, n_sides, h_prev.shape[-1]))

        z_proj = nn.relu(nn.Dense(self.hidden_dim, name="z_proj")(z_tokens))
        side_proj = nn.relu(nn.Dense(self.hidden_dim, name="side_proj")(side_id))
        mid_proj = nn.relu(nn.Dense(self.hidden_dim, name="mid_proj")(mid_context))
        h_proj = nn.relu(nn.Dense(self.hidden_dim, name="h_proj")(h_prev))

        x = jnp.concatenate([z_proj, side_proj, mid_proj, h_proj], axis=-1)
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
