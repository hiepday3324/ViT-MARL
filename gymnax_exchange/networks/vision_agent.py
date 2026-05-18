import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict


class VisionAgent(nn.Module):
    """Encode orderbook matrices into embeddings for contrastive learning."""

    embed_dim: int
    hidden_size: int = 128

    @nn.compact
    def __call__(self, x, *, train: bool = False, return_tokens: bool = False):
        """Encode LOB observations.

        Inputs are shaped ``(..., levels, features, sides)``. The default output
        is pooled over levels; ``return_tokens=True`` preserves level tokens for
        cross-attention.
        """
        x = jnp.asarray(x, dtype=jnp.float32)
        if x.ndim == 2:
            x = x[None, ...]
        if x.ndim < 3:
            raise ValueError(f"VisionAgent expects at least 3 dims, got shape {x.shape}")

        level_tokens = x.reshape(*x.shape[:-3], x.shape[-3], -1)
        level_tokens = nn.Dense(self.hidden_size, name="token_fc")(level_tokens)
        level_tokens = nn.relu(level_tokens)
        level_tokens = nn.Dense(self.embed_dim, name="token_embed")(level_tokens)

        if return_tokens:
            return level_tokens

        return jnp.mean(level_tokens, axis=-2)


def _l2_normalize(x: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    return x / jnp.maximum(norm, eps)


def contrastive_loss_fn(
    params: Dict[str, jnp.ndarray],
    model: nn.Module,
    batch: Dict[str, jnp.ndarray],
    *,
    temperature: float = 0.1,
) -> jnp.ndarray:
    z_a = model.apply({"params": params}, batch["obs_a"], train=True)
    z_b = model.apply({"params": params}, batch["obs_b"], train=True)
    z_a = _l2_normalize(z_a)
    z_b = _l2_normalize(z_b)

    logits = (z_a @ z_b.T) / temperature
    labels = jnp.arange(logits.shape[0])

    log_probs_a = jax.nn.log_softmax(logits, axis=-1)
    loss_a = -jnp.take_along_axis(log_probs_a, labels[:, None], axis=-1).squeeze(-1).mean()

    log_probs_b = jax.nn.log_softmax(logits.T, axis=-1)
    loss_b = -jnp.take_along_axis(log_probs_b, labels[:, None], axis=-1).squeeze(-1).mean()

    return 0.5 * (loss_a + loss_b)


def loss_fn(
    params: Dict[str, jnp.ndarray],
    model: nn.Module,
    batch: Dict[str, jnp.ndarray],
    *,
    temperature: float = 0.1,
) -> jnp.ndarray:
    return contrastive_loss_fn(params, model, batch, temperature=temperature)


def prepare_raw_orderbook(world_state, n_levels: int = 10) -> jnp.ndarray:
    """Extract [2*n_levels, 8] matrix from WorldState."""
    asks = world_state.ask_raw_orders[:n_levels]
    bids = world_state.bid_raw_orders[:n_levels]
    return jnp.concatenate([asks, bids], axis=0)


def prepare_obs_vector(obs: jnp.ndarray, shape: tuple = (4, 5)) -> jnp.ndarray:
    """Pad and reshape 1D observation to 2D matrix."""
    size = shape[0] * shape[1]
    if obs.shape[0] < size:
        obs = jnp.concatenate([obs, jnp.zeros(size - obs.shape[0])])
    return obs[:size].reshape(shape)


def supervised_contrastive_loss(embeddings, labels, temperature=0.1):
    """Compute supervised contrastive loss for a batch of embeddings."""
    eps = 1e-8
    norm = jnp.linalg.norm(embeddings, axis=-1, keepdims=True)
    embeddings = embeddings / jnp.maximum(norm, eps)
    logits = jnp.matmul(embeddings, embeddings.T) / temperature
    batch_size = embeddings.shape[0]
    labels = labels.reshape(-1)
    mask = jnp.equal(labels[:, None], labels[None, :]).astype(jnp.float32)

    self_mask = jnp.eye(batch_size, dtype=jnp.float32)
    mask = mask - self_mask
    logits = jnp.where(self_mask.astype(bool), -jnp.inf, logits)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    num_positives = jnp.maximum(mask.sum(axis=1), eps)
    log_prob_positives = jnp.sum(mask * log_probs, axis=1) / num_positives

    return -log_prob_positives.mean()
