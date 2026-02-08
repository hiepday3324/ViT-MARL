import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict


class VisionAgent(nn.Module):
    """Encode orderbook matrices into embeddings for contrastive learning."""
    embed_dim: int
    hidden_size: int = 128

    @nn.compact
    def __call__(self, x, *, train: bool = False):
        """Forward pass: matrix → embedding vector."""
        x = jnp.asarray(x, dtype=jnp.float32)
        if x.ndim == 2:  # Single matrix [M, N] without batch
            x = x[None, ...]  # Add batch dim [1, M, N] - neural nets expect [batch, features]
        # Now x is always 3D: either [1, M, N] or [B, M, N] if already batched
        x = x.reshape(x.shape[0], -1)  # [batch_size, M*N] - just REARRANGES data, no learning
        x = nn.Dense(self.hidden_size)(x)  # [B, M*N] → [B, 128] - LEARNS W matrix + bias
        # reshape vs Dense:
        #   reshape: just reorganizes existing numbers, no parameters
        #   Dense: multiplies by learned weight matrix W of shape [M*N, 128], adds bias
        x = nn.relu(x)  # Nonlinearity: max(0, x). Allows network to learn complex patterns
        return nn.Dense(self.embed_dim)(x)  # [B, 128] → [B, embed_dim] via W of shape [128, embed_dim]

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
