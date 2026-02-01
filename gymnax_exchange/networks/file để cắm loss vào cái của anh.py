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
        x = x.reshape(x.shape[0], -1)  # [batch_size, M*N] - just REARRANGES data, no learning
        x = nn.Dense(self.hidden_size)(x)  # [B, M*N] → [B, 128] - LEARNS W matrix + bias
        x = nn.relu(x)  # Nonlinearity: max(0, x). Allows network to learn complex patterns
        return nn.Dense(self.embed_dim)(x)
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