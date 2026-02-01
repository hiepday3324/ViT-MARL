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
        # Flax syntax: nn.Dense(output_size) creates layer, then (input) applies it
        #   Step 1: nn.Dense(64) → creates layer object with 64 output units
        #   Step 2: layer(x) → applies transformation: output = x @ W + b
        #   Combined: nn.Dense(64)(x) - create and apply in one line
    # Why return Dense(embed_dim)(x) not just x?
    #   After ReLU, x is [B, 128] but we want [B, embed_dim] (e.g., 64 or 256)
    #   This final Dense layer projects to the desired embedding size
    # Why Dense + ReLU + Dense (2 layers)?
    #   Dense alone is linear: f(x) = Wx + b
    #   Without ReLU, stacking Dense layers = one big linear transformation
    #   ReLU breaks linearity → network can learn nonlinear orderbook patterns


# Concrete Example of reshape vs Dense:
# Suppose x = [1, 160] (1 example, 160 features from flattened [20,8] orderbook)
#
# x.reshape(1, -1):
#   Just rearranges: [1, 160] stays [1, 160]
#   No parameters, no learning
#
# nn.Dense(128)(x):
#   Creates weight matrix W of shape [160, 128] (learns during training!)
#   Creates bias vector b of shape [128]
#   Computes: output = x @ W + b → shape [1, 128]
#   This TRANSFORMS the 160 features into 128 new learned features
#
# Similarly, nn.Dense(64)(x) where x is [1, 128]:
#   W has shape [128, 64], b has shape [64]
#   output = x @ W + b → shape [1, 64]


# Shape Flow Example:
# Input Case 1: Single matrix [20, 8]
#   → After if statement: [1, 20, 8]  (added batch dim, x.shape[0] = 1)
#   → After reshape:      [1, 160]    (1 example, 160 features)
#
# Input Case 2: Batch of 5 matrices [5, 20, 8]
#   → After if statement: [5, 20, 8]  (already batched, unchanged, x.shape[0] = 5)
#   → After reshape:      [5, 160]    (5 examples, each with 160 features)
#
# "B" in comments = generic placeholder for batch_size (could be 1, 5, 32, etc.)

def _l2_normalize(x: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    return x / jnp.maximum(norm, eps)


# ============================================================================
# What does __call__ do?
# ============================================================================
# __call__ is the FORWARD PASS - it transforms input → output embeddings
#
# Input:  orderbook matrix [B, M, N] (raw data)
# Output: embedding vector [B, embed_dim] (learned representation)
#
# It does NOT create parameters! The parameters (W, b in Dense layers) already exist.
# __call__ USES those parameters to compute embeddings from input.
#
# Training flow:
#   1. __call__(orderbook) → produces embeddings using current parameters
#   2. loss_fn(embeddings) → evaluates how good the embeddings are
#   3. Optimizer → updates parameters (W, b) to make better embeddings next time
#
# So: __call__ = compute embeddings | loss_fn = evaluate quality | optimizer = improve parameters
# ============================================================================


def contrastive_loss_fn(
    params: Dict[str, jnp.ndarray],
    model: nn.Module,
    batch: Dict[str, jnp.ndarray],
    *,
    temperature: float = 0.1,
) -> jnp.ndarray:
    """InfoNCE loss: expects batch["obs_a"] and batch["obs_b"].
    
    Yes, this IS the exact InfoNCE formula implementation:
    
    For each example i with positive pair i:
        loss_i = -log(exp(sim(z_i, z_i)/τ) / Σ_k exp(sim(z_i, z_k)/τ))
    
    What each function contributes:
    
    1. sim(z_i, z_j) = cosine similarity (dot product of L2-normalized vectors)
       • Measures embedding similarity: +1 (identical), 0 (orthogonal), -1 (opposite)
       • Example: sim([1,0], [1,0]) = 1, sim([1,0], [0,1]) = 0
    
    2. Division by τ (temperature, before exp):
       • Controls confidence sharpness
       • τ=0.1 (small): sim=0.8 → 8 after division → model very confident
       • τ=1.0 (large): sim=0.8 → 0.8 after division → model less certain
       • Smaller τ = stronger penalties for incorrect pairs
    
    3. exp(...) = exponential function:
       • Converts similarities to positive values: exp(8) ≈ 2981, exp(0) = 1
       • Amplifies differences: small similarity differences → large probability differences
       • Creates a "soft max" effect: highest similarity dominates
    
    4. Σ_k = sum over all k examples in batch:
       • Denominator normalization (partition function)
       • Sum of exp(sim_all) ensures output is a valid probability distribution
       • Includes both positive (i=j) and negative (i≠j) pairs
    
    5. Division (numerator/denominator):
       • Creates probability: P(pair i is positive) = exp(sim_pos) / Σ_k exp(sim_k)
       • If positive pair has highest similarity → P close to 1
       • If negative pair has higher similarity → P close to 0
    
    6. log(...) = natural logarithm:
       • Converts probability to log-probability
       • log(0.9) ≈ -0.1, log(0.5) ≈ -0.7, log(0.1) ≈ -2.3
       • Makes loss roughly linear in similarity (better gradients)
       • Numerical stability: avoids multiplying many small probabilities
    
    7. Negative sign (-):
       • Converts log-probability to loss (minimize loss = maximize probability)
       • P=0.9 → log(0.9)≈-0.1 → loss≈0.1 (good, low loss)
       • P=0.1 → log(0.1)≈-2.3 → loss≈2.3 (bad, high loss)
    
    Combined effect: Model learns to make positive pairs have highest similarity
    while negative pairs have lower similarity, in a probabilistic framework.
    
    where sim() = cosine similarity (dot product of normalized vectors)
          τ = temperature parameter
          Σ_k sums over all examples in batch (positives + negatives)
    """
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


# ============================================================================
# reshape() Explanation
# ============================================================================
# x.reshape(x.shape[0], -1) means:
#   - Keep the first dimension (batch size) as-is
#   - Flatten all other dimensions into one
#   - The -1 is "auto-compute": JAX calculates what size this dimension needs to be
#
# Example: x with shape [2, 10, 8] (2 examples, 10 rows, 8 cols)
#   Total elements = 2 * 10 * 8 = 160
#   x.reshape(2, -1) → [2, 80] because 160 ÷ 2 = 80
#   Each of the 2 examples is now a flat vector of 80 features
#
# Example: x with shape [1, 20, 8] (1 example from orderbook)
#   x.reshape(1, -1) → [1, 160]
#   The single example is flattened to 160 features
# ============================================================================


# ============================================================================
# Matrix Preparation (orderbook state or observation → model input)
# ============================================================================

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


# Usage:
#   matrix = prepare_raw_orderbook(world_state, n_levels=10)  # [20, 8]
#   matrix = prepare_obs_vector(obs_array, shape=(4, 5))      # [4, 5]
#   batch = {"obs_a": matrix_t0, "obs_b": matrix_t1}
#   loss = loss_fn(params, model, batch)
