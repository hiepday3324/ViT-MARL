import unittest

import jax
import jax.numpy as jnp
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from gymnax_exchange.networks.gate_fusion import EMASmoothing, StableGatedCrossAttention
from gymnax_exchange.networks.reliability_head import LevelWiseReliabilityHead
from gymnax_exchange.networks.vision_agent import VisionAgent, supervised_contrastive_loss


class VisionPipelineShapeTest(unittest.TestCase):
    def test_vision_tokens_fusion_and_supcon_shapes(self):
        rng = jax.random.PRNGKey(0)
        time_steps = 4
        batch_size = 3
        embed_dim = 128

        exec_obs = jnp.ones((time_steps, batch_size, 28), dtype=jnp.float32)
        vision_obs = jnp.ones((time_steps, batch_size, 10, 3, 2), dtype=jnp.float32)

        vision = VisionAgent(embed_dim=embed_dim)
        vision_params = vision.init(rng, vision_obs, return_tokens=True)
        tokens = vision.apply(vision_params, vision_obs, return_tokens=True)
        pooled = vision.apply(vision_params, vision_obs)

        self.assertEqual(tokens.shape, (time_steps, batch_size, 10, embed_dim))
        self.assertEqual(pooled.shape, (time_steps, batch_size, embed_dim))
        self.assertTrue(bool(jnp.allclose(pooled, jnp.mean(tokens, axis=-2), atol=1e-5)))

        ema = EMASmoothing(alpha=0.5)
        ema_params = ema.init(rng, exec_obs)
        smoothed = ema.apply(ema_params, exec_obs)
        self.assertEqual(smoothed.shape, exec_obs.shape)

        fusion = StableGatedCrossAttention(d_model=embed_dim)
        fusion_params = fusion.init(rng, smoothed, tokens)
        fused = fusion.apply(fusion_params, smoothed, tokens)
        self.assertEqual(fused.shape, (time_steps, batch_size, embed_dim // 2))

        reliability = LevelWiseReliabilityHead(hidden_dim=64)
        h_prev = jnp.ones((batch_size, embed_dim), dtype=jnp.float32)
        tick_shift = jnp.ones((time_steps, batch_size, 1), dtype=jnp.float32)
        reliability_params = reliability.init(
            rng,
            z_tokens=tokens,
            obs_exec=exec_obs,
            h_prev=h_prev,
            tick_shift=tick_shift,
        )
        reliability_scores, filtered_tokens = reliability.apply(
            reliability_params,
            z_tokens=tokens,
            obs_exec=exec_obs,
            h_prev=h_prev,
            tick_shift=tick_shift,
        )
        self.assertEqual(reliability_scores.shape, (time_steps, batch_size, 10, 1))
        self.assertEqual(filtered_tokens.shape, tokens.shape)
        self.assertTrue(bool(jnp.all(reliability_scores >= 0.0)))
        self.assertTrue(bool(jnp.all(reliability_scores <= 1.0)))

        filtered_fused = fusion.apply(fusion_params, smoothed, filtered_tokens)
        self.assertEqual(filtered_fused.shape, (time_steps, batch_size, embed_dim // 2))

        single_scores, single_filtered = reliability.apply(
            reliability_params,
            z_tokens=tokens[0],
            obs_exec=exec_obs[0],
            h_prev=h_prev,
            tick_shift=tick_shift[0],
        )
        self.assertEqual(single_scores.shape, (batch_size, 10, 1))
        self.assertEqual(single_filtered.shape, tokens[0].shape)
        single_fused = fusion.apply(fusion_params, smoothed[0], single_filtered)
        self.assertEqual(single_fused.shape, (batch_size, embed_dim // 2))

        labels = jnp.zeros((time_steps, batch_size), dtype=jnp.int32)
        loss = supervised_contrastive_loss(pooled.reshape(-1, embed_dim), labels.reshape(-1))
        self.assertEqual(loss.shape, ())


if __name__ == "__main__":
    unittest.main()
