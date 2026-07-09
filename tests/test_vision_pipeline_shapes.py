import unittest
import inspect

import jax
import jax.numpy as jnp
import os
import sys
from flax.traverse_util import flatten_dict

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from gymnax_exchange.networks.gate_fusion import EMASmoothing, StableGatedCrossAttention
from gymnax_exchange.networks.reliability_head import (
    LevelWiseReliabilityHead,
    build_side_id_from_tokens,
    select_h_prev_for_reliability,
)
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

        self.assertEqual(tokens.shape, (time_steps, batch_size, 10, 2, embed_dim))
        self.assertEqual(pooled.shape, (time_steps, batch_size, embed_dim))
        self.assertTrue(bool(jnp.allclose(pooled, jnp.mean(tokens, axis=(-3, -2)), atol=1e-5)))

        ema = EMASmoothing(alpha=0.5)
        ema_params = ema.init(rng, exec_obs)
        smoothed = ema.apply(ema_params, exec_obs)
        self.assertEqual(smoothed.shape, exec_obs.shape)

        fusion = StableGatedCrossAttention(d_model=embed_dim)
        fusion_params = fusion.init(rng, smoothed, tokens)
        fused = fusion.apply(fusion_params, smoothed, tokens)
        self.assertEqual(fused.shape, (time_steps, batch_size, embed_dim // 2))

        legacy_tokens = jnp.mean(tokens, axis=-2)
        legacy_fused = fusion.apply(fusion_params, smoothed, legacy_tokens)
        self.assertEqual(legacy_fused.shape, (time_steps, batch_size, embed_dim // 2))

        reliability = LevelWiseReliabilityHead(hidden_dim=64)
        self.assertNotIn("tick_shift", inspect.signature(reliability.__call__).parameters)
        self.assertNotIn("obs_exec", inspect.signature(reliability.__call__).parameters)
        h_prev = jnp.ones((batch_size, embed_dim), dtype=jnp.float32)
        side_id = build_side_id_from_tokens(tokens)
        mid_context = jnp.ones((time_steps, batch_size, 4), dtype=jnp.float32)
        reliability_params = reliability.init(
            rng,
            z_tokens=tokens,
            side_id=side_id,
            mid_context=mid_context,
            h_prev=h_prev,
        )
        flat_params = flatten_dict(reliability_params["params"])
        param_names = ["/".join(key) for key in flat_params]
        self.assertFalse(any("level_embed" in name for name in param_names))
        self.assertFalse(any("level_proj" in name for name in param_names))
        self.assertFalse(any("side_embed" in name for name in param_names))
        self.assertFalse(any("shift_proj" in name for name in param_names))
        self.assertTrue(any("side_proj" in name for name in param_names))
        self.assertTrue(any("mid_proj" in name for name in param_names))
        reliability_scores, filtered_tokens = reliability.apply(
            reliability_params,
            z_tokens=tokens,
            side_id=side_id,
            mid_context=mid_context,
            h_prev=h_prev,
        )
        self.assertEqual(reliability_scores.shape, (time_steps, batch_size, 10, 2, 1))
        self.assertEqual(filtered_tokens.shape, tokens.shape)
        self.assertTrue(bool(jnp.all(reliability_scores >= 0.0)))
        self.assertTrue(bool(jnp.all(reliability_scores <= 1.0)))

        filtered_fused = fusion.apply(fusion_params, smoothed, filtered_tokens)
        self.assertEqual(filtered_fused.shape, (time_steps, batch_size, embed_dim // 2))

        single_scores, single_filtered = reliability.apply(
            reliability_params,
            z_tokens=tokens[0],
            side_id=side_id[0],
            mid_context=mid_context[0],
            h_prev=h_prev,
        )
        self.assertEqual(single_scores.shape, (batch_size, 10, 2, 1))
        self.assertEqual(single_filtered.shape, tokens[0].shape)
        single_fused = fusion.apply(fusion_params, smoothed[0], single_filtered)
        self.assertEqual(single_fused.shape, (batch_size, embed_dim // 2))

        labels = jnp.zeros((time_steps, batch_size), dtype=jnp.int32)
        loss = supervised_contrastive_loss(pooled.reshape(-1, embed_dim), labels.reshape(-1))
        self.assertEqual(loss.shape, ())
        self.assertTrue(bool(jnp.isfinite(loss)))

    def test_reliability_head_accepts_mid_context_side_id(self):
        rng = jax.random.PRNGKey(1)
        time_steps = 2
        batch_size = 3
        n_levels = 10
        n_sides = 2
        embed_dim = 64
        z_tokens = jnp.ones((time_steps, batch_size, n_levels, n_sides, embed_dim), dtype=jnp.float32)
        side_id = build_side_id_from_tokens(z_tokens)
        mid_context = jnp.ones((time_steps, batch_size, 4), dtype=jnp.float32)
        h_prev = jnp.ones((batch_size, embed_dim), dtype=jnp.float32)
        reliability = LevelWiseReliabilityHead(hidden_dim=32)

        params = reliability.init(
            rng,
            z_tokens=z_tokens,
            side_id=side_id,
            mid_context=mid_context,
            h_prev=h_prev,
        )
        scores, filtered = reliability.apply(
            params,
            z_tokens=z_tokens,
            side_id=side_id,
            mid_context=mid_context,
            h_prev=h_prev,
        )

        self.assertEqual(scores.shape, (time_steps, batch_size, n_levels, n_sides, 1))
        self.assertEqual(filtered.shape, z_tokens.shape)

    def test_side_id_order(self):
        z_tokens = jnp.ones((2, 3, 10, 2, 64), dtype=jnp.float32)
        side_id = build_side_id_from_tokens(z_tokens)

        self.assertTrue(bool(jnp.all(side_id[..., 0, :] == 1.0)))
        self.assertTrue(bool(jnp.all(side_id[..., 1, :] == -1.0)))

    def test_obs_exec_not_used_in_reliability(self):
        params = inspect.signature(LevelWiseReliabilityHead().__call__).parameters

        self.assertNotIn("obs_exec", params)

    def test_h_prev_reliability_default_true(self):
        h_prev = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)

        selected = select_h_prev_for_reliability(h_prev)

        self.assertTrue(bool(jnp.allclose(selected, h_prev)))

    def test_h_prev_reliability_false_zeroes_h_prev(self):
        rng = jax.random.PRNGKey(2)
        time_steps = 2
        batch_size = 3
        n_levels = 10
        n_sides = 2
        embed_dim = 64
        z_tokens = jnp.ones((time_steps, batch_size, n_levels, n_sides, embed_dim), dtype=jnp.float32)
        side_id = build_side_id_from_tokens(z_tokens)
        mid_context = jnp.ones((time_steps, batch_size, 4), dtype=jnp.float32)
        h_prev = jnp.ones((batch_size, embed_dim), dtype=jnp.float32)

        selected = select_h_prev_for_reliability(
            h_prev,
            use_h_prev_in_reliability=False,
        )

        self.assertTrue(bool(jnp.allclose(selected, jnp.zeros_like(h_prev))))

        reliability = LevelWiseReliabilityHead(hidden_dim=32)
        params = reliability.init(
            rng,
            z_tokens=z_tokens,
            side_id=side_id,
            mid_context=mid_context,
            h_prev=selected,
        )
        scores, filtered = reliability.apply(
            params,
            z_tokens=z_tokens,
            side_id=side_id,
            mid_context=mid_context,
            h_prev=selected,
        )

        self.assertEqual(scores.shape, (time_steps, batch_size, n_levels, n_sides, 1))
        self.assertEqual(filtered.shape, z_tokens.shape)


if __name__ == "__main__":
    unittest.main()
