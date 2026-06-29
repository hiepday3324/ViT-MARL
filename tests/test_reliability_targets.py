import os
import sys
import unittest

import jax.numpy as jnp

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from gymnax_exchange.jaxrl.MARL.reliability_targets import (
    build_liquidity_survival_targets,
    masked_reliability_loss,
    resolve_rollout_is_sell_task,
)


class ReliabilityTargetTest(unittest.TestCase):
    tick_size = 100.0

    def _make_vision_obs(self, time_steps, batch_size=1, levels=4, volume=10.0):
        vision_obs = jnp.zeros((time_steps, batch_size, levels, 3, 2), dtype=jnp.float32)
        gaps = jnp.arange(1, levels + 1, dtype=jnp.float32)
        vision_obs = vision_obs.at[:, :, :, 0, 0].set(gaps)
        vision_obs = vision_obs.at[:, :, :, 0, 1].set(gaps)
        vision_obs = vision_obs.at[:, :, :, 1, :].set(jnp.log1p(volume))
        return vision_obs

    def _build_targets(self, vision_obs, *, mode, is_sell_task=None, delta=2, **kwargs):
        num_steps = vision_obs.shape[0] - delta
        mid_prices = jnp.full((vision_obs.shape[0], vision_obs.shape[1]), 10_000.0)
        return build_liquidity_survival_targets(
            vision_obs,
            mid_prices,
            tick_size=self.tick_size,
            survival_delta_steps=delta,
            survival_min_volume=1.0,
            survival_ratio=0.5,
            num_steps=num_steps,
            survival_target_mode=mode,
            is_sell_task=is_sell_task,
            **kwargs,
        )

    def test_shape_and_min_horizon_uses_intermediate_frame(self):
        vision_obs = self._make_vision_obs(time_steps=5)
        # At t=0, Ask level 0 disappears at tau=2, then recovers by the final frame.
        vision_obs = vision_obs.at[2, 0, 0, 1, 0].set(0.0)
        labels, mask = self._build_targets(vision_obs, mode="min_horizon_soft", delta=3)

        self.assertEqual(labels.shape, (2, 1, 4, 2))
        self.assertEqual(mask.shape, labels.shape)
        self.assertEqual(labels.dtype, jnp.float32)
        self.assertEqual(mask.dtype, jnp.float32)
        self.assertLess(float(labels[0, 0, 0, 0]), 1e-6)
        self.assertGreater(float(labels[0, 0, 1, 0]), 0.99)

    def test_missing_future_price_has_zero_target(self):
        vision_obs = self._make_vision_obs(time_steps=3)
        # Remove the current Ask level-0 price from every future Top-K frame.
        vision_obs = vision_obs.at[1:, 0, :, 0, 0].add(10.0)
        labels, _ = self._build_targets(vision_obs, mode="min_horizon_soft", delta=2)

        self.assertLess(float(labels[0, 0, 0, 0]), 1e-6)

    def test_passive_limit_actionability_respects_side_and_depth(self):
        vision_obs = self._make_vision_obs(time_steps=2)
        common_kwargs = {
            "delta": 1,
            "actionability_eta": 0.1,
            "actionability_depth": 3,
            "actionability_far_level_weight": 0.25,
        }

        buy_labels, _ = self._build_targets(
            vision_obs,
            mode="actionability_weighted_min_horizon",
            is_sell_task=jnp.zeros((1, 1), dtype=jnp.float32),
            **common_kwargs,
        )
        self.assertAlmostEqual(float(buy_labels[0, 0, 0, 0]), 0.1, places=5)
        self.assertAlmostEqual(float(buy_labels[0, 0, 0, 1]), 1.0, places=5)
        self.assertAlmostEqual(float(buy_labels[0, 0, 3, 0]), 0.025, places=5)
        self.assertAlmostEqual(float(buy_labels[0, 0, 3, 1]), 0.25, places=5)

        sell_labels, _ = self._build_targets(
            vision_obs,
            mode="actionability_weighted_min_horizon",
            is_sell_task=jnp.ones((1, 1), dtype=jnp.float32),
            **common_kwargs,
        )
        self.assertAlmostEqual(float(sell_labels[0, 0, 0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(sell_labels[0, 0, 0, 1]), 0.1, places=5)

    def test_episode_done_masks_the_inclusive_horizon_without_changing_labels(self):
        vision_obs = self._make_vision_obs(time_steps=4)
        labels_without_done, mask_without_done = self._build_targets(
            vision_obs,
            mode="min_horizon_soft",
            delta=2,
        )

        for done_index in (0, 1, 2):
            with self.subTest(done_index=done_index):
                episode_done = jnp.zeros((4, 1, 1), dtype=jnp.bool_)
                episode_done = episode_done.at[done_index, 0, 0].set(True)
                labels_with_done, mask_with_done = self._build_targets(
                    vision_obs,
                    mode="min_horizon_soft",
                    delta=2,
                    episode_done=episode_done,
                )
                self.assertTrue(bool(jnp.allclose(labels_with_done, labels_without_done)))
                self.assertTrue(bool(jnp.all(mask_with_done[0, 0] == 0.0)))

        done_after_horizon = jnp.zeros((4, 1), dtype=jnp.bool_)
        done_after_horizon = done_after_horizon.at[3, 0].set(True)
        labels_after_horizon, mask_after_horizon = self._build_targets(
            vision_obs,
            mode="min_horizon_soft",
            delta=2,
            episode_done=done_after_horizon,
        )
        self.assertTrue(bool(jnp.allclose(labels_after_horizon, labels_without_done)))
        self.assertTrue(bool(jnp.all(mask_after_horizon[0, 0] == mask_without_done[0, 0])))
        self.assertTrue(bool(jnp.any(mask_after_horizon[0, 0] == 1.0)))

    def test_masked_reliability_loss_accepts_soft_labels_for_bce_and_mse(self):
        scores = jnp.array([[[[[0.2], [0.8]], [[0.5], [0.4]]]]], dtype=jnp.float32)
        labels = jnp.array([[[[0.1, 0.9], [0.3, 0.7]]]], dtype=jnp.float32)
        mask = jnp.ones_like(labels)

        bce = masked_reliability_loss(scores, labels, mask, loss_type="bce")
        mse = masked_reliability_loss(scores, labels, mask, loss_type="mse")

        self.assertEqual(bce.shape, ())
        self.assertEqual(mse.shape, ())
        self.assertTrue(bool(jnp.isfinite(bce)))
        self.assertTrue(bool(jnp.isfinite(mse)))

    def test_resolve_rollout_is_sell_task_accepts_padded_rollout(self):
        padded = jnp.zeros((42, 2), dtype=jnp.float32)
        padded = padded.at[1, 0].set(1.0)
        padded = padded.at[33:, :].set(1.0)

        resolved = resolve_rollout_is_sell_task(
            {"is_sell_task": padded},
            task="random",
            num_steps=32,
            batch_size=2,
        )

        self.assertEqual(resolved.shape, (32, 2))
        self.assertEqual(float(resolved[1, 0]), 1.0)
        self.assertEqual(float(resolved[-1, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
