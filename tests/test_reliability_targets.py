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
    mid_price = 10_000.0

    def _make_vision_obs(self, time_steps, batch_size=1, levels=4, volume=100.0):
        vision_obs = jnp.zeros((time_steps, batch_size, levels, 3, 2), dtype=jnp.float32)
        gaps = jnp.arange(1, levels + 1, dtype=jnp.float32)
        vision_obs = vision_obs.at[:, :, :, 0, 0].set(gaps)
        vision_obs = vision_obs.at[:, :, :, 0, 1].set(gaps)
        vision_obs = vision_obs.at[:, :, :, 1, :].set(jnp.log1p(volume))
        vision_obs = vision_obs.at[:, :, :, 2, :].set(jnp.log1p(volume))
        return vision_obs

    def _make_raw_orders(self, time_steps, batch_size=1, n_orders=16):
        return jnp.zeros((time_steps, batch_size, n_orders, 6), dtype=jnp.float32)

    def _ask_price(self, level):
        return self.mid_price + (level + 1) * self.tick_size

    def _bid_price(self, level):
        return self.mid_price - (level + 1) * self.tick_size

    def _expected_robust_survival(self, ratios, survival_ratio=0.5, temperature=0.15):
        ratios = jnp.asarray(ratios, dtype=jnp.float32)
        mean_survival = jnp.mean(ratios)
        availability = jnp.mean(
            1.0 / (1.0 + jnp.exp(-((ratios - survival_ratio) / temperature)))
        )
        return float(mean_survival * availability)

    def _build_targets(
        self,
        vision_obs,
        *,
        delta=1,
        ask_raw_orders=None,
        bid_raw_orders=None,
        episode_done=None,
        survival_ratio=0.5,
        temperature=0.15,
    ):
        num_steps = vision_obs.shape[0] - delta
        mid_prices = jnp.full((vision_obs.shape[0], vision_obs.shape[1]), self.mid_price)
        return build_liquidity_survival_targets(
            vision_obs,
            mid_prices,
            tick_size=self.tick_size,
            survival_delta_steps=delta,
            survival_min_volume=1.0,
            survival_ratio=survival_ratio,
            survival_availability_temperature=temperature,
            num_steps=num_steps,
            episode_done=episode_done,
            ask_raw_orders=ask_raw_orders,
            bid_raw_orders=bid_raw_orders,
        )

    def _set_abs_price_sum_ratio(
        self,
        ask_raw_orders,
        bid_raw_orders,
        *,
        time_index,
        level,
        side,
        ratio,
        current_volume=100.0,
        slot,
    ):
        price = self._ask_price(level) if side == "ask" else self._bid_price(level)
        future_volume = current_volume * ratio
        ask_qty = future_volume * 0.25
        bid_qty = future_volume * 0.75
        ask_raw_orders = ask_raw_orders.at[time_index, 0, slot, 0].set(price)
        ask_raw_orders = ask_raw_orders.at[time_index, 0, slot, 1].set(ask_qty)
        bid_raw_orders = bid_raw_orders.at[time_index, 0, slot, 0].set(price)
        bid_raw_orders = bid_raw_orders.at[time_index, 0, slot, 1].set(bid_qty)
        return ask_raw_orders, bid_raw_orders

    def test_fullbook_absolute_price_sum_matches_both_sides_at_token_price(self):
        vision_obs = self._make_vision_obs(time_steps=2, levels=1, volume=100.0)
        ask_raw_orders = self._make_raw_orders(time_steps=2)
        bid_raw_orders = self._make_raw_orders(time_steps=2)

        ask_raw_orders = ask_raw_orders.at[1, 0, 0, 0].set(self._ask_price(0))
        ask_raw_orders = ask_raw_orders.at[1, 0, 0, 1].set(20.0)
        bid_raw_orders = bid_raw_orders.at[1, 0, 0, 0].set(self._ask_price(0))
        bid_raw_orders = bid_raw_orders.at[1, 0, 0, 1].set(60.0)

        ask_raw_orders = ask_raw_orders.at[1, 0, 1, 0].set(self._bid_price(0))
        ask_raw_orders = ask_raw_orders.at[1, 0, 1, 1].set(30.0)
        bid_raw_orders = bid_raw_orders.at[1, 0, 1, 0].set(self._bid_price(0))
        bid_raw_orders = bid_raw_orders.at[1, 0, 1, 1].set(50.0)

        labels, mask = self._build_targets(
            vision_obs,
            ask_raw_orders=ask_raw_orders,
            bid_raw_orders=bid_raw_orders,
        )
        expected = self._expected_robust_survival([0.8])

        self.assertEqual(labels.shape, (1, 1, 1, 2))
        self.assertEqual(mask.shape, labels.shape)
        self.assertAlmostEqual(float(labels[0, 0, 0, 0]), expected, places=5)
        self.assertAlmostEqual(float(labels[0, 0, 0, 1]), expected, places=5)

    def test_target_equals_rho_robust_on_valid_mask(self):
        ratios = [0.2, 0.6, 1.0]
        vision_obs = self._make_vision_obs(time_steps=len(ratios) + 1, levels=1)
        ask_raw_orders = self._make_raw_orders(time_steps=len(ratios) + 1)
        bid_raw_orders = self._make_raw_orders(time_steps=len(ratios) + 1)

        for idx, ratio in enumerate(ratios, start=1):
            ask_raw_orders, bid_raw_orders = self._set_abs_price_sum_ratio(
                ask_raw_orders,
                bid_raw_orders,
                time_index=idx,
                level=0,
                side="ask",
                ratio=ratio,
                slot=0,
            )
            ask_raw_orders, bid_raw_orders = self._set_abs_price_sum_ratio(
                ask_raw_orders,
                bid_raw_orders,
                time_index=idx,
                level=0,
                side="bid",
                ratio=ratio,
                slot=1,
            )

        labels, mask = self._build_targets(
            vision_obs,
            delta=len(ratios),
            ask_raw_orders=ask_raw_orders,
            bid_raw_orders=bid_raw_orders,
        )
        expected = self._expected_robust_survival(ratios)
        valid_labels = labels[mask > 0.0]

        self.assertTrue(bool(jnp.allclose(valid_labels, expected, atol=1e-6)))

    def test_no_task_side_or_far_level_discount(self):
        levels = 4
        vision_obs = self._make_vision_obs(time_steps=2, levels=levels, volume=100.0)
        ask_raw_orders = self._make_raw_orders(time_steps=2, n_orders=levels * 2)
        bid_raw_orders = self._make_raw_orders(time_steps=2, n_orders=levels * 2)

        for level in range(levels):
            ask_raw_orders, bid_raw_orders = self._set_abs_price_sum_ratio(
                ask_raw_orders,
                bid_raw_orders,
                time_index=1,
                level=level,
                side="ask",
                ratio=0.8,
                slot=level,
            )
            ask_raw_orders, bid_raw_orders = self._set_abs_price_sum_ratio(
                ask_raw_orders,
                bid_raw_orders,
                time_index=1,
                level=level,
                side="bid",
                ratio=0.8,
                slot=level + levels,
            )

        labels, mask = self._build_targets(
            vision_obs,
            ask_raw_orders=ask_raw_orders,
            bid_raw_orders=bid_raw_orders,
        )
        expected = self._expected_robust_survival([0.8])
        valid_labels = labels[mask > 0.0]

        self.assertTrue(bool(jnp.allclose(valid_labels, expected, atol=1e-6)))
        self.assertAlmostEqual(float(labels[0, 0, 3, 0]), expected, places=5)
        self.assertAlmostEqual(float(labels[0, 0, 3, 1]), expected, places=5)

    def test_missing_raw_orders_raises(self):
        vision_obs = self._make_vision_obs(time_steps=2, levels=1, volume=100.0)
        with self.assertRaisesRegex(ValueError, "ask_raw_orders and bid_raw_orders"):
            self._build_targets(vision_obs)

    def test_missing_future_price_has_zero_target(self):
        vision_obs = self._make_vision_obs(time_steps=2, levels=1, volume=100.0)
        ask_raw_orders = self._make_raw_orders(time_steps=2)
        bid_raw_orders = self._make_raw_orders(time_steps=2)

        labels, mask = self._build_targets(
            vision_obs,
            ask_raw_orders=ask_raw_orders,
            bid_raw_orders=bid_raw_orders,
        )

        self.assertEqual(float(mask[0, 0, 0, 0]), 1.0)
        self.assertLess(float(labels[0, 0, 0, 0]), 1e-6)

    def test_episode_done_masks_the_inclusive_horizon_without_changing_labels(self):
        vision_obs = self._make_vision_obs(time_steps=4, levels=1, volume=100.0)
        ask_raw_orders = self._make_raw_orders(time_steps=4)
        bid_raw_orders = self._make_raw_orders(time_steps=4)
        for time_index in range(1, 4):
            ask_raw_orders, bid_raw_orders = self._set_abs_price_sum_ratio(
                ask_raw_orders,
                bid_raw_orders,
                time_index=time_index,
                level=0,
                side="ask",
                ratio=0.8,
                slot=0,
            )
            ask_raw_orders, bid_raw_orders = self._set_abs_price_sum_ratio(
                ask_raw_orders,
                bid_raw_orders,
                time_index=time_index,
                level=0,
                side="bid",
                ratio=0.8,
                slot=1,
            )

        labels_without_done, mask_without_done = self._build_targets(
            vision_obs,
            delta=2,
            ask_raw_orders=ask_raw_orders,
            bid_raw_orders=bid_raw_orders,
        )

        for done_index in (0, 1, 2):
            with self.subTest(done_index=done_index):
                episode_done = jnp.zeros((4, 1, 1), dtype=jnp.bool_)
                episode_done = episode_done.at[done_index, 0, 0].set(True)
                labels_with_done, mask_with_done = self._build_targets(
                    vision_obs,
                    delta=2,
                    ask_raw_orders=ask_raw_orders,
                    bid_raw_orders=bid_raw_orders,
                    episode_done=episode_done,
                )
                self.assertTrue(bool(jnp.allclose(labels_with_done, labels_without_done)))
                self.assertTrue(bool(jnp.all(mask_with_done[0, 0] == 0.0)))

        done_after_horizon = jnp.zeros((4, 1), dtype=jnp.bool_)
        done_after_horizon = done_after_horizon.at[3, 0].set(True)
        labels_after_horizon, mask_after_horizon = self._build_targets(
            vision_obs,
            delta=2,
            ask_raw_orders=ask_raw_orders,
            bid_raw_orders=bid_raw_orders,
            episode_done=done_after_horizon,
        )
        self.assertTrue(bool(jnp.allclose(labels_after_horizon, labels_without_done)))
        self.assertTrue(bool(jnp.all(mask_after_horizon[0, 0] == mask_without_done[0, 0])))
        self.assertTrue(bool(jnp.any(mask_after_horizon[0, 0] == 1.0)))

    def test_masked_reliability_loss_accepts_soft_labels_for_bce_and_mse(self):
        scores = jnp.array([[[[[0.2], [0.8]], [[0.5], [0.4]]]]], dtype=jnp.float32)
        labels = jnp.array([[[[0.1, 0.9], [0.3, 0.7]]]], dtype=jnp.float32)
        mask = jnp.ones_like(labels)
        scores_without_trailing_singleton = jnp.squeeze(scores, axis=-1)

        bce = masked_reliability_loss(scores, labels, mask, loss_type="bce")
        mse = masked_reliability_loss(scores, labels, mask, loss_type="mse")
        bce_same_shape = masked_reliability_loss(
            scores_without_trailing_singleton,
            labels,
            mask,
            loss_type="bce",
        )

        self.assertEqual(bce.shape, ())
        self.assertEqual(mse.shape, ())
        self.assertEqual(bce_same_shape.shape, ())
        self.assertTrue(bool(jnp.isfinite(bce)))
        self.assertTrue(bool(jnp.isfinite(mse)))
        self.assertTrue(bool(jnp.isfinite(bce_same_shape)))

    def test_masked_reliability_loss_rejects_side_axis_broadcast(self):
        scores = jnp.ones((1, 1, 2, 1), dtype=jnp.float32)
        labels = jnp.ones((1, 1, 2, 2), dtype=jnp.float32)
        mask = jnp.ones_like(labels)

        with self.assertRaisesRegex(ValueError, "reliability_scores must match labels"):
            masked_reliability_loss(scores, labels, mask, loss_type="bce")

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
