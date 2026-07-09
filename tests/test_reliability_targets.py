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

    def _make_ratio_vision_obs(self, ratios, current_volume=10.0):
        ratios = jnp.asarray(ratios, dtype=jnp.float32)
        vision_obs = self._make_vision_obs(
            time_steps=ratios.shape[0] + 1,
            batch_size=1,
            levels=1,
            volume=current_volume,
        )
        future_volumes = current_volume * ratios
        vision_obs = vision_obs.at[1:, 0, 0, 1, :].set(jnp.log1p(future_volumes[:, None]))
        vision_obs = vision_obs.at[1:, 0, 0, 2, :].set(jnp.log1p(future_volumes[:, None]))
        return vision_obs

    def _make_raw_orders(self, time_steps, batch_size=1, n_orders=4):
        return jnp.zeros((time_steps, batch_size, n_orders, 6), dtype=jnp.float32)

    def _expected_robust_survival(self, ratios, survival_ratio=0.5, temperature=0.15):
        ratios = jnp.asarray(ratios, dtype=jnp.float32)
        mean_survival = jnp.mean(ratios)
        availability = jnp.mean(
            1.0 / (1.0 + jnp.exp(-((ratios - survival_ratio) / temperature)))
        )
        return float(mean_survival * availability)

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

    def test_vision_topk_source_keeps_topk_lookup_behavior(self):
        visible = self._make_ratio_vision_obs([0.8], current_volume=100.0)
        visible_labels, _ = self._build_targets(
            visible,
            mode="actionability_weighted_min_horizon",
            is_sell_task=jnp.ones((1, 1), dtype=jnp.float32),
            delta=1,
            actionability_depth=1,
            survival_target_book_source="vision_topk",
        )
        self.assertAlmostEqual(
            float(visible_labels[0, 0, 0, 0]),
            self._expected_robust_survival([0.8]),
            places=5,
        )

        missing = self._make_vision_obs(time_steps=2, levels=1, volume=100.0)
        missing = missing.at[1, 0, 0, 0, 0].set(10.0)
        missing_labels, _ = self._build_targets(
            missing,
            mode="actionability_weighted_min_horizon",
            is_sell_task=jnp.ones((1, 1), dtype=jnp.float32),
            delta=1,
            actionability_depth=1,
            survival_target_book_source="vision_topk",
        )
        self.assertLess(float(missing_labels[0, 0, 0, 0]), 1e-6)

    def test_fullbook_recovers_price_outside_future_topk(self):
        vision_obs = self._make_vision_obs(time_steps=2, levels=1, volume=100.0)
        vision_obs = vision_obs.at[1, 0, 0, 0, 0].set(10.0)
        ask_raw_orders = self._make_raw_orders(time_steps=2)
        bid_raw_orders = self._make_raw_orders(time_steps=2)
        ask_raw_orders = ask_raw_orders.at[1, 0, 0, 0].set(10_100.0)
        ask_raw_orders = ask_raw_orders.at[1, 0, 0, 1].set(80.0)

        labels, mask = self._build_targets(
            vision_obs,
            mode="actionability_weighted_min_horizon",
            is_sell_task=jnp.ones((1, 1), dtype=jnp.float32),
            delta=1,
            actionability_depth=1,
            survival_target_book_source="fullbook",
            ask_raw_orders=ask_raw_orders,
            bid_raw_orders=bid_raw_orders,
        )

        self.assertAlmostEqual(
            float(labels[0, 0, 0, 0]),
            self._expected_robust_survival([0.8]),
            places=5,
        )
        self.assertEqual(float(mask[0, 0, 0, 0]), 1.0)

    def test_fullbook_alignment_does_not_swap_ask_bid(self):
        vision_obs = self._make_vision_obs(time_steps=2, levels=1, volume=100.0)
        ask_raw_orders = self._make_raw_orders(time_steps=2)
        bid_raw_orders = self._make_raw_orders(time_steps=2)
        ask_raw_orders = ask_raw_orders.at[1, 0, 0, 0].set(10_100.0)
        ask_raw_orders = ask_raw_orders.at[1, 0, 0, 1].set(80.0)

        labels, _ = self._build_targets(
            vision_obs,
            mode="actionability_weighted_min_horizon",
            is_sell_task=jnp.ones((1, 1), dtype=jnp.float32),
            delta=1,
            actionability_depth=1,
            survival_target_book_source="fullbook",
            ask_raw_orders=ask_raw_orders,
            bid_raw_orders=bid_raw_orders,
        )

        self.assertAlmostEqual(
            float(labels[0, 0, 0, 0]),
            self._expected_robust_survival([0.8]),
            places=5,
        )
        self.assertLess(float(labels[0, 0, 0, 1]), 1e-6)

    def test_fullbook_same_side_does_not_sum_opposite_side_price(self):
        vision_obs = self._make_vision_obs(time_steps=2, levels=1, volume=100.0)
        ask_raw_orders = self._make_raw_orders(time_steps=2)
        bid_raw_orders = self._make_raw_orders(time_steps=2)

        # Current Ask price is 10_100. same_side must use only future Ask volume.
        ask_raw_orders = ask_raw_orders.at[1, 0, 0, 0].set(10_100.0)
        ask_raw_orders = ask_raw_orders.at[1, 0, 0, 1].set(30.0)
        bid_raw_orders = bid_raw_orders.at[1, 0, 0, 0].set(10_100.0)
        bid_raw_orders = bid_raw_orders.at[1, 0, 0, 1].set(70.0)

        # Current Bid price is 9_900. same_side must use only future Bid volume.
        ask_raw_orders = ask_raw_orders.at[1, 0, 1, 0].set(9_900.0)
        ask_raw_orders = ask_raw_orders.at[1, 0, 1, 1].set(80.0)
        bid_raw_orders = bid_raw_orders.at[1, 0, 1, 0].set(9_900.0)
        bid_raw_orders = bid_raw_orders.at[1, 0, 1, 1].set(20.0)

        labels, _ = self._build_targets(
            vision_obs,
            mode="actionability_weighted_min_horizon",
            is_sell_task=jnp.ones((1, 1), dtype=jnp.float32),
            delta=1,
            actionability_depth=1,
            survival_target_book_source="fullbook",
            survival_fullbook_match_mode="same_side",
            ask_raw_orders=ask_raw_orders,
            bid_raw_orders=bid_raw_orders,
        )

        self.assertAlmostEqual(
            float(labels[0, 0, 0, 0]),
            self._expected_robust_survival([0.3]),
            places=5,
        )
        self.assertAlmostEqual(
            float(labels[0, 0, 0, 1]),
            self._expected_robust_survival([0.2]),
            places=5,
        )

    def test_fullbook_absolute_price_sum_matches_both_sides_at_token_price(self):
        vision_obs = self._make_vision_obs(time_steps=2, levels=1, volume=100.0)
        ask_raw_orders = self._make_raw_orders(time_steps=2)
        bid_raw_orders = self._make_raw_orders(time_steps=2)

        ask_raw_orders = ask_raw_orders.at[1, 0, 0, 0].set(10_100.0)
        ask_raw_orders = ask_raw_orders.at[1, 0, 0, 1].set(20.0)
        bid_raw_orders = bid_raw_orders.at[1, 0, 0, 0].set(10_100.0)
        bid_raw_orders = bid_raw_orders.at[1, 0, 0, 1].set(60.0)

        ask_raw_orders = ask_raw_orders.at[1, 0, 1, 0].set(9_900.0)
        ask_raw_orders = ask_raw_orders.at[1, 0, 1, 1].set(30.0)
        bid_raw_orders = bid_raw_orders.at[1, 0, 1, 0].set(9_900.0)
        bid_raw_orders = bid_raw_orders.at[1, 0, 1, 1].set(50.0)

        labels, _ = self._build_targets(
            vision_obs,
            mode="actionability_weighted_min_horizon",
            is_sell_task=jnp.ones((1, 1), dtype=jnp.float32),
            delta=1,
            actionability_depth=1,
            survival_target_book_source="fullbook",
            survival_fullbook_match_mode="absolute_price_sum",
            ask_raw_orders=ask_raw_orders,
            bid_raw_orders=bid_raw_orders,
        )
        expected = self._expected_robust_survival([0.8])

        self.assertAlmostEqual(float(labels[0, 0, 0, 0]), expected, places=5)
        self.assertAlmostEqual(float(labels[0, 0, 0, 1]), expected, places=5)

    def test_fullbook_alignment_does_not_swap_levels(self):
        vision_obs = self._make_vision_obs(time_steps=2, levels=2, volume=100.0)
        ask_raw_orders = self._make_raw_orders(time_steps=2)
        bid_raw_orders = self._make_raw_orders(time_steps=2)
        ask_raw_orders = ask_raw_orders.at[1, 0, 0, 0].set(10_200.0)
        ask_raw_orders = ask_raw_orders.at[1, 0, 0, 1].set(80.0)

        labels, _ = self._build_targets(
            vision_obs,
            mode="actionability_weighted_min_horizon",
            is_sell_task=jnp.ones((1, 1), dtype=jnp.float32),
            delta=1,
            actionability_depth=2,
            survival_target_book_source="fullbook",
            ask_raw_orders=ask_raw_orders,
            bid_raw_orders=bid_raw_orders,
        )

        self.assertLess(float(labels[0, 0, 0, 0]), 1e-6)
        self.assertAlmostEqual(
            float(labels[0, 0, 1, 0]),
            self._expected_robust_survival([0.8]),
            places=5,
        )

    def test_fullbook_zero_when_price_missing_from_raw_book(self):
        vision_obs = self._make_vision_obs(time_steps=2, levels=1, volume=100.0)
        vision_obs = vision_obs.at[1, 0, 0, 0, 0].set(10.0)
        ask_raw_orders = self._make_raw_orders(time_steps=2)
        bid_raw_orders = self._make_raw_orders(time_steps=2)

        labels, _ = self._build_targets(
            vision_obs,
            mode="actionability_weighted_min_horizon",
            is_sell_task=jnp.ones((1, 1), dtype=jnp.float32),
            delta=1,
            actionability_depth=1,
            survival_target_book_source="fullbook",
            ask_raw_orders=ask_raw_orders,
            bid_raw_orders=bid_raw_orders,
        )

        self.assertLess(float(labels[0, 0, 0, 0]), 1e-6)

    def test_fullbook_source_requires_raw_orders_and_valid_source(self):
        vision_obs = self._make_vision_obs(time_steps=2, levels=1, volume=100.0)
        with self.assertRaisesRegex(ValueError, "ask_raw_orders and bid_raw_orders"):
            self._build_targets(
                vision_obs,
                mode="actionability_weighted_min_horizon",
                is_sell_task=jnp.ones((1, 1), dtype=jnp.float32),
                delta=1,
                actionability_depth=1,
                survival_target_book_source="fullbook",
            )

        with self.assertRaisesRegex(ValueError, "Unknown survival_target_book_source"):
            self._build_targets(
                vision_obs,
                mode="actionability_weighted_min_horizon",
                is_sell_task=jnp.ones((1, 1), dtype=jnp.float32),
                delta=1,
                actionability_depth=1,
                survival_target_book_source="unknown",
            )

        with self.assertRaisesRegex(ValueError, "Unknown survival_fullbook_match_mode"):
            self._build_targets(
                vision_obs,
                mode="actionability_weighted_min_horizon",
                is_sell_task=jnp.ones((1, 1), dtype=jnp.float32),
                delta=1,
                actionability_depth=1,
                survival_target_book_source="fullbook",
                survival_fullbook_match_mode="bad_mode",
                ask_raw_orders=self._make_raw_orders(time_steps=2),
                bid_raw_orders=self._make_raw_orders(time_steps=2),
            )

    def test_fullbook_absolute_price_sum_keeps_rho_robust_formula_without_discounts(self):
        levels = 4
        vision_obs = self._make_vision_obs(time_steps=2, levels=levels, volume=100.0)
        ask_raw_orders = self._make_raw_orders(time_steps=2, n_orders=levels * 2)
        bid_raw_orders = self._make_raw_orders(time_steps=2, n_orders=levels * 2)
        for level in range(levels):
            ask_price = 10_000.0 + (level + 1) * self.tick_size
            bid_price = 10_000.0 - (level + 1) * self.tick_size
            ask_raw_orders = ask_raw_orders.at[1, 0, level, 0].set(ask_price)
            ask_raw_orders = ask_raw_orders.at[1, 0, level, 1].set(20.0)
            bid_raw_orders = bid_raw_orders.at[1, 0, level, 0].set(ask_price)
            bid_raw_orders = bid_raw_orders.at[1, 0, level, 1].set(60.0)

            bid_slot = level + levels
            ask_raw_orders = ask_raw_orders.at[1, 0, bid_slot, 0].set(bid_price)
            ask_raw_orders = ask_raw_orders.at[1, 0, bid_slot, 1].set(30.0)
            bid_raw_orders = bid_raw_orders.at[1, 0, bid_slot, 0].set(bid_price)
            bid_raw_orders = bid_raw_orders.at[1, 0, bid_slot, 1].set(50.0)

        labels, mask = self._build_targets(
            vision_obs,
            mode="actionability_weighted_min_horizon",
            is_sell_task=jnp.zeros((1, 1), dtype=jnp.float32),
            delta=1,
            actionability_depth=1,
            actionability_far_level_weight=0.25,
            survival_target_book_source="fullbook",
            survival_fullbook_match_mode="absolute_price_sum",
            ask_raw_orders=ask_raw_orders,
            bid_raw_orders=bid_raw_orders,
        )
        expected = self._expected_robust_survival([0.8])
        valid_labels = labels[mask > 0.0]

        self.assertTrue(bool(jnp.allclose(valid_labels, expected, atol=1e-6)))
        self.assertAlmostEqual(float(labels[0, 0, 3, 0]), expected, places=5)
        self.assertAlmostEqual(float(labels[0, 0, 3, 1]), expected, places=5)

    def test_actionability_side_weight_no_longer_affects_label(self):
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
        full_ratio_target = self._expected_robust_survival([1.0])
        self.assertAlmostEqual(float(buy_labels[0, 0, 0, 0]), full_ratio_target, places=5)
        self.assertAlmostEqual(float(buy_labels[0, 0, 0, 1]), full_ratio_target, places=5)
        self.assertAlmostEqual(float(buy_labels[0, 0, 3, 0]), full_ratio_target, places=5)
        self.assertAlmostEqual(float(buy_labels[0, 0, 3, 1]), full_ratio_target, places=5)

        sell_labels, _ = self._build_targets(
            vision_obs,
            mode="actionability_weighted_min_horizon",
            is_sell_task=jnp.ones((1, 1), dtype=jnp.float32),
            **common_kwargs,
        )
        self.assertAlmostEqual(float(sell_labels[0, 0, 0, 0]), full_ratio_target, places=5)
        self.assertAlmostEqual(float(sell_labels[0, 0, 0, 1]), full_ratio_target, places=5)
        self.assertAlmostEqual(float(sell_labels[0, 0, 3, 0]), full_ratio_target, places=5)
        self.assertAlmostEqual(float(sell_labels[0, 0, 3, 1]), full_ratio_target, places=5)
        self.assertTrue(bool(jnp.allclose(buy_labels, sell_labels)))

    def test_actionability_level_weight_no_longer_affects_label(self):
        vision_obs = self._make_vision_obs(time_steps=2, levels=6)
        full_ratio_target = self._expected_robust_survival([1.0])

        labels, _ = self._build_targets(
            vision_obs,
            mode="actionability_weighted_min_horizon",
            is_sell_task=jnp.ones((1, 1), dtype=jnp.float32),
            delta=1,
            actionability_eta=0.1,
            actionability_depth=3,
            actionability_far_level_weight=0.25,
        )

        self.assertAlmostEqual(float(labels[0, 0, 0, 0]), full_ratio_target, places=5)
        self.assertAlmostEqual(float(labels[0, 0, 5, 0]), full_ratio_target, places=5)
        self.assertAlmostEqual(float(labels[0, 0, 5, 1]), full_ratio_target, places=5)
        self.assertGreater(float(labels[0, 0, 5, 0]), 0.9)
        self.assertNotAlmostEqual(float(labels[0, 0, 5, 0]), 0.25 * full_ratio_target, places=4)

    def test_actionability_weighted_target_equals_rho_robust_on_valid_mask(self):
        ratios = [0.8, 0.7, 0.9]
        vision_obs = self._make_ratio_vision_obs(ratios)
        labels, mask = self._build_targets(
            vision_obs,
            mode="actionability_weighted_min_horizon",
            is_sell_task=jnp.ones((1, 1), dtype=jnp.float32),
            delta=len(ratios),
            actionability_eta=0.1,
            actionability_depth=1,
            actionability_far_level_weight=0.25,
        )
        expected_target = self._expected_robust_survival(ratios)
        valid_labels = labels[mask > 0.0]

        self.assertTrue(bool(jnp.allclose(valid_labels, expected_target, atol=1e-6)))

    def test_no_target_formula_change(self):
        ratios = [0.2, 0.6, 1.0]
        vision_obs = self._make_ratio_vision_obs(ratios)
        labels, mask = self._build_targets(
            vision_obs,
            mode="actionability_weighted_min_horizon",
            is_sell_task=jnp.zeros((1, 1), dtype=jnp.float32),
            delta=len(ratios),
            actionability_eta=0.1,
            actionability_depth=1,
            actionability_far_level_weight=0.25,
        )
        expected_target = self._expected_robust_survival(ratios)
        valid_labels = labels[mask > 0.0]

        self.assertTrue(bool(jnp.allclose(valid_labels, expected_target, atol=1e-6)))

    def test_actionability_weighted_target_uses_sigmoid_availability(self):
        cases = [
            [0.5] * 10,
            [0.4] * 10,
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]

        for ratios in cases:
            with self.subTest(ratios=ratios):
                vision_obs = self._make_ratio_vision_obs(ratios)
                labels, mask = self._build_targets(
                    vision_obs,
                    mode="actionability_weighted_min_horizon",
                    is_sell_task=jnp.ones((1, 1), dtype=jnp.float32),
                    delta=len(ratios),
                    actionability_eta=0.1,
                    actionability_depth=1,
                    actionability_far_level_weight=0.25,
                )
                expected_target = self._expected_robust_survival(ratios)

                self.assertAlmostEqual(float(labels[0, 0, 0, 0]), expected_target, places=5)
                self.assertEqual(float(mask[0, 0, 0, 0]), 1.0)

    def test_sigmoid_availability_target_is_monotonic(self):
        targets = []
        for ratios in ([0.4] * 10, [0.5] * 10, [0.8] * 10):
            vision_obs = self._make_ratio_vision_obs(ratios)
            labels, _ = self._build_targets(
                vision_obs,
                mode="actionability_weighted_min_horizon",
                is_sell_task=jnp.ones((1, 1), dtype=jnp.float32),
                delta=len(ratios),
                actionability_eta=0.1,
                actionability_depth=1,
                actionability_far_level_weight=0.25,
            )
            targets.append(float(labels[0, 0, 0, 0]))

        self.assertLess(targets[0], targets[1])
        self.assertLess(targets[1], targets[2])

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
