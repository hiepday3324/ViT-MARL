import os
import sys
import unittest

import jax
import jax.numpy as jnp

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from gymnax_exchange.jaxob import jaxob_constants as cst
from gymnax_exchange.jaxrl.MARL.reliability_targets import (
    build_liquidity_survival_targets,
    masked_reliability_loss,
    resolve_rollout_is_sell_task,
)


class ReliabilityTargetTest(unittest.TestCase):
    tick_size = 100.0
    mid_price = 10_000.0
    q0 = 100.0

    def _ask_price(self, level=0):
        return self.mid_price + (level + 1) * self.tick_size

    def _bid_price(self, level=0):
        return self.mid_price - (level + 1) * self.tick_size

    def _make_case(self, *, delta=1, levels=1, n_trades=4):
        time_steps = 1 + delta
        vision_obs = jnp.zeros((time_steps, 1, levels, 3, 2), dtype=jnp.float32)
        gaps = jnp.arange(1, levels + 1, dtype=jnp.float32)
        vision_obs = vision_obs.at[:, 0, :, 0, 0].set(gaps)
        vision_obs = vision_obs.at[:, 0, :, 0, 1].set(gaps)
        vision_obs = vision_obs.at[:, 0, :, 1, :].set(jnp.log1p(self.q0))
        vision_obs = vision_obs.at[:, 0, :, 2, :].set(jnp.log1p(self.q0))

        n_orders = max(8, levels * 2)
        ask_orders = jnp.full((time_steps, 1, n_orders, 6), -1, dtype=jnp.float32)
        bid_orders = jnp.full((time_steps, 1, n_orders, 6), -1, dtype=jnp.float32)
        for level in range(levels):
            ask_orders = self._set_order(
                ask_orders,
                time_index=0,
                slot=level,
                price=self._ask_price(level),
                quantity=self.q0,
                oid=-2 - level,
            )
            bid_orders = self._set_order(
                bid_orders,
                time_index=0,
                slot=level,
                price=self._bid_price(level),
                quantity=self.q0,
                oid=-2 - level,
            )

        trades = jnp.full(
            (time_steps, 1, n_trades, cst.TRADE_FEAT),
            cst.EMPTY_SLOT,
            dtype=jnp.float32,
        )
        trade_valid_mask = jnp.zeros((time_steps, 1, n_trades), dtype=jnp.bool_)
        saturated = jnp.zeros((time_steps, 1), dtype=jnp.bool_)
        done = jnp.zeros((time_steps, 1), dtype=jnp.bool_)
        return {
            "vision_obs": vision_obs,
            "ask_orders": ask_orders,
            "bid_orders": bid_orders,
            "trades": trades,
            "trade_valid_mask": trade_valid_mask,
            "saturated": saturated,
            "done": done,
            "delta": delta,
        }

    @staticmethod
    def _set_order(raw_orders, *, time_index, slot, price, quantity, oid):
        raw_orders = raw_orders.at[time_index, 0, slot, cst.OrderSideFeat.P.value].set(price)
        raw_orders = raw_orders.at[time_index, 0, slot, cst.OrderSideFeat.Q.value].set(quantity)
        raw_orders = raw_orders.at[time_index, 0, slot, cst.OrderSideFeat.OID.value].set(oid)
        return raw_orders

    @staticmethod
    def _set_trade(case, *, transition, slot, price, signed_quantity, passive_oid=-2):
        case["trades"] = case["trades"].at[
            transition, 0, slot, cst.TradesFeat.P.value
        ].set(price)
        case["trades"] = case["trades"].at[
            transition, 0, slot, cst.TradesFeat.Q.value
        ].set(signed_quantity)
        case["trades"] = case["trades"].at[
            transition, 0, slot, cst.TradesFeat.PASS_OID.value
        ].set(passive_oid)
        case["trade_valid_mask"] = case["trade_valid_mask"].at[
            transition, 0, slot
        ].set(True)

    def _set_future_side_volume(
        self,
        case,
        *,
        tau,
        side,
        quantity,
        level=0,
        price=None,
        oid=-100,
        slot=0,
    ):
        if price is None:
            price = self._ask_price(level) if side == "ask" else self._bid_price(level)
        key = "ask_orders" if side == "ask" else "bid_orders"
        case[key] = self._set_order(
            case[key],
            time_index=tau,
            slot=slot,
            price=price,
            quantity=quantity,
            oid=oid,
        )

    def _build_targets(self, case, *, return_diagnostics=False):
        time_steps = case["vision_obs"].shape[0]
        mid_prices = jnp.full((time_steps, 1), self.mid_price, dtype=jnp.float32)
        return build_liquidity_survival_targets(
            case["vision_obs"],
            mid_prices,
            tick_size=self.tick_size,
            survival_delta_steps=case["delta"],
            survival_min_volume=1.0,
            num_steps=1,
            episode_done=case["done"],
            ask_raw_orders=case["ask_orders"],
            bid_raw_orders=case["bid_orders"],
            new_trades=case["trades"],
            trade_valid_mask=case["trade_valid_mask"],
            trade_buffer_saturated=case["saturated"],
            return_diagnostics=return_diagnostics,
        )

    def test_resting_liquidity_has_unit_target(self):
        case = self._make_case()
        self._set_future_side_volume(case, tau=1, side="ask", quantity=100.0)
        self._set_future_side_volume(case, tau=1, side="bid", quantity=100.0)

        labels, mask = self._build_targets(case)

        self.assertEqual(labels.shape, (1, 1, 1, 2))
        self.assertEqual(mask.shape, (1, 1, 1, 2))
        self.assertTrue(bool(jnp.all(mask == 1.0)))
        self.assertTrue(bool(jnp.allclose(labels, 1.0)))

    def test_full_execution_has_unit_target_for_both_sides(self):
        for side, side_index, signed_quantity, price in (
            ("ask", 0, -100.0, self._ask_price()),
            ("bid", 1, 100.0, self._bid_price()),
        ):
            with self.subTest(side=side):
                case = self._make_case()
                self._set_trade(
                    case,
                    transition=0,
                    slot=0,
                    price=price,
                    signed_quantity=signed_quantity,
                )
                labels, mask = self._build_targets(case)
                self.assertEqual(float(mask[0, 0, 0, side_index]), 1.0)
                self.assertAlmostEqual(float(labels[0, 0, 0, side_index]), 1.0, places=6)

    def test_cumulative_execution_persists_after_book_liquidity_is_consumed(self):
        case = self._make_case(delta=10)
        for tau in range(1, 5):
            self._set_future_side_volume(case, tau=tau, side="ask", quantity=100.0)
        self._set_trade(
            case,
            transition=4,
            slot=0,
            price=self._ask_price(),
            signed_quantity=-100.0,
        )

        labels, mask = self._build_targets(case)

        self.assertEqual(float(mask[0, 0, 0, 0]), 1.0)
        self.assertAlmostEqual(float(labels[0, 0, 0, 0]), 1.0, places=6)

    def test_partial_cancel_with_execution_uses_execution_plus_resting(self):
        case = self._make_case()
        self._set_trade(
            case,
            transition=0,
            slot=0,
            price=self._ask_price(),
            signed_quantity=-20.0,
        )
        self._set_future_side_volume(case, tau=1, side="ask", quantity=50.0)

        labels, _mask = self._build_targets(case)

        self.assertAlmostEqual(float(labels[0, 0, 0, 0]), 0.7, places=6)

    def test_execution_plus_resting_is_clipped_to_one(self):
        case = self._make_case()
        self._set_trade(
            case,
            transition=0,
            slot=0,
            price=self._ask_price(),
            signed_quantity=-80.0,
        )
        self._set_future_side_volume(case, tau=1, side="ask", quantity=50.0)

        labels, _mask = self._build_targets(case)

        self.assertAlmostEqual(float(labels[0, 0, 0, 0]), 1.0, places=6)

    def test_complete_cancel_has_zero_target(self):
        case = self._make_case()

        labels, mask = self._build_targets(case)

        self.assertEqual(float(mask[0, 0, 0, 0]), 1.0)
        self.assertAlmostEqual(float(labels[0, 0, 0, 0]), 0.0, places=6)

    def test_immediate_refill_by_new_oid_has_unit_target(self):
        case = self._make_case()
        self._set_future_side_volume(
            case,
            tau=1,
            side="ask",
            quantity=100.0,
            oid=-999,
        )

        labels, _mask = self._build_targets(case)

        self.assertAlmostEqual(float(labels[0, 0, 0, 0]), 1.0, places=6)

    def test_late_refill_scores_below_immediate_refill(self):
        late = self._make_case(delta=3)
        self._set_future_side_volume(late, tau=3, side="ask", quantity=100.0)
        immediate = self._make_case(delta=3)
        for tau in range(1, 4):
            self._set_future_side_volume(immediate, tau=tau, side="ask", quantity=100.0)

        late_labels, _ = self._build_targets(late)
        immediate_labels, _ = self._build_targets(immediate)

        self.assertAlmostEqual(float(late_labels[0, 0, 0, 0]), 1.0 / 3.0, places=6)
        self.assertAlmostEqual(float(immediate_labels[0, 0, 0, 0]), 1.0, places=6)
        self.assertLess(float(late_labels[0, 0, 0, 0]), float(immediate_labels[0, 0, 0, 0]))

    def test_wrong_price_trade_and_refill_do_not_count(self):
        case = self._make_case()
        wrong_price = self._ask_price() + self.tick_size
        self._set_trade(
            case,
            transition=0,
            slot=0,
            price=wrong_price,
            signed_quantity=-100.0,
        )
        self._set_future_side_volume(
            case,
            tau=1,
            side="ask",
            quantity=100.0,
            price=wrong_price,
        )

        labels, _mask = self._build_targets(case)

        self.assertAlmostEqual(float(labels[0, 0, 0, 0]), 0.0, places=6)

    def test_opposite_side_resting_does_not_count(self):
        case = self._make_case()
        self._set_future_side_volume(
            case,
            tau=1,
            side="bid",
            quantity=100.0,
            price=self._ask_price(),
        )

        labels, _mask = self._build_targets(case)

        self.assertAlmostEqual(float(labels[0, 0, 0, 0]), 0.0, places=6)

    def test_opposite_side_execution_does_not_count(self):
        case = self._make_case()
        self._set_trade(
            case,
            transition=0,
            slot=0,
            price=self._ask_price(),
            signed_quantity=100.0,
        )

        labels, _mask = self._build_targets(case)

        self.assertAlmostEqual(float(labels[0, 0, 0, 0]), 0.0, places=6)

    def test_done_masks_only_execution_transitions_in_horizon(self):
        for done_index, expected_mask in ((0, 0.0), (1, 0.0), (2, 1.0)):
            with self.subTest(done_index=done_index):
                case = self._make_case(delta=2)
                for tau in (1, 2):
                    self._set_future_side_volume(case, tau=tau, side="ask", quantity=100.0)
                case["done"] = case["done"].at[done_index, 0].set(True)
                _labels, mask = self._build_targets(case)
                self.assertEqual(float(mask[0, 0, 0, 0]), expected_mask)

    def test_saturated_trade_buffer_masks_only_execution_horizon(self):
        for saturated_index, expected_mask in ((0, 0.0), (1, 0.0), (2, 1.0)):
            with self.subTest(saturated_index=saturated_index):
                case = self._make_case(delta=2)
                for tau in (1, 2):
                    self._set_future_side_volume(case, tau=tau, side="ask", quantity=100.0)
                case["saturated"] = case["saturated"].at[saturated_index, 0].set(True)
                _labels, mask = self._build_targets(case)
                self.assertEqual(float(mask[0, 0, 0, 0]), expected_mask)

    def test_synthetic_reset_order_ids_are_not_masked_or_tracked(self):
        case = self._make_case()
        self._set_future_side_volume(
            case,
            tau=1,
            side="ask",
            quantity=100.0,
            oid=-12345,
        )

        labels, mask = self._build_targets(case)

        self.assertEqual(float(mask[0, 0, 0, 0]), 1.0)
        self.assertAlmostEqual(float(labels[0, 0, 0, 0]), 1.0, places=6)

    def test_trade_at_t_plus_delta_is_excluded(self):
        case = self._make_case(delta=2)
        self._set_trade(
            case,
            transition=2,
            slot=0,
            price=self._ask_price(),
            signed_quantity=-100.0,
        )

        labels, mask = self._build_targets(case)

        self.assertEqual(float(mask[0, 0, 0, 0]), 1.0)
        self.assertAlmostEqual(float(labels[0, 0, 0, 0]), 0.0, places=6)

    def test_q0_at_minimum_volume_is_masked(self):
        case = self._make_case()
        case["ask_orders"] = case["ask_orders"].at[
            0, 0, 0, cst.OrderSideFeat.Q.value
        ].set(1.0)

        _labels, mask = self._build_targets(case)

        self.assertEqual(float(mask[0, 0, 0, 0]), 0.0)

    def test_nonfinite_trade_tensor_masks_target(self):
        case = self._make_case()
        case["trades"] = case["trades"].at[
            0, 0, 0, cst.TradesFeat.P.value
        ].set(jnp.nan)

        labels, mask = self._build_targets(case)

        self.assertEqual(float(mask[0, 0, 0, 0]), 0.0)
        self.assertTrue(bool(jnp.all(jnp.isfinite(labels))))

    def test_nonfinite_future_raw_book_masks_target(self):
        case = self._make_case()
        case["ask_orders"] = case["ask_orders"].at[
            1, 0, 0, cst.OrderSideFeat.Q.value
        ].set(jnp.nan)

        labels, mask = self._build_targets(case)

        self.assertEqual(float(mask[0, 0, 0, 0]), 0.0)
        self.assertTrue(bool(jnp.all(jnp.isfinite(labels))))

    def test_diagnostics_report_execution_aware_components(self):
        case = self._make_case()
        self._set_trade(
            case,
            transition=0,
            slot=0,
            price=self._ask_price(),
            signed_quantity=-20.0,
        )
        self._set_future_side_volume(case, tau=1, side="ask", quantity=50.0)

        labels, mask, diag = self._build_targets(case, return_diagnostics=True)

        self.assertEqual(labels.shape, mask.shape)
        self.assertTrue(bool(jnp.isfinite(diag["q0_mean"])))
        self.assertTrue(bool(jnp.isfinite(diag["q_tau_mean"])))
        self.assertTrue(bool(jnp.isfinite(diag["cumulative_executed_mean"])))
        self.assertTrue(bool(jnp.isfinite(diag["cancel_star_mean"])))
        self.assertTrue(bool(jnp.isfinite(diag["ask_target_std"])))
        self.assertTrue(bool(jnp.isfinite(diag["bid_target_std"])))
        self.assertEqual(diag["target_level_mean_ask"].shape, (1,))
        self.assertEqual(diag["target_level_mean_bid"].shape, (1,))
        self.assertAlmostEqual(float(diag["ask_target_mean"]), 0.7, places=6)

    def test_missing_execution_log_raises(self):
        case = self._make_case()
        time_steps = case["vision_obs"].shape[0]
        mid_prices = jnp.full((time_steps, 1), self.mid_price, dtype=jnp.float32)

        with self.assertRaisesRegex(ValueError, "new_trades"):
            build_liquidity_survival_targets(
                case["vision_obs"],
                mid_prices,
                tick_size=self.tick_size,
                survival_delta_steps=1,
                survival_min_volume=1.0,
                num_steps=1,
                ask_raw_orders=case["ask_orders"],
                bid_raw_orders=case["bid_orders"],
            )

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

    def test_logits_bce_keeps_gradient_for_saturated_probabilities(self):
        logits = jnp.array([[-100.0, 100.0]], dtype=jnp.float32)
        labels = jnp.array([[1.0, 0.0]], dtype=jnp.float32)
        mask = jnp.ones_like(labels)

        def loss_fn(candidate_logits):
            return masked_reliability_loss(
                jax.nn.sigmoid(candidate_logits),
                labels,
                mask,
                loss_type="bce",
                reliability_logits=candidate_logits,
            )

        loss, grad = jax.value_and_grad(loss_fn)(logits)

        self.assertTrue(bool(jnp.isfinite(loss)))
        self.assertTrue(bool(jnp.all(jnp.isfinite(grad))))
        self.assertGreater(float(jnp.linalg.norm(grad)), 0.0)

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
