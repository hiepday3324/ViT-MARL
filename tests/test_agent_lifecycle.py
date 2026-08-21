import os
import sys
import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from gymnax_exchange.jaxen.marl_env import (
    MARLEnv,
    mask_inactive_agent_value,
    resolve_agent_lifecycle,
)
from gymnax_exchange.jaxen.vision_env import ExecutionAgent
from gymnax_exchange.jaxob import JaxOrderBookArrays as job
from gymnax_exchange.jaxob import jaxob_constants as cst
from gymnax_exchange.jaxob.jaxob_config import (
    Execution_EnvironmentConfig,
    World_EnvironmentConfig,
)
from gymnax_exchange.jaxrl.MARL.execution_episode_metrics import (
    accumulate_execution_episode_metrics,
)
from gymnax_exchange.jaxrl.MARL.ppo_lifecycle import (
    calculate_gae,
    compute_masked_ppo_terms,
    next_rnn_reset,
)
from tests.test_shadow_twap_teacher import _synthetic_marl_fixture


class AgentLifecycleTest(unittest.TestCase):
    def test_early_execution_completion_is_absorbing_until_world_horizon(self):
        active = jnp.array([True])
        rnn_reset = jnp.array([False])
        rows = []

        for step in range(8):
            task_completed = jnp.array([step >= 2])
            global_done = jnp.array(step == 7)
            agent_done, next_active = resolve_agent_lifecycle(
                active,
                task_completed,
                global_done,
            )
            reward = mask_inactive_agent_value(
                jnp.array([3.0 if step == 2 else 1.0]),
                active,
            )
            rows.append(
                (
                    bool(active[0]),
                    bool(agent_done[0]),
                    bool(rnn_reset[0]),
                    bool(global_done),
                    float(reward[0]),
                )
            )
            rnn_reset = next_rnn_reset(agent_done, global_done)
            active = next_active

        self.assertEqual(rows[2], (True, True, False, False, 3.0))
        self.assertEqual(rows[3], (False, False, True, False, 0.0))
        self.assertEqual(rows[4], (False, False, False, False, 0.0))
        self.assertEqual(rows[7], (False, False, False, True, 0.0))
        self.assertTrue(bool(active[0]))
        self.assertTrue(bool(rnn_reset[0]))

    def test_world_horizon_is_independent_of_execution_completion(self):
        env_like = SimpleNamespace(
            multi_agent_config=SimpleNamespace(
                world_config=World_EnvironmentConfig(
                    ep_type="fixed_steps",
                    episode_time=50,
                )
            )
        )
        state = SimpleNamespace(
            max_steps_in_episode=jnp.array(51),
            step_counter=jnp.array(32),
        )
        self.assertFalse(bool(MARLEnv._world_horizon_done(env_like, state)))
        state.step_counter = jnp.array(50)
        self.assertTrue(bool(MARLEnv._world_horizon_done(env_like, state)))

    def test_absorbing_execution_keeps_world_mm_and_shadow_running(self):
        env, params, _obs, state = _synthetic_marl_fixture(seed=5)
        exe_state = state.agent_states[1]
        exe_tid = params.agent_params[1].trader_id[0]
        completed_exe_state = exe_state.replace(
            quant_executed=exe_state.task_to_execute,
        )
        own_bid = jnp.asarray(
            [9600, 5, -800, exe_tid, 34_200, 0],
            dtype=jnp.int32,
        )
        world_state = state.world_state.replace(
            bid_raw_orders=state.world_state.bid_raw_orders.at[3].set(own_bid),
        )
        state = state.replace(
            world_state=world_state,
            agent_states=[state.agent_states[0], completed_exe_state],
        )
        shadow_fill_before = float(
            completed_exe_state.shadow_cumulative_filled_quantity[0]
        )

        _next_obs, next_state, rewards, dones, _info = env.step_env(
            jax.random.PRNGKey(17),
            state,
            [
                jnp.asarray(0, dtype=jnp.int32),
                jnp.zeros((3,), dtype=jnp.float32),
            ],
            params,
        )

        self.assertFalse(bool(dones["__all__"]))
        self.assertTrue(bool(dones["active"][0][0]))
        self.assertFalse(bool(dones["active"][1][0]))
        self.assertEqual(int(next_state.world_state.step_counter), 1)
        self.assertEqual(float(rewards[1][0]), 0.0)
        self.assertEqual(
            float(next_state.agent_states[1].quant_executed[0]),
            float(completed_exe_state.task_to_execute[0]),
        )
        self.assertFalse(
            bool(
                jnp.any(
                    next_state.world_state.bid_raw_orders[
                        :, cst.OrderSideFeat.TID.value
                    ]
                    == exe_tid
                )
            )
        )
        self.assertGreater(
            float(
                next_state.agent_states[
                    1
                ].shadow_cumulative_filled_quantity[0]
            ),
            shadow_fill_before,
        )

    def test_real_terminal_transition_precedes_absorbing_transition(self):
        env, params, _obs, state = _synthetic_marl_fixture(seed=7)
        exe_state = state.agent_states[1].replace(
            quant_executed=jnp.asarray([70.0], dtype=jnp.float32),
        )
        state = state.replace(
            agent_states=[state.agent_states[0], exe_state],
        )
        aggressive_sell = jnp.asarray(
            [4, 1, 1_000, 9_700, 9_999, 900, 34_201, 0],
            dtype=jnp.int32,
        )
        loaded_params = params.loaded_params.replace(
            message_data=params.loaded_params.message_data.at[0].set(
                aggressive_sell
            )
        )
        params = params.replace(loaded_params=loaded_params)
        actions = [
            jnp.asarray(0, dtype=jnp.int32),
            jnp.zeros((3,), dtype=jnp.float32),
        ]

        _obs_1, state_1, rewards_1, dones_1, _info_1 = env.step_env(
            jax.random.PRNGKey(31),
            state,
            actions,
            params,
        )
        terminal_reward = float(rewards_1[1][0])
        terminal_quantity = float(state_1.agent_states[1].quant_executed[0])

        self.assertTrue(bool(dones_1["active"][1][0]))
        self.assertTrue(bool(dones_1["agents"][1][0]))
        self.assertFalse(bool(dones_1["__all__"]))
        self.assertTrue(np.isfinite(terminal_reward))
        self.assertAlmostEqual(terminal_quantity, 80.0, places=6)

        shadow_fill_at_terminal = float(
            state_1.agent_states[1].shadow_cumulative_filled_quantity[0]
        )
        _obs_2, state_2, rewards_2, dones_2, _info_2 = env.step_env(
            jax.random.PRNGKey(32),
            state_1,
            actions,
            params,
        )

        self.assertFalse(bool(dones_2["active"][1][0]))
        self.assertFalse(bool(dones_2["agents"][1][0]))
        self.assertFalse(bool(dones_2["__all__"]))
        self.assertEqual(float(rewards_2[1][0]), 0.0)
        self.assertEqual(
            float(state_2.agent_states[1].quant_executed[0]),
            terminal_quantity,
        )
        self.assertEqual(int(state_2.world_state.step_counter), 2)
        self.assertTrue(bool(dones_2["active"][0][0]))
        self.assertGreater(
            float(state_2.agent_states[1].shadow_cumulative_filled_quantity[0]),
            shadow_fill_at_terminal,
        )

    def test_gae_stops_at_agent_terminal_and_ignores_absorbing_tail(self):
        rewards = jnp.array([[1.0], [2.0], [3.0], [0.0], [0.0]])
        values = jnp.array([[0.5], [0.6], [0.7], [5.0], [6.0]])
        agent_done = jnp.array([[False], [False], [True], [False], [False]])
        advantages, _targets = calculate_gae(
            0.9,
            0.8,
            rewards,
            values,
            agent_done,
            jnp.array([7.0]),
        )

        expected = np.zeros((5, 1), dtype=np.float32)
        gae = np.zeros((1,), dtype=np.float32)
        next_value = np.array([7.0], dtype=np.float32)
        for index in range(4, -1, -1):
            not_done = 1.0 - float(agent_done[index, 0])
            delta = (
                float(rewards[index, 0])
                + 0.9 * float(next_value[0]) * not_done
                - float(values[index, 0])
            )
            gae = delta + 0.9 * 0.8 * not_done * gae
            expected[index, 0] = gae[0]
            next_value = np.asarray(values[index])

        np.testing.assert_allclose(np.asarray(advantages), expected, atol=1e-6)

        huge_tail_rewards = rewards.at[3:].set(1e6)
        huge_tail_values = values.at[3:].set(-1e6)
        changed, _ = calculate_gae(
            0.9,
            0.8,
            huge_tail_rewards,
            huge_tail_values,
            agent_done,
            jnp.array([1e6]),
        )
        np.testing.assert_allclose(
            np.asarray(changed[:3]),
            np.asarray(advantages[:3]),
            atol=1e-5,
        )

    def test_inactive_samples_cannot_change_ppo_terms(self):
        active = jnp.array([[True], [True], [True], [False], [False]])
        kwargs = {
            "ratio": jnp.array([[1.0], [1.1], [0.9], [1.2], [0.8]]),
            "logratio": jnp.log(
                jnp.array([[1.0], [1.1], [0.9], [1.2], [0.8]])
            ),
            "advantage": jnp.array([[1.0], [2.0], [4.0], [8.0], [16.0]]),
            "value_loss_samples": jnp.array(
                [[0.1], [0.2], [0.3], [0.4], [0.5]]
            ),
            "entropy_samples": jnp.array([[0.7], [0.8], [0.9], [1.0], [1.1]]),
            "agent_active": active,
            "clip_eps": 0.2,
        }
        baseline = compute_masked_ppo_terms(**kwargs)
        perturbed = dict(kwargs)
        for key in (
            "ratio",
            "logratio",
            "advantage",
            "value_loss_samples",
            "entropy_samples",
        ):
            perturbed[key] = kwargs[key].at[3:].set(
                jnp.array([[1234.0], [-987.0]])
            )
        changed = compute_masked_ppo_terms(**perturbed)

        for field in (
            "actor_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "clip_frac",
        ):
            self.assertAlmostEqual(
                float(getattr(baseline, field)),
                float(getattr(changed, field)),
                places=6,
            )
        np.testing.assert_allclose(
            np.asarray(baseline.normalized_advantage[:3]),
            np.asarray(changed.normalized_advantage[:3]),
            atol=1e-6,
        )
        self.assertTrue(
            bool(jnp.all(changed.normalized_advantage[3:] == 0.0))
        )

    def test_masked_ppo_is_finite_with_zero_or_one_active_sample(self):
        common = {
            "ratio": jnp.ones((3, 1)),
            "logratio": jnp.zeros((3, 1)),
            "advantage": jnp.array([[1.0], [2.0], [3.0]]),
            "value_loss_samples": jnp.ones((3, 1)),
            "entropy_samples": jnp.ones((3, 1)),
            "clip_eps": 0.2,
        }
        for mask in (
            jnp.zeros((3, 1), dtype=jnp.bool_),
            jnp.array([[True], [False], [False]]),
        ):
            terms = compute_masked_ppo_terms(agent_active=mask, **common)
            for value in terms[:-1]:
                self.assertTrue(bool(jnp.isfinite(value)))
            self.assertTrue(bool(jnp.all(jnp.isfinite(terms.normalized_advantage))))

    def test_masked_ppo_helper_compiles_under_single_device_pmap(self):
        devices = jax.local_devices()[:1]

        def mapped(mask):
            return compute_masked_ppo_terms(
                ratio=jnp.ones((3,)),
                logratio=jnp.zeros((3,)),
                advantage=jnp.asarray([1.0, 2.0, 3.0]),
                value_loss_samples=jnp.asarray([0.1, 0.2, 0.3]),
                entropy_samples=jnp.asarray([0.4, 0.5, 0.6]),
                agent_active=mask,
                clip_eps=0.2,
                axis_name="device_batch",
            )

        result = jax.pmap(
            mapped,
            axis_name="device_batch",
            devices=devices,
        )(jnp.asarray([[True, False, True]]))
        for value in result[:-1]:
            self.assertTrue(bool(jnp.all(jnp.isfinite(value))))

    def test_absorbing_execution_messages_cancel_without_new_orders(self):
        world_config = World_EnvironmentConfig(
            nOrders=8,
            nTrades=8,
        )
        agent = ExecutionAgent(
            cfg=Execution_EnvironmentConfig(
                action_space="policy_blending",
                observation_space="execution_policy",
            ),
            world_config=world_config,
        )
        asks = jnp.full((8, 6), cst.EMPTY_SLOT, dtype=jnp.int32)
        bids = jnp.full((8, 6), cst.EMPTY_SLOT, dtype=jnp.int32)
        trader_id = -101
        bids = bids.at[0].set(
            jnp.array([10_000, 12, -250, trader_id, 34_200, 0], dtype=jnp.int32)
        )
        world_state = SimpleNamespace(
            ask_raw_orders=asks,
            bid_raw_orders=bids,
            time=jnp.array([34_201, 5], dtype=jnp.int32),
        )
        agent_state = SimpleNamespace(is_sell_task=jnp.array(False))
        agent_params = SimpleNamespace(trader_id=jnp.array(trader_id))

        action_msgs, cancel_msgs = agent._get_absorbing_messages(
            world_state,
            agent_state,
            agent_params,
        )
        self.assertTrue(bool(jnp.all(action_msgs == 0)))
        self.assertEqual(int(cancel_msgs[0, 0]), 2)
        self.assertEqual(int(cancel_msgs[0, 4]), -250)
        self.assertEqual(int(cancel_msgs[0, 5]), trader_id)

        trades = jnp.full(
            (world_config.nTrades, cst.TRADE_FEAT),
            cst.EMPTY_SLOT,
            dtype=jnp.int32,
        )
        _asks_after, bids_after, trades_after = job.scan_through_entire_array(
            world_config,
            jax.random.PRNGKey(0),
            cancel_msgs,
            (asks, bids, trades),
        )
        self.assertFalse(
            bool(
                jnp.any(
                    bids_after[:, cst.OrderSideFeat.TID.value] == trader_id
                )
            )
        )
        self.assertFalse(
            bool(
                jnp.any(
                    trades_after[:, cst.TradesFeat.PASS_TID.value] == trader_id
                )
            )
        )

    def test_execution_episode_metrics_finalize_before_world_horizon(self):
        rewards = jnp.array([[1.0], [2.0], [3.0], [0.0], [0.0]])
        agent_done = jnp.array([[False], [False], [True], [False], [False]])
        quant_left = jnp.array([[50.0], [25.0], [0.0], [0.0], [0.0]])
        zeros = jnp.zeros_like(rewards)
        invalid = jnp.zeros_like(agent_done)
        running, metrics = accumulate_execution_episode_metrics(
            jnp.zeros((1,), dtype=jnp.float32),
            rewards,
            agent_done,
            quant_left,
            jnp.full_like(quant_left, 100.0),
            full_completion=agent_done & (quant_left == 0.0),
            realized_is_bps=zeros,
            realized_is_valid=invalid,
            forced_liquidation_is_bps=zeros,
            forced_liquidation_is_valid=invalid,
            twap_forced_liquidation_is_bps=zeros,
            twap_forced_liquidation_is_valid=invalid,
            twap_advantage_bps=zeros,
            twap_comparison_valid=invalid,
            twap_win=zeros,
        )
        self.assertEqual(int(metrics.episode_count), 1)
        self.assertAlmostEqual(float(metrics.episode_return_mean), 6.0, places=6)
        self.assertAlmostEqual(float(running[0]), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
