import dataclasses
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from gymnax_exchange.jaxen.StatesandParams import (
    ExecEnvParams,
    ExecEnvState,
    LoadedEnvParams,
    MultiAgentParams,
    MultiAgentState,
    WorldState,
)
from gymnax_exchange.jaxen.marl_env import MARLEnv
from gymnax_exchange.jaxen.mm_env import MarketMakingAgent
from gymnax_exchange.jaxen.shadow_twap import (
    compute_terminal_execution_benchmark,
    shadow_market_order_sweep,
    simulate_shadow_twap_interval,
)
from gymnax_exchange.jaxen.twap_schedule import fixed_step_twap_child_quantity
from gymnax_exchange.jaxen.vision_env import (
    ExecutionAgent,
    _compute_windowed_itt_reward,
    _update_itt_reward_window,
    _zero_itt_reward_window,
)
from gymnax_exchange.jaxob.jaxob_config import (
    Execution_EnvironmentConfig,
    MarketMaking_EnvironmentConfig,
    MultiAgentConfig,
    World_EnvironmentConfig,
)


def _orders(prices, quantities, capacity=None, trader_id=-100):
    capacity = capacity or len(prices)
    orders = jnp.full((capacity, 6), -1, dtype=jnp.int32)
    count = len(prices)
    if count:
        orders = orders.at[:count, 0].set(jnp.asarray(prices, dtype=jnp.int32))
        orders = orders.at[:count, 1].set(jnp.asarray(quantities, dtype=jnp.int32))
        orders = orders.at[:count, 2].set(-2 - jnp.arange(count, dtype=jnp.int32))
        orders = orders.at[:count, 3].set(trader_id)
        orders = orders.at[:count, 4:].set(0)
    return orders


def _run_interval(
    asks,
    bids,
    scheduled,
    remaining=100.0,
    cumulative_fill=0.0,
    cumulative_cost=0.0,
    is_sell=False,
    tick_size=1,
):
    return simulate_shadow_twap_interval(
        asks,
        bids,
        scheduled,
        remaining,
        cumulative_fill,
        cumulative_cost,
        is_sell,
        tick_size,
    )


def _trades(rows, capacity=8, trader_id=-101):
    trades = jnp.full((capacity, 8), -1, dtype=jnp.int32)
    for index, (price, quantity) in enumerate(rows):
        trades = trades.at[index].set(jnp.asarray([
            price,
            quantity,
            1000 + index,
            2000 + index,
            1,
            index,
            500,
            trader_id,
        ], dtype=jnp.int32))
    return trades


class _PostActionShadowReferenceMARLEnv(MARLEnv):
    """Test-only reference that reproduces the pre-fix shadow timing."""

    def _get_shadow_twap_snapshot(
        self,
        pre_step_world_state,
        post_step_asks,
        post_step_bids,
    ):
        del pre_step_world_state
        return post_step_asks, post_step_bids


class _SyntheticBaseEnv:
    def __init__(self, config):
        self.cfg = config
        self.n_data_msg_per_step = config.n_data_msg_per_step

    def _get_data_messages(self, message_data, start, step_counter, end_time_s):
        del end_time_s
        offset = start + self.n_data_msg_per_step * step_counter
        return jax.lax.dynamic_slice_in_dim(
            message_data,
            offset,
            self.n_data_msg_per_step,
            axis=0,
        )


def _synthetic_marl_fixture(seed):
    world_config = World_EnvironmentConfig(
        nOrders=32,
        nTrades=100,
        n_data_msg_per_step=1,
        episode_time=8,
        ep_type="fixed_steps",
        shuffle_action_messages=True,
        use_pickles_for_init=False,
    )
    mm_config = MarketMaking_EnvironmentConfig(
        action_space="spread_skew",
        observation_space="engineered",
    )
    exe_config = Execution_EnvironmentConfig(
        action_space="policy_blending",
        observation_space="execution_policy",
        task="buy",
        task_size=80,
    )
    config = MultiAgentConfig(
        world_config=world_config,
        dict_of_agents_configs={
            "MarketMaking": mm_config,
            "Execution": exe_config,
        },
        number_of_agents_per_type=[1, 1],
    )

    env = object.__new__(MARLEnv)
    env.multi_agent_config = config
    env.num_agents = 2
    env.type_names = ["MM", "EXE"]
    env.instance_list = [
        MarketMakingAgent(mm_config, world_config),
        ExecutionAgent(exe_config, world_config),
    ]
    env.list_of_agents_configs = [mm_config, exe_config]
    env.action_spaces = [instance.action_space() for instance in env.instance_list]
    env.observation_spaces = [
        instance.observation_space() for instance in env.instance_list
    ]
    env.num_msgs_per_step = (
        world_config.n_data_msg_per_step
        + mm_config.num_messages_by_agent
        + exe_config.num_messages_by_agent
    )
    env.num_action_msgs_per_step_by_all_agents = (
        mm_config.num_action_messages_by_agent
        + exe_config.num_action_messages_by_agent
    )
    env.base_env = _SyntheticBaseEnv(world_config)

    asks = _orders(
        [10000, 10100, 10200],
        [15, 20, 30],
        capacity=world_config.nOrders,
        trader_id=700,
    )
    bids = _orders(
        [9900, 9800, 9700],
        [15, 20, 30],
        capacity=world_config.nOrders,
        trader_id=700,
    )
    best_asks = jnp.tile(
        jnp.asarray([[10000, 15]], dtype=jnp.int32),
        (env.num_msgs_per_step, 1),
    )
    best_bids = jnp.tile(
        jnp.asarray([[9900, 15]], dtype=jnp.int32),
        (env.num_msgs_per_step, 1),
    )
    world_state = WorldState(
        ask_raw_orders=asks,
        bid_raw_orders=bids,
        trades=jnp.full((world_config.nTrades, 8), -1, dtype=jnp.int32),
        init_time=jnp.asarray([34200, 0], dtype=jnp.int32),
        window_index=0,
        max_steps_in_episode=world_config.episode_time + 1,
        start_index=0,
        step_counter=0,
        best_bids=best_bids,
        best_asks=best_asks,
        time=jnp.asarray([34200, 0], dtype=jnp.int32),
        order_id_counter=-200,
        mid_price=9950.0,
        delta_time=0.0,
    )

    data_messages = jnp.asarray([
        [1, -1, 2, 10500, 3000 + step, 900, 34201 + step, 0]
        if step % 2 == 0
        else [1, 1, 2, 9500, 3000 + step, 900, 34201 + step, 0]
        for step in range(world_config.episode_time)
    ], dtype=jnp.int32)
    mm_params, next_trader_id = env.instance_list[0].default_params(
        mm_config,
        world_config.trader_id_range_start,
        1,
    )
    exe_params, _ = env.instance_list[1].default_params(
        exe_config,
        next_trader_id,
        1,
    )
    params = MultiAgentParams(
        loaded_params=LoadedEnvParams(
            message_data=data_messages,
            book_data=jnp.zeros((1, world_config.book_depth * 4), dtype=jnp.int32),
            init_states_array=jnp.zeros((1, 1), dtype=jnp.int32),
        ),
        agent_params=[mm_params, exe_params],
    )

    mm_param = jax.tree_util.tree_map(lambda x: x[0], mm_params)
    exe_param = jax.tree_util.tree_map(lambda x: x[0], exe_params)
    mm_obs, mm_state = env.instance_list[0].reset_env(
        mm_param,
        jax.random.PRNGKey(seed + 1),
        world_state,
        env.num_msgs_per_step,
    )
    exe_obs, exe_state = env.instance_list[1].reset_env(
        exe_param,
        jax.random.PRNGKey(seed + 2),
        world_state,
        env.num_msgs_per_step,
    )
    add_batch_axis = lambda tree: jax.tree_util.tree_map(
        lambda x: jnp.asarray(x)[None],
        tree,
    )
    state = MultiAgentState(
        world_state=world_state,
        agent_states=[add_batch_axis(mm_state), add_batch_axis(exe_state)],
    )
    obs = [add_batch_axis(mm_obs), add_batch_axis(exe_obs)]
    return env, params, obs, state


class ShadowTWAPTeacherTest(unittest.TestCase):
    def test_buy_sweeps_multiple_ask_levels(self):
        step = _run_interval(
            _orders([100, 101, 102], [5, 5, 20]),
            _orders([99], [100], capacity=3),
            20,
        )
        self.assertAlmostEqual(float(step.filled_quantity), 20.0)
        self.assertAlmostEqual(float(step.execution_cost), 2025.0)
        self.assertAlmostEqual(
            float(step.execution_cost / step.filled_quantity),
            101.25,
        )

    def test_sell_sweeps_multiple_bid_levels(self):
        step = _run_interval(
            _orders([101], [100], capacity=3),
            _orders([100, 99, 98], [3, 4, 10]),
            10,
            is_sell=True,
        )
        self.assertAlmostEqual(float(step.filled_quantity), 10.0)
        self.assertAlmostEqual(float(step.execution_cost), 990.0)

    def test_partial_fill_records_only_available_liquidity(self):
        step = _run_interval(
            _orders([100, 101], [4, 6]),
            _orders([99], [100], capacity=2),
            20,
        )
        self.assertAlmostEqual(float(step.filled_quantity), 10.0)
        self.assertAlmostEqual(float(step.execution_cost), 1006.0)
        self.assertAlmostEqual(float(step.unexecuted_quantity), 10.0)

    def test_partial_fill_is_not_carried_to_next_interval(self):
        first_scheduled = fixed_step_twap_child_quantity(20, 0, 3)
        first = _run_interval(
            _orders([100], [7]),
            _orders([99], [100]),
            scheduled=first_scheduled,
            remaining=20,
        )
        second_scheduled = fixed_step_twap_child_quantity(20, 1, 3)
        second = _run_interval(
            _orders([100], [20]),
            _orders([99], [100]),
            scheduled=second_scheduled,
            remaining=first.remaining_task_quantity,
            cumulative_fill=first.cumulative_filled_quantity,
            cumulative_cost=first.cumulative_execution_cost,
        )
        self.assertAlmostEqual(float(first.filled_quantity), 7.0)
        self.assertAlmostEqual(float(first.unexecuted_quantity), 3.0)
        self.assertEqual(int(first_scheduled), 10)
        self.assertEqual(int(second_scheduled), 10)
        self.assertAlmostEqual(float(second.child_quantity), 10.0)
        self.assertAlmostEqual(float(second.filled_quantity), 10.0)

    def test_child_quantity_is_clipped_by_shadow_remaining_task(self):
        step = _run_interval(
            _orders([100], [20]),
            _orders([99], [20]),
            scheduled=10,
            remaining=4,
        )
        self.assertAlmostEqual(float(step.child_quantity), 4.0)
        self.assertAlmostEqual(float(step.filled_quantity), 4.0)

    def test_best_level_fill_matches_level_one_multiplication(self):
        step = _run_interval(
            _orders([100, 101], [50, 50]),
            _orders([99], [100], capacity=2),
            20,
        )
        self.assertAlmostEqual(float(step.filled_quantity), 20.0)
        self.assertAlmostEqual(float(step.execution_cost), 2000.0)

    def test_empty_book_is_finite_and_zero(self):
        empty = _orders([], [], capacity=4)
        step = _run_interval(empty, empty, 20)
        self.assertAlmostEqual(float(step.filled_quantity), 0.0)
        self.assertAlmostEqual(float(step.execution_cost), 0.0)
        self.assertTrue(np.isfinite(float(step.filled_quantity)))
        self.assertTrue(np.isfinite(float(step.execution_cost)))

    def test_padded_levels_are_ignored(self):
        asks = _orders([100, 101], [5, 5], capacity=4)
        step = _run_interval(asks, _orders([], [], capacity=4), 20)
        self.assertAlmostEqual(float(step.filled_quantity), 10.0)
        self.assertAlmostEqual(float(step.execution_cost), 1005.0)

    def test_shadow_state_updates_from_actual_fill_and_cost(self):
        step = _run_interval(
            _orders([100, 101], [3, 4]),
            _orders([99], [100], capacity=2),
            scheduled=10,
            remaining=20,
        )
        self.assertAlmostEqual(float(step.filled_quantity), 7.0)
        self.assertAlmostEqual(float(step.execution_cost), 704.0)
        self.assertAlmostEqual(float(step.remaining_task_quantity), 13.0)
        self.assertAlmostEqual(float(step.cumulative_filled_quantity), 7.0)
        self.assertAlmostEqual(float(step.cumulative_execution_cost), 704.0)

    def test_temporal_window_accumulates_actual_shadow_results(self):
        windows = (
            _zero_itt_reward_window(),
            _zero_itt_reward_window(),
            _zero_itt_reward_window(),
            _zero_itt_reward_window(),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )
        for volume, cost in zip([5.0, 7.0, 4.0], [500.0, 707.0, 408.0]):
            windows = _update_itt_reward_window(
                *windows,
                V_RL_step=0.0,
                C_RL_step=0.0,
                V_base_step=volume,
                C_base_step=cost,
            )
        self.assertAlmostEqual(float(jnp.sum(windows[2])), 16.0)
        self.assertAlmostEqual(float(jnp.sum(windows[3])), 1615.0)

    def test_reward_path_uses_depth_aware_shadow_cost(self):
        agent, world_state, agent_state, agent_params, trades = self._reward_fixture()
        asks = _orders([10000, 10100, 10200], [5, 5, 20])
        bids = _orders([9900], [100], capacity=3)
        _, info = agent._get_reward(
            world_state,
            agent_state,
            agent_params,
            trades,
            asks,
            bids,
            jnp.asarray([[10000, 30]], dtype=jnp.int32),
            jnp.asarray([[9900, 100]], dtype=jnp.int32),
            jnp.asarray([1, 0], dtype=jnp.int32),
        )
        self.assertAlmostEqual(float(info["V_base_step"]), 20.0)
        self.assertAlmostEqual(float(info["C_base_step"]), 2025.0)
        self.assertNotAlmostEqual(float(info["C_base_step"]), 20.0 * 100.0)

    def test_policy_and_shadow_receive_the_same_integer_twap_child(self):
        agent, world_state, agent_state, agent_params, trades = self._reward_fixture(
            task="buy",
            task_size=500,
        )
        world_state = world_state.replace(
            max_steps_in_episode=51,
            step_counter=0,
        )
        expected_child = fixed_step_twap_child_quantity(500, 0, 51)
        action_messages = agent._getActionMsgs_PolicyBlending(
            jnp.zeros((3,), dtype=jnp.float32),
            world_state,
            agent_state,
            agent_params,
        )
        _, reward_info = agent._get_reward(
            world_state,
            agent_state,
            agent_params,
            trades,
            world_state.ask_raw_orders,
            world_state.bid_raw_orders,
            world_state.best_asks,
            world_state.best_bids,
            jnp.asarray([1, 0], dtype=jnp.int32),
        )

        self.assertEqual(int(expected_child), 10)
        self.assertEqual(int(action_messages[0, 2]), int(expected_child))
        self.assertEqual(
            int(reward_info["shadow_scheduled_child_quantity"]),
            int(expected_child),
        )

    def test_buy_reward_terms_have_expected_cost_and_sign(self):
        agent, world_state, agent_state, agent_params, _ = self._reward_fixture(
            task="buy",
            task_size=100,
        )
        trades = _trades([(10000, -5), (10100, -15)])
        _, info = agent._get_reward(
            world_state,
            agent_state,
            agent_params,
            trades,
            _orders([10000, 10100, 10200], [5, 5, 20]),
            _orders([9900], [100], capacity=3),
            jnp.asarray([[10000, 30]], dtype=jnp.int32),
            jnp.asarray([[9900, 100]], dtype=jnp.int32),
            jnp.asarray([1, 0], dtype=jnp.int32),
        )

        self.assertAlmostEqual(float(info["V_base_step"]), 20.0)
        self.assertAlmostEqual(float(info["C_base_step"]), 2025.0)
        self.assertAlmostEqual(float(info["V_RL_step"]), 20.0)
        self.assertAlmostEqual(float(info["C_RL_step"]), 2015.0)
        self.assertAlmostEqual(float(info["matched_base_cost"]), 2025.0)
        self.assertEqual(float(jnp.where(agent_state.is_sell_task, 1.0, -1.0)), -1.0)
        self.assertAlmostEqual(float(info["r_comp_raw"]), 10.0)
        self.assertAlmostEqual(float(info["r_comp"]), 0.5)

    def test_sell_reward_terms_have_expected_cost_and_sign(self):
        agent, world_state, agent_state, agent_params, _ = self._reward_fixture(
            task="sell",
            task_size=50,
        )
        trades = _trades([(10000, 5), (9900, 5)])
        _, info = agent._get_reward(
            world_state,
            agent_state,
            agent_params,
            trades,
            _orders([10100], [100], capacity=3),
            _orders([10000, 9900, 9800], [3, 4, 10]),
            jnp.asarray([[10100, 100]], dtype=jnp.int32),
            jnp.asarray([[10000, 30]], dtype=jnp.int32),
            jnp.asarray([1, 0], dtype=jnp.int32),
        )

        self.assertAlmostEqual(float(info["V_base_step"]), 10.0)
        self.assertAlmostEqual(float(info["C_base_step"]), 990.0)
        self.assertAlmostEqual(float(info["V_RL_step"]), 10.0)
        self.assertAlmostEqual(float(info["C_RL_step"]), 995.0)
        self.assertAlmostEqual(float(info["matched_base_cost"]), 990.0)
        self.assertEqual(float(jnp.where(agent_state.is_sell_task, 1.0, -1.0)), 1.0)
        self.assertAlmostEqual(float(info["r_comp_raw"]), 5.0)
        self.assertAlmostEqual(float(info["r_comp"]), 0.5)

    def test_hypothetical_volume_matching_still_uses_level_one_price(self):
        terms = _compute_windowed_itt_reward(
            V_RL_k=25.0,
            C_RL_k=2500.0,
            V_base_k=20.0,
            C_base_k=2025.0,
            p_benchmark_tick=100.0,
            is_sell_task=False,
            doom_quant=0.0,
            task_to_execute=100.0,
            reward_lambda=0.5,
            terminal_penalty_beta=1.0,
        )
        self.assertAlmostEqual(float(terms["matched_base_cost"]), 2525.0)

    def test_imitation_reward_uses_actual_shadow_volume(self):
        terms = _compute_windowed_itt_reward(
            V_RL_k=15.0,
            C_RL_k=1500.0,
            V_base_k=10.0,
            C_base_k=1000.0,
            p_benchmark_tick=100.0,
            is_sell_task=False,
            doom_quant=0.0,
            task_to_execute=100.0,
            reward_lambda=0.5,
            terminal_penalty_beta=1.0,
        )
        self.assertAlmostEqual(float(terms["r_mimic"]), -0.5)

    def test_shadow_sweep_does_not_mutate_inputs(self):
        asks = _orders([100, 101], [5, 5], capacity=4)
        bids = _orders([99, 98], [5, 5], capacity=4)
        asks_before = np.asarray(asks).copy()
        bids_before = np.asarray(bids).copy()
        _run_interval(asks, bids, 7)
        np.testing.assert_array_equal(np.asarray(asks), asks_before)
        np.testing.assert_array_equal(np.asarray(bids), bids_before)

        agent, world_state, agent_state, agent_params, trades = self._reward_fixture()
        world_before = jax.tree_util.tree_map(lambda x: np.asarray(x).copy(), world_state)
        agent_before = jax.tree_util.tree_map(lambda x: np.asarray(x).copy(), agent_state)
        trades_before = np.asarray(trades).copy()
        agent._get_reward(
            world_state,
            agent_state,
            agent_params,
            trades,
            _orders([10000, 10100], [5, 5], capacity=3),
            _orders([9900, 9800], [5, 5], capacity=3),
            jnp.asarray([[10000, 5]], dtype=jnp.int32),
            jnp.asarray([[9900, 5]], dtype=jnp.int32),
            jnp.asarray([1, 0], dtype=jnp.int32),
        )
        for before, after in zip(
            jax.tree_util.tree_leaves(world_before),
            jax.tree_util.tree_leaves(world_state),
        ):
            np.testing.assert_array_equal(before, np.asarray(after))
        for before, after in zip(
            jax.tree_util.tree_leaves(agent_before),
            jax.tree_util.tree_leaves(agent_state),
        ):
            np.testing.assert_array_equal(before, np.asarray(after))
        np.testing.assert_array_equal(np.asarray(trades), trades_before)

    def test_vectorized_environments_are_independent(self):
        asks = jnp.stack(
            [
                _orders([100], [5], capacity=3),
                _orders([200], [30], capacity=3),
                _orders([300, 301], [2, 10], capacity=3),
            ]
        )
        bids = jnp.stack(
            [
                _orders([99], [30], capacity=3),
                _orders([199, 198], [4, 20], capacity=3),
                _orders([299], [30], capacity=3),
            ]
        )
        vmapped = jax.vmap(
            simulate_shadow_twap_interval,
            in_axes=(0, 0, 0, 0, 0, 0, 0, None),
        )
        result = vmapped(
            asks,
            bids,
            jnp.asarray([10.0, 10.0, 20.0]),
            jnp.asarray([20.0, 20.0, 4.0]),
            jnp.zeros((3,), dtype=jnp.float32),
            jnp.zeros((3,), dtype=jnp.float32),
            jnp.asarray([False, True, False]),
            1,
        )
        np.testing.assert_allclose(
            np.asarray(result.filled_quantity),
            np.asarray([5.0, 10.0, 4.0]),
        )
        np.testing.assert_allclose(
            np.asarray(result.execution_cost),
            np.asarray([500.0, 1984.0, 1202.0]),
        )

    def test_shadow_sweep_and_reward_path_jit(self):
        asks = _orders([100, 101, 102], [5, 5, 20])
        bids = _orders([99], [100], capacity=3)

        @jax.jit
        def jitted_path(ask_orders, bid_orders):
            step = simulate_shadow_twap_interval(
                ask_orders,
                bid_orders,
                20.0,
                100.0,
                0.0,
                0.0,
                False,
                1,
            )
            windows = _update_itt_reward_window(
                _zero_itt_reward_window(),
                _zero_itt_reward_window(),
                _zero_itt_reward_window(),
                _zero_itt_reward_window(),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
                20.0,
                2015.0,
                step.filled_quantity,
                step.execution_cost,
            )
            return _compute_windowed_itt_reward(
                jnp.sum(windows[0]),
                jnp.sum(windows[1]),
                jnp.sum(windows[2]),
                jnp.sum(windows[3]),
                100.0,
                False,
                0.0,
                100.0,
                0.5,
                1.0,
            )

        terms = jitted_path(asks, bids)
        self.assertTrue(np.isfinite(float(terms["reward"])))
        self.assertAlmostEqual(float(terms["matched_base_cost"]), 2025.0)

    def test_pre_action_shadow_snapshot_preserves_first_real_rollout(self):
        seed = 2
        env, params, obs, state = _synthetic_marl_fixture(seed)
        reference_env = object.__new__(_PostActionShadowReferenceMARLEnv)
        reference_env.__dict__.update(env.__dict__)
        reference_obs = obs
        reference_state = state
        book_changed = False

        for step in range(8):
            self._assert_tree_equal(obs, reference_obs)
            action_key = jax.random.fold_in(jax.random.PRNGKey(seed + 2), step)
            mm_key, exe_key, step_key = jax.random.split(action_key, 3)
            actions = [
                jax.random.randint(mm_key, (), 0, 6, dtype=jnp.int32),
                jax.random.uniform(
                    exe_key,
                    (3,),
                    minval=jnp.asarray([-1.0, 0.0, 0.0], dtype=jnp.float32),
                    maxval=jnp.asarray([3.0, 1.0, 1.0], dtype=jnp.float32),
                ),
            ]
            reference_actions = [
                jax.random.randint(mm_key, (), 0, 6, dtype=jnp.int32),
                jax.random.uniform(
                    exe_key,
                    (3,),
                    minval=jnp.asarray([-1.0, 0.0, 0.0], dtype=jnp.float32),
                    maxval=jnp.asarray([3.0, 1.0, 1.0], dtype=jnp.float32),
                ),
            ]
            self._assert_tree_equal(actions, reference_actions)

            old_world = state.world_state
            obs, state, rewards, dones, info = env.step_env(
                step_key,
                state,
                actions,
                params,
            )
            (
                reference_obs,
                reference_state,
                reference_rewards,
                reference_dones,
                reference_info,
            ) = reference_env.step_env(
                step_key,
                reference_state,
                reference_actions,
                params,
            )

            book_changed = book_changed or not np.array_equal(
                np.asarray(old_world.ask_raw_orders),
                np.asarray(state.world_state.ask_raw_orders),
            ) or not np.array_equal(
                np.asarray(old_world.bid_raw_orders),
                np.asarray(state.world_state.bid_raw_orders),
            )

            # Real exchange trajectory and MM behavior are invariant.
            self._assert_tree_equal(state.world_state, reference_state.world_state)
            self._assert_tree_equal(info["world"], reference_info["world"])
            self._assert_tree_equal(state.agent_states[0], reference_state.agent_states[0])
            self._assert_tree_equal(info["agents"][0], reference_info["agents"][0])
            self._assert_tree_equal(rewards[0], reference_rewards[0])
            self._assert_tree_equal(dones, reference_dones)

            # EXE real fills/cost/accounting remain identical; only its shadow and
            # reward-derived fields are allowed to differ.
            for key in (
                "agentQuant",
                "V_RL_step",
                "C_RL_step",
                "V_RL_k",
                "C_RL_k",
                "quant_left",
                "quant_left_before_unwind",
            ):
                self._assert_tree_equal(
                    info["agents"][1][key],
                    reference_info["agents"][1][key],
                )
            for field in (
                "quant_executed",
                "total_revenue",
                "task_to_execute",
                "is_sell_task",
                "rl_vol_window",
                "rl_cost_window",
            ):
                self._assert_tree_equal(
                    getattr(state.agent_states[1], field),
                    getattr(reference_state.agent_states[1], field),
                )

            allowed_exe_state_differences = {
                "advantage_return",
                "price_adv_rm",
                "base_vol_window",
                "base_cost_window",
                "shadow_cumulative_filled_quantity",
                "shadow_remaining_task_quantity",
                "shadow_cumulative_execution_cost",
            }
            for field in dataclasses.fields(state.agent_states[1]):
                if field.name not in allowed_exe_state_differences:
                    self._assert_tree_equal(
                        getattr(state.agent_states[1], field.name),
                        getattr(reference_state.agent_states[1], field.name),
                    )

            allowed_exe_info_differences = {
                "revenue_direction_normalised",
                "advantage",
                "reward",
                "reward_main",
                "r_comp",
                "r_comp_raw",
                "r_mimic",
                "V_base_step",
                "C_base_step",
                "V_base_k",
                "C_base_k",
                "matched_base_cost",
                "denom_comp",
                "denom_base",
                "v_base_step",
                "terminal_twap_forced_liquidation_is_bps",
                "terminal_twap_forced_liquidation_is_valid",
                "terminal_twap_advantage_bps",
                "terminal_twap_comparison_valid",
                "terminal_twap_win",
            }
            self.assertEqual(
                set(info["agents"][1]),
                set(reference_info["agents"][1]),
            )
            for key in info["agents"][1]:
                if key not in allowed_exe_info_differences:
                    self._assert_tree_equal(
                        info["agents"][1][key],
                        reference_info["agents"][1][key],
                    )

        self.assertTrue(book_changed)
        self._assert_tree_equal(obs, reference_obs)

    def test_shadow_snapshot_selector_uses_pre_action_book(self):
        _, world_state, _, _, _ = self._reward_fixture()
        post_asks = _orders([9900], [1], capacity=3)
        post_bids = _orders([10100], [1], capacity=3)
        env = object.__new__(MARLEnv)
        selected_asks, selected_bids = env._get_shadow_twap_snapshot(
            world_state,
            post_asks,
            post_bids,
        )
        np.testing.assert_array_equal(selected_asks, world_state.ask_raw_orders)
        np.testing.assert_array_equal(selected_bids, world_state.bid_raw_orders)
        self.assertFalse(np.array_equal(np.asarray(selected_asks), np.asarray(post_asks)))
        self.assertFalse(np.array_equal(np.asarray(selected_bids), np.asarray(post_bids)))

    def test_terminal_benchmark_uses_terminal_book_before_auto_reset(self):
        agent, pre_world, agent_state, agent_params, trades = self._reward_fixture(
            task="buy",
            task_size=10,
        )
        pre_world = pre_world.replace(step_counter=4)
        agent_state = agent_state.replace(init_price=10_000)
        _, extras = agent._get_reward(
            pre_world,
            agent_state,
            agent_params,
            trades,
            pre_world.ask_raw_orders,
            pre_world.bid_raw_orders,
            pre_world.best_asks,
            pre_world.best_bids,
            jnp.asarray([4, 0], dtype=jnp.int32),
        )

        terminal_asks = _orders([10_100], [10], capacity=3)
        terminal_bids = _orders([9_900], [10], capacity=3)
        terminal_world = pre_world.replace(
            ask_raw_orders=terminal_asks,
            bid_raw_orders=terminal_bids,
            step_counter=5,
        )
        _, done, info = agent.update_state_and_get_done_and_info(
            terminal_world,
            agent_state,
            extras,
        )

        reset_asks = _orders([20_000], [10], capacity=3)
        reset_result = compute_terminal_execution_benchmark(
            reset_asks,
            terminal_bids,
            task_quantity=10.0,
            terminal_quant_left=10.0,
            rl_cumulative_filled_quantity=0.0,
            rl_cumulative_execution_cost=0.0,
            shadow_cumulative_filled_quantity=extras[
                "shadow_cumulative_filled_quantity"
            ],
            shadow_remaining_task_quantity=extras[
                "shadow_remaining_task_quantity"
            ],
            shadow_cumulative_execution_cost=extras[
                "shadow_cumulative_execution_cost"
            ],
            arrival_price=10_000.0,
            is_sell_task=False,
            tick_size=100,
        )

        self.assertTrue(bool(done))
        self.assertTrue(bool(info["terminal_forced_liquidation_is_valid"]))
        self.assertAlmostEqual(
            float(info["terminal_forced_liquidation_is_bps"]),
            100.0,
            places=4,
        )
        self.assertNotAlmostEqual(
            float(info["terminal_forced_liquidation_is_bps"]),
            float(reset_result.forced_liquidation_is_bps),
        )

    def _assert_tree_equal(self, actual, expected):
        actual_leaves, actual_tree = jax.tree_util.tree_flatten(actual)
        expected_leaves, expected_tree = jax.tree_util.tree_flatten(expected)
        self.assertEqual(actual_tree, expected_tree)
        for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves):
            np.testing.assert_array_equal(
                np.asarray(actual_leaf),
                np.asarray(expected_leaf),
            )

    @staticmethod
    def _reward_fixture(task="buy", task_size=100):
        world_config = World_EnvironmentConfig(
            nOrders=3,
            episode_time=5,
            ep_type="fixed_steps",
        )
        agent_config = Execution_EnvironmentConfig(
            action_space="policy_blending",
            observation_space="execution_policy",
            task=task,
            task_size=task_size,
        )
        agent = ExecutionAgent(agent_config, world_config)
        asks = _orders([10000], [30], capacity=3)
        bids = _orders([9900], [30], capacity=3)
        trades = jnp.full((8, 8), -1, dtype=jnp.int32)
        world_state = WorldState(
            ask_raw_orders=asks,
            bid_raw_orders=bids,
            trades=trades,
            init_time=jnp.asarray([0, 0], dtype=jnp.int32),
            window_index=0,
            max_steps_in_episode=6,
            start_index=0,
            step_counter=0,
            best_bids=jnp.asarray([[9900, 30]], dtype=jnp.int32),
            best_asks=jnp.asarray([[10000, 30]], dtype=jnp.int32),
            time=jnp.asarray([0, 0], dtype=jnp.int32),
            order_id_counter=-200,
            mid_price=9950.0,
            delta_time=0.0,
        )
        agent_state = ExecEnvState(
            init_price=100,
            task_to_execute=task_size,
            quant_executed=0,
            total_revenue=0.0,
            drift_return=0.0,
            advantage_return=0.0,
            slippage_rm=0.0,
            price_adv_rm=0.0,
            price_drift_rm=0.0,
            vwap_rm=0.0,
            is_sell_task=(task == "sell"),
            trade_duration=0.0,
            rl_vol_window=_zero_itt_reward_window(),
            rl_cost_window=_zero_itt_reward_window(),
            base_vol_window=_zero_itt_reward_window(),
            base_cost_window=_zero_itt_reward_window(),
            reward_window_ptr=jnp.asarray(0, dtype=jnp.int32),
            reward_window_count=jnp.asarray(0, dtype=jnp.int32),
            shadow_cumulative_filled_quantity=0.0,
            shadow_remaining_task_quantity=float(task_size),
            shadow_cumulative_execution_cost=0.0,
        )
        agent_params = ExecEnvParams(
            trader_id=-101,
            task_size=task_size,
            reward_lambda=0.5,
            time_delay_obs_act=0,
            normalize=True,
        )
        return agent, world_state, agent_state, agent_params, trades


if __name__ == "__main__":
    unittest.main()
