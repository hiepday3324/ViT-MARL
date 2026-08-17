import jax
import jax.numpy as jnp
import numpy as np

from gymnax_exchange.jaxen.shadow_twap import (
    compute_terminal_execution_benchmark,
    signed_implementation_shortfall_bps,
)
from gymnax_exchange.jaxrl.MARL.execution_episode_metrics import (
    accumulate_execution_episode_metrics,
)


def _orders(prices, quantities, capacity=None):
    capacity = capacity or len(prices)
    orders = jnp.full((capacity, 6), -1, dtype=jnp.int32)
    count = len(prices)
    if count:
        orders = orders.at[:count, 0].set(jnp.asarray(prices, dtype=jnp.int32))
        orders = orders.at[:count, 1].set(jnp.asarray(quantities, dtype=jnp.int32))
        orders = orders.at[:count, 2].set(-2 - jnp.arange(count, dtype=jnp.int32))
        orders = orders.at[:count, 3].set(-100)
        orders = orders.at[:count, 4:].set(0)
    return orders


def _accumulate(
    running_return,
    rewards,
    terminals,
    quant_left,
    task_size=100.0,
    full_completion=None,
    realized_is_bps=None,
    realized_is_valid=None,
    forced_is_bps=None,
    forced_is_valid=None,
    twap_forced_is_bps=None,
    twap_forced_is_valid=None,
    twap_advantage_bps=None,
    comparison_valid=None,
    twap_win=None,
):
    rewards = jnp.asarray(rewards, dtype=jnp.float32)
    if rewards.ndim == 1:
        rewards = rewards[:, None]
    shape = rewards.shape

    def _array_or_default(value, default, dtype):
        if value is None:
            return jnp.full(shape, default, dtype=dtype)
        return jnp.asarray(value, dtype=dtype).reshape(shape)

    return accumulate_execution_episode_metrics(
        jnp.asarray(running_return, dtype=jnp.float32),
        rewards,
        jnp.asarray(terminals, dtype=jnp.bool_).reshape(shape),
        jnp.asarray(quant_left, dtype=jnp.float32).reshape(shape),
        jnp.full(shape, task_size, dtype=jnp.float32),
        full_completion=_array_or_default(
            full_completion,
            False,
            jnp.bool_,
        ),
        realized_is_bps=_array_or_default(
            realized_is_bps,
            0.0,
            jnp.float32,
        ),
        realized_is_valid=_array_or_default(
            realized_is_valid,
            False,
            jnp.bool_,
        ),
        forced_liquidation_is_bps=_array_or_default(
            forced_is_bps,
            0.0,
            jnp.float32,
        ),
        forced_liquidation_is_valid=_array_or_default(
            forced_is_valid,
            False,
            jnp.bool_,
        ),
        twap_forced_liquidation_is_bps=_array_or_default(
            twap_forced_is_bps,
            0.0,
            jnp.float32,
        ),
        twap_forced_liquidation_is_valid=_array_or_default(
            twap_forced_is_valid,
            False,
            jnp.bool_,
        ),
        twap_advantage_bps=_array_or_default(
            twap_advantage_bps,
            0.0,
            jnp.float32,
        ),
        twap_comparison_valid=_array_or_default(
            comparison_valid,
            False,
            jnp.bool_,
        ),
        twap_win=_array_or_default(twap_win, 0.0, jnp.float32),
    )


def test_completed_episode_return_and_full_fill():
    running, metrics = _accumulate(
        jnp.zeros((1,), dtype=jnp.float32),
        rewards=[1.0, 2.0, 3.0],
        terminals=[False, False, True],
        quant_left=[100.0, 100.0, 0.0],
    )

    assert int(metrics.episode_count) == 1
    assert float(metrics.episode_return_mean) == 6.0
    assert float(metrics.terminal_quant_left_mean) == 0.0
    assert float(metrics.terminal_fill_ratio_mean) == 1.0
    np.testing.assert_array_equal(np.asarray(running), np.asarray([0.0]))


def test_partial_terminal_fill_ratio():
    _, metrics = _accumulate(
        jnp.zeros((1,), dtype=jnp.float32),
        rewards=[1.0],
        terminals=[True],
        quant_left=[25.0],
    )

    assert float(metrics.terminal_quant_left_mean) == 25.0
    assert float(metrics.terminal_fill_ratio_mean) == 0.75


def test_episode_return_crosses_rollout_boundary():
    running, first_metrics = _accumulate(
        jnp.zeros((1,), dtype=jnp.float32),
        rewards=[1.0, 2.0],
        terminals=[False, False],
        quant_left=[100.0, 100.0],
    )
    running, second_metrics = _accumulate(
        running,
        rewards=[3.0, 4.0],
        terminals=[False, True],
        quant_left=[100.0, 20.0],
    )

    assert int(first_metrics.episode_count) == 0
    assert float(first_metrics.episode_return_mean) == 0.0
    assert float(second_metrics.episode_return_mean) == 10.0
    np.testing.assert_array_equal(np.asarray(running), np.asarray([0.0]))


def test_terminal_metric_uses_pre_reset_transition_quant_left():
    reset_state_quant_left = 100.0
    terminal_transition_quant_left = 20.0
    _, metrics = _accumulate(
        jnp.zeros((1,), dtype=jnp.float32),
        rewards=[1.0],
        terminals=[True],
        quant_left=[terminal_transition_quant_left],
    )

    assert reset_state_quant_left != terminal_transition_quant_left
    assert float(metrics.terminal_quant_left_mean) == terminal_transition_quant_left
    assert np.isclose(float(metrics.terminal_fill_ratio_mean), 0.8)


def test_no_terminal_has_safe_zero_metrics_and_preserves_return():
    running, metrics = _accumulate(
        jnp.zeros((2,), dtype=jnp.float32),
        rewards=[[1.0, 10.0], [2.0, 20.0]],
        terminals=[[False, False], [False, False]],
        quant_left=[[100.0, 100.0], [75.0, 50.0]],
    )

    assert int(metrics.episode_count) == 0
    assert float(metrics.episode_return_mean) == 0.0
    assert float(metrics.terminal_quant_left_mean) == 0.0
    assert float(metrics.terminal_fill_ratio_mean) == 0.0
    assert np.all(np.isfinite(np.asarray(metrics)))
    np.testing.assert_array_equal(np.asarray(running), np.asarray([3.0, 30.0]))


def test_full_completion_rate_is_distinct_from_fill_ratio():
    _, metrics = _accumulate(
        jnp.zeros((2,), dtype=jnp.float32),
        rewards=[[1.0, 1.0]],
        terminals=[[True, True]],
        quant_left=[[0.0, 10.0]],
        full_completion=[[True, False]],
    )

    assert np.isclose(float(metrics.terminal_fill_ratio_mean), 0.95)
    assert float(metrics.full_completion_rate) == 0.5


def test_signed_realized_is_buy_sell_and_zero_fill_semantics():
    costs = jnp.asarray([1001.0, 999.5, 1001.0, 999.5, 0.0])
    quantities = jnp.asarray([10.0, 10.0, 10.0, 10.0, 0.0])
    is_sell = jnp.asarray([False, False, True, True, False])
    values, valid = jax.vmap(signed_implementation_shortfall_bps)(
        costs,
        quantities,
        jnp.full((5,), 100.0),
        is_sell,
        jnp.ones((5,)),
    )

    np.testing.assert_allclose(
        np.asarray(values[:4]),
        np.asarray([10.0, -5.0, -10.0, 5.0]),
        atol=1e-3,
    )
    np.testing.assert_array_equal(
        np.asarray(valid),
        np.asarray([True, True, True, True, False]),
    )
    assert float(values[-1]) == 0.0
    assert np.all(np.isfinite(np.asarray(values)))


def test_full_rl_completion_forced_is_equals_whole_order_realized_is():
    result = compute_terminal_execution_benchmark(
        _orders([101], [100]),
        _orders([99], [100]),
        task_quantity=100.0,
        terminal_quant_left=0.0,
        rl_cumulative_filled_quantity=100.0,
        rl_cumulative_execution_cost=10_100.0,
        shadow_cumulative_filled_quantity=100.0,
        shadow_remaining_task_quantity=0.0,
        shadow_cumulative_execution_cost=10_200.0,
        arrival_price=100.0,
        is_sell_task=False,
        tick_size=1,
    )

    assert bool(result.full_completion)
    assert bool(result.realized_is_valid)
    assert bool(result.forced_liquidation_is_valid)
    assert np.isclose(
        float(result.forced_liquidation_is_bps),
        float(result.realized_is_bps),
    )
    assert float(result.twap_advantage_bps) > 0.0
    assert float(result.twap_win) == 1.0


def test_incomplete_buy_and_sell_use_terminal_depth_and_preserve_real_cost():
    buy = compute_terminal_execution_benchmark(
        _orders([101, 102], [30, 30]),
        _orders([99], [100], capacity=2),
        task_quantity=100.0,
        terminal_quant_left=60.0,
        rl_cumulative_filled_quantity=40.0,
        rl_cumulative_execution_cost=4_000.0,
        shadow_cumulative_filled_quantity=100.0,
        shadow_remaining_task_quantity=0.0,
        shadow_cumulative_execution_cost=10_100.0,
        arrival_price=100.0,
        is_sell_task=False,
        tick_size=1,
    )
    sell = compute_terminal_execution_benchmark(
        _orders([101], [100], capacity=2),
        _orders([99, 98], [30, 30]),
        task_quantity=100.0,
        terminal_quant_left=60.0,
        rl_cumulative_filled_quantity=40.0,
        rl_cumulative_execution_cost=4_000.0,
        shadow_cumulative_filled_quantity=100.0,
        shadow_remaining_task_quantity=0.0,
        shadow_cumulative_execution_cost=9_900.0,
        arrival_price=100.0,
        is_sell_task=True,
        tick_size=1,
    )

    assert bool(buy.forced_liquidation_is_valid)
    assert bool(sell.forced_liquidation_is_valid)
    np.testing.assert_allclose(
        [buy.forced_liquidation_is_bps, sell.forced_liquidation_is_bps],
        [90.0, 90.0],
        atol=1e-3,
    )


def test_insufficient_liquidity_is_finite_invalid_and_excluded_from_mean():
    result = compute_terminal_execution_benchmark(
        _orders([101], [10]),
        _orders([99], [10]),
        task_quantity=100.0,
        terminal_quant_left=60.0,
        rl_cumulative_filled_quantity=40.0,
        rl_cumulative_execution_cost=4_000.0,
        shadow_cumulative_filled_quantity=100.0,
        shadow_remaining_task_quantity=0.0,
        shadow_cumulative_execution_cost=10_000.0,
        arrival_price=100.0,
        is_sell_task=False,
        tick_size=1,
    )
    assert not bool(result.forced_liquidation_is_valid)
    assert float(result.forced_liquidation_is_bps) == 0.0
    assert np.all(np.isfinite(np.asarray(result)))

    _, metrics = _accumulate(
        jnp.zeros((2,), dtype=jnp.float32),
        rewards=[[1.0, 1.0]],
        terminals=[[True, True]],
        quant_left=[[10.0, 10.0]],
        forced_is_bps=[[25.0, 99_999.0]],
        forced_is_valid=[[True, False]],
    )
    assert float(metrics.forced_liquidation_is_bps_mean) == 25.0


def test_twap_advantage_and_win_rate_ignore_invalid_comparisons():
    _, metrics = _accumulate(
        jnp.zeros((3,), dtype=jnp.float32),
        rewards=[[1.0, 1.0, 1.0]],
        terminals=[[True, True, True]],
        quant_left=[[0.0, 0.0, 0.0]],
        twap_advantage_bps=[[20.0, -10.0, 500.0]],
        comparison_valid=[[True, True, False]],
        twap_win=[[1.0, 0.0, 1.0]],
    )

    assert float(metrics.twap_advantage_bps_mean) == 5.0
    assert float(metrics.twap_win_rate) == 0.5

    _, empty_metrics = _accumulate(
        jnp.zeros((1,), dtype=jnp.float32),
        rewards=[1.0],
        terminals=[True],
        quant_left=[0.0],
        twap_advantage_bps=[500.0],
        comparison_valid=[False],
        twap_win=[1.0],
    )
    assert float(empty_metrics.twap_advantage_bps_mean) == 0.0
    assert float(empty_metrics.twap_win_rate) == 0.0


def test_twap_advantage_sign_is_correct_for_buy_and_sell():
    empty_asks = _orders([], [], capacity=1)
    empty_bids = _orders([], [], capacity=1)

    def completed(rl_cost, twap_cost, is_sell):
        return compute_terminal_execution_benchmark(
            empty_asks,
            empty_bids,
            task_quantity=100.0,
            terminal_quant_left=0.0,
            rl_cumulative_filled_quantity=100.0,
            rl_cumulative_execution_cost=rl_cost,
            shadow_cumulative_filled_quantity=100.0,
            shadow_remaining_task_quantity=0.0,
            shadow_cumulative_execution_cost=twap_cost,
            arrival_price=100.0,
            is_sell_task=is_sell,
            tick_size=1,
        )

    buy_rl_better = completed(9_900.0, 10_100.0, False)
    sell_rl_better = completed(10_100.0, 9_900.0, True)
    buy_rl_worse = completed(10_100.0, 9_900.0, False)
    sell_rl_worse = completed(9_900.0, 10_100.0, True)

    assert float(buy_rl_better.twap_advantage_bps) > 0.0
    assert float(sell_rl_better.twap_advantage_bps) > 0.0
    assert float(buy_rl_worse.twap_advantage_bps) < 0.0
    assert float(sell_rl_worse.twap_advantage_bps) < 0.0
    assert float(buy_rl_better.twap_win) == 1.0
    assert float(sell_rl_better.twap_win) == 1.0
    assert float(buy_rl_worse.twap_win) == 0.0
    assert float(sell_rl_worse.twap_win) == 0.0


def test_terminal_benchmark_is_read_only_jittable_and_vmappable():
    asks = jnp.stack([
        _orders([101, 102], [30, 30]),
        _orders([201, 202], [40, 20]),
    ])
    bids = jnp.stack([
        _orders([99, 98], [30, 30]),
        _orders([199, 198], [40, 20]),
    ])
    asks_before = np.asarray(asks).copy()
    bids_before = np.asarray(bids).copy()

    def evaluate(ask_orders, bid_orders, arrival, is_sell):
        return compute_terminal_execution_benchmark(
            ask_orders,
            bid_orders,
            task_quantity=100.0,
            terminal_quant_left=60.0,
            rl_cumulative_filled_quantity=40.0,
            rl_cumulative_execution_cost=40.0 * arrival,
            shadow_cumulative_filled_quantity=100.0,
            shadow_remaining_task_quantity=0.0,
            shadow_cumulative_execution_cost=100.0 * arrival,
            arrival_price=arrival,
            is_sell_task=is_sell,
            tick_size=1,
        )

    results = jax.jit(jax.vmap(evaluate))(
        asks,
        bids,
        jnp.asarray([100.0, 200.0]),
        jnp.asarray([False, True]),
    )
    assert np.all(np.isfinite(np.asarray(results)))
    assert np.all(np.asarray(results.forced_liquidation_is_valid))
    np.testing.assert_array_equal(np.asarray(asks), asks_before)
    np.testing.assert_array_equal(np.asarray(bids), bids_before)


def test_episode_metric_accumulator_compiles_under_pmap():
    device_count = jax.local_device_count()
    shape = (device_count, 1, 1)

    def aggregate(running_return, rewards, terminals, benchmark_value):
        _, metrics = accumulate_execution_episode_metrics(
            running_return,
            rewards,
            terminals,
            jnp.zeros_like(rewards),
            jnp.full_like(rewards, 100.0),
            full_completion=jnp.ones_like(terminals),
            realized_is_bps=benchmark_value,
            realized_is_valid=jnp.ones_like(terminals),
            forced_liquidation_is_bps=benchmark_value,
            forced_liquidation_is_valid=jnp.ones_like(terminals),
            twap_forced_liquidation_is_bps=benchmark_value,
            twap_forced_liquidation_is_valid=jnp.ones_like(terminals),
            twap_advantage_bps=benchmark_value,
            twap_comparison_valid=jnp.ones_like(terminals),
            twap_win=jnp.ones_like(rewards),
            axis_name="devices",
        )
        return metrics

    aggregate = jax.pmap(aggregate, axis_name="devices")

    metrics = aggregate(
        jnp.zeros((device_count, 1), dtype=jnp.float32),
        jnp.ones(shape, dtype=jnp.float32),
        jnp.ones(shape, dtype=jnp.bool_),
        jnp.arange(1, device_count + 1, dtype=jnp.float32).reshape(shape),
    )
    expected_mean = float((device_count + 1) / 2)
    assert np.all(np.asarray(metrics.episode_count) == device_count)
    np.testing.assert_allclose(
        np.asarray(metrics.realized_is_bps_mean),
        expected_mean,
    )
