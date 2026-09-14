from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gymnax_exchange.jaxen.vision_env import (
    ExecutionAgent,
    _reconcile_policy_blending_messages,
)
from gymnax_exchange.jaxob import JaxOrderBookArrays as job
from gymnax_exchange.jaxob import jaxob_constants as cst
from gymnax_exchange.jaxob.jaxob_config import (
    Execution_EnvironmentConfig,
    World_EnvironmentConfig,
)


def _action_rows(rows, capacity=3):
    messages = jnp.zeros((capacity, 8), dtype=jnp.int32)
    for index, (side, price, quantity) in enumerate(rows):
        messages = messages.at[index].set(
            jnp.asarray([1, side, quantity, price, -900 - index, -101, 10, 0])
        )
    return messages


def _resting_rows(rows, capacity=8, trader_id=-101):
    orders = jnp.full((capacity, 6), cst.EMPTY_SLOT, dtype=jnp.int32)
    for index, (price, quantity, order_id, sec, nsec) in enumerate(rows):
        orders = orders.at[index].set(
            jnp.asarray([price, quantity, order_id, trader_id, sec, nsec])
        )
    return orders


def _reconcile(actions, resting, side=1, cancel_capacity=3):
    return _reconcile_policy_blending_messages(
        actions,
        resting,
        jnp.asarray(-101, dtype=jnp.int32),
        jnp.asarray(side, dtype=jnp.int32),
        jnp.asarray(20, dtype=jnp.int32),
        jnp.asarray(5, dtype=jnp.int32),
        cancel_capacity=cancel_capacity,
    )


def _action_map(messages):
    result = {}
    for row in np.asarray(messages):
        quantity = int(row[cst.LOBMSGFEAT.Quant.value])
        if quantity > 0:
            key = (
                int(row[cst.LOBMSGFEAT.Side.value]),
                int(row[cst.LOBMSGFEAT.Price.value]),
            )
            result[key] = result.get(key, 0) + quantity
    return result


def _cancel_map(messages):
    return {
        int(row[cst.LOBMSGFEAT.OID.value]): int(
            row[cst.LOBMSGFEAT.Quant.value]
        )
        for row in np.asarray(messages)
        if row[cst.LOBMSGFEAT.Quant.value] > 0
    }


def test_target_equal_resting_keeps_existing_order_without_repost():
    action, cancel = _reconcile(
        _action_rows([(1, 100, 15)]),
        _resting_rows([(100, 15, -10, 1, 0)]),
    )
    assert _action_map(action) == {}
    assert _cancel_map(cancel) == {}


def test_target_below_resting_cancels_only_excess():
    action, cancel = _reconcile(
        _action_rows([(1, 100, 6)]),
        _resting_rows([(100, 10, -10, 1, 0)]),
    )
    assert _action_map(action) == {}
    assert _cancel_map(cancel) == {-10: 4}


def test_target_above_resting_preserves_fifo_and_submits_only_delta():
    action, cancel = _reconcile(
        _action_rows([(1, 100, 20)]),
        _resting_rows(
            [(100, 10, -10, 1, 0), (100, 5, -11, 2, 0)]
        ),
    )
    assert _action_map(action) == {(1, 100): 5}
    assert _cancel_map(cancel) == {}


def test_zero_target_cancels_all_resting_at_key():
    action, cancel = _reconcile(
        _action_rows([(1, 100, 0)]),
        _resting_rows([(100, 6, -10, 1, 0)]),
    )
    assert _action_map(action) == {}
    assert _cancel_map(cancel) == {-10: 6}


def test_multi_price_reconciliation_uses_exact_keys():
    action, cancel = _reconcile(
        _action_rows([(1, 100, 10), (1, 99, 5)]),
        _resting_rows([(99, 100, -20, 1, 0), (100, 1, -10, 2, 0)]),
    )
    assert _action_map(action) == {(1, 100): 9}
    assert _cancel_map(cancel) == {-20: 95}


def test_raw_resting_permutation_is_semantically_invariant():
    actions = _action_rows([(1, 100, 7), (1, 99, 5)])
    rows = [
        (100, 4, -10, 1, 0),
        (100, 6, -11, 2, 0),
        (99, 8, -12, 3, 0),
    ]
    action_a, cancel_a = _reconcile(actions, _resting_rows(rows))
    action_b, cancel_b = _reconcile(
        actions, _resting_rows([rows[2], rows[0], rows[1]])
    )
    assert _action_map(action_a) == _action_map(action_b)
    assert _cancel_map(cancel_a) == _cancel_map(cancel_b)


def test_counterexample_does_not_pair_quantities_across_prices():
    action, cancel = _reconcile(
        _action_rows([(1, 100, 10), (1, 99, 5)]),
        _resting_rows([(99, 100, -20, 1, 0), (100, 1, -10, 2, 0)]),
    )
    assert _action_map(action)[(1, 100)] == 9
    assert (1, 99) not in _action_map(action)
    assert -10 not in _cancel_map(cancel)
    assert _cancel_map(cancel)[-20] == 95


def test_duplicate_policy_prices_are_aggregated():
    action, cancel = _reconcile(
        _action_rows([(1, 100, 7), (1, 100, 5)]),
        _resting_rows([(100, 5, -10, 1, 0)]),
    )
    assert _action_map(action) == {(1, 100): 7}
    assert _cancel_map(cancel) == {}


@pytest.mark.parametrize(
    "target,expected",
    [
        (12, {-11: 3}),
        (2, {-11: 5, -10: 8}),
    ],
)
def test_duplicate_resting_orders_cancel_newest_first(target, expected):
    action, cancel = _reconcile(
        _action_rows([(1, 100, target)]),
        _resting_rows([(100, 10, -10, 1, 0), (100, 5, -11, 2, 0)]),
    )
    assert _action_map(action) == {}
    assert _cancel_map(cancel) == expected


def test_decrease_crossing_multiple_oids_cancels_newest_first():
    action, cancel = _reconcile(
        _action_rows([(1, 100, 11)]),
        _resting_rows(
            [
                (100, 10, -10, 1, 0),
                (100, 5, -11, 2, 0),
                (100, 7, -12, 3, 0),
            ]
        ),
    )
    cancel_array = np.asarray(cancel)
    active = cancel_array[cancel_array[:, cst.LOBMSGFEAT.Quant.value] > 0]
    assert active[:, cst.LOBMSGFEAT.OID.value].tolist() == [-12, -11]
    assert active[:, cst.LOBMSGFEAT.Quant.value].tolist() == [7, 4]
    assert active[:, cst.LOBMSGFEAT.Price.value].tolist() == [100, 100]
    assert active[:, cst.LOBMSGFEAT.Side.value].tolist() == [1, 1]
    assert -10 not in _cancel_map(cancel)
    assert _action_map(action) == {}


def test_zero_target_cancels_multiple_oids_newest_first():
    action, cancel = _reconcile(
        _action_rows([(1, 100, 0)]),
        _resting_rows(
            [
                (100, 10, -10, 1, 0),
                (100, 5, -11, 2, 0),
                (100, 7, -12, 3, 0),
            ]
        ),
    )
    cancel_array = np.asarray(cancel)
    active = cancel_array[cancel_array[:, cst.LOBMSGFEAT.Quant.value] > 0]
    assert active[:, cst.LOBMSGFEAT.OID.value].tolist() == [-12, -11, -10]
    assert active[:, cst.LOBMSGFEAT.Quant.value].tolist() == [7, 5, 10]
    assert _action_map(action) == {}


def test_same_numeric_price_on_opposite_side_is_not_matched():
    action, cancel = _reconcile(
        _action_rows([(-1, 100, 7)]),
        _resting_rows([(100, 7, -10, 1, 0)]),
        side=1,
    )
    assert _action_map(action) == {(-1, 100): 7}
    assert _cancel_map(cancel) == {-10: 7}


def test_old_price_absent_from_targets_is_fully_cancelled():
    action, cancel = _reconcile(
        _action_rows([(1, 100, 5)]),
        _resting_rows([(99, 8, -20, 1, 0)]),
    )
    assert _action_map(action) == {(1, 100): 5}
    assert _cancel_map(cancel) == {-20: 8}


def test_no_resting_order_submits_target_normally():
    action, cancel = _reconcile(
        _action_rows([(1, 100, 9)]),
        _resting_rows([]),
    )
    assert _action_map(action) == {(1, 100): 9}
    assert _cancel_map(cancel) == {}


def test_helper_is_jittable_and_vmappable():
    actions = jnp.stack(
        [_action_rows([(1, 100, 6)]), _action_rows([(1, 101, 10)])]
    )
    resting = jnp.stack(
        [
            _resting_rows([(100, 10, -10, 1, 0)]),
            _resting_rows([(101, 4, -11, 1, 0)]),
        ]
    )

    compiled = jax.jit(
        jax.vmap(
            lambda a, r: _reconcile_policy_blending_messages(
                a,
                r,
                jnp.asarray(-101, dtype=jnp.int32),
                jnp.asarray(1, dtype=jnp.int32),
                jnp.asarray(20, dtype=jnp.int32),
                jnp.asarray(5, dtype=jnp.int32),
                cancel_capacity=3,
            )
        )
    )
    action, cancel = compiled(actions, resting)
    assert action.shape == (2, 3, 8)
    assert cancel.shape == (2, 3, 8)
    assert _cancel_map(cancel[0]) == {-10: 4}
    assert _action_map(action[1]) == {(1, 101): 6}


def test_policy_blending_get_messages_dispatches_to_keyed_reconciliation():
    config = Execution_EnvironmentConfig(
        action_space="policy_blending",
        observation_space="execution_policy",
    )
    agent = ExecutionAgent(config, World_EnvironmentConfig(nOrders=8))
    actions = _action_rows([(1, 100, 10), (1, 99, 5)])
    agent.action_fn = lambda **_kwargs: actions
    resting = _resting_rows([(99, 100, -20, 1, 0), (100, 1, -10, 2, 0)])
    world_state = SimpleNamespace(
        ask_raw_orders=jnp.full_like(resting, cst.EMPTY_SLOT),
        bid_raw_orders=resting,
        time=jnp.asarray([20, 5], dtype=jnp.int32),
    )
    agent_state = SimpleNamespace(is_sell_task=jnp.asarray(False))
    agent_params = SimpleNamespace(trader_id=jnp.asarray(-101))

    action, cancel = agent._get_messages(
        jnp.asarray(0), world_state, agent_state, agent_params
    )
    assert _action_map(action) == {(1, 100): 9}
    assert _cancel_map(cancel) == {-20: 95}


@pytest.mark.parametrize(
    "action_space", ["fixed_quants", "fixed_quants_complex", "twap"]
)
def test_non_policy_blending_modes_keep_generic_filter_path(action_space):
    config = Execution_EnvironmentConfig(
        action_space=action_space,
        observation_space="engineered",
    )
    agent = ExecutionAgent(config, World_EnvironmentConfig(nOrders=8))
    actions = _action_rows(
        [(1, 100, 5)], capacity=config.num_action_messages_by_agent
    )
    agent.action_fn = lambda **_kwargs: actions
    resting = _resting_rows([(100, 7, -10, 1, 0)])
    world_state = SimpleNamespace(
        ask_raw_orders=jnp.full_like(resting, cst.EMPTY_SLOT),
        bid_raw_orders=resting,
        time=jnp.asarray([20, 5], dtype=jnp.int32),
    )
    agent_state = SimpleNamespace(is_sell_task=jnp.asarray(False))
    agent_params = SimpleNamespace(trader_id=jnp.asarray(-101))

    expected_cancel = job.getCancelMsgs(
        resting,
        agent_params.trader_id,
        config.num_messages_by_agent // 2,
        1,
        world_state.time[0],
        world_state.time[1],
    )
    expected = agent._filter_messages(actions, expected_cancel)
    actual = agent._get_messages(
        jnp.asarray(0), world_state, agent_state, agent_params
    )
    np.testing.assert_array_equal(np.asarray(actual[0]), np.asarray(expected[0]))
    np.testing.assert_array_equal(np.asarray(actual[1]), np.asarray(expected[1]))
