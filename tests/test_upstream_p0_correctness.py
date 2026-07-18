from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from gymnax_exchange.jaxen.base_env import (
    build_initial_orders_from_l2,
    pre_reset_cache_path,
)
from gymnax_exchange.jaxlobster.lobster_loader import (
    LOBSTER_CACHE_SCHEMA_VERSION,
    LoadLOBSTER_resample,
    merge_market_orders,
    validate_lobster_file_pair,
    validate_orderbook_columns,
)
from gymnax_exchange.jaxob import JaxOrderBookArrays as job
from gymnax_exchange.jaxob.jaxob_config import (
    JAXLOB_Configuration,
    World_EnvironmentConfig,
)


def _loader_for_preprocessing():
    loader = object.__new__(LoadLOBSTER_resample)
    loader.day_start = 34200
    loader.day_end = 57600
    return loader


def _raw_messages(types):
    rows = []
    for index, event_type in enumerate(types):
        rows.append([
            34200.0 + index / 10.0,
            event_type,
            100 + index,
            index + 1,
            1000 + index,
            -1,
        ])
    return pd.DataFrame(rows)


def test_valid_index_alignment_and_pre_message_shift():
    loader = _loader_for_preprocessing()
    messages = _raw_messages([1, 5, 3, 1])
    post_message_books = pd.DataFrame([
        [1000, 10, 900, 9],
        [1100, 11, 800, 8],
        [1200, 12, 700, 7],
        [1300, 13, 600, 6],
    ])

    processed_messages, aligned_books = loader._pre_process_msg_ob(
        messages, post_message_books
    )

    # Type 5 is filtered. Message raw[2] is paired with raw book[0], the
    # state immediately before the next retained event; raw[3] sees book[2].
    np.testing.assert_array_equal(processed_messages["order_id"], [102, 103])
    np.testing.assert_array_equal(processed_messages["type"], [2, 1])
    np.testing.assert_array_equal(aligned_books.iloc[:, 0], [1000, 1200])


def test_message_book_shape_mismatch_is_rejected():
    loader = _loader_for_preprocessing()
    with pytest.raises(ValueError, match="row mismatch"):
        loader._pre_process_msg_ob(
            _raw_messages([1, 1, 1]),
            pd.DataFrame(np.zeros((2, 4), dtype=np.int32)),
        )


def test_file_pair_and_book_depth_guards():
    validate_lobster_file_pair(
        "AMZN_2012-06-21_message_10.csv",
        "AMZN_2012-06-21_orderbook_10.csv",
    )
    with pytest.raises(ValueError, match="file mismatch"):
        validate_lobster_file_pair("A_message_10.csv", "B_orderbook_10.csv")
    validate_orderbook_columns(pd.DataFrame(np.zeros((2, 8))), n_levels=2)
    with pytest.raises(ValueError, match="expected 8"):
        validate_orderbook_columns(pd.DataFrame(np.zeros((2, 4))), n_levels=2)


def test_visible_executions_merge_by_exact_timestamp_and_side():
    messages = pd.DataFrame({
        "time": [34200.1, 34200.1, 34200.2],
        "type": [4, 4, 1],
        "order_id": [10, 11, 12],
        "qty": [3, 7, 2],
        "price": [1000, 1100, 900],
        "direction": [-1, -1, 1],
        "time_s": [34200, 34200, 34200],
        "time_ns": [100, 100, 200],
    })

    merged = merge_market_orders(messages)
    execution = merged[merged["type"] == 4].iloc[0]
    assert len(merged) == 2
    assert execution["order_id"] == 11
    assert execution["qty"] == 10
    assert execution["price"] == 1100


def test_type4_is_ioc_and_preserves_passive_trade_identity():
    cfg = JAXLOB_Configuration(book_depth=2, nOrders=4, nTrades=4)
    asks = job.init_orderside(4).at[0].set(
        jnp.array([1000, 5, 77, 700, 34200, 0], dtype=jnp.int32)
    )
    bids = job.init_orderside(4)
    trades = jnp.full((4, 8), -1, dtype=jnp.int32)
    # Raw type-4 side -1 means visible ask liquidity was executed. The engine
    # flips it to an aggressive bid and must discard the unmatched quantity.
    execution = jnp.array([[4, -1, 9, 1000, 88, 880, 34200, 1]], dtype=jnp.int32)

    new_asks, new_bids, new_trades = job.scan_through_entire_array(
        cfg, jax.random.PRNGKey(0), execution, (asks, bids, trades)
    )

    assert not np.any(np.asarray(new_bids)[:, 2] == 88)
    assert not np.any(np.asarray(new_asks)[:, 2] == 77)
    assert np.asarray(new_trades)[0, 2] == 77
    assert abs(np.asarray(new_trades)[0, 1]) == 5


def test_full_book_capacity_guard_and_invalid_direct_insert():
    cfg = JAXLOB_Configuration(book_depth=1, nOrders=2, nTrades=2)
    full_bids = jnp.array([
        [1000, 5, 1, 1, 1, 0],
        [900, 5, 2, 2, 2, 0],
    ], dtype=jnp.int32)
    asks = job.init_orderside(2)
    trades = jnp.full((2, 8), -1, dtype=jnp.int32)
    message = {
        "side": jnp.int32(1),
        "type": jnp.int32(1),
        "price": jnp.int32(1100),
        "quantity": jnp.int32(3),
        "orderid": jnp.int32(3),
        "traderid": jnp.int32(3),
        "time": jnp.int32(3),
        "time_ns": jnp.int32(0),
    }

    unchanged = job.add_order(full_bids, message)
    np.testing.assert_array_equal(unchanged, full_bids)

    _asks, guarded_bids, _trades = job.bid_lim(
        cfg, message, asks, full_bids, trades
    )
    assert set(np.asarray(guarded_bids)[:, 2].tolist()) == {1, 3}


def test_synthetic_fallback_does_not_cancel_agent_order():
    cfg = JAXLOB_Configuration(book_depth=2, nOrders=2)
    orders = jnp.array([
        [1000, 10, -200, -200, 1, 0],
        [-1, -1, -1, -1, -1, -1],
    ], dtype=jnp.int32)
    cancel = {
        "price": jnp.int32(1000),
        "quantity": jnp.int32(5),
        "orderid": jnp.int32(999),
    }

    result = job.cancel_order(cfg, jax.random.PRNGKey(0), orders, cancel)
    np.testing.assert_array_equal(result, orders)


def test_dynamic_book_depth_builds_expected_initial_messages():
    book = jnp.array([1010, 5, 990, 6, 1020, 7, 980, 8], dtype=jnp.int32)
    initial = build_initial_orders_from_l2(
        book, jnp.array([34200, 0], dtype=jnp.int32), book_depth=2, init_id=-2
    )

    assert initial.shape == (4, 8)
    np.testing.assert_array_equal(initial[:, 1], [-1, 1, -1, 1])
    np.testing.assert_array_equal(initial[:, 3], [1010, 990, 1020, 980])
    np.testing.assert_array_equal(initial[:, 2], [5, 6, 7, 8])


def test_cache_schema_versions_saved_npz_and_pre_reset_paths(tmp_path):
    loader = object.__new__(LoadLOBSTER_resample)
    loader.alphatrade_path = str(tmp_path)
    loader.stock = "AMZN"
    loader.time_period = "2012June_oneday"
    loader.n_Levels = 10
    loader.window_type = "fixed_steps"
    loader.window_length = 50
    loader.window_resolution = 50
    loader.n_data_msg_per_step = 100
    loader.day_start = 34200
    loader.day_end = 57600
    npz_path = Path(loader._get_save_filename())

    cfg = World_EnvironmentConfig(book_depth=10)
    pkl_path = Path(pre_reset_cache_path(str(tmp_path), cfg))

    assert LOBSTER_CACHE_SCHEMA_VERSION in npz_path.name
    assert LOBSTER_CACHE_SCHEMA_VERSION in pkl_path.name
    assert npz_path.parent.name == "saved_npz"
    assert pkl_path.parent.name == "pre_reset_states"
