import jax
import jax.numpy as jnp
import numpy as np

from gymnax_exchange.jaxen.twap_schedule import (
    fixed_step_twap_child_quantity,
    fixed_step_twap_cumulative_quantity,
    fixed_step_twap_execution_intervals,
)
from gymnax_exchange.jaxlobster.lobster_loader import LoadLOBSTER_resample


def _schedule(task_size, max_steps_in_episode):
    interval_count = int(fixed_step_twap_execution_intervals(max_steps_in_episode))
    steps = jnp.arange(interval_count, dtype=jnp.int32)
    return jax.vmap(
        lambda step: fixed_step_twap_child_quantity(
            task_size,
            step,
            max_steps_in_episode,
        )
    )(steps)


def test_normal_500_share_schedule_excludes_sentinel():
    children = _schedule(task_size=500, max_steps_in_episode=51)

    assert int(fixed_step_twap_execution_intervals(51)) == 50
    np.testing.assert_array_equal(np.asarray(children), np.full((50,), 10))
    assert int(jnp.sum(children)) == 500
    assert int(fixed_step_twap_child_quantity(500, 50, 51)) == 0


def test_non_divisible_task_is_apportioned_exactly_and_deterministically():
    children = _schedule(task_size=503, max_steps_in_episode=51)
    repeated = _schedule(task_size=503, max_steps_in_episode=51)

    np.testing.assert_array_equal(np.unique(np.asarray(children)), [10, 11])
    np.testing.assert_array_equal(children, repeated)
    assert bool(jnp.all(children >= 0))
    assert jnp.issubdtype(children.dtype, jnp.integer)
    assert int(jnp.sum(children)) == 503


def test_final_normal_interval_reaches_parent_quantity():
    before_final = fixed_step_twap_cumulative_quantity(503, 49, 51)
    final_child = fixed_step_twap_child_quantity(503, 49, 51)
    after_final = fixed_step_twap_cumulative_quantity(503, 50, 51)

    assert int(before_final + final_child) == 503
    assert int(after_final) == 503
    assert int(fixed_step_twap_child_quantity(503, 50, 51)) == 0


def test_schedule_is_jittable_and_vmappable():
    compiled = jax.jit(
        jax.vmap(fixed_step_twap_child_quantity, in_axes=(0, 0, 0))
    )
    children = compiled(
        jnp.asarray([500, 503, 500], dtype=jnp.int32),
        jnp.asarray([0, 49, 50], dtype=jnp.int32),
        jnp.asarray([51, 51, 51], dtype=jnp.int32),
    )

    np.testing.assert_array_equal(np.asarray(children), [10, 11, 0])


def _padding_loader(n_data_msg_per_step=100):
    loader = object.__new__(LoadLOBSTER_resample)
    loader.n_data_msg_per_step = n_data_msg_per_step
    return loader


def test_last_episode_padding_does_not_add_divisible_block():
    loader = _padding_loader()
    messages = np.arange(200 * 8, dtype=np.int32).reshape(200, 8)
    max_messages = np.asarray([200], dtype=np.int64)

    padded, padded_max_messages = loader._pad_last_ep(messages, max_messages)

    np.testing.assert_array_equal(padded, messages)
    np.testing.assert_array_equal(padded_max_messages, [200])


def test_last_episode_padding_reaches_next_multiple_only():
    loader = _padding_loader()
    messages = np.arange(150 * 8, dtype=np.int32).reshape(150, 8)
    messages[-1, -2:] = np.asarray([1234, 567], dtype=np.int32)
    max_messages = np.asarray([150], dtype=np.int64)

    padded, padded_max_messages = loader._pad_last_ep(messages, max_messages)

    assert padded.shape == (200, 8)
    np.testing.assert_array_equal(padded[:150], messages)
    np.testing.assert_array_equal(padded[150:, :-2], 0)
    np.testing.assert_array_equal(
        padded[150:, -2:],
        np.tile(np.asarray([1235, 0], dtype=np.int32), (50, 1)),
    )
    np.testing.assert_array_equal(padded_max_messages, [200])
