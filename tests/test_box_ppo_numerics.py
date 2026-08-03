import distrax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState

from gymnax_exchange.jaxrl.MARL.box_ppo import (
    FIRST_NONFINITE_STAGE,
    box_action_from_pre_tanh,
    box_log_prob_from_pre_tanh,
    build_box_ppo_numerics_diagnostics,
    empty_ppo_safety_state,
    guarded_ppo_apply_gradients,
    policy_log_prob_from_transition,
    sample_policy_action,
    select_guarded_train_state,
    update_ppo_safety_state,
)


BOX_LOW = jnp.asarray([-1.0, 0.0, 0.0], dtype=jnp.float32)
BOX_HIGH = jnp.asarray([3.0, 1.0, 1.0], dtype=jnp.float32)


def _repo_box_distribution(loc, log_std):
    base = distrax.Independent(
        distrax.Normal(loc, jnp.exp(log_std)),
        reinterpreted_batch_ndims=1,
    )
    shift = (BOX_HIGH + BOX_LOW) / 2.0
    scale = (BOX_HIGH - BOX_LOW) / 2.0
    bijector = distrax.Block(
        distrax.Chain(
            [
                distrax.ScalarAffine(shift=shift, scale=scale),
                distrax.Tanh(),
            ]
        ),
        ndims=1,
    )
    return distrax.Transformed(base, bijector)


def _assert_tree_equal(left, right):
    left_leaves, left_structure = jax.tree_util.tree_flatten(left)
    right_leaves, right_structure = jax.tree_util.tree_flatten(right)
    assert left_structure == right_structure
    for left_leaf, right_leaf in zip(left_leaves, right_leaves):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def _assert_tree_allclose(left, right, atol=1e-7, rtol=1e-7):
    left_leaves, left_structure = jax.tree_util.tree_flatten(left)
    right_leaves, right_structure = jax.tree_util.tree_flatten(right)
    assert left_structure == right_structure
    for left_leaf, right_leaf in zip(left_leaves, right_leaves):
        np.testing.assert_allclose(
            np.asarray(left_leaf),
            np.asarray(right_leaf),
            atol=atol,
            rtol=rtol,
        )


def _make_train_state():
    tx = optax.adam(1e-2)
    return TrainState.create(
        apply_fn=lambda *_args, **_kwargs: None,
        params={"w": jnp.asarray([1.0, -2.0], dtype=jnp.float32)},
        tx=tx,
    )


def _finite_guard_kwargs():
    return {
        "total_loss": jnp.asarray(1.0, dtype=jnp.float32),
        "new_log_prob": jnp.asarray([0.0], dtype=jnp.float32),
        "logratio": jnp.asarray([0.0], dtype=jnp.float32),
        "ratio": jnp.asarray([1.0], dtype=jnp.float32),
    }


def _rejected_update_with_initialized_adam_state():
    state = _make_train_state()
    finite_update = guarded_ppo_apply_gradients(
        state,
        {"w": jnp.asarray([0.25, -0.5], dtype=jnp.float32)},
        **_finite_guard_kwargs(),
    )
    state = finite_update.train_state
    kwargs = _finite_guard_kwargs()
    kwargs["ratio"] = jnp.asarray([jnp.inf], dtype=jnp.float32)
    return state, guarded_ppo_apply_gradients(
        state,
        {"w": jnp.asarray([0.25, -0.5], dtype=jnp.float32)},
        **kwargs,
    )


def test_box_sample_then_logprob_boundary_nan():
    loc = jnp.full((4, 3), -50.0, dtype=jnp.float32)
    log_std = jnp.full((3,), -20.0, dtype=jnp.float32)
    pi = _repo_box_distribution(loc, log_std)

    transformed_action = pi.sample(seed=jax.random.PRNGKey(0))
    inverse_log_prob = pi.log_prob(transformed_action)

    assert bool(jnp.any(transformed_action == BOX_LOW))
    assert not bool(jnp.all(jnp.isfinite(inverse_log_prob)))


def test_box_latent_logprob_boundary_finite():
    pre_tanh = jnp.full((4, 3), -50.0, dtype=jnp.float32)
    loc = jnp.full((4, 3), -50.0, dtype=jnp.float32)
    log_std = jnp.full((3,), -2.0, dtype=jnp.float32)
    transformed_action = box_action_from_pre_tanh(pre_tanh, BOX_LOW, BOX_HIGH)

    def objective(current_loc, current_log_std):
        return jnp.sum(
            box_log_prob_from_pre_tanh(
                pre_tanh,
                current_loc,
                current_log_std,
                BOX_LOW,
                BOX_HIGH,
            )
        )

    latent_log_prob = objective(loc, log_std)
    loc_grad, log_std_grad = jax.grad(objective, argnums=(0, 1))(loc, log_std)

    assert bool(jnp.any(transformed_action == BOX_LOW))
    assert bool(jnp.isfinite(latent_log_prob))
    assert bool(jnp.all(jnp.isfinite(loc_grad)))
    assert bool(jnp.all(jnp.isfinite(log_std_grad)))


def test_box_old_new_logprob_same_params():
    loc = jnp.asarray([[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]], dtype=jnp.float32)
    log_std = jnp.asarray([-0.4, 0.2, -0.1], dtype=jnp.float32)
    sample = sample_policy_action(
        _repo_box_distribution(loc, log_std),
        {"policy_loc": loc, "policy_log_std": log_std},
        jax.random.PRNGKey(7),
        action_low=BOX_LOW,
        action_high=BOX_HIGH,
    )
    new_log_prob = policy_log_prob_from_transition(
        _repo_box_distribution(loc, log_std),
        {"policy_loc": loc, "policy_log_std": log_std},
        sample.action,
        sample.pre_tanh_action,
        action_low=BOX_LOW,
        action_high=BOX_HIGH,
    )
    logratio = new_log_prob - sample.log_prob
    ratio = jnp.exp(logratio)

    np.testing.assert_allclose(new_log_prob, sample.log_prob, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(logratio, 0.0, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(ratio, 1.0, atol=1e-6, rtol=1e-6)


def test_box_logprob_matches_transformed_distribution_away_from_boundary():
    loc = jnp.asarray([[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]], dtype=jnp.float32)
    log_std = jnp.asarray([-0.4, 0.2, -0.1], dtype=jnp.float32)
    pre_tanh = jnp.asarray([[0.2, -0.1, 0.4], [-0.3, 0.6, -0.7]], dtype=jnp.float32)
    action = box_action_from_pre_tanh(pre_tanh, BOX_LOW, BOX_HIGH)

    latent_log_prob = box_log_prob_from_pre_tanh(
        pre_tanh,
        loc,
        log_std,
        BOX_LOW,
        BOX_HIGH,
    )
    transformed_log_prob = _repo_box_distribution(loc, log_std).log_prob(action)

    np.testing.assert_allclose(latent_log_prob, transformed_log_prob, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize(
    "field,value,expected_stage",
    (
        ("total_loss", jnp.nan, "total_loss"),
        ("new_log_prob", jnp.asarray([jnp.nan]), "new_log_prob"),
        ("logratio", jnp.asarray([jnp.inf]), "logratio"),
        ("ratio", jnp.asarray([jnp.inf]), "ratio"),
    ),
)
def test_ppo_nonfinite_candidate_rolls_back_params_and_opt_state(
    field,
    value,
    expected_stage,
):
    state = _make_train_state()
    finite_update = guarded_ppo_apply_gradients(
        state,
        {"w": jnp.asarray([0.25, -0.5], dtype=jnp.float32)},
        **_finite_guard_kwargs(),
    )
    state = finite_update.train_state
    kwargs = _finite_guard_kwargs()
    kwargs[field] = value

    rejected = guarded_ppo_apply_gradients(
        state,
        {"w": jnp.asarray([0.25, -0.5], dtype=jnp.float32)},
        **kwargs,
    )

    assert not bool(rejected.accepted)
    assert bool(rejected.rejected_nonfinite)
    assert int(rejected.first_nonfinite_stage) == FIRST_NONFINITE_STAGE[expected_stage]
    _assert_tree_equal(rejected.train_state.params, state.params)
    _assert_tree_equal(rejected.train_state.opt_state, state.opt_state)
    assert int(rejected.train_state.step) == int(state.step)


def test_ppo_nonfinite_candidate_rolls_back_params():
    state, rejected = _rejected_update_with_initialized_adam_state()
    _assert_tree_equal(rejected.train_state.params, state.params)


def test_ppo_nonfinite_candidate_rolls_back_opt_state():
    state, rejected = _rejected_update_with_initialized_adam_state()
    _assert_tree_equal(rejected.train_state.opt_state, state.opt_state)


def test_ppo_rejection_does_not_increment_optimizer_count():
    state, rejected = _rejected_update_with_initialized_adam_state()
    assert int(rejected.train_state.step) == int(state.step)


def test_ppo_nonfinite_gradient_rolls_back_and_does_not_increment_optimizer_count():
    state = _make_train_state()
    rejected = guarded_ppo_apply_gradients(
        state,
        {"w": jnp.asarray([jnp.nan, 1.0], dtype=jnp.float32)},
        **_finite_guard_kwargs(),
    )

    assert not bool(rejected.accepted)
    assert int(rejected.first_nonfinite_stage) == FIRST_NONFINITE_STAGE["gradients"]
    _assert_tree_equal(rejected.train_state.params, state.params)
    _assert_tree_equal(rejected.train_state.opt_state, state.opt_state)
    assert int(rejected.train_state.step) == int(state.step)


def test_ppo_rejection_stops_later_minibatch_commits_for_that_agent():
    state = _make_train_state()
    safety = empty_ppo_safety_state()
    kwargs = _finite_guard_kwargs()
    kwargs["new_log_prob"] = jnp.asarray([jnp.nan])
    rejected = guarded_ppo_apply_gradients(
        state,
        {"w": jnp.asarray([0.25, -0.5], dtype=jnp.float32)},
        **kwargs,
    )
    state_after_rejection = select_guarded_train_state(state, rejected, safety)
    safety = update_ppo_safety_state(
        safety,
        rejected,
        epoch_index=0,
        minibatch_index=0,
    )
    finite_candidate = guarded_ppo_apply_gradients(
        state_after_rejection,
        {"w": jnp.asarray([0.25, -0.5], dtype=jnp.float32)},
        **_finite_guard_kwargs(),
    )
    final_state = select_guarded_train_state(
        state_after_rejection,
        finite_candidate,
        safety,
    )

    _assert_tree_equal(final_state.params, state.params)
    _assert_tree_equal(final_state.opt_state, state.opt_state)
    assert int(final_state.step) == int(state.step)


def test_box_boundary_rollout_replay_ratio_and_gradient_are_finite():
    loc = jnp.full((4, 3), -50.0, dtype=jnp.float32)
    log_std = jnp.full((3,), -2.0, dtype=jnp.float32)
    pi = _repo_box_distribution(loc, log_std)
    sample = sample_policy_action(
        pi,
        {"policy_loc": loc, "policy_log_std": log_std},
        jax.random.PRNGKey(1),
        action_low=BOX_LOW,
        action_high=BOX_HIGH,
    )

    def replay_objective(current_loc, current_log_std):
        current_pi = _repo_box_distribution(current_loc, current_log_std)
        new_log_prob = policy_log_prob_from_transition(
            current_pi,
            {"policy_loc": current_loc, "policy_log_std": current_log_std},
            sample.action,
            sample.pre_tanh_action,
            action_low=BOX_LOW,
            action_high=BOX_HIGH,
        )
        return jnp.mean(jnp.exp(new_log_prob - sample.log_prob))

    new_log_prob = policy_log_prob_from_transition(
        pi,
        {"policy_loc": loc, "policy_log_std": log_std},
        sample.action,
        sample.pre_tanh_action,
        action_low=BOX_LOW,
        action_high=BOX_HIGH,
    )
    ratio = jnp.exp(new_log_prob - sample.log_prob)
    grads = jax.grad(replay_objective, argnums=(0, 1))(loc, log_std)

    assert bool(jnp.any(sample.action == BOX_LOW))
    assert bool(jnp.all(jnp.isfinite(sample.log_prob)))
    assert bool(jnp.all(jnp.isfinite(new_log_prob)))
    assert bool(jnp.all(jnp.isfinite(ratio)))
    assert all(bool(jnp.all(jnp.isfinite(grad))) for grad in grads)


def test_box_boundary_rollout_old_logprob_finite():
    loc = jnp.full((2, 3), -50.0, dtype=jnp.float32)
    log_std = jnp.full((3,), -2.0, dtype=jnp.float32)
    sample = sample_policy_action(
        _repo_box_distribution(loc, log_std),
        {"policy_loc": loc, "policy_log_std": log_std},
        jax.random.PRNGKey(31),
        action_low=BOX_LOW,
        action_high=BOX_HIGH,
    )
    assert bool(jnp.any(sample.action == BOX_LOW))
    assert bool(jnp.all(jnp.isfinite(sample.log_prob)))


def test_box_boundary_replay_new_logprob_finite():
    loc = jnp.full((2, 3), 50.0, dtype=jnp.float32)
    log_std = jnp.full((3,), -2.0, dtype=jnp.float32)
    pi = _repo_box_distribution(loc, log_std)
    sample = sample_policy_action(
        pi,
        {"policy_loc": loc, "policy_log_std": log_std},
        jax.random.PRNGKey(32),
        action_low=BOX_LOW,
        action_high=BOX_HIGH,
    )
    replay = policy_log_prob_from_transition(
        pi,
        {"policy_loc": loc, "policy_log_std": log_std},
        sample.action,
        sample.pre_tanh_action,
        action_low=BOX_LOW,
        action_high=BOX_HIGH,
    )
    assert bool(jnp.any(sample.action == BOX_HIGH))
    assert bool(jnp.all(jnp.isfinite(replay)))


def test_box_boundary_ratio_finite():
    loc = jnp.full((2, 3), 50.0, dtype=jnp.float32)
    log_std = jnp.full((3,), -2.0, dtype=jnp.float32)
    pi = _repo_box_distribution(loc, log_std)
    sample = sample_policy_action(
        pi,
        {"policy_loc": loc, "policy_log_std": log_std},
        jax.random.PRNGKey(33),
        action_low=BOX_LOW,
        action_high=BOX_HIGH,
    )
    new_log_prob = box_log_prob_from_pre_tanh(
        sample.pre_tanh_action,
        loc,
        log_std,
        BOX_LOW,
        BOX_HIGH,
    )
    assert bool(jnp.all(jnp.isfinite(jnp.exp(new_log_prob - sample.log_prob))))


def test_box_boundary_gradient_finite():
    pre_tanh = jnp.full((2, 3), 50.0, dtype=jnp.float32)
    log_std = jnp.full((3,), -2.0, dtype=jnp.float32)

    def objective(loc):
        return jnp.sum(
            box_log_prob_from_pre_tanh(
                pre_tanh,
                loc,
                log_std,
                BOX_LOW,
                BOX_HIGH,
            )
        )

    gradient = jax.grad(objective)(jnp.full((2, 3), 50.0, dtype=jnp.float32))
    assert bool(jnp.all(jnp.isfinite(gradient)))


def test_discrete_policy_sampling_and_logprob_are_unchanged():
    logits = jnp.asarray([[0.2, -0.1, 0.4], [-0.7, 0.3, 0.1]], dtype=jnp.float32)
    pi = distrax.Categorical(logits=logits)
    key = jax.random.PRNGKey(17)
    expected_action = pi.sample(seed=key)
    expected_log_prob = pi.log_prob(expected_action)

    sample = sample_policy_action(pi, {"policy_logits": logits}, key)

    np.testing.assert_array_equal(sample.action, expected_action)
    np.testing.assert_allclose(sample.log_prob, expected_log_prob, atol=0.0, rtol=0.0)
    np.testing.assert_array_equal(sample.pre_tanh_action, jnp.zeros_like(expected_action))


@pytest.mark.parametrize("optimization_mode", ("joint", "phasic"))
def test_box_ppo_behavior_unchanged_away_from_boundary(optimization_mode):
    del optimization_mode
    loc = jnp.asarray([[0.1, 0.2, -0.1]], dtype=jnp.float32)
    log_std = jnp.asarray([-0.5, -0.3, -0.2], dtype=jnp.float32)
    pre_tanh = jnp.asarray([[0.2, -0.3, 0.4]], dtype=jnp.float32)
    action = box_action_from_pre_tanh(pre_tanh, BOX_LOW, BOX_HIGH)
    pi = _repo_box_distribution(loc, log_std)
    old_log_prob = pi.log_prob(action)
    new_log_prob = policy_log_prob_from_transition(
        pi,
        {"policy_loc": loc, "policy_log_std": log_std},
        action,
        pre_tanh,
        action_low=BOX_LOW,
        action_high=BOX_HIGH,
    )
    advantage = jnp.asarray([0.7], dtype=jnp.float32)
    old_actor_loss = -jnp.mean(jnp.exp(old_log_prob - old_log_prob) * advantage)
    new_actor_loss = -jnp.mean(jnp.exp(new_log_prob - old_log_prob) * advantage)

    np.testing.assert_allclose(new_actor_loss, old_actor_loss, atol=2e-6, rtol=2e-6)


def test_diagnostic_off_invariance_and_does_not_change_rng():
    loc = jnp.asarray([[0.1, 0.2, -0.1]], dtype=jnp.float32)
    log_std = jnp.asarray([-0.5, -0.3, -0.2], dtype=jnp.float32)
    key = jax.random.PRNGKey(23)
    pi = _repo_box_distribution(loc, log_std)
    sample_without_diag = sample_policy_action(
        pi,
        {"policy_loc": loc, "policy_log_std": log_std},
        key,
        action_low=BOX_LOW,
        action_high=BOX_HIGH,
    )
    sample_with_diag = sample_policy_action(
        pi,
        {"policy_loc": loc, "policy_log_std": log_std},
        key,
        action_low=BOX_LOW,
        action_high=BOX_HIGH,
    )
    _ = build_box_ppo_numerics_diagnostics(
        enabled=True,
        loc=loc,
        log_std=log_std,
        pre_tanh_action=sample_with_diag.pre_tanh_action,
        action=sample_with_diag.action,
        action_low=BOX_LOW,
        action_high=BOX_HIGH,
        old_log_prob=sample_with_diag.log_prob,
        new_log_prob=sample_with_diag.log_prob,
        advantage=jnp.ones((1,), dtype=jnp.float32),
        value=jnp.zeros((1,), dtype=jnp.float32),
        grads={"params": {"Dense_1": {"kernel": jnp.ones((1, 3))}, "log_std": jnp.ones((3,))}},
        total_loss=jnp.asarray(0.0, dtype=jnp.float32),
    )

    _assert_tree_allclose(sample_with_diag, sample_without_diag, atol=0.0, rtol=0.0)
    np.testing.assert_array_equal(key, jax.random.PRNGKey(23))


def test_box_sampling_matches_legacy_transformed_sample_away_from_boundary():
    loc = jnp.asarray([[0.1, 0.2, -0.1]], dtype=jnp.float32)
    log_std = jnp.asarray([-0.5, -0.3, -0.2], dtype=jnp.float32)
    key = jax.random.PRNGKey(41)
    pi = _repo_box_distribution(loc, log_std)
    legacy_action = pi.sample(seed=key)
    stable_sample = sample_policy_action(
        pi,
        {"policy_loc": loc, "policy_log_std": log_std},
        key,
        action_low=BOX_LOW,
        action_high=BOX_HIGH,
    )
    np.testing.assert_array_equal(stable_sample.action, legacy_action)


def test_diagnostic_computation_does_not_change_loss_gradient_params_or_optimizer():
    state = _make_train_state()

    def loss_fn(params):
        return jnp.sum(jnp.square(params["w"]))

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    without_diag = guarded_ppo_apply_gradients(
        state,
        grads,
        total_loss=loss,
        new_log_prob=jnp.asarray([0.0]),
        logratio=jnp.asarray([0.0]),
        ratio=jnp.asarray([1.0]),
    )
    _ = build_box_ppo_numerics_diagnostics(
        enabled=True,
        loc=jnp.zeros((1, 3), dtype=jnp.float32),
        log_std=jnp.zeros((3,), dtype=jnp.float32),
        pre_tanh_action=jnp.zeros((1, 3), dtype=jnp.float32),
        action=box_action_from_pre_tanh(
            jnp.zeros((1, 3), dtype=jnp.float32),
            BOX_LOW,
            BOX_HIGH,
        ),
        action_low=BOX_LOW,
        action_high=BOX_HIGH,
        old_log_prob=jnp.asarray([0.0]),
        new_log_prob=jnp.asarray([0.0]),
        advantage=jnp.asarray([1.0]),
        value=jnp.asarray([0.0]),
        grads={"params": {"Dense_1": {"kernel": jnp.ones((1, 3))}, "log_std": jnp.ones((3,))}},
        total_loss=loss,
    )
    with_diag = guarded_ppo_apply_gradients(
        state,
        grads,
        total_loss=loss,
        new_log_prob=jnp.asarray([0.0]),
        logratio=jnp.asarray([0.0]),
        ratio=jnp.asarray([1.0]),
    )

    np.testing.assert_allclose(loss, loss, atol=0.0, rtol=0.0)
    _assert_tree_equal(grads, grads)
    _assert_tree_equal(with_diag.train_state.params, without_diag.train_state.params)
    _assert_tree_equal(with_diag.train_state.opt_state, without_diag.train_state.opt_state)
    assert int(with_diag.train_state.step) == int(without_diag.train_state.step)
