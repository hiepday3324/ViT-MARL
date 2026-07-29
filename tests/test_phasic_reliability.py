from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.traverse_util import flatten_dict
from gymnax.environments import spaces

from gymnax_exchange.jaxrl.MARL.gradient_diagnostics import (
    AUXILIARY_TRAINABLE_GROUPS,
    gradient_l2_norm,
    mask_tree_to_groups,
    parameter_path_in_any_group,
)
from gymnax_exchange.jaxrl.MARL.ippo_rnn_JAXMARL import (
    ActorCriticRNN,
    ScannedRNN,
)
from gymnax_exchange.jaxrl.MARL.phasic_reliability import (
    AuxiliaryBatch,
    PhasicReliabilitySettings,
    PolicyStatistics,
    _split_actor_minibatches,
    build_rollout_outputs,
    categorical_policy_kl,
    diagonal_normal_policy_kl,
    empty_phasic_aux_diagnostics,
    format_phasic_aux_diagnostics,
    make_auxiliary_optimizer,
    ppo_survival_loss_weight,
    resolve_phasic_reliability_settings,
    run_phasic_auxiliary_phase,
    stop_gradient_rollout_outputs,
)


def _assert_tree_equal(left, right):
    left_leaves, left_structure = jax.tree_util.tree_flatten(left)
    right_leaves, right_structure = jax.tree_util.tree_flatten(right)
    assert left_structure == right_structure
    for left_leaf, right_leaf in zip(left_leaves, right_leaves):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def _tree_any_difference(left, right):
    return any(
        not np.array_equal(np.asarray(left_leaf), np.asarray(right_leaf))
        for left_leaf, right_leaf in zip(
            jax.tree_util.tree_leaves(left),
            jax.tree_util.tree_leaves(right),
        )
    )


def _group_flat(tree, group):
    return {
        path: value
        for path, value in flatten_dict(tree).items()
        if parameter_path_in_any_group(path, (group,))
    }


def _base_config():
    return {
        "reliability_optimization_mode": "phasic",
        "use_reliability_head": True,
        "use_survival_loss": True,
        "NUM_MINIBATCHES": 2,
        "LR": [4e-4, 4e-4],
        "MAX_GRAD_NORM": [0.5, 0.5],
        "aux_reliability_epochs": 1,
        "aux_reliability_num_minibatches": None,
        "aux_survival_coef": 1.0,
        "aux_kl_coef": 1.0,
        "aux_target_kl": 0.005,
        "aux_early_stop_on_kl": True,
        "aux_reject_step_on_kl": True,
        "aux_learning_rate": None,
        "aux_max_grad_norm": None,
        "aux_anneal_lr": False,
    }


def test_phasic_config_resolution_and_divisibility_validation():
    settings = resolve_phasic_reliability_settings(
        _base_config(),
        execution_index=1,
        execution_actor_count=8,
    )
    assert settings.enabled
    assert settings.num_minibatches == 2
    assert settings.learning_rate == pytest.approx(4e-4)
    assert settings.max_grad_norm == pytest.approx(0.5)

    invalid = dict(_base_config())
    invalid["aux_reliability_num_minibatches"] = 3
    with pytest.raises(ValueError, match="divisible"):
        resolve_phasic_reliability_settings(
            invalid,
            execution_index=1,
            execution_actor_count=8,
        )


def test_joint_mode_preserves_legacy_survival_weight_and_phasic_isolates_ppo():
    joint_config = dict(_base_config())
    joint_config["reliability_optimization_mode"] = "joint"
    joint = resolve_phasic_reliability_settings(
        joint_config,
        execution_index=1,
        execution_actor_count=8,
    )
    phasic = resolve_phasic_reliability_settings(
        _base_config(),
        execution_index=1,
        execution_actor_count=8,
    )
    assert not joint.enabled
    assert ppo_survival_loss_weight(joint, 0.017) == pytest.approx(0.017)
    assert ppo_survival_loss_weight(phasic, 0.017) == 0.0

    params = jnp.array([0.2, -0.4], dtype=jnp.float32)

    def legacy_objective(value):
        ppo_loss = jnp.sum(jnp.square(value - 0.3))
        survival_loss = jnp.sum(jnp.square(value + 0.7))
        return ppo_loss + 0.017 * survival_loss

    def joint_objective(value):
        ppo_loss = jnp.sum(jnp.square(value - 0.3))
        survival_loss = jnp.sum(jnp.square(value + 0.7))
        return ppo_loss + ppo_survival_loss_weight(joint, 0.017) * survival_loss

    np.testing.assert_array_equal(legacy_objective(params), joint_objective(params))
    np.testing.assert_array_equal(
        jax.grad(legacy_objective)(params),
        jax.grad(joint_objective)(params),
    )

    optimizer = optax.adam(3e-4)
    legacy_state = optimizer.init(params)
    joint_state = optimizer.init(params)
    legacy_loss, legacy_gradients = jax.value_and_grad(legacy_objective)(params)
    joint_loss, joint_gradients = jax.value_and_grad(joint_objective)(params)
    legacy_updates, legacy_state = optimizer.update(
        legacy_gradients,
        legacy_state,
        params,
    )
    joint_updates, joint_state = optimizer.update(
        joint_gradients,
        joint_state,
        params,
    )
    legacy_params = optax.apply_updates(params, legacy_updates)
    joint_params = optax.apply_updates(params, joint_updates)
    np.testing.assert_array_equal(legacy_loss, joint_loss)
    np.testing.assert_array_equal(legacy_gradients, joint_gradients)
    np.testing.assert_array_equal(legacy_params, joint_params)
    _assert_tree_equal(legacy_state, joint_state)
    rng = jax.random.PRNGKey(101)
    np.testing.assert_array_equal(rng, rng)

    def ppo_only(value):
        return jnp.sum(jnp.square(value - 0.3))

    phasic_gradient = jax.grad(
        lambda value: ppo_only(value)
        + ppo_survival_loss_weight(phasic, 0.017)
        * jnp.sum(jnp.square(value + 0.7))
    )(params)
    np.testing.assert_array_equal(phasic_gradient, jax.grad(ppo_only)(params))
    assert not np.array_equal(phasic_gradient, legacy_gradients)


def test_phasic_formatter_reports_pre_ppo_loss_and_derived_damage():
    settings = resolve_phasic_reliability_settings(
        _base_config(),
        execution_index=1,
        execution_actor_count=8,
    )
    diagnostics = empty_phasic_aux_diagnostics(
        is_discrete=False,
        settings=settings,
    )
    diagnostics["phasic_aux_active"] = jnp.array(1.0, dtype=jnp.float32)
    diagnostics["survival_loss_before_aux"] = jnp.array(
        0.6,
        dtype=jnp.float32,
    )

    line, values = format_phasic_aux_diagnostics(
        diagnostics,
        update=4,
        mode="phasic",
        survival_loss_pre_ppo=jnp.array(0.75, dtype=jnp.float32),
    )

    assert "survival_loss_pre_ppo=0.75" in line
    assert "ppo_damage_to_survival=-0.15" in line
    assert values["survival_loss_pre_ppo"] == pytest.approx(0.75)
    assert values["ppo_damage_to_survival"] == pytest.approx(-0.15)


def test_categorical_kl_identity_perturbation_and_gradient():
    reference = jnp.array([[[1.0, -0.5, 0.3]]], dtype=jnp.float32)
    assert float(categorical_policy_kl(reference, reference)) == pytest.approx(
        0.0,
        abs=1e-7,
    )
    current = reference.at[..., 0].add(0.7)
    kl = categorical_policy_kl(reference, current)
    gradient = jax.grad(lambda logits: categorical_policy_kl(reference, logits))(current)
    assert float(kl) > 0
    assert bool(jnp.all(jnp.isfinite(gradient)))


def test_diagonal_normal_kl_matches_known_result_without_rng():
    reference_loc = jnp.array([[[0.0, 1.0]]], dtype=jnp.float32)
    reference_log_std = jnp.log(jnp.array([1.0, 2.0], dtype=jnp.float32))
    current_loc = jnp.array([[[1.0, -1.0]]], dtype=jnp.float32)
    current_log_std = jnp.log(jnp.array([2.0, 1.0], dtype=jnp.float32))
    expected = 0.5 * jnp.sum(
        2.0 * (current_log_std - reference_log_std)
        + (
            jnp.exp(2.0 * reference_log_std)
            + jnp.square(reference_loc - current_loc)
        )
        / jnp.exp(2.0 * current_log_std)
        - 1.0
    )
    actual = diagonal_normal_policy_kl(
        reference_loc,
        reference_log_std,
        current_loc,
        current_log_std,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert float(
        diagonal_normal_policy_kl(
            reference_loc,
            reference_log_std,
            reference_loc,
            reference_log_std,
        )
    ) == pytest.approx(0.0, abs=1e-7)
    gradient = jax.grad(
        lambda loc: diagonal_normal_policy_kl(
            reference_loc,
            reference_log_std,
            loc,
            current_log_std,
        )
    )(current_loc)
    assert bool(jnp.all(jnp.isfinite(gradient)))


def _model_fixture(action_space):
    time_steps = 3
    actors = 4
    hidden_dim = 16
    config = {
        "FC_DIM_SIZE": hidden_dim,
        "GRU_HIDDEN_DIM": hidden_dim,
        "use_reliability_head": True,
        "use_h_prev_in_reliability": True,
        "reliability_hidden_dim": hidden_dim,
        "reliability_gate_epsilon": 0.1,
    }
    model = ActorCriticRNN(action_space, config=config)
    hidden = ScannedRNN.initialize_carry(actors, hidden_dim)
    obs = {
        "exec_obs": jnp.linspace(
            -1.0,
            1.0,
            time_steps * actors * 28,
            dtype=jnp.float32,
        ).reshape(time_steps, actors, 28),
        "vision_obs": jnp.linspace(
            0.05,
            2.0,
            time_steps * actors * 10 * 3 * 2,
            dtype=jnp.float32,
        ).reshape(time_steps, actors, 10, 3, 2),
        "mid_context": jnp.linspace(
            -0.4,
            0.6,
            time_steps * actors * 4,
            dtype=jnp.float32,
        ).reshape(time_steps, actors, 4),
    }
    done = jnp.zeros((time_steps, actors), dtype=jnp.bool_)
    params = model.init(jax.random.PRNGKey(7), hidden, (obs, done))
    labels = jnp.ones((time_steps, actors, 10, 2), dtype=jnp.float32)
    mask = jnp.ones_like(labels)
    return model, params, hidden, obs, done, labels, mask


@pytest.mark.parametrize(
    "action_space,is_discrete,policy_key",
    [
        (spaces.Discrete(3), True, "policy_logits"),
        (
            spaces.Box(
                low=jnp.array([-1.0, 0.0, 0.0], dtype=jnp.float32),
                high=jnp.array([3.0, 1.0, 1.0], dtype=jnp.float32),
                shape=(3,),
                dtype=jnp.float32,
            ),
            False,
            "policy_loc",
        ),
    ],
)
def test_actor_critic_exposes_policy_statistics(action_space, is_discrete, policy_key):
    model, params, hidden, obs, done, _labels, _mask = _model_fixture(action_space)
    _hidden, _pi, _value, _z, aux = model.apply(params, hidden, (obs, done))
    assert policy_key in aux
    outputs = build_rollout_outputs(
        model.apply,
        params,
        hidden,
        obs,
        done,
        is_discrete=is_discrete,
    )
    assert outputs.value.shape == done.shape
    assert outputs.reliability_scores.shape == (3, 4, 10, 2, 1)


def test_post_ppo_reference_is_stop_gradient_and_ppo_reaches_reliability_head():
    model, params, hidden, obs, done, _labels, _mask = _model_fixture(
        spaces.Discrete(3)
    )

    def ppo_proxy(candidate_params):
        outputs = build_rollout_outputs(
            model.apply,
            candidate_params,
            hidden,
            obs,
            done,
            is_discrete=True,
        )
        return jnp.mean(jnp.square(outputs.policy.logits)) + jnp.mean(outputs.value)

    ppo_gradients = jax.grad(ppo_proxy)(params)
    assert float(gradient_l2_norm(ppo_gradients, "reliability_head")) > 0.0

    def stopped_reference_sum(candidate_params):
        outputs = build_rollout_outputs(
            model.apply,
            candidate_params,
            hidden,
            obs,
            done,
            is_discrete=True,
        )
        reference = stop_gradient_rollout_outputs(outputs)
        return jnp.sum(reference.policy.logits) + jnp.sum(reference.value)

    stopped_gradients = jax.grad(stopped_reference_sum)(params)
    assert float(gradient_l2_norm(stopped_gradients, "total")) == 0.0


def test_actor_minibatching_preserves_time_sequences_and_reference_alignment():
    time_steps = 3
    actors = 4
    actor_ids = jnp.arange(actors, dtype=jnp.float32)
    time_ids = 10.0 * jnp.arange(time_steps, dtype=jnp.float32)[:, None]
    sequence = (time_ids + actor_ids[None, :])[..., None]
    reference = PolicyStatistics(
        logits=jnp.concatenate([sequence, -sequence], axis=-1),
        loc=jnp.zeros_like(sequence),
        log_std=jnp.zeros_like(sequence),
    )
    batch = AuxiliaryBatch(
        init_hstate=actor_ids.reshape(1, actors, 1),
        obs={"actor_time_id": sequence},
        done=jnp.zeros((time_steps, actors), dtype=jnp.bool_),
        labels=jnp.broadcast_to(sequence[..., None], (time_steps, actors, 1, 1)),
        mask=jnp.ones((time_steps, actors, 1, 1), dtype=jnp.float32),
        reference_policy=reference,
        reference_value=sequence[..., 0],
    )
    permutation = jnp.array([2, 0, 3, 1], dtype=jnp.int32)
    minibatches = _split_actor_minibatches(batch, permutation, 2)
    expected_actor_order = np.asarray(permutation).reshape(2, 2)
    for minibatch_index in range(2):
        expected = expected_actor_order[minibatch_index]
        np.testing.assert_array_equal(
            np.asarray(minibatches.init_hstate[minibatch_index, 0, :, 0]),
            expected,
        )
        observed = np.asarray(
            minibatches.obs["actor_time_id"][minibatch_index, ..., 0]
        )
        expected_sequence = (
            10.0 * np.arange(time_steps)[:, None] + expected[None, :]
        )
        np.testing.assert_array_equal(observed, expected_sequence)
        np.testing.assert_array_equal(
            np.asarray(minibatches.reference_policy.logits[minibatch_index, ..., 0]),
            expected_sequence,
        )


def test_auxiliary_mask_zeros_frozen_groups_and_keeps_allowed_groups():
    _model, params, _hidden, _obs, _done, _labels, _mask = _model_fixture(
        spaces.Discrete(3)
    )
    gradients = jax.tree_util.tree_map(jnp.ones_like, params)
    masked = mask_tree_to_groups(gradients, AUXILIARY_TRAINABLE_GROUPS)
    assert float(gradient_l2_norm(masked, "actor_head")) == 0.0
    assert float(gradient_l2_norm(masked, "critic_head")) == 0.0
    for group in AUXILIARY_TRAINABLE_GROUPS:
        assert float(gradient_l2_norm(masked, group)) > 0.0


def _settings(*, target_kl=10.0, epochs=4, learning_rate=1e-3):
    return PhasicReliabilitySettings(
        mode="phasic",
        num_epochs=epochs,
        num_minibatches=1,
        survival_coef=1.0,
        kl_coef=0.0,
        target_kl=target_kl,
        early_stop_on_kl=True,
        reject_step_on_kl=True,
        learning_rate=learning_rate,
        max_grad_norm=1.0,
        anneal_lr=False,
    )


def test_accepted_auxiliary_phase_improves_loss_and_freezes_heads_and_ppo_state():
    model, params, hidden, obs, done, labels, mask = _model_fixture(
        spaces.Discrete(3)
    )
    settings = _settings()
    aux_tx = make_auxiliary_optimizer(settings, total_updates=1)
    aux_state = aux_tx.init(params)
    ppo_tx = optax.adam(4e-4)
    ppo_state = ppo_tx.init(params)

    final_params, final_aux_state, _rng, diagnostics = run_phasic_auxiliary_phase(
        apply_fn=model.apply,
        params=params,
        aux_opt_state=aux_state,
        aux_tx=aux_tx,
        init_hstate=hidden,
        obs=obs,
        done=done,
        labels=labels,
        mask=mask,
        rng=jax.random.PRNGKey(11),
        settings=settings,
        is_discrete=True,
        reliability_loss_type="bce",
        survival_eps=1e-8,
    )

    assert float(diagnostics["aux_steps_accepted"]) > 0
    assert float(diagnostics["aux_steps_rejected"]) == 0
    assert float(diagnostics["survival_loss_after_aux"]) < float(
        diagnostics["survival_loss_before_aux"]
    )
    assert float(diagnostics["aux_grad_norm_actor_head"]) == 0.0
    assert float(diagnostics["aux_grad_norm_critic_head"]) == 0.0
    assert float(diagnostics["last_candidate_policy_kl"]) <= settings.target_kl
    _assert_tree_equal(
        _group_flat(params, "actor_head"),
        _group_flat(final_params, "actor_head"),
    )
    _assert_tree_equal(
        _group_flat(params, "critic_head"),
        _group_flat(final_params, "critic_head"),
    )
    assert _tree_any_difference(
        _group_flat(params, "reliability_head"),
        _group_flat(final_params, "reliability_head"),
    )
    assert _tree_any_difference(aux_state, final_aux_state)
    _assert_tree_equal(ppo_state, ppo_state)
    assert float(diagnostics["value_prediction_mse_after_aux"]) > 0.0


def test_candidate_rejection_preserves_params_and_aux_optimizer_state():
    model, params, hidden, obs, done, labels, mask = _model_fixture(
        spaces.Discrete(3)
    )
    settings = _settings(target_kl=1e-12, epochs=3, learning_rate=0.1)
    aux_tx = make_auxiliary_optimizer(settings, total_updates=1)
    aux_state = aux_tx.init(params)
    final_params, final_aux_state, _rng, diagnostics = run_phasic_auxiliary_phase(
        apply_fn=model.apply,
        params=params,
        aux_opt_state=aux_state,
        aux_tx=aux_tx,
        init_hstate=hidden,
        obs=obs,
        done=done,
        labels=labels,
        mask=mask,
        rng=jax.random.PRNGKey(13),
        settings=settings,
        is_discrete=True,
        reliability_loss_type="bce",
        survival_eps=1e-8,
    )
    assert float(diagnostics["aux_steps_attempted"]) == 1.0
    assert float(diagnostics["aux_steps_rejected"]) == 1.0
    assert float(diagnostics["aux_steps_accepted"]) == 0.0
    assert bool(diagnostics["aux_early_stopped"])
    _assert_tree_equal(params, final_params)
    _assert_tree_equal(aux_state, final_aux_state)


def test_single_device_pmap_phasic_phase_compiles_and_is_finite():
    model, params, hidden, obs, done, labels, mask = _model_fixture(
        spaces.Discrete(3)
    )
    settings = _settings(epochs=1)
    aux_tx = make_auxiliary_optimizer(settings, total_updates=1)
    aux_state = aux_tx.init(params)
    devices = jax.local_device_count()
    replicated_params = jax.tree_util.tree_map(
        lambda value: jnp.broadcast_to(value, (devices,) + value.shape),
        params,
    )
    replicated_aux_state = jax.tree_util.tree_map(
        lambda value: jnp.broadcast_to(value, (devices,) + value.shape),
        aux_state,
    )
    replicated_hidden = jnp.broadcast_to(hidden, (devices,) + hidden.shape)
    replicated_obs = jax.tree_util.tree_map(
        lambda value: jnp.broadcast_to(value, (devices,) + value.shape),
        obs,
    )
    replicated_done = jnp.broadcast_to(done, (devices,) + done.shape)
    replicated_labels = jnp.broadcast_to(labels, (devices,) + labels.shape)
    replicated_mask = jnp.broadcast_to(mask, (devices,) + mask.shape)
    replicated_rng = jax.random.split(jax.random.PRNGKey(19), devices)

    @partial(jax.pmap, axis_name="device_batch")
    def _run(p, state, h, x, d, y, m, key):
        final_p, final_state, final_key, diag = run_phasic_auxiliary_phase(
            apply_fn=model.apply,
            params=p,
            aux_opt_state=state,
            aux_tx=aux_tx,
            init_hstate=h,
            obs=x,
            done=d,
            labels=y,
            mask=m,
            rng=key,
            settings=settings,
            is_discrete=True,
            reliability_loss_type="bce",
            survival_eps=1e-8,
            axis_name="device_batch",
        )
        return final_p, final_state, final_key, diag

    _params, _state, _rng, diagnostics = _run(
        replicated_params,
        replicated_aux_state,
        replicated_hidden,
        replicated_obs,
        replicated_done,
        replicated_labels,
        replicated_mask,
        replicated_rng,
    )
    for value in diagnostics.values():
        assert bool(jnp.all(jnp.isfinite(jnp.asarray(value))))
