from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from gymnax.environments import spaces

from gymnax_exchange.jaxrl.MARL.gradient_diagnostics import (
    ACTOR_CRITIC_GRADIENT_GROUPS,
    CRITIC_OPTIMIZER_DIAGNOSTIC_GROUPS,
    GRADIENT_GROUPS,
    actor_critic_grad_diag_should_run,
    add_gradient_trees,
    critic_optimizer_diag_should_run,
    empty_actor_critic_gradient_diagnostics,
    empty_critic_optimizer_diagnostics,
    empty_gradient_interaction_diagnostics,
    empty_value_representation_probe_diagnostics,
    empty_value_clip_diagnostics,
    fit_value_representation_probe_variants,
    format_gradient_interaction_diagnostics,
    gradient_cosine,
    gradient_diag_should_run,
    gradient_dot,
    gradient_l2_norm,
    matching_parameter_paths,
    flatten_tree_with_paths,
    parameter_groups_mask,
    VALUE_PROBE_B_SEPARATE_TRAINABLE_GROUPS,
    scale_gradient_tree,
    subtract_gradient_trees,
    run_value_representation_probe,
    summarize_actor_critic_gradient_interaction,
    summarize_critic_optimizer_diagnostics,
    summarize_gradient_interaction,
    summarize_phasic_gradient_interaction,
    summarize_value_clip_diagnostics,
    validate_actor_critic_grad_diag_config,
    validate_critic_optimizer_diag_config,
    validate_gradient_diag_config,
    validate_required_parameter_groups,
    validate_value_representation_probe_config,
    validate_value_clip_diag_config,
    value_probe_statistics,
    value_clip_diag_should_run,
    trajectory_train_holdout_masks,
)
from gymnax_exchange.jaxrl.MARL.ippo_rnn_JAXMARL import (
    ActorCriticRNN,
    ScannedRNN,
)
from gymnax_exchange.jaxrl.MARL.reliability_targets import (
    masked_reliability_loss,
)
from gymnax_exchange.jaxrl.MARL.phasic_reliability import (
    make_auxiliary_optimizer,
    resolve_phasic_reliability_settings,
    run_phasic_auxiliary_phase,
)


def _toy_tree(value):
    return {"params": {"x": jnp.asarray(value, dtype=jnp.float32)}}


def _assert_tree_allclose(left, right, atol=1e-7, rtol=1e-7):
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    assert left_tree == right_tree
    for lhs, rhs in zip(left_leaves, right_leaves):
        np.testing.assert_allclose(np.asarray(lhs), np.asarray(rhs), atol=atol, rtol=rtol)


def test_gradient_norm_dot_and_cosine_known_values():
    grad = _toy_tree([3.0, 4.0])
    assert float(gradient_l2_norm(grad)) == 5.0

    same = _toy_tree([3.0, 4.0])
    opposite = _toy_tree([-3.0, -4.0])
    orthogonal_left = _toy_tree([1.0, 0.0])
    orthogonal_right = _toy_tree([0.0, 1.0])
    same_cos, same_valid = gradient_cosine(grad, same)
    opposite_cos, opposite_valid = gradient_cosine(grad, opposite)
    orthogonal_cos, orthogonal_valid = gradient_cosine(
        orthogonal_left,
        orthogonal_right,
    )

    np.testing.assert_allclose(float(gradient_dot(grad, same)), 25.0)
    np.testing.assert_allclose(float(same_cos), 1.0, atol=1e-6)
    np.testing.assert_allclose(float(opposite_cos), -1.0, atol=1e-6)
    np.testing.assert_allclose(float(orthogonal_cos), 0.0, atol=1e-6)
    assert bool(same_valid)
    assert bool(opposite_valid)
    assert bool(orthogonal_valid)


def test_zero_gradient_cosine_is_finite_and_invalid():
    zero = _toy_tree([0.0, 0.0])
    nonzero = _toy_tree([1.0, 0.0])
    cosine, valid = gradient_cosine(zero, nonzero)
    assert np.isfinite(float(cosine))
    assert float(cosine) == 0.0
    assert not bool(valid)


def test_gradient_tree_scaling_addition_and_subtraction():
    ppo = _toy_tree([1.0, 2.0])
    survival = _toy_tree([4.0, -2.0])
    weighted = scale_gradient_tree(survival, 0.25)
    reconstructed = add_gradient_trees(ppo, weighted)
    residual = subtract_gradient_trees(reconstructed, ppo)
    np.testing.assert_allclose(reconstructed["params"]["x"], [2.0, 1.5])
    np.testing.assert_allclose(residual["params"]["x"], [1.0, -0.5])


def test_toy_loss_decomposition_is_exact():
    params = _toy_tree([0.2, -0.3, 0.7])
    lambda_surv = 0.01

    def ppo_objective(p):
        x = p["params"]["x"]
        return jnp.sum(jnp.square(x + 0.5))

    def survival_objective(p):
        x = p["params"]["x"]
        return jnp.sum(jnp.square(x - 1.5))

    def total_objective(p):
        return ppo_objective(p) + lambda_surv * survival_objective(p)

    ppo_grads = jax.grad(ppo_objective)(params)
    survival_grads = jax.grad(survival_objective)(params)
    total_grads = jax.grad(total_objective)(params)
    diagnostics = summarize_gradient_interaction(
        params,
        total_grads,
        ppo_grads,
        survival_grads,
        lambda_surv,
    )
    assert float(
        diagnostics["groups"]["total"]["decomposition_rel_error"]
    ) < 1e-6


def _make_reliability_model_fixture():
    time_steps = 1
    batch_size = 3
    hidden_dim = 16
    config = {
        "FC_DIM_SIZE": hidden_dim,
        "GRU_HIDDEN_DIM": hidden_dim,
        "use_reliability_head": True,
        "use_h_prev_in_reliability": True,
        "reliability_hidden_dim": hidden_dim,
        "reliability_gate_epsilon": 0.1,
    }
    model = ActorCriticRNN(spaces.Discrete(3), config=config)
    hidden = ScannedRNN.initialize_carry(batch_size, hidden_dim)
    obs = {
        "exec_obs": jnp.linspace(
            -1.0,
            1.0,
            time_steps * batch_size * 28,
            dtype=jnp.float32,
        ).reshape(time_steps, batch_size, 28),
        "vision_obs": jnp.linspace(
            0.1,
            2.0,
            time_steps * batch_size * 10 * 3 * 2,
            dtype=jnp.float32,
        ).reshape(time_steps, batch_size, 10, 3, 2),
        "mid_context": jnp.linspace(
            -0.5,
            0.5,
            time_steps * batch_size * 4,
            dtype=jnp.float32,
        ).reshape(time_steps, batch_size, 4),
    }
    done = jnp.zeros((time_steps, batch_size), dtype=jnp.bool_)
    variables = model.init(jax.random.PRNGKey(17), hidden, (obs, done))
    labels = jnp.linspace(
        0.05,
        0.95,
        time_steps * batch_size * 10 * 2,
        dtype=jnp.float32,
    ).reshape(time_steps, batch_size, 10, 2)
    mask = jnp.ones_like(labels)
    actions = jnp.zeros((time_steps, batch_size), dtype=jnp.int32)
    value_targets = jnp.ones((time_steps, batch_size), dtype=jnp.float32) * 0.3

    def components(params):
        _hidden, pi, value, _z, aux = model.apply(params, hidden, (obs, done))
        ppo_loss = 100.0 * (
            -pi.log_prob(actions).mean()
            + 0.5 * jnp.mean(jnp.square(value - value_targets))
            - 0.01 * pi.entropy().mean()
        )
        survival_loss = masked_reliability_loss(
            aux["reliability_scores"],
            labels,
            mask,
            loss_type="bce",
            reliability_logits=aux["reliability_logits"],
        )
        return ppo_loss, survival_loss

    return variables, components


def test_fixed_reliability_batch_gradient_interaction_and_group_assignment():
    params, components = _make_reliability_model_fixture()
    counts = validate_required_parameter_groups(params)
    assert all(counts[group] > 0 for group in GRADIENT_GROUPS)
    lambda_surv = 0.01

    def total_objective(p):
        ppo_loss, survival_loss = components(p)
        return ppo_loss + lambda_surv * survival_loss

    @jax.jit
    def compute_grads(p):
        def weighted_objective(value, survival_weight):
            ppo_loss, survival_loss = components(value)
            return ppo_loss + survival_weight * survival_loss

        ppo_grads = jax.grad(weighted_objective)(p, 0.0)
        joint_unit_grads = jax.grad(weighted_objective)(p, 1.0)
        return (
            jax.grad(total_objective)(p),
            ppo_grads,
            # The objective is affine in survival_weight.
            subtract_gradient_trees(joint_unit_grads, ppo_grads),
        )

    total_grads, ppo_grads, survival_grads = compute_grads(params)
    diagnostics = summarize_gradient_interaction(
        params,
        total_grads,
        ppo_grads,
        survival_grads,
        lambda_surv,
    )

    for group in GRADIENT_GROUPS:
        for value in diagnostics["groups"][group].values():
            assert bool(jnp.all(jnp.isfinite(jnp.asarray(value))))
        group_diag = diagnostics["groups"][group]
        # Keep this below the maximum tolerance allowed for float32 reverse
        # passes while using a non-degenerate PPO gradient in every group.
        assert group_diag["decomposition_rel_error"] < 1e-4, (
            group,
            float(group_diag["decomposition_abs_error"]),
            float(group_diag["joint_grad_norm"]),
            float(group_diag["decomposition_rel_error"]),
        )

    assert diagnostics["groups"]["total"]["ppo_grad_norm"] > 0
    assert diagnostics["groups"]["reliability_head"]["survival_grad_norm_raw"] > 0
    raw_norm = diagnostics["groups"]["vision_encoder"]["survival_grad_norm_raw"]
    weighted_norm = diagnostics["groups"]["vision_encoder"][
        "survival_grad_norm_weighted"
    ]
    np.testing.assert_allclose(
        float(weighted_norm),
        abs(lambda_surv) * float(raw_norm),
        rtol=1e-5,
        atol=1e-7,
    )
    for group in ("actor_head", "critic_head"):
        assert diagnostics["groups"][group]["survival_grad_norm_raw"] < 1e-12
        assert not bool(diagnostics["groups"][group]["cosine_valid"])


@pytest.mark.parametrize("optimization_mode", ("joint", "phasic"))
def test_diagnostics_do_not_change_applied_update_or_rng(optimization_mode):
    params, components = _make_reliability_model_fixture()
    lambda_surv = 0.01

    def ppo_objective(p):
        ppo_loss, _survival_loss = components(p)
        return ppo_loss

    def survival_objective(p):
        _ppo_loss, survival_loss = components(p)
        return survival_loss

    def applied_objective(p):
        ppo_loss, survival_loss = components(p)
        if optimization_mode == "phasic":
            return ppo_loss
        return ppo_loss + lambda_surv * survival_loss

    tx = optax.adam(1e-3)
    opt_state = tx.init(params)
    initial_rng = jax.random.PRNGKey(99)

    def run_update(enable_diagnostics):
        next_rng, _ = jax.random.split(initial_rng)
        total_loss, total_grads = jax.value_and_grad(applied_objective)(params)
        if enable_diagnostics:
            ppo_grads = jax.grad(ppo_objective)(params)
            survival_grads = jax.grad(survival_objective)(params)
            if optimization_mode == "phasic":
                diagnostics = summarize_phasic_gradient_interaction(
                    params,
                    ppo_grads,
                    survival_grads,
                    survival_loss_pre_ppo=survival_objective(params),
                )
            else:
                diagnostics = summarize_gradient_interaction(
                    params,
                    total_grads,
                    ppo_grads,
                    survival_grads,
                    lambda_surv,
                )
            assert bool(diagnostics["grad_diag_active"])
        updates, next_opt_state = tx.update(total_grads, opt_state, params)
        next_params = optax.apply_updates(params, updates)
        return total_loss, total_grads, next_params, next_opt_state, next_rng

    disabled = run_update(False)
    enabled = run_update(True)
    np.testing.assert_allclose(
        np.asarray(disabled[0]),
        np.asarray(enabled[0]),
        rtol=0.0,
        atol=0.0,
    )
    # Compiling the graph with extra reverse passes can move the applied
    # float32 gradient by sub-ULP amounts (observed max abs: 4.66e-10).
    _assert_tree_allclose(disabled[1], enabled[1], atol=1e-9, rtol=1e-6)
    _assert_tree_allclose(disabled[2], enabled[2], atol=1e-8, rtol=1e-7)
    _assert_tree_allclose(disabled[3], enabled[3], atol=1e-9, rtol=1e-6)
    np.testing.assert_array_equal(np.asarray(disabled[4]), np.asarray(enabled[4]))


def test_status_branches_have_static_pytree_and_first_minibatch_gate():
    params = _toy_tree([1.0, 2.0])
    disabled = empty_gradient_interaction_diagnostics(params, enabled=False)
    skipped = empty_gradient_interaction_diagnostics(
        params,
        enabled=True,
        skipped_by_cadence=True,
    )
    not_applicable = empty_gradient_interaction_diagnostics(
        params,
        enabled=True,
        not_applicable=True,
        reason_survival_disabled=True,
    )
    ppo = jax.grad(lambda p: jnp.sum(p["params"]["x"] ** 2))(params)
    survival = jax.grad(lambda p: jnp.sum((p["params"]["x"] - 1) ** 2))(params)
    active = summarize_gradient_interaction(params, ppo, ppo, survival, 0.0)
    phasic_active = summarize_phasic_gradient_interaction(
        params,
        ppo,
        survival,
        survival_loss_pre_ppo=0.25,
    )
    structures = [
        jax.tree_util.tree_structure(item)
        for item in (disabled, skipped, not_applicable, active, phasic_active)
    ]
    assert all(structure == structures[0] for structure in structures[1:])

    assert bool(gradient_diag_should_run(4, 2, 0, 0))
    assert not bool(gradient_diag_should_run(3, 2, 0, 0))
    assert not bool(gradient_diag_should_run(4, 2, 1, 0))
    assert not bool(gradient_diag_should_run(4, 2, 0, 1))
    with pytest.raises(ValueError, match="must be >= 1"):
        validate_gradient_diag_config(
            {"grad_interaction_diag_every_updates": 0}
        )


def test_phasic_gradient_diagnostics_log_only_raw_interaction_metrics():
    params = {
        "params": {
            "ReliabilityFusionRNN_0": {
                "LevelWiseReliabilityHead_0": {
                    "kernel": jnp.array([1.0, 2.0], dtype=jnp.float32),
                },
                "fusion": {
                    "kernel": jnp.array([3.0], dtype=jnp.float32),
                },
            },
            "VisionAgent_0": {
                "kernel": jnp.array([4.0], dtype=jnp.float32),
            },
        }
    }
    ppo_grads = jax.tree_util.tree_map(jnp.ones_like, params)
    survival_grads = jax.tree_util.tree_map(
        lambda value: -2.0 * jnp.ones_like(value),
        params,
    )
    diagnostics = summarize_phasic_gradient_interaction(
        params,
        ppo_grads,
        survival_grads,
        survival_loss_pre_ppo=jnp.array(0.75, dtype=jnp.float32),
    )
    lines, metrics = format_gradient_interaction_diagnostics(
        diagnostics,
        update=3,
        optimization_mode="phasic",
    )
    assert len(lines) == 4
    assert all("status=active" in line for line in lines)
    assert all("optimization_mode=phasic" in line for line in lines)
    assert all("ppo_grad_norm=" in line for line in lines)
    assert all("survival_grad_norm_raw=" in line for line in lines)
    assert all("ppo_survival_dot_raw=" in line for line in lines)
    assert all("ppo_survival_cosine_raw=" in line for line in lines)
    forbidden = (
        "joint_grad_norm",
        "survival_grad_norm_weighted",
        "weighted_survival_to_ppo_grad_ratio",
        "decomposition_abs_error",
        "decomposition_rel_error",
    )
    assert all(
        forbidden_name not in line
        for line in lines
        for forbidden_name in forbidden
    )
    assert all(
        forbidden_name not in metric_name
        for metric_name in metrics
        for forbidden_name in forbidden
    )
    assert float(diagnostics["survival_loss_pre_ppo"]) == pytest.approx(0.75)
    assert float(
        diagnostics["groups"]["total"]["ppo_survival_cosine_raw"]
    ) == pytest.approx(-1.0)


def test_single_device_pmap_gradient_diagnostics_compile_and_are_finite():
    device_count = jax.local_device_count()
    initial = jnp.tile(jnp.array([[0.2, -0.4]], dtype=jnp.float32), (device_count, 1))

    @partial(jax.pmap, axis_name="device")
    def pmapped(values):
        params = {"params": {"x": values}}
        ppo = jax.grad(lambda p: jnp.sum(p["params"]["x"] ** 2))(params)
        survival = jax.grad(
            lambda p: jnp.sum((p["params"]["x"] - 1.0) ** 2)
        )(params)
        total = jax.grad(
            lambda p: (
                jnp.sum(p["params"]["x"] ** 2)
                + 0.01 * jnp.sum((p["params"]["x"] - 1.0) ** 2)
            )
        )(params)
        ppo = jax.lax.pmean(ppo, "device")
        survival = jax.lax.pmean(survival, "device")
        total = jax.lax.pmean(total, "device")
        return summarize_gradient_interaction(
            params,
            total,
            ppo,
            survival,
            0.01,
        )["groups"]["total"]

    metrics = pmapped(initial)
    for value in metrics.values():
        assert bool(jnp.all(jnp.isfinite(jnp.asarray(value))))
    assert bool(jnp.all(metrics["decomposition_rel_error"] < 1e-5))


def test_single_device_pmap_phasic_gradient_diagnostics_are_finite():
    device_count = jax.local_device_count()
    initial = jnp.tile(
        jnp.array([[0.2, -0.4]], dtype=jnp.float32),
        (device_count, 1),
    )

    @partial(jax.pmap, axis_name="device")
    def pmapped(values):
        params = {
            "params": {
                "ReliabilityFusionRNN_0": {
                    "LevelWiseReliabilityHead_0": {"kernel": values},
                    "fusion": {"kernel": values},
                },
                "VisionAgent_0": {"kernel": values},
            }
        }

        def objective(tree, offset):
            return sum(
                jnp.sum(jnp.square(leaf - offset))
                for leaf in jax.tree_util.tree_leaves(tree)
            )

        ppo = jax.grad(objective)(params, 0.0)
        survival = jax.grad(objective)(params, 1.0)
        ppo = jax.lax.pmean(ppo, "device")
        survival = jax.lax.pmean(survival, "device")
        return summarize_phasic_gradient_interaction(
            params,
            ppo,
            survival,
            survival_loss_pre_ppo=jax.lax.pmean(
                jnp.array(0.4, dtype=jnp.float32),
                "device",
            ),
        )

    diagnostics = pmapped(initial)
    assert bool(jnp.all(diagnostics["grad_diag_active"]))
    np.testing.assert_allclose(
        np.asarray(diagnostics["survival_loss_pre_ppo"]),
        0.4,
        atol=1e-7,
    )
    for group in (
        "total",
        "reliability_head",
        "vision_encoder",
        "fusion_shared_trunk",
    ):
        for key in (
            "ppo_grad_norm",
            "survival_grad_norm_raw",
            "ppo_survival_dot_raw",
            "ppo_survival_cosine_raw",
        ):
            assert bool(
                jnp.all(jnp.isfinite(diagnostics["groups"][group][key]))
            )


ACTOR_CRITIC_SHARED_GROUPS = (
    "reliability_head",
    "vision_encoder",
    "fusion_shared_trunk",
)


def _actor_critic_tree(head, vision, fusion, actor=(0.0, 0.0)):
    as_array = lambda values: jnp.asarray(values, dtype=jnp.float32)
    return {
        "params": {
            "ReliabilityFusionRNN_0": {
                "LevelWiseReliabilityHead_0": {"kernel": as_array(head)},
                "SharedBlock_0": {"kernel": as_array(fusion)},
            },
            "VisionAgent_0": {"kernel": as_array(vision)},
            "Dense_0": {"kernel": as_array(actor)},
        }
    }


def _actor_critic_params():
    return _actor_critic_tree(
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
    )


def _actor_critic_summary(
    policy,
    value,
    vf_coef=0.25,
    actor_policy=(0.0, 0.0),
):
    params = _actor_critic_params()
    policy_grads = _actor_critic_tree(
        policy,
        policy,
        policy,
        actor=actor_policy,
    )
    value_grads = _actor_critic_tree(value, value, value)
    ppo_grads = add_gradient_trees(
        policy_grads,
        scale_gradient_tree(value_grads, vf_coef),
    )
    return summarize_actor_critic_gradient_interaction(
        params,
        ppo_grads,
        ppo_grads,
        policy_grads,
        value_grads,
        vf_coef,
        max_grad_norm=0.5,
    )


@pytest.mark.parametrize(
    "value,expected_cosine",
    [
        ((3.0, 4.0), 1.0),
        ((-3.0, -4.0), -1.0),
        ((-4.0, 3.0), 0.0),
    ],
)
def test_actor_critic_cosine_known_directions(value, expected_cosine):
    diagnostics = _actor_critic_summary((3.0, 4.0), value)
    for group in ACTOR_CRITIC_SHARED_GROUPS:
        group_metrics = diagnostics["groups"][group]
        np.testing.assert_allclose(
            float(group_metrics["policy_value_cosine_raw"]),
            expected_cosine,
            atol=1e-6,
        )
        assert bool(group_metrics["cosine_valid"])


def test_actor_critic_weighted_value_norm_and_ratio_use_vf_coef():
    diagnostics = _actor_critic_summary(
        (3.0, 4.0),
        (6.0, 8.0),
        vf_coef=0.25,
    )
    for group in ACTOR_CRITIC_SHARED_GROUPS:
        group_metrics = diagnostics["groups"][group]
        np.testing.assert_allclose(
            float(group_metrics["value_grad_norm_raw"]),
            10.0,
        )
        np.testing.assert_allclose(
            float(group_metrics["value_grad_norm_weighted"]),
            2.5,
        )
        np.testing.assert_allclose(
            float(group_metrics["weighted_value_to_policy_grad_ratio"]),
            0.5,
        )


def test_actor_critic_direct_ppo_gradient_decomposition_is_near_exact():
    params = _actor_critic_tree(
        (0.2, -0.3),
        (0.4, -0.5),
        (0.6, -0.7),
        actor=(0.8, -0.9),
    )
    vf_coef = 0.125

    def policy_objective(tree):
        return sum(
            jnp.sum(jnp.square(leaf + 0.25))
            for leaf in jax.tree_util.tree_leaves(tree)
        )

    def value_objective(tree):
        return sum(
            jnp.sum(jnp.square(leaf - 0.75))
            for leaf in jax.tree_util.tree_leaves(tree)
        )

    def ppo_objective(tree):
        return policy_objective(tree) + vf_coef * value_objective(tree)

    policy_grads = jax.grad(policy_objective)(params)
    value_grads = jax.grad(value_objective)(params)
    ppo_grads = jax.grad(ppo_objective)(params)
    diagnostics = summarize_actor_critic_gradient_interaction(
        params,
        ppo_grads,
        ppo_grads,
        policy_grads,
        value_grads,
        vf_coef,
        max_grad_norm=0.5,
    )
    for group in ACTOR_CRITIC_GRADIENT_GROUPS:
        assert (
            float(
                diagnostics["groups"][group][
                    "ppo_decomposition_rel_error"
                ]
            )
            < 1e-6
        )


def test_actor_critic_zero_gradient_is_finite_and_invalid():
    diagnostics = _actor_critic_summary((0.0, 0.0), (1.0, -2.0))
    for group in ACTOR_CRITIC_SHARED_GROUPS:
        group_metrics = diagnostics["groups"][group]
        assert np.isfinite(float(group_metrics["policy_value_cosine_raw"]))
        assert float(group_metrics["policy_value_cosine_raw"]) == 0.0
        assert not bool(group_metrics["policy_grad_nonzero"])
        assert bool(group_metrics["value_grad_nonzero"])
        assert not bool(group_metrics["cosine_valid"])


def test_actor_critic_shared_groups_exclude_disjoint_actor_head():
    diagnostics = _actor_critic_summary(
        (3.0, 4.0),
        (6.0, 8.0),
        actor_policy=(100.0, 100.0),
    )
    assert tuple(diagnostics["groups"]) == ACTOR_CRITIC_GRADIENT_GROUPS
    for group in ACTOR_CRITIC_SHARED_GROUPS:
        group_metrics = diagnostics["groups"][group]
        assert int(group_metrics["param_leaf_count"]) == 1
        np.testing.assert_allclose(
            float(group_metrics["policy_grad_norm"]),
            5.0,
        )
    assert diagnostics["groups"]["total"]["policy_grad_norm"] > 100.0


def test_actor_critic_disabled_and_skipped_paths_have_static_pytree():
    params = _actor_critic_params()
    disabled = empty_actor_critic_gradient_diagnostics(
        params,
        enabled=False,
        max_grad_norm=0.5,
    )
    skipped = empty_actor_critic_gradient_diagnostics(
        params,
        enabled=True,
        skipped_by_cadence=True,
        max_grad_norm=0.5,
    )
    active = _actor_critic_summary((1.0, 2.0), (2.0, 1.0))
    assert jax.tree_util.tree_structure(disabled) == jax.tree_util.tree_structure(
        skipped
    )
    assert jax.tree_util.tree_structure(disabled) == jax.tree_util.tree_structure(
        active
    )
    assert not bool(disabled["actor_critic_grad_diag_enabled"])
    assert not bool(disabled["actor_critic_grad_diag_active"])
    assert bool(skipped["actor_critic_grad_diag_skipped_by_cadence"])
    for group in ACTOR_CRITIC_GRADIENT_GROUPS:
        assert float(disabled["groups"][group]["policy_grad_norm"]) == 0.0


def test_actor_critic_cadence_is_first_minibatch_first_epoch_only():
    assert validate_actor_critic_grad_diag_config({}) == 1
    assert validate_actor_critic_grad_diag_config(
        {"actor_critic_grad_diag_every_updates": 3}
    ) == 3
    with pytest.raises(ValueError):
        validate_actor_critic_grad_diag_config(
            {"actor_critic_grad_diag_every_updates": 0}
        )
    assert bool(actor_critic_grad_diag_should_run(6, 3, 0, 0))
    assert not bool(actor_critic_grad_diag_should_run(5, 3, 0, 0))
    assert not bool(actor_critic_grad_diag_should_run(6, 3, 1, 0))
    assert not bool(actor_critic_grad_diag_should_run(6, 3, 0, 1))


def test_actor_critic_summary_is_read_only_and_reports_clip_estimate():
    params = _actor_critic_params()
    policy_grads = _actor_critic_tree(
        (3.0, 4.0),
        (0.0, 0.0),
        (0.0, 0.0),
    )
    value_grads = _actor_critic_tree(
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
    )
    ppo_grads = add_gradient_trees(policy_grads, value_grads)
    before = jax.tree_util.tree_map(
        lambda value: np.asarray(value).copy(),
        ppo_grads,
    )
    diagnostics = summarize_actor_critic_gradient_interaction(
        params,
        ppo_grads,
        ppo_grads,
        policy_grads,
        value_grads,
        vf_coef=0.25,
        max_grad_norm=0.5,
    )
    for old, new in zip(
        jax.tree_util.tree_leaves(before),
        jax.tree_util.tree_leaves(ppo_grads),
    ):
        np.testing.assert_array_equal(old, np.asarray(new))
    np.testing.assert_allclose(
        float(diagnostics["pre_clip_total_grad_norm"]),
        5.0,
    )
    np.testing.assert_allclose(
        float(diagnostics["estimated_global_clip_scale"]),
        0.1,
    )


def _value_clip_summary(old, new, target, active=None, clip_eps=0.2):
    old = jnp.asarray(old, dtype=jnp.float32)
    new = jnp.asarray(new, dtype=jnp.float32)
    target = jnp.asarray(target, dtype=jnp.float32)
    if active is None:
        active = jnp.ones_like(target, dtype=jnp.bool_)
    active = jnp.asarray(active, dtype=jnp.bool_)
    clipped = old + jnp.clip(new - old, -clip_eps, clip_eps)
    raw_losses = jnp.square(new - target)
    clipped_losses = jnp.square(clipped - target)
    return summarize_value_clip_diagnostics(
        old_value=old,
        new_value=new,
        target=target,
        value_pred_clipped=clipped,
        value_losses=raw_losses,
        value_losses_clipped=clipped_losses,
        agent_active=active,
        clip_eps=clip_eps,
    )


def test_value_clip_no_saturation_and_boundary_behavior():
    diagnostics = _value_clip_summary(
        old=[0.0, 0.0, 0.0],
        new=[0.1, 0.2, 0.200001],
        target=[1.0, 1.0, 1.0],
    )
    np.testing.assert_allclose(
        float(diagnostics["clip_saturated_rate"]),
        1.0 / 3.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        float(diagnostics["branch_tie_rate"]),
        2.0 / 3.0,
        atol=1e-6,
    )


def test_value_clip_saturated_clipped_branch_locks_gradient():
    diagnostics = _value_clip_summary(old=[0.0], new=[1.0], target=[10.0])
    assert float(diagnostics["clip_saturated_rate"]) == 1.0
    assert float(diagnostics["clipped_branch_selected_rate"]) == 1.0
    assert float(diagnostics["unclipped_branch_selected_rate"]) == 0.0
    assert float(diagnostics["value_gradient_locked_rate"]) == 1.0


def test_value_clip_saturated_unclipped_branch_keeps_gradient():
    diagnostics = _value_clip_summary(old=[0.0], new=[-1.0], target=[10.0])
    assert float(diagnostics["clip_saturated_rate"]) == 1.0
    assert float(diagnostics["unclipped_branch_selected_rate"]) == 1.0
    assert float(diagnostics["clipped_branch_selected_rate"]) == 0.0
    assert float(diagnostics["value_gradient_locked_rate"]) == 0.0
    assert (
        float(diagnostics["clip_saturated_and_unclipped_branch_rate"])
        == 1.0
    )


def test_value_clip_explained_variance_and_degenerate_target():
    diagnostics = _value_clip_summary(
        old=[0.0, 0.0, 0.0],
        new=[1.0, 2.0, 3.0],
        target=[1.0, 2.0, 3.0],
    )
    assert bool(diagnostics["explained_variance_valid"])
    np.testing.assert_allclose(
        float(diagnostics["new_value_explained_variance"]),
        1.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        float(diagnostics["old_value_explained_variance"]),
        0.0,
        atol=1e-6,
    )

    degenerate = _value_clip_summary(
        old=[0.0, 0.0],
        new=[1.0, 1.0],
        target=[2.0, 2.0],
    )
    assert not bool(degenerate["explained_variance_valid"])
    assert float(degenerate["old_value_explained_variance"]) == 0.0
    assert float(degenerate["new_value_explained_variance"]) == 0.0
    for value in jax.tree_util.tree_leaves(degenerate):
        assert bool(jnp.all(jnp.isfinite(value)))

    empty = _value_clip_summary(
        old=[-1000.0, 1000.0],
        new=[1000.0, -1000.0],
        target=[5000.0, -5000.0],
        active=[False, False],
    )
    assert float(empty["active_sample_count"]) == 0.0
    assert not bool(empty["explained_variance_valid"])
    for value in jax.tree_util.tree_leaves(empty):
        assert bool(jnp.all(jnp.isfinite(value)))


def test_value_clip_active_mask_excludes_extreme_samples_and_quantiles():
    diagnostics = _value_clip_summary(
        old=[0.0, 0.0, -10000.0],
        new=[0.0, 0.0, 10000.0],
        target=[0.2, 0.4, 10000.0],
        active=[True, True, False],
    )
    np.testing.assert_allclose(float(diagnostics["active_sample_count"]), 2.0)
    np.testing.assert_allclose(float(diagnostics["target_mean"]), 0.3)
    np.testing.assert_allclose(
        float(diagnostics["target_distance_to_clip_ratio_mean"]),
        1.5,
    )
    np.testing.assert_allclose(
        float(diagnostics["target_distance_to_clip_ratio_p50"]),
        1.5,
    )
    np.testing.assert_allclose(
        float(diagnostics["target_distance_to_clip_ratio_p95"]),
        1.95,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        float(diagnostics["target_std_to_clip_ratio"]),
        0.5,
        atol=1e-6,
    )


def test_value_clip_disabled_pytree_and_cadence_are_static():
    disabled = empty_value_clip_diagnostics(
        enabled=False,
        clip_eps=0.2,
    )
    skipped = empty_value_clip_diagnostics(
        enabled=True,
        skipped_by_cadence=True,
        clip_eps=0.2,
    )
    active = _value_clip_summary(old=[0.0], new=[0.1], target=[1.0])
    assert jax.tree_util.tree_structure(disabled) == jax.tree_util.tree_structure(
        skipped
    )
    assert jax.tree_util.tree_structure(disabled) == jax.tree_util.tree_structure(
        active
    )
    assert not bool(disabled["value_clip_diag_enabled"])
    assert bool(skipped["value_clip_diag_skipped_by_cadence"])
    assert validate_value_clip_diag_config({}) == 1
    assert validate_value_clip_diag_config(
        {"value_clip_diag_every_updates": 3}
    ) == 3
    with pytest.raises(ValueError):
        validate_value_clip_diag_config({"value_clip_diag_every_updates": 0})
    assert bool(value_clip_diag_should_run(6, 3, 0, 0))
    assert not bool(value_clip_diag_should_run(5, 3, 0, 0))
    assert not bool(value_clip_diag_should_run(6, 3, 1, 0))
    assert not bool(value_clip_diag_should_run(6, 3, 0, 1))


def test_value_clip_summary_is_jittable_and_read_only():
    old = jnp.asarray([0.0, 0.25], dtype=jnp.float32)
    new = jnp.asarray([1.0, -0.5], dtype=jnp.float32)
    target = jnp.asarray([10.0, -2.0], dtype=jnp.float32)
    active = jnp.asarray([True, True])
    clipped = old + jnp.clip(new - old, -0.2, 0.2)
    raw_losses = jnp.square(new - target)
    clipped_losses = jnp.square(clipped - target)
    inputs = (old, new, target, clipped, raw_losses, clipped_losses, active)
    before = tuple(np.asarray(value).copy() for value in inputs)

    @jax.jit
    def summarize(values):
        return summarize_value_clip_diagnostics(
            old_value=values[0],
            new_value=values[1],
            target=values[2],
            value_pred_clipped=values[3],
            value_losses=values[4],
            value_losses_clipped=values[5],
            agent_active=values[6],
            clip_eps=0.2,
        )

    diagnostics = summarize(inputs)
    for expected, actual in zip(before, inputs):
        np.testing.assert_array_equal(expected, np.asarray(actual))
    for value in jax.tree_util.tree_leaves(diagnostics):
        assert bool(jnp.all(jnp.isfinite(value)))


def _critic_optimizer_tree(
    reliability,
    vision,
    fusion,
    critic,
):
    as_array = lambda values: jnp.asarray(values, dtype=jnp.float32)
    return {
        "params": {
            "ReliabilityFusionRNN_0": {
                "LevelWiseReliabilityHead_0": {
                    "kernel": as_array(reliability)
                },
                "SharedBlock_0": {"kernel": as_array(fusion)},
            },
            "VisionAgent_0": {"kernel": as_array(vision)},
            "Dense_2": {"kernel": as_array(critic)},
        }
    }


def _critic_optimizer_fixture(max_grad_norm=20.0):
    params = _critic_optimizer_tree(
        (1.0, 1.0),
        (1.0, 1.0),
        (1.0, 1.0),
        (1.0, 1.0),
    )
    policy_grads = _critic_optimizer_tree(
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
    )
    value_grads = _critic_optimizer_tree(
        (3.0, 4.0),
        (3.0, 4.0),
        (3.0, 4.0),
        (3.0, 4.0),
    )
    tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(1e-2, eps=1e-5),
    )
    opt_state = tx.init(params)
    return params, policy_grads, value_grads, tx, opt_state


def _critic_optimizer_summary(max_grad_norm=20.0):
    params, policy_grads, value_grads, tx, opt_state = (
        _critic_optimizer_fixture(max_grad_norm)
    )
    diagnostics = summarize_critic_optimizer_diagnostics(
        params=params,
        optimizer_state=opt_state,
        optimizer=tx,
        policy_gradients=policy_grads,
        value_gradients_raw=value_grads,
        current_vf_coef=1.0,
        max_grad_norm=max_grad_norm,
    )
    return diagnostics, params, policy_grads, value_grads, tx, opt_state


def test_critic_optimizer_counterfactual_scaling_and_global_clipping():
    diagnostics, *_ = _critic_optimizer_summary(max_grad_norm=20.0)
    current = diagnostics["variants"]["current"]
    ten = diagnostics["variants"]["10x"]
    hundred = diagnostics["variants"]["100x"]

    np.testing.assert_allclose(float(current["coefficient"]), 1.0)
    np.testing.assert_allclose(float(ten["coefficient"]), 10.0)
    np.testing.assert_allclose(float(hundred["coefficient"]), 100.0)

    np.testing.assert_allclose(
        float(current["groups"]["critic_head"]["preclip_grad_norm"]),
        5.0,
    )
    np.testing.assert_allclose(
        float(ten["groups"]["critic_head"]["preclip_grad_norm"]),
        50.0,
    )
    np.testing.assert_allclose(
        float(hundred["groups"]["critic_head"]["preclip_grad_norm"]),
        500.0,
    )
    np.testing.assert_allclose(float(current["preclip_total_grad_norm"]), 10.0)
    np.testing.assert_allclose(float(current["estimated_global_clip_scale"]), 1.0)
    np.testing.assert_allclose(float(ten["estimated_global_clip_scale"]), 0.2)
    np.testing.assert_allclose(
        float(hundred["estimated_global_clip_scale"]),
        0.02,
    )
    assert not bool(current["global_clip_active"])
    assert bool(ten["global_clip_active"])
    assert bool(hundred["global_clip_active"])
    np.testing.assert_allclose(float(ten["postclip_total_grad_norm"]), 20.0)
    np.testing.assert_allclose(
        float(hundred["postclip_total_grad_norm"]),
        20.0,
    )

    response = diagnostics["scale_response"]["critic_head"]
    np.testing.assert_allclose(
        float(response["preclip_grad_ratio_10x_to_current"]),
        10.0,
    )
    np.testing.assert_allclose(
        float(response["preclip_grad_ratio_100x_to_current"]),
        100.0,
    )
    np.testing.assert_allclose(
        float(response["postclip_grad_ratio_10x_to_current"]),
        2.0,
    )
    np.testing.assert_allclose(
        float(response["postclip_grad_ratio_100x_to_current"]),
        2.0,
        atol=1e-6,
    )


def test_critic_optimizer_reuses_exact_optax_updates_and_preserves_inputs():
    diagnostics, params, policy_grads, value_grads, tx, opt_state = (
        _critic_optimizer_summary(max_grad_norm=20.0)
    )
    params_before = jax.tree_util.tree_map(
        lambda value: np.asarray(value).copy(),
        params,
    )
    state_before = jax.tree_util.tree_map(
        lambda value: np.asarray(value).copy(),
        opt_state,
    )
    actual_training_grad = add_gradient_trees(
        policy_grads,
        scale_gradient_tree(value_grads, 0.25),
    )
    actual_before = jax.tree_util.tree_map(
        lambda value: np.asarray(value).copy(),
        actual_training_grad,
    )

    for variant, coefficient in (("current", 1.0), ("10x", 10.0), ("100x", 100.0)):
        counterfactual_grad = add_gradient_trees(
            policy_grads,
            scale_gradient_tree(value_grads, coefficient),
        )
        direct_updates, _ = tx.update(counterfactual_grad, opt_state, params)
        for group in CRITIC_OPTIMIZER_DIAGNOSTIC_GROUPS:
            parameter_group = "total" if group == "total_network" else group
            np.testing.assert_allclose(
                float(
                    diagnostics["variants"][variant]["groups"][group][
                        "param_step_norm"
                    ]
                ),
                float(gradient_l2_norm(direct_updates, parameter_group)),
                rtol=1e-6,
                atol=1e-8,
            )

    _assert_tree_allclose(params_before, params, atol=0.0, rtol=0.0)
    _assert_tree_allclose(state_before, opt_state, atol=0.0, rtol=0.0)
    _assert_tree_allclose(
        actual_before,
        actual_training_grad,
        atol=0.0,
        rtol=0.0,
    )
    for group in CRITIC_OPTIMIZER_DIAGNOSTIC_GROUPS:
        response = diagnostics["scale_response"][group]
        current_step = diagnostics["variants"]["current"]["groups"][group][
            "param_step_norm"
        ]
        ten_step = diagnostics["variants"]["10x"]["groups"][group][
            "param_step_norm"
        ]
        np.testing.assert_allclose(
            float(response["step_ratio_10x_to_current"]),
            float(ten_step / jnp.maximum(current_step, 1e-12)),
        )
        current_group = diagnostics["variants"]["current"]["groups"][group]
        np.testing.assert_allclose(
            float(current_group["relative_param_step"]),
            float(
                current_group["param_step_norm"]
                / jnp.maximum(current_group["param_norm"], 1e-12)
            ),
        )


def test_critic_optimizer_disabled_pytree_cadence_and_group_norms():
    disabled = empty_critic_optimizer_diagnostics(
        enabled=False,
        max_grad_norm=0.5,
    )
    skipped = empty_critic_optimizer_diagnostics(
        enabled=True,
        skipped_by_cadence=True,
        max_grad_norm=0.5,
    )
    active, *_ = _critic_optimizer_summary(max_grad_norm=20.0)
    assert jax.tree_util.tree_structure(disabled) == jax.tree_util.tree_structure(
        skipped
    )
    assert jax.tree_util.tree_structure(disabled) == jax.tree_util.tree_structure(
        active
    )
    assert tuple(active["variants"]["current"]["groups"]) == (
        CRITIC_OPTIMIZER_DIAGNOSTIC_GROUPS
    )
    np.testing.assert_allclose(
        float(active["variants"]["current"]["groups"]["critic_head"]["param_norm"]),
        np.sqrt(2.0),
    )
    assert not bool(disabled["critic_optimizer_diag_enabled"])
    assert bool(skipped["critic_optimizer_diag_skipped_by_cadence"])
    assert validate_critic_optimizer_diag_config({}) == 1
    assert validate_critic_optimizer_diag_config(
        {"critic_optimizer_diag_every_updates": 3}
    ) == 3
    with pytest.raises(ValueError):
        validate_critic_optimizer_diag_config(
            {"critic_optimizer_diag_every_updates": 0}
        )
    assert bool(critic_optimizer_diag_should_run(6, 3, 0, 0))
    assert not bool(critic_optimizer_diag_should_run(5, 3, 0, 0))
    assert not bool(critic_optimizer_diag_should_run(6, 3, 1, 0))
    assert not bool(critic_optimizer_diag_should_run(6, 3, 0, 1))


def test_critic_optimizer_diagnostics_are_jittable_and_finite():
    params, policy_grads, value_grads, tx, opt_state = (
        _critic_optimizer_fixture(max_grad_norm=20.0)
    )

    @jax.jit
    def run(p, state, policy, value):
        return summarize_critic_optimizer_diagnostics(
            params=p,
            optimizer_state=state,
            optimizer=tx,
            policy_gradients=policy,
            value_gradients_raw=value,
            current_vf_coef=1.0,
            max_grad_norm=20.0,
        )

    diagnostics = run(params, opt_state, policy_grads, value_grads)
    assert bool(diagnostics["critic_optimizer_diag_active"])
    assert not bool(diagnostics["adam_internals_available"])
    for value in jax.tree_util.tree_leaves(diagnostics):
        assert bool(jnp.all(jnp.isfinite(value)))


def _value_probe_params(shared_scale, critic_scale=1.0):
    scalar = lambda value: jnp.asarray([value], dtype=jnp.float32)
    return {
        "params": {
            "Dense_0": {"kernel": scalar(7.0)},
            "Dense_1": {"bias": scalar(-3.0)},
            "Dense_2": {"kernel": scalar(critic_scale)},
            "Dense_3": {"bias": scalar(0.0)},
            "log_std": scalar(0.25),
            "VisionAgent_0": {"kernel": scalar(shared_scale)},
            "ReliabilityFusionRNN_0": {
                "SharedBlock_0": {"kernel": scalar(shared_scale)},
                "LevelWiseReliabilityHead_0": {
                    "kernel": scalar(shared_scale)
                },
            },
        }
    }


def _value_probe_apply(params, init_hstate, obs, rnn_reset):
    del init_hstate, rnn_reset
    model_params = params["params"]
    shared_scale = (
        model_params["VisionAgent_0"]["kernel"][0]
        + model_params["ReliabilityFusionRNN_0"]["SharedBlock_0"]["kernel"][0]
        + model_params["ReliabilityFusionRNN_0"][
            "LevelWiseReliabilityHead_0"
        ]["kernel"][0]
    )
    representation = shared_scale * obs["x"]
    return (
        model_params["Dense_2"]["kernel"][0] * representation
        + model_params["Dense_3"]["bias"][0]
    )


def _value_probe_data():
    per_trajectory = jnp.asarray([-2.0, -1.0, 1.0, 2.0], dtype=jnp.float32)
    x = jnp.tile(per_trajectory[:, None], (1, 10))
    return {
        "obs": {"x": x},
        "rnn_reset": jnp.zeros_like(x, dtype=jnp.bool_),
        "init_hstate": jnp.zeros((10, 1), dtype=jnp.float32),
        "active": jnp.ones_like(x, dtype=jnp.bool_),
        "target": x,
    }


def test_value_probe_trajectory_split_is_deterministic_and_disjoint():
    active = jnp.ones((4, 12), dtype=jnp.bool_).at[2, 8].set(False)
    training_key = jax.random.PRNGKey(123)
    training_key_before = np.asarray(training_key).copy()
    first = trajectory_train_holdout_masks(
        active,
        num_environments=6,
        train_fraction=2.0 / 3.0,
        split_seed=0,
    )
    second = trajectory_train_holdout_masks(
        active,
        num_environments=6,
        train_fraction=2.0 / 3.0,
        split_seed=0,
    )
    train_mask, holdout_mask, train_count, holdout_count = first
    _assert_tree_allclose(first, second, atol=0.0, rtol=0.0)
    assert train_count == 4
    assert holdout_count == 2
    assert not bool(jnp.any(train_mask & holdout_mask))
    np.testing.assert_array_equal(train_mask | holdout_mask, active)
    for environment_id in range(6):
        actor_slice = slice(2 * environment_id, 2 * environment_id + 2)
        assert bool(jnp.all(train_mask[0, actor_slice])) or bool(
            jnp.all(holdout_mask[0, actor_slice])
        )
    for actor_index in range(active.shape[1]):
        active_timesteps = active[:, actor_index]
        assert not (
            bool(jnp.any(train_mask[:, actor_index] & active_timesteps))
            and bool(jnp.any(holdout_mask[:, actor_index] & active_timesteps))
        )
    np.testing.assert_array_equal(training_key, training_key_before)


def test_value_probe_trajectory_split_changes_with_seed():
    active = jnp.ones((3, 40), dtype=jnp.bool_)
    seed_zero = trajectory_train_holdout_masks(
        active,
        num_environments=20,
        train_fraction=0.8,
        split_seed=0,
    )
    seed_one = trajectory_train_holdout_masks(
        active,
        num_environments=20,
        train_fraction=0.8,
        split_seed=1,
    )
    assert not np.array_equal(np.asarray(seed_zero[0]), np.asarray(seed_one[0]))
    assert seed_zero[2:] == seed_one[2:] == (16, 4)


def test_value_probe_statistics_mse_mae_and_explained_variance():
    target = jnp.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=jnp.float32)
    prediction = jnp.asarray([[1.0, 1.0, 3.0, 3.0]], dtype=jnp.float32)
    metrics = value_probe_statistics(
        prediction,
        target,
        jnp.ones_like(target, dtype=jnp.bool_),
    )
    np.testing.assert_allclose(float(metrics["mse"]), 0.5)
    np.testing.assert_allclose(float(metrics["rmse"]), np.sqrt(0.5))
    np.testing.assert_allclose(float(metrics["mae"]), 0.5)
    np.testing.assert_allclose(float(metrics["explained_variance"]), 0.8)
    assert bool(metrics["explained_variance_valid"])


def test_value_probe_a_changes_only_critic_and_b_freezes_actor_parameters():
    data = _value_probe_data()
    params = _value_probe_params(shared_scale=1.0 / 3.0, critic_scale=0.1)
    params_before = jax.tree_util.tree_map(lambda value: np.asarray(value).copy(), params)
    real_optimizer_state = {"count": jnp.asarray(11, dtype=jnp.int32)}
    optimizer_state_before = jax.tree_util.tree_map(
        lambda value: np.asarray(value).copy(),
        real_optimizer_state,
    )
    train_mask, _, _, _ = trajectory_train_holdout_masks(
        data["active"],
        num_environments=10,
        train_fraction=0.8,
        split_seed=0,
    )
    params_a, params_b = fit_value_representation_probe_variants(
        _value_probe_apply,
        params,
        data["init_hstate"],
        data["obs"],
        data["rnn_reset"],
        data["target"] * 2.0,
        train_mask,
        steps=100,
        learning_rate=0.03,
    )
    _assert_tree_allclose(params_before, params, atol=0.0, rtol=0.0)
    _assert_tree_allclose(
        optimizer_state_before,
        real_optimizer_state,
        atol=0.0,
        rtol=0.0,
    )
    for path in ("Dense_0", "Dense_1"):
        _assert_tree_allclose(
            params_a["params"][path],
            params["params"][path],
            atol=0.0,
            rtol=0.0,
        )
        _assert_tree_allclose(
            params_b["params"][path],
            params["params"][path],
            atol=0.0,
            rtol=0.0,
        )
    _assert_tree_allclose(
        params_a["params"]["log_std"],
        params["params"]["log_std"],
        atol=0.0,
        rtol=0.0,
    )
    _assert_tree_allclose(
        params_b["params"]["log_std"],
        params["params"]["log_std"],
        atol=0.0,
        rtol=0.0,
    )
    for path in ("VisionAgent_0", "ReliabilityFusionRNN_0"):
        _assert_tree_allclose(
            params_a["params"][path],
            params["params"][path],
            atol=0.0,
            rtol=0.0,
        )
    assert not np.allclose(
        np.asarray(params_a["params"]["Dense_2"]["kernel"]),
        np.asarray(params["params"]["Dense_2"]["kernel"]),
    )
    assert not np.allclose(
        np.asarray(params_b["params"]["VisionAgent_0"]["kernel"]),
        np.asarray(params["params"]["VisionAgent_0"]["kernel"]),
    )
    assert not np.allclose(
        np.asarray(
            params_b["params"]["ReliabilityFusionRNN_0"]["SharedBlock_0"][
                "kernel"
            ]
        ),
        np.asarray(
            params["params"]["ReliabilityFusionRNN_0"]["SharedBlock_0"][
                "kernel"
            ]
        ),
    )


def test_value_probe_b_outperforms_a_when_frozen_features_are_insufficient():
    data = _value_probe_data()
    params = _value_probe_params(shared_scale=0.0, critic_scale=1.0)
    diagnostics = run_value_representation_probe(
        _value_probe_apply,
        params,
        data["init_hstate"],
        data["obs"],
        data["rnn_reset"],
        data["target"],
        data["active"],
        num_environments=10,
        steps=150,
        learning_rate=0.03,
        train_fraction=0.8,
        split_seed=0,
    )
    holdout_a = diagnostics["probe_a"]["holdout"]["final"]
    holdout_b = diagnostics["probe_b"]["holdout"]["final"]
    assert int(diagnostics["split_seed"]) == 0
    assert bool(diagnostics["probes_start_identical"])
    assert float(holdout_b["explained_variance"]) > (
        float(holdout_a["explained_variance"]) + 0.5
    )
    assert float(holdout_b["mae"]) < float(holdout_a["mae"])
    assert float(
        diagnostics["probe_b"]["parameter_change"]["vision_encoder"]
    ) > 0.0
    assert float(
        diagnostics["probe_b"]["parameter_change"]["fusion_shared_trunk"]
    ) > 0.0
    assert float(
        diagnostics["probe_b"]["parameter_change"]["reliability_head"]
    ) > 0.0


def test_value_probe_sufficient_features_fit_and_disabled_path_is_static():
    data = _value_probe_data()
    params = _value_probe_params(shared_scale=1.0 / 3.0, critic_scale=0.1)

    @jax.jit
    def run_probe(probe_params):
        return run_value_representation_probe(
            _value_probe_apply,
            probe_params,
            data["init_hstate"],
            data["obs"],
            data["rnn_reset"],
            data["target"] * 2.0,
            data["active"],
            num_environments=10,
            steps=100,
            learning_rate=0.03,
            train_fraction=0.8,
            split_seed=0,
        )

    diagnostics = run_probe(params)
    for probe_name in ("probe_a", "probe_b"):
        assert float(
            diagnostics[probe_name]["holdout"]["final"]["mae"]
        ) < 0.1
    disabled = empty_value_representation_probe_diagnostics(enabled=False)
    skipped = empty_value_representation_probe_diagnostics(
        enabled=True,
        skipped_by_schedule=True,
        steps=150,
        learning_rate=0.001,
        train_fraction=0.8,
        split_seed=0,
    )
    assert jax.tree_util.tree_structure(disabled) == jax.tree_util.tree_structure(
        skipped
    )
    assert jax.tree_util.tree_structure(disabled) == jax.tree_util.tree_structure(
        diagnostics
    )
    assert not bool(disabled["enabled"])
    assert bool(skipped["skipped_by_schedule"])
    validated = validate_value_representation_probe_config({})
    assert validated["updates"] == (5, 10, 15)
    assert validated["steps"] == 150
    np.testing.assert_allclose(validated["learning_rate"], 0.001)
    np.testing.assert_allclose(validated["train_fraction"], 0.8)
    assert validated["split_seed"] == 0
    assert validate_value_representation_probe_config(
        {"value_representation_probe_split_seed": 17}
    )["split_seed"] == 17


@pytest.fixture(scope="module")
def separate_critic_fixture():
    config = {
        "FC_DIM_SIZE": 16, "GRU_HIDDEN_DIM": 16,
        "use_reliability_head": True, "use_h_prev_in_reliability": True,
        "reliability_hidden_dim": 16, "reliability_gate_epsilon": 0.1,
        "use_separate_critic_representation": True,
    }
    action_space = spaces.Box(
        low=jnp.array([-1., 0., 0.]), high=jnp.array([3., 1., 1.]),
        shape=(3,), dtype=jnp.float32,
    )
    model = ActorCriticRNN(action_space, config=config, is_execution=True)
    hidden = model.initialize_carry(2)
    obs = {
        "exec_obs": jnp.linspace(-1., 1., 4 * 2 * 28).reshape(4, 2, 28),
        "vision_obs": jnp.linspace(0.1, 2., 4 * 2 * 60).reshape(4, 2, 10, 3, 2),
        "mid_context": jnp.linspace(-0.5, 0.5, 4 * 2 * 4).reshape(4, 2, 4),
    }
    resets = jnp.zeros((4, 2), dtype=jnp.bool_).at[2, 0].set(True)
    params = model.init(jax.random.PRNGKey(37), hidden, (obs, resets))
    return model, params, hidden, obs, resets


def test_separate_critic_topology_actor_parity_and_legacy(separate_critic_fixture):
    model, params, hidden, obs, resets = separate_critic_fixture
    legacy = ActorCriticRNN(
        model.action_space,
        config={**model.config, "use_separate_critic_representation": False},
        is_execution=True,
    )
    legacy_hidden = legacy.initialize_carry(2)
    legacy_params = legacy.init(jax.random.PRNGKey(37), legacy_hidden, (obs, resets))
    assert hidden.shape == (2, 32)
    assert legacy_hidden.shape == (2, 16)
    assert "CriticValueRNN_0" not in legacy_params["params"]
    assert set(params["params"]) == set(legacy_params["params"]) | {"CriticValueRNN_0"}
    assert sum(key.startswith("VisionAgent") for key in params["params"]) == 1
    for key in legacy_params["params"]:
        _assert_tree_allclose(params["params"][key], legacy_params["params"][key], atol=0, rtol=0)
    with jax.default_matmul_precision("highest"):
        new_hidden, _, value, _, aux = jax.jit(model.apply)(params, hidden, (obs, resets))
        old_hidden, _, _, _, old_aux = jax.jit(legacy.apply)(legacy_params, legacy_hidden, (obs, resets))
    assert value.shape == (4, 2)
    assert aux["policy_loc"].shape == (4, 2, 3)
    for key in ("policy_loc", "policy_log_std", "reliability_logits", "reliability_scores"):
        np.testing.assert_allclose(aux[key], old_aux[key], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(new_hidden[:, :16], old_hidden, atol=1e-6, rtol=1e-6)
    for leaf in jax.tree_util.tree_leaves((new_hidden, value, aux)):
        assert np.isfinite(np.asarray(leaf)).all()
    critic_paths = set(matching_parameter_paths(params, "critic_representation"))
    assert critic_paths == {
        path for path in flatten_tree_with_paths(params)
        if path[:2] == ("params", "CriticValueRNN_0")
    }
    assert {path[2] for path in critic_paths} == {
        "StableGatedCrossAttention_0", "Dense_0", "GRUCell_0",
    }
    for group in GRADIENT_GROUPS[1:]:
        assert critic_paths.isdisjoint(matching_parameter_paths(params, group))


@pytest.mark.parametrize("objective", ["value", "actor", "reliability"])
def test_separate_critic_gradient_routing(separate_critic_fixture, objective):
    model, params, hidden, obs, resets = separate_critic_fixture

    def loss(p):
        _, _, value, _, aux = model.apply(p, hidden, (obs, resets))
        if objective == "value":
            return 0.5 * jnp.mean(jnp.square(value - 2.0))
        if objective == "actor":
            return jnp.mean(jnp.square(aux["policy_loc"] - 0.3))
        return masked_reliability_loss(
            aux["reliability_scores"], jnp.ones((4, 2, 10, 2)),
            jnp.ones((4, 2, 10, 2)), loss_type="bce",
            reliability_logits=aux["reliability_logits"],
        )

    gradients = jax.jit(jax.grad(loss))(params)
    positive = {
        "value": ("critic_representation", "critic_head", "vision_encoder"),
        "actor": ("actor_head", "fusion_shared_trunk", "reliability_head", "vision_encoder"),
        "reliability": ("reliability_head", "vision_encoder"),
    }[objective]
    zero = (
        ("reliability_head", "fusion_shared_trunk", "actor_head")
        if objective == "value" else ("critic_representation", "critic_head")
    )
    for group in positive:
        assert float(gradient_l2_norm(gradients, group)) > 0.0, (objective, group)
    for group in zero:
        assert float(gradient_l2_norm(gradients, group)) == 0.0, (objective, group)


def test_separate_critic_reset_rollout_replay_and_raw_tokens(separate_critic_fixture):
    model, params, hidden, obs, resets = separate_critic_fixture
    with jax.default_matmul_precision("highest"):
        full_hidden, _, full_value, _, full_aux = model.apply(params, hidden, (obs, resets))
        carry, values, locs = hidden, [], []
        for t in range(4):
            step_obs = jax.tree_util.tree_map(lambda x: x[t:t + 1], obs)
            carry, _, value, _, aux = model.apply(params, carry, (step_obs, resets[t:t + 1]))
            values.append(value)
            locs.append(aux["policy_loc"])
        np.testing.assert_allclose(carry, full_hidden, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(jnp.concatenate(values), full_value, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(jnp.concatenate(locs), full_aux["policy_loc"], atol=1e-6, rtol=1e-6)
        suffix_obs = jax.tree_util.tree_map(lambda x: x[2:, :1], obs)
        suffix_reset = resets[2:, :1]
        reset_out = model.apply(params, jnp.ones((1, 32)) * 17, (suffix_obs, suffix_reset))
        fresh_out = model.apply(params, model.initialize_carry(1), (suffix_obs, suffix_reset))
        _assert_tree_allclose(reset_out[0], fresh_out[0], atol=0, rtol=0)
        np.testing.assert_allclose(reset_out[2], full_value[2:, :1], atol=1e-6, rtol=1e-6)

    perturbed = {"params": dict(params["params"])}
    perturbed["params"]["ReliabilityFusionRNN_0"] = jax.tree_util.tree_map(
        lambda x: x + 0.25, params["params"]["ReliabilityFusionRNN_0"]
    )
    with jax.default_matmul_precision("highest"):
        _, _, other_value, _, other_aux = model.apply(perturbed, hidden, (obs, resets))
    np.testing.assert_array_equal(other_value, full_value)
    assert not np.allclose(other_aux["reliability_scores"], full_aux["reliability_scores"])


def test_separate_critic_phasic_isolation(separate_critic_fixture):
    model, params, hidden, obs, resets = separate_critic_fixture
    settings = resolve_phasic_reliability_settings(
        {**model.config, "reliability_optimization_mode": "phasic",
         "use_survival_loss": True, "NUM_MINIBATCHES": 1,
         "LR": [0.0004], "MAX_GRAD_NORM": [0.5]},
        execution_index=0, execution_actor_count=2,
    )
    tx = make_auxiliary_optimizer(settings, total_updates=1)
    result = jax.jit(lambda p, state: run_phasic_auxiliary_phase(
        apply_fn=model.apply, params=p, aux_opt_state=state, aux_tx=tx,
        init_hstate=hidden, obs=obs, rnn_reset=resets,
        labels=jnp.ones((4, 2, 10, 2)), mask=jnp.ones((4, 2, 10, 2)),
        rng=jax.random.PRNGKey(92), settings=settings, is_discrete=False,
        reliability_loss_type="bce", survival_eps=1e-8,
    ))(params, tx.init(params))
    updated, _, _, diagnostics = result
    assert float(diagnostics["aux_steps_accepted"]) == 1
    for subtree in ("CriticValueRNN_0", "Dense_2", "Dense_3"):
        _assert_tree_allclose(updated["params"][subtree], params["params"][subtree], atol=0, rtol=0)
    delta = subtract_gradient_trees(updated, params)
    assert float(gradient_l2_norm(delta, "vision_encoder")) > 0


def test_separate_critic_probe_masks_and_fitting(separate_critic_fixture):
    model, params, hidden, obs, resets = separate_critic_fixture
    mask = flatten_tree_with_paths(parameter_groups_mask(params, VALUE_PROBE_B_SEPARATE_TRAINABLE_GROUPS))
    expected_roots = {"CriticValueRNN_0", "Dense_2", "Dense_3", "VisionAgent_0"}
    for path, enabled in mask.items():
        assert enabled == (path[1] in expected_roots)
    before = jax.tree_util.tree_map(lambda x: np.array(x, copy=True), params)

    def value_apply(p, h, o, r):
        return model.apply(p, h, (o, r))[2]

    params_a, params_b = jax.jit(lambda p: fit_value_representation_probe_variants(
        value_apply, p, hidden, obs, resets, jnp.ones((4, 2)) * 2,
        jnp.ones((4, 2), dtype=jnp.bool_), steps=3, learning_rate=0.001,
        use_separate_critic_representation=True,
    ))(params)
    _assert_tree_allclose(before, params, atol=0, rtol=0)
    for fitted, allowed in ((params_a, {"Dense_2", "Dense_3"}), (params_b, expected_roots)):
        for root in params["params"]:
            if root not in allowed:
                _assert_tree_allclose(fitted["params"][root], params["params"][root], atol=0, rtol=0)
        assert float(gradient_l2_norm(subtract_gradient_trees(fitted, params), "critic_head")) > 0
    for group in ("critic_representation", "vision_encoder"):
        assert float(gradient_l2_norm(subtract_gradient_trees(params_b, params), group)) > 0


def test_separate_critic_is_execution_only_and_supports_no_reliability(separate_critic_fixture):
    model, _, _, obs, resets = separate_critic_fixture
    other_agent = ActorCriticRNN(model.action_space, model.config, is_execution=False)
    hidden = other_agent.initialize_carry(2)
    assert hidden.shape == (2, 16)
    params = other_agent.init(jax.random.PRNGKey(37), hidden, (obs, resets))
    assert "CriticValueRNN_0" not in params["params"]
    numeric_obs = obs["exec_obs"]
    numeric_params = other_agent.init(jax.random.PRNGKey(37), hidden, (numeric_obs, resets))
    assert other_agent.apply(numeric_params, hidden, (numeric_obs, resets))[0].shape == (2, 16)
    no_rel = ActorCriticRNN(
        model.action_space, {**model.config, "use_reliability_head": False}, is_execution=True,
    )
    hidden = no_rel.initialize_carry(2)
    params = no_rel.init(jax.random.PRNGKey(37), hidden, (obs, resets))
    assert "ReliabilityFusionRNN_0" not in params["params"]
    assert "CriticValueRNN_0" in params["params"]
    assert no_rel.apply(params, hidden, (obs, resets))[0].shape == (2, 32)
    with pytest.raises(ValueError, match="hidden width"):
        no_rel.apply(params, hidden[:, :16], (obs, resets))
