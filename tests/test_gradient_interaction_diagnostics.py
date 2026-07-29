from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from gymnax.environments import spaces

from gymnax_exchange.jaxrl.MARL.gradient_diagnostics import (
    GRADIENT_GROUPS,
    add_gradient_trees,
    empty_gradient_interaction_diagnostics,
    format_gradient_interaction_diagnostics,
    gradient_cosine,
    gradient_diag_should_run,
    gradient_dot,
    gradient_l2_norm,
    scale_gradient_tree,
    subtract_gradient_trees,
    summarize_gradient_interaction,
    summarize_phasic_gradient_interaction,
    validate_gradient_diag_config,
    validate_required_parameter_groups,
)
from gymnax_exchange.jaxrl.MARL.ippo_rnn_JAXMARL import (
    ActorCriticRNN,
    ScannedRNN,
)
from gymnax_exchange.jaxrl.MARL.reliability_targets import (
    masked_reliability_loss,
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
