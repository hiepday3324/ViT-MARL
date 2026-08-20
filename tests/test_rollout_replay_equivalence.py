import unittest

import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments import spaces

from gymnax_exchange.jaxrl.MARL.box_ppo import (
    box_action_from_pre_tanh,
    policy_log_prob_from_transition,
)
from gymnax_exchange.jaxrl.MARL.ippo_rnn_JAXMARL import (
    ActorCriticRNN,
    ScannedRNN,
)


class RolloutReplayEquivalenceTest(unittest.TestCase):
    def test_sequential_rollout_matches_full_sequence_replay(self):
        with jax.default_matmul_precision("highest"):
            self._assert_sequential_rollout_matches_full_sequence_replay()

    def _assert_sequential_rollout_matches_full_sequence_replay(self):
        time_steps = 8
        batch_size = 2
        hidden_dim = 16
        action_dim = 3
        action_low = jnp.asarray([-1.0, 0.0, 0.0], dtype=jnp.float32)
        action_high = jnp.asarray([3.0, 1.0, 1.0], dtype=jnp.float32)
        action_space = spaces.Box(
            low=action_low,
            high=action_high,
            shape=(action_dim,),
            dtype=jnp.float32,
        )
        config = {
            "FC_DIM_SIZE": hidden_dim,
            "GRU_HIDDEN_DIM": hidden_dim,
            "use_reliability_head": True,
            "use_h_prev_in_reliability": True,
            "reliability_hidden_dim": hidden_dim,
            "reliability_gate_epsilon": 0.1,
        }
        model = ActorCriticRNN(action_space, config=config)
        initial_hidden = ScannedRNN.initialize_carry(batch_size, hidden_dim)
        observations = {
            "exec_obs": jnp.linspace(
                -1.0,
                1.0,
                time_steps * batch_size * 28,
                dtype=jnp.float32,
            ).reshape(time_steps, batch_size, 28),
            "vision_obs": jnp.linspace(
                0.05,
                2.0,
                time_steps * batch_size * 10 * 3 * 2,
                dtype=jnp.float32,
            ).reshape(time_steps, batch_size, 10, 3, 2),
            "mid_context": jnp.linspace(
                -0.5,
                0.75,
                time_steps * batch_size * 4,
                dtype=jnp.float32,
            ).reshape(time_steps, batch_size, 4),
        }
        dones = jnp.zeros((time_steps, batch_size), dtype=jnp.bool_)
        dones = dones.at[3, 0].set(True)
        dones = dones.at[5, 1].set(True)

        params = model.init(
            jax.random.PRNGKey(17),
            initial_hidden,
            (observations, dones),
        )
        hidden_full, _pi_full, value_full, _z_full, aux_full = model.apply(
            params,
            initial_hidden,
            (observations, dones),
        )

        hidden_sequential = initial_hidden
        sequential_locs = []
        sequential_values = []
        sequential_scores = []
        sequential_log_stds = []
        for time_index in range(time_steps):
            observation_t = jax.tree_util.tree_map(
                lambda value: value[time_index : time_index + 1],
                observations,
            )
            done_t = dones[time_index : time_index + 1]
            (
                hidden_sequential,
                _pi_t,
                value_t,
                _z_t,
                aux_t,
            ) = model.apply(
                params,
                hidden_sequential,
                (observation_t, done_t),
            )
            sequential_locs.append(aux_t["policy_loc"])
            sequential_values.append(value_t)
            sequential_scores.append(aux_t["reliability_scores"])
            sequential_log_stds.append(aux_t["policy_log_std"])

        policy_loc_sequential = jnp.concatenate(sequential_locs, axis=0)
        value_sequential = jnp.concatenate(sequential_values, axis=0)
        reliability_sequential = jnp.concatenate(sequential_scores, axis=0)
        log_std_sequential = jnp.stack(sequential_log_stds, axis=0)

        pre_tanh_action = jnp.linspace(
            -1.25,
            1.25,
            time_steps * batch_size * action_dim,
            dtype=jnp.float32,
        ).reshape(time_steps, batch_size, action_dim)
        action = box_action_from_pre_tanh(
            pre_tanh_action,
            action_low,
            action_high,
        )
        log_prob_sequential = policy_log_prob_from_transition(
            None,
            {
                "policy_loc": policy_loc_sequential,
                "policy_log_std": aux_full["policy_log_std"],
            },
            action,
            pre_tanh_action,
            action_low=action_low,
            action_high=action_high,
        )
        log_prob_full = policy_log_prob_from_transition(
            None,
            aux_full,
            action,
            pre_tanh_action,
            action_low=action_low,
            action_high=action_high,
        )
        logratio = log_prob_full - log_prob_sequential

        errors = {
            "max_abs_policy_loc_error": jnp.max(
                jnp.abs(policy_loc_sequential - aux_full["policy_loc"])
            ),
            "max_abs_value_error": jnp.max(jnp.abs(value_sequential - value_full)),
            "max_abs_reliability_score_error": jnp.max(
                jnp.abs(reliability_sequential - aux_full["reliability_scores"])
            ),
            "max_abs_logprob_error": jnp.max(
                jnp.abs(log_prob_sequential - log_prob_full)
            ),
            "max_abs_preupdate_logratio": jnp.max(jnp.abs(logratio)),
        }
        print(
            "ROLLOUT_REPLAY_EQUIVALENCE",
            " ".join(f"{name}={float(value):.9g}" for name, value in errors.items()),
        )

        for name, value in errors.items():
            self.assertTrue(bool(jnp.isfinite(value)), msg=name)
            self.assertLessEqual(float(value), 1e-6, msg=name)
        np.testing.assert_allclose(
            log_std_sequential,
            jnp.broadcast_to(aux_full["policy_log_std"], log_std_sequential.shape),
            atol=0.0,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            hidden_sequential,
            hidden_full,
            atol=1e-6,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            jnp.exp(logratio),
            jnp.ones_like(logratio),
            atol=1e-6,
            rtol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
