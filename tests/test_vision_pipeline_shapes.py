import unittest
import inspect

import jax
import jax.numpy as jnp
import optax
import os
import sys
from flax.traverse_util import flatten_dict

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from gymnax_exchange.networks.gate_fusion import StableGatedCrossAttention
from gymnax_exchange.networks.reliability_head import (
    LevelWiseReliabilityHead,
    build_side_id_from_tokens,
    select_h_prev_for_reliability,
)
from gymnax_exchange.networks.vision_agent import VisionAgent, supervised_contrastive_loss
from gymnax_exchange.jaxen.StatesandParams import ExecEnvState, WorldState
from gymnax_exchange.jaxen.vision_env import ExecutionAgent
from gymnax_exchange.jaxob.jaxob_config import (
    Execution_EnvironmentConfig,
    World_EnvironmentConfig,
)
from gymnax_exchange.jaxrl.MARL.reliability_targets import masked_reliability_loss


class VisionPipelineShapeTest(unittest.TestCase):
    def test_vision_tokens_fusion_and_supcon_shapes(self):
        rng = jax.random.PRNGKey(0)
        time_steps = 4
        batch_size = 3
        embed_dim = 128

        exec_obs = jnp.ones((time_steps, batch_size, 28), dtype=jnp.float32)
        vision_obs = jnp.ones((time_steps, batch_size, 10, 3, 2), dtype=jnp.float32)

        vision = VisionAgent(embed_dim=embed_dim)
        vision_params = vision.init(rng, vision_obs, return_tokens=True)
        tokens = vision.apply(vision_params, vision_obs, return_tokens=True)
        pooled = vision.apply(vision_params, vision_obs)

        self.assertEqual(tokens.shape, (time_steps, batch_size, 10, 2, embed_dim))
        self.assertEqual(pooled.shape, (time_steps, batch_size, embed_dim))
        self.assertTrue(bool(jnp.allclose(pooled, jnp.mean(tokens, axis=(-3, -2)), atol=1e-5)))

        fusion = StableGatedCrossAttention(d_model=embed_dim)
        fusion_params = fusion.init(rng, exec_obs, tokens)
        fused = fusion.apply(fusion_params, exec_obs, tokens)
        self.assertEqual(fused.shape, (time_steps, batch_size, embed_dim // 2))

        legacy_tokens = jnp.mean(tokens, axis=-2)
        legacy_fused = fusion.apply(fusion_params, exec_obs, legacy_tokens)
        self.assertEqual(legacy_fused.shape, (time_steps, batch_size, embed_dim // 2))

        reliability = LevelWiseReliabilityHead(hidden_dim=64)
        self.assertNotIn("tick_shift", inspect.signature(reliability.__call__).parameters)
        self.assertNotIn("obs_exec", inspect.signature(reliability.__call__).parameters)
        h_prev = jnp.ones((batch_size, embed_dim), dtype=jnp.float32)
        side_id = build_side_id_from_tokens(tokens)
        mid_context = jnp.ones((time_steps, batch_size, 4), dtype=jnp.float32)
        reliability_params = reliability.init(
            rng,
            z_tokens=tokens,
            side_id=side_id,
            mid_context=mid_context,
            h_prev=h_prev,
        )
        flat_params = flatten_dict(reliability_params["params"])
        param_names = ["/".join(key) for key in flat_params]
        self.assertFalse(any("level_embed" in name for name in param_names))
        self.assertFalse(any("level_proj" in name for name in param_names))
        self.assertFalse(any("side_embed" in name for name in param_names))
        self.assertFalse(any("shift_proj" in name for name in param_names))
        self.assertTrue(any("side_proj" in name for name in param_names))
        self.assertTrue(any("mid_proj" in name for name in param_names))
        reliability_logits, reliability_scores, filtered_tokens = reliability.apply(
            reliability_params,
            z_tokens=tokens,
            side_id=side_id,
            mid_context=mid_context,
            h_prev=h_prev,
        )
        self.assertEqual(reliability_logits.shape, (time_steps, batch_size, 10, 2, 1))
        self.assertEqual(reliability_scores.shape, (time_steps, batch_size, 10, 2, 1))
        self.assertEqual(filtered_tokens.shape, tokens.shape)
        self.assertTrue(bool(jnp.all(reliability_scores >= 0.0)))
        self.assertTrue(bool(jnp.all(reliability_scores <= 1.0)))

        filtered_fused = fusion.apply(fusion_params, exec_obs, filtered_tokens)
        self.assertEqual(filtered_fused.shape, (time_steps, batch_size, embed_dim // 2))

        single_logits, single_scores, single_filtered = reliability.apply(
            reliability_params,
            z_tokens=tokens[0],
            side_id=side_id[0],
            mid_context=mid_context[0],
            h_prev=h_prev,
        )
        self.assertEqual(single_logits.shape, (batch_size, 10, 2, 1))
        self.assertEqual(single_scores.shape, (batch_size, 10, 2, 1))
        self.assertEqual(single_filtered.shape, tokens[0].shape)
        single_fused = fusion.apply(fusion_params, exec_obs[0], single_filtered)
        self.assertEqual(single_fused.shape, (batch_size, embed_dim // 2))

        labels = jnp.zeros((time_steps, batch_size), dtype=jnp.int32)
        loss = supervised_contrastive_loss(pooled.reshape(-1, embed_dim), labels.reshape(-1))
        self.assertEqual(loss.shape, ())
        self.assertTrue(bool(jnp.isfinite(loss)))

    def test_reliability_head_accepts_mid_context_side_id(self):
        rng = jax.random.PRNGKey(1)
        time_steps = 2
        batch_size = 3
        n_levels = 10
        n_sides = 2
        embed_dim = 64
        z_tokens = jnp.ones((time_steps, batch_size, n_levels, n_sides, embed_dim), dtype=jnp.float32)
        side_id = build_side_id_from_tokens(z_tokens)
        mid_context = jnp.ones((time_steps, batch_size, 4), dtype=jnp.float32)
        h_prev = jnp.ones((batch_size, embed_dim), dtype=jnp.float32)
        reliability = LevelWiseReliabilityHead(hidden_dim=32)

        params = reliability.init(
            rng,
            z_tokens=z_tokens,
            side_id=side_id,
            mid_context=mid_context,
            h_prev=h_prev,
        )
        logits, scores, filtered = reliability.apply(
            params,
            z_tokens=z_tokens,
            side_id=side_id,
            mid_context=mid_context,
            h_prev=h_prev,
        )

        self.assertEqual(logits.shape, (time_steps, batch_size, n_levels, n_sides, 1))
        self.assertEqual(scores.shape, (time_steps, batch_size, n_levels, n_sides, 1))
        self.assertEqual(filtered.shape, z_tokens.shape)

    def test_side_id_order(self):
        z_tokens = jnp.ones((2, 3, 10, 2, 64), dtype=jnp.float32)
        side_id = build_side_id_from_tokens(z_tokens)

        self.assertTrue(bool(jnp.all(side_id[..., 0, :] == 1.0)))
        self.assertTrue(bool(jnp.all(side_id[..., 1, :] == -1.0)))

    def test_obs_exec_not_used_in_reliability(self):
        params = inspect.signature(LevelWiseReliabilityHead().__call__).parameters

        self.assertNotIn("obs_exec", params)

    def test_h_prev_reliability_default_true(self):
        h_prev = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)

        selected = select_h_prev_for_reliability(h_prev)

        self.assertTrue(bool(jnp.allclose(selected, h_prev)))

    def test_h_prev_reliability_false_zeroes_h_prev(self):
        rng = jax.random.PRNGKey(2)
        time_steps = 2
        batch_size = 3
        n_levels = 10
        n_sides = 2
        embed_dim = 64
        z_tokens = jnp.ones((time_steps, batch_size, n_levels, n_sides, embed_dim), dtype=jnp.float32)
        side_id = build_side_id_from_tokens(z_tokens)
        mid_context = jnp.ones((time_steps, batch_size, 4), dtype=jnp.float32)
        h_prev = jnp.ones((batch_size, embed_dim), dtype=jnp.float32)

        selected = select_h_prev_for_reliability(
            h_prev,
            use_h_prev_in_reliability=False,
        )

        self.assertTrue(bool(jnp.allclose(selected, jnp.zeros_like(h_prev))))

        reliability = LevelWiseReliabilityHead(hidden_dim=32)
        params = reliability.init(
            rng,
            z_tokens=z_tokens,
            side_id=side_id,
            mid_context=mid_context,
            h_prev=selected,
        )
        logits, scores, filtered = reliability.apply(
            params,
            z_tokens=z_tokens,
            side_id=side_id,
            mid_context=mid_context,
            h_prev=selected,
        )

        self.assertEqual(logits.shape, (time_steps, batch_size, n_levels, n_sides, 1))
        self.assertEqual(scores.shape, (time_steps, batch_size, n_levels, n_sides, 1))
        self.assertEqual(filtered.shape, z_tokens.shape)

    def test_actor_reliability_enabled_and_disabled_paths(self):
        from gymnax.environments import spaces
        from gymnax_exchange.jaxrl.MARL.ippo_rnn_JAXMARL import (
            ActorCriticRNN,
            ScannedRNN,
        )

        time_steps = 2
        batch_size = 2
        hidden_dim = 16
        obs = {
            "exec_obs": jnp.ones((time_steps, batch_size, 28), dtype=jnp.float32),
            "vision_obs": jnp.ones((time_steps, batch_size, 10, 3, 2), dtype=jnp.float32),
            "mid_context": jnp.ones((time_steps, batch_size, 4), dtype=jnp.float32),
        }
        done = jnp.zeros((time_steps, batch_size), dtype=jnp.bool_)
        hidden = ScannedRNN.initialize_carry(batch_size, hidden_dim)
        action_space = spaces.Discrete(3)

        def init_and_apply(use_reliability_head):
            config = {
                "FC_DIM_SIZE": hidden_dim,
                "GRU_HIDDEN_DIM": hidden_dim,
                "use_reliability_head": use_reliability_head,
                "use_h_prev_in_reliability": True,
                "reliability_hidden_dim": hidden_dim,
                "reliability_gate_epsilon": 0.1,
            }
            model = ActorCriticRNN(action_space, config=config)
            variables = model.init(
                jax.random.PRNGKey(int(use_reliability_head) + 10),
                hidden,
                (obs, done),
            )
            outputs = model.apply(variables, hidden, (obs, done))
            return variables, outputs[-1]

        enabled_variables, enabled_aux = init_and_apply(True)
        enabled_paths = [
            "/".join(key)
            for key in flatten_dict(enabled_variables["params"]).keys()
        ]
        self.assertTrue(any("LevelWiseReliabilityHead" in path for path in enabled_paths))
        self.assertFalse(bool(jnp.all(enabled_aux["reliability_scores"] == 0.0)))
        self.assertTrue(bool(jnp.all(enabled_aux["reliability_path_active"] == 1.0)))

        disabled_variables, disabled_aux = init_and_apply(False)
        disabled_paths = [
            "/".join(key)
            for key in flatten_dict(disabled_variables["params"]).keys()
        ]
        self.assertFalse(any("LevelWiseReliabilityHead" in path for path in disabled_paths))
        self.assertTrue(bool(jnp.all(disabled_aux["reliability_scores"] == 0.0)))
        self.assertTrue(bool(jnp.all(disabled_aux["reliability_path_active"] == 0.0)))

    def test_fixed_batch_reliability_gradient_update_and_overfit(self):
        rng = jax.random.PRNGKey(20)
        time_steps = 2
        batch_size = 3
        n_levels = 4
        n_sides = 2
        embed_dim = 16
        rng, token_rng, context_rng = jax.random.split(rng, 3)
        z_tokens = jax.random.normal(
            token_rng,
            (time_steps, batch_size, n_levels, n_sides, embed_dim),
        )
        side_id = build_side_id_from_tokens(z_tokens)
        mid_context = jax.random.normal(
            context_rng,
            (time_steps, batch_size, 4),
        )
        h_prev = jnp.zeros((batch_size, embed_dim), dtype=jnp.float32)
        labels = (
            jnp.arange(time_steps * batch_size * n_levels * n_sides)
            .reshape(time_steps, batch_size, n_levels, n_sides)
            % 3
            == 0
        ).astype(jnp.float32)
        mask = jnp.ones_like(labels)
        reliability = LevelWiseReliabilityHead(hidden_dim=16)
        variables = reliability.init(
            rng,
            z_tokens=z_tokens,
            side_id=side_id,
            mid_context=mid_context,
            h_prev=h_prev,
        )

        def loss_fn(params):
            logits, scores, _filtered = reliability.apply(
                {"params": params},
                z_tokens=z_tokens,
                side_id=side_id,
                mid_context=mid_context,
                h_prev=h_prev,
            )
            return masked_reliability_loss(
                scores,
                labels,
                mask,
                loss_type="bce",
                reliability_logits=logits,
            )

        params = variables["params"]
        initial_loss, grads = jax.value_and_grad(loss_fn)(params)
        grad_norm = optax.global_norm(grads)
        self.assertGreater(float(grad_norm), 0.0)

        tx = optax.adam(1e-2)
        opt_state = tx.init(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        updated_params = optax.apply_updates(params, updates)
        update_norm = optax.global_norm(
            jax.tree_util.tree_map(lambda new, old: new - old, updated_params, params)
        )
        self.assertGreater(float(update_norm), 0.0)

        params = updated_params
        for _ in range(19):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = tx.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
        final_loss = loss_fn(params)
        self.assertLess(float(final_loss), float(initial_loss))


class ExecutionPolicyTimeRemainingTest(unittest.TestCase):
    @staticmethod
    def _orders(prices):
        orders = jnp.full((8, 6), -1, dtype=jnp.int32)
        count = len(prices)
        orders = orders.at[:count, 0].set(jnp.asarray(prices, dtype=jnp.int32))
        orders = orders.at[:count, 1].set(10)
        orders = orders.at[:count, 2].set(
            -2 - jnp.arange(count, dtype=jnp.int32)
        )
        orders = orders.at[:count, 3].set(-100)
        orders = orders.at[:count, 4:].set(0)
        return orders

    @classmethod
    def _fixture(cls, ep_type="fixed_steps", episode_time=50):
        world_config = World_EnvironmentConfig(
            ep_type=ep_type,
            episode_time=episode_time,
            nOrders=8,
        )
        agent_config = Execution_EnvironmentConfig(
            action_space="policy_blending",
            observation_space="execution_policy",
            task="buy",
            task_size=500,
        )
        agent = ExecutionAgent(agent_config, world_config)
        agent_state = ExecEnvState(
            init_price=9_950,
            task_to_execute=500,
            quant_executed=0,
            total_revenue=0.0,
            drift_return=0.0,
            advantage_return=0.0,
            slippage_rm=0.0,
            price_adv_rm=0.0,
            price_drift_rm=0.0,
            vwap_rm=0.0,
            is_sell_task=False,
            trade_duration=0.0,
            rl_vol_window=jnp.zeros((1,), dtype=jnp.float32),
            rl_cost_window=jnp.zeros((1,), dtype=jnp.float32),
            base_vol_window=jnp.zeros((1,), dtype=jnp.float32),
            base_cost_window=jnp.zeros((1,), dtype=jnp.float32),
            reward_window_ptr=jnp.asarray(0, dtype=jnp.int32),
            reward_window_count=jnp.asarray(0, dtype=jnp.int32),
            shadow_cumulative_filled_quantity=0.0,
            shadow_remaining_task_quantity=500.0,
            shadow_cumulative_execution_cost=0.0,
        )
        return agent, agent_state

    @classmethod
    def _world_state(cls, step_counter, elapsed_seconds, max_steps=51):
        return WorldState(
            ask_raw_orders=cls._orders([10_000, 10_100, 10_200]),
            bid_raw_orders=cls._orders([9_900, 9_800, 9_700]),
            trades=jnp.full((4, 8), -1, dtype=jnp.int32),
            init_time=jnp.asarray([1_000, 0], dtype=jnp.int32),
            window_index=jnp.asarray(0, dtype=jnp.int32),
            max_steps_in_episode=jnp.asarray(max_steps, dtype=jnp.int32),
            start_index=jnp.asarray(0, dtype=jnp.int32),
            step_counter=jnp.asarray(step_counter, dtype=jnp.int32),
            best_bids=jnp.asarray([[9_900, 10]], dtype=jnp.int32),
            best_asks=jnp.asarray([[10_000, 10]], dtype=jnp.int32),
            time=jnp.asarray([1_000 + elapsed_seconds, 0], dtype=jnp.int32),
            order_id_counter=jnp.asarray(-200, dtype=jnp.int32),
            mid_price=jnp.asarray(9_950.0, dtype=jnp.float32),
            delta_time=jnp.asarray(7.0, dtype=jnp.float32),
        )

    @staticmethod
    def _horizon_features(observation):
        return {
            "step_counter": observation[5],
            "remaining_ratio": observation[6],
            "time": observation[7],
            "delta_time": observation[8],
            "time_remaining": observation[9],
        }

    def test_fixed_steps_boundaries_and_nonnegative_time_remaining(self):
        agent, agent_state = self._fixture()
        expected = {
            0: (50.0, 1.0),
            25: (25.0, 0.5),
            49: (1.0, 0.02),
            50: (0.0, 0.0),
            75: (0.0, 0.0),
        }

        for step_counter, (remaining, ratio) in expected.items():
            with self.subTest(step_counter=step_counter):
                observation = agent._get_obs_execution_policy(
                    agent_state,
                    self._world_state(step_counter, elapsed_seconds=300),
                    normalize=False,
                )
                features = self._horizon_features(observation)
                self.assertAlmostEqual(float(features["time_remaining"]), remaining)
                self.assertAlmostEqual(float(features["remaining_ratio"]), ratio)
                self.assertGreaterEqual(float(features["time_remaining"]), 0.0)

    def test_fixed_steps_horizon_is_independent_of_wall_clock(self):
        agent, agent_state = self._fixture()
        observations = [
            agent._get_obs_execution_policy(
                agent_state,
                self._world_state(25, elapsed_seconds=elapsed),
                normalize=False,
            )
            for elapsed in (20, 300)
        ]

        for observation in observations:
            features = self._horizon_features(observation)
            self.assertAlmostEqual(float(features["time_remaining"]), 25.0)
            self.assertAlmostEqual(float(features["remaining_ratio"]), 0.5)
        self.assertNotEqual(float(observations[0][7]), float(observations[1][7]))
        self.assertEqual(float(observations[0][8]), float(observations[1][8]))

    def test_fixed_steps_normalization_uses_decision_horizon(self):
        agent, agent_state = self._fixture()
        observation = agent._get_obs_execution_policy(
            agent_state,
            self._world_state(25, elapsed_seconds=300),
            normalize=True,
        )
        features = self._horizon_features(observation)

        self.assertAlmostEqual(float(features["step_counter"]), 0.5)
        self.assertAlmostEqual(float(features["remaining_ratio"]), 0.5)
        self.assertAlmostEqual(float(features["time_remaining"]), 0.5)

    def test_fixed_time_uses_elapsed_seconds_and_clips_at_zero(self):
        agent, agent_state = self._fixture(ep_type="fixed_time", episode_time=50)
        active = agent._get_obs_execution_policy(
            agent_state,
            self._world_state(12, elapsed_seconds=20),
            normalize=False,
        )
        expired = agent._get_obs_execution_policy(
            agent_state,
            self._world_state(12, elapsed_seconds=300),
            normalize=False,
        )

        self.assertAlmostEqual(float(active[9]), 30.0)
        self.assertAlmostEqual(float(active[6]), 0.6)
        self.assertAlmostEqual(float(expired[9]), 0.0)
        self.assertAlmostEqual(float(expired[6]), 0.0)

        normalized = agent._get_obs_execution_policy(
            agent_state,
            self._world_state(12, elapsed_seconds=20),
            normalize=True,
        )
        self.assertAlmostEqual(float(normalized[5]), 12.0 / 30.0)
        self.assertAlmostEqual(float(normalized[6]), 0.6)
        self.assertAlmostEqual(float(normalized[9]), 0.6)

    def test_execution_policy_horizon_is_jittable_and_vmappable(self):
        agent, agent_state = self._fixture()
        worlds = [
            self._world_state(step, elapsed_seconds=elapsed)
            for step, elapsed in ((0, 10), (25, 300), (49, 20), (50, 500))
        ]
        batched_worlds = jax.tree_util.tree_map(
            lambda *values: jnp.stack(values),
            *worlds,
        )
        apply_observation = jax.jit(
            jax.vmap(
                lambda world: agent._get_obs_execution_policy(
                    agent_state,
                    world,
                    normalize=False,
                )
            )
        )

        observations = apply_observation(batched_worlds)
        self.assertEqual(observations.shape, (4, 22))
        self.assertTrue(
            bool(
                jnp.allclose(
                    observations[:, 9],
                    jnp.asarray([50.0, 25.0, 1.0, 0.0]),
                )
            )
        )
        self.assertTrue(
            bool(
                jnp.allclose(
                    observations[:, 6],
                    jnp.asarray([1.0, 0.5, 0.02, 0.0]),
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
