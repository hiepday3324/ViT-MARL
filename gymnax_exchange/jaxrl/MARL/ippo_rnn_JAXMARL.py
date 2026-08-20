"""
Based on PureJaxRL Implementation of PPO
"""

import os
import sys
import copy

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
import pandas as pd
import csv
import wandb.sdk


os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
# os.environ["JAX_CHECK_TRACER_LEAKS"] = "true"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


import time
import jax # type: ignorepip 
jax.config.update('jax_disable_jit', False)
from flax import serialization
import jax, os
import jax.numpy as jnp # type: ignore
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal # type: ignore
from typing import Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import distrax
import orbax.checkpoint as oxcp
import hydra
from omegaconf import DictConfig, OmegaConf
import gc

#from jaxmarl.wrappers.baselines import SMAXLogWrapper
#from jaxmarl.environments.smax import map_name_to_scenario, HeuristicEnemySMAX
from gymnax_exchange.jaxen.marl_env import MARLEnv
from gymnax.environments import spaces
from gymnax_exchange.jaxob.jaxob_config import MultiAgentConfig,Execution_EnvironmentConfig, World_EnvironmentConfig,MarketMaking_EnvironmentConfig
from gymnax_exchange.networks.gate_fusion import StableGatedCrossAttention
from gymnax_exchange.networks.reliability_head import (
    LevelWiseReliabilityHead,
    build_side_id_from_tokens,
    select_h_prev_for_reliability,
)
from gymnax_exchange.networks.vision_agent import VisionAgent
from gymnax_exchange.jaxrl.MARL.reliability_targets import (
    build_liquidity_survival_targets,
    empty_liquidity_survival_diagnostics,
    masked_reliability_loss,
    resolve_rollout_is_sell_task,
)
from gymnax_exchange.jaxrl.MARL.execution_episode_metrics import (
    accumulate_execution_episode_metrics,
    empty_execution_episode_metrics,
)
from gymnax_exchange.jaxrl.MARL.gradient_diagnostics import (
    GRADIENT_GROUPS,
    PARAMETER_GROUP_RULES,
    empty_gradient_interaction_diagnostics,
    format_gradient_interaction_diagnostics,
    gradient_diag_should_run,
    subtract_gradient_trees,
    summarize_gradient_interaction,
    summarize_phasic_gradient_interaction,
    validate_gradient_diag_config,
    validate_required_parameter_groups,
)
from gymnax_exchange.jaxrl.MARL.box_ppo import (
    FIRST_NONFINITE_STAGE_NAME,
    build_box_ppo_numerics_diagnostics,
    empty_box_ppo_numerics_diagnostics,
    empty_ppo_safety_state,
    guarded_ppo_apply_gradients,
    policy_log_prob_from_transition,
    sample_policy_action,
    select_guarded_train_state,
    update_ppo_safety_state,
)
from gymnax_exchange.jaxrl.MARL.phasic_reliability import (
    build_rollout_outputs,
    empty_phasic_aux_diagnostics,
    format_phasic_aux_diagnostics,
    make_auxiliary_optimizer,
    ppo_survival_loss_weight,
    resolve_phasic_reliability_settings,
    run_phasic_auxiliary_phase,
)
import wandb
import functools
import matplotlib.pyplot as plt



class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, jnp.newaxis],
            self.initialize_carry(*rnn_state.shape),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ReliabilityFusionRNN(nn.Module):
    config: Dict

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        obs_exec_t, z_tokens_t, mid_context_t, done_t = x
        rnn_state = jnp.where(done_t[:, jnp.newaxis], jnp.zeros_like(carry), carry)
        side_id_t = build_side_id_from_tokens(z_tokens_t)
        use_h_prev_in_reliability = self.config.get("use_h_prev_in_reliability", True)
        h_prev_for_reliability = select_h_prev_for_reliability(
            rnn_state,
            use_h_prev_in_reliability=use_h_prev_in_reliability,
        )

        reliability = LevelWiseReliabilityHead(
            hidden_dim=self.config.get("reliability_hidden_dim", self.config["FC_DIM_SIZE"]),
            gate_epsilon=self.config.get("reliability_gate_epsilon", 0.1),
        )
        reliability_logits_t, reliability_scores_t, filtered_tokens_t = reliability(
            z_tokens=z_tokens_t,
            side_id=side_id_t,
            mid_context=mid_context_t,
            h_prev=h_prev_for_reliability,
        )

        fusion = StableGatedCrossAttention(d_model=self.config["FC_DIM_SIZE"])
        fused_t = fusion(obs_exec_t, filtered_tokens_t)
        embedding_t = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(fused_t)
        embedding_t = nn.relu(embedding_t)

        new_rnn_state, y_t = nn.GRUCell(features=self.config["FC_DIM_SIZE"])(rnn_state, embedding_t)
        rvd_diag_t = {
            "z_tokens_norm": jnp.linalg.norm(z_tokens_t, axis=-1),
            "filtered_tokens_norm": jnp.linalg.norm(filtered_tokens_t, axis=-1),
            "fusion_output_norm": jnp.linalg.norm(fused_t, axis=-1),
            "pre_rnn_embedding_norm": jnp.linalg.norm(embedding_t, axis=-1),
            "mid_return_from_init": mid_context_t[..., 0],
            "spread_ticks": mid_context_t[..., 1],
            "mid_delta_ticks": mid_context_t[..., 2],
            "mid_volatility_ticks": mid_context_t[..., 3],
            "h_prev_reliability_norm": jnp.linalg.norm(h_prev_for_reliability, axis=-1),
            "use_h_prev_in_reliability": jnp.full(
                done_t.shape,
                float(bool(use_h_prev_in_reliability)),
                dtype=jnp.float32,
            ),
            "h_prev_used_in_reliability": jnp.full(
                done_t.shape,
                float(bool(use_h_prev_in_reliability)),
                dtype=jnp.float32,
            ),
            "h_prev_reliability_zeroed": jnp.full(
                done_t.shape,
                float(not bool(use_h_prev_in_reliability)),
                dtype=jnp.float32,
            ),
        }
        return new_rnn_state, (
            y_t,
            reliability_logits_t,
            reliability_scores_t,
            rvd_diag_t,
        )

# FIXME: APPLY VISION 
class ActorCriticRNN(nn.Module):
    action_space: spaces.Space
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        
        if isinstance(obs, dict):
            obs_exec = obs['exec_obs']
            obs_vision = obs['vision_obs']
            mid_context = obs['mid_context']

            vision_encoder = VisionAgent(embed_dim=self.config["FC_DIM_SIZE"])
            z_tokens = vision_encoder(obs_vision, return_tokens=True)
            z_vision = jnp.mean(z_tokens, axis=(-3, -2))

            use_reliability_head = self.config.get("use_reliability_head", False)
            if use_reliability_head:
                hidden, (
                    embedding,
                    reliability_logits,
                    reliability_scores,
                    rvd_diag,
                ) = ReliabilityFusionRNN(config=self.config)(
                    hidden,
                    (obs_exec, z_tokens, mid_context, dones),
                )
                aux_info = {
                    "reliability_logits": reliability_logits,
                    "reliability_scores": reliability_scores,
                    "reliability_path_active": jnp.ones(
                        reliability_scores.shape[:2],
                        dtype=jnp.float32,
                    ),
                    **rvd_diag,
                }
            else:
                fusion = StableGatedCrossAttention(d_model=self.config["FC_DIM_SIZE"])
                fused_obs = fusion(obs_exec, z_tokens)
                embedding = nn.Dense(
                    self.config["FC_DIM_SIZE"], kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
                )(fused_obs)
                embedding = nn.relu(embedding)

                rnn_in = (embedding, dones)

                hidden, embedding = ScannedRNN()(hidden, rnn_in)
                aux_info = {
                    "reliability_logits": jnp.zeros((*z_tokens.shape[:-1], 1), dtype=z_tokens.dtype),
                    "reliability_scores": jnp.zeros((*z_tokens.shape[:-1], 1), dtype=z_tokens.dtype),
                    "reliability_path_active": jnp.zeros(z_tokens.shape[:2], dtype=jnp.float32),
                    "z_tokens_norm": jnp.linalg.norm(z_tokens, axis=-1),
                    "filtered_tokens_norm": jnp.linalg.norm(z_tokens, axis=-1),
                    "fusion_output_norm": jnp.linalg.norm(fused_obs, axis=-1),
                    "pre_rnn_embedding_norm": jnp.linalg.norm(embedding, axis=-1),
                }

            aux_info.update({
                "exec_obs_norm": jnp.linalg.norm(obs_exec, axis=-1),
                "vision_token_pooled_norm": jnp.linalg.norm(z_vision, axis=-1),
                "actor_input_norm": jnp.linalg.norm(embedding, axis=-1),
                "rel_z_shape": jnp.asarray(z_tokens.shape, dtype=jnp.float32),
                "rel_side_id_shape": jnp.asarray(z_tokens.shape[:-1] + (1,), dtype=jnp.float32),
                "rel_mid_context_shape": jnp.asarray(mid_context.shape, dtype=jnp.float32),
                "mid_return_from_init": mid_context[..., 0],
                "spread_ticks": mid_context[..., 1],
                "mid_delta_ticks": mid_context[..., 2],
                "mid_volatility_ticks": mid_context[..., 3],
            })
        else:
            fused_obs = obs
            z_vision = jnp.zeros((1,))
            embedding = nn.Dense(
                self.config["FC_DIM_SIZE"], kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
            )(fused_obs)
            embedding = nn.relu(embedding)

            rnn_in = (embedding, dones)

            hidden, embedding = ScannedRNN()(hidden, rnn_in)
            zero_diag = jnp.zeros(embedding.shape[:-1], dtype=embedding.dtype)
            aux_info = {
                "reliability_logits": jnp.zeros((*embedding.shape[:-1], 1, 1, 1), dtype=embedding.dtype),
                "reliability_scores": jnp.zeros((*embedding.shape[:-1], 1, 1, 1), dtype=embedding.dtype),
                "reliability_path_active": jnp.zeros(embedding.shape[:-1], dtype=jnp.float32),
                "z_tokens_norm": zero_diag,
                "filtered_tokens_norm": zero_diag,
                "fusion_output_norm": zero_diag,
                "pre_rnn_embedding_norm": zero_diag,
                "exec_obs_norm": jnp.linalg.norm(obs, axis=-1),
                "vision_token_pooled_norm": zero_diag,
                "actor_input_norm": jnp.linalg.norm(embedding, axis=-1),
            }
        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )

        actor_mean = nn.relu(actor_mean)

        if isinstance(self.action_space, spaces.Discrete):
            action_logits = nn.Dense(
                self.action_space.n, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
            )(actor_mean)
            pi = distrax.Categorical(logits=action_logits)
            aux_info["policy_logits"] = action_logits
        elif isinstance(self.action_space, spaces.Box):
            action_loc = nn.Dense(
                self.action_space.shape[-1], kernel_init=orthogonal(0.01), bias_init=constant(0.0)
            )(actor_mean)
            actor_logstd = self.param("log_std", nn.initializers.zeros, (self.action_space.shape[-1],))
            base_dist = distrax.Independent(
                distrax.Normal(action_loc, jnp.exp(actor_logstd)),
                reinterpreted_batch_ndims=1,
            )
            action_low = jnp.asarray(self.action_space.low, dtype=jnp.float32)
            action_high = jnp.asarray(self.action_space.high, dtype=jnp.float32)
            action_shift = (action_high + action_low) / 2.0
            action_scale = (action_high - action_low) / 2.0
            action_bijector = distrax.Block(
                distrax.Chain([
                    distrax.ScalarAffine(shift=action_shift, scale=action_scale),
                    distrax.Tanh(),
                ]),
                ndims=1,
            )
            pi = distrax.Transformed(base_dist, action_bijector)
            aux_info["policy_loc"] = action_loc
            aux_info["policy_log_std"] = actor_logstd
        else:
            raise ValueError(f"Unknown action space type {type(self.action_space)}")

        critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1), z_vision, aux_info

# FIXME: APPLY VISION
class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    pre_tanh_action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: Dict[str, jnp.ndarray]  
    info: Dict[str, Any]
    # avail_actions: jnp.ndarray


def batchify(x, num_actors):
    return jax.tree_util.tree_map(lambda y: y.reshape((num_actors, *y.shape[2:])), x)


def batchify_action(x, num_actors):
    def _batchify_action(y):
        if y.shape[0] == num_actors:
            return y
        return y.reshape((num_actors, *y.shape[2:]))

    return jax.tree_util.tree_map(_batchify_action, x)


def unbatchify(x, num_envs, num_agents):
    def _unbatchify(y):
        if y.ndim >= 2 and y.shape[0] == 1:
            y = jnp.squeeze(y, axis=0)
        return y.reshape((num_envs, num_agents, *y.shape[1:]))

    return jax.tree_util.tree_map(_unbatchify, x)


def _is_execution_agent(agent_config):
    return isinstance(agent_config, Execution_EnvironmentConfig)


def make_train(config):
    # scenario = map_name_to_scenario(config["MAP_NAME"])
    grad_diag_cadence = validate_gradient_diag_config(config)
    grad_diag_enabled = bool(config.get("enable_grad_interaction_diag", False))
    box_ppo_diag_enabled = bool(config.get("enable_box_ppo_numerics_diag", False))
    init_key = jax.random.PRNGKey(config["SEED"])
    config_dict={"MarketMaking": MarketMaking_EnvironmentConfig,"Execution": Execution_EnvironmentConfig}
    print("init_key: ", init_key)
    # Create a MultiAgentConfig object with parameters from the config
    agent_configs = {}
    if "AGENT_CONFIGS" in config:
        agent_configs = {
            agent_type: config_dict[agent_type](**{k.lower(): v for k, v in agent_cfg.items()})
            for agent_type, agent_cfg in config["AGENT_CONFIGS"].items()
        }
    elif "dict_of_agents_configs" in config:
        agent_configs = {
            agent_type: config_dict[agent_type](**agent_cfg)
            for agent_type, agent_cfg in config["dict_of_agents_configs"].items()
            if agent_type in config_dict
        }
    print("agent_configs:", agent_configs)
    


    ma_config = MultiAgentConfig(
        number_of_agents_per_type=config["NUM_AGENTS_PER_TYPE"],
        dict_of_agents_configs=agent_configs,
        world_config=World_EnvironmentConfig(
            seed=config["SEED"],
            # Only override parameters that exist in both config and World_EnvironmentConfig
            **{k.lower(): v for k, v in config.items() 
               if hasattr(World_EnvironmentConfig(), k.lower()) and k != "SEED"}
        )
    )
    print(ma_config)

    print("MultiAgentInventoryPenalty",ma_config.dict_of_agents_configs["MarketMaking"].inv_penalty)

    # For evaluation, create a separate config with evaluation-specific parameters
    eval_ma_config = None
    if config["CALC_EVAL"]:
        # Reuse agent_configs from above if it exists
        eval_agent_configs = {}
        if "AGENT_CONFIGS" in config:
            eval_agent_configs = {
                agent_type: config_dict[agent_type](**{k.lower(): v for k, v in agent_cfg.items()})
                for agent_type, agent_cfg in config["AGENT_CONFIGS"].items()
            }
            
        eval_ma_config = MultiAgentConfig(
            number_of_agents_per_type=config["NUM_AGENTS_PER_TYPE"],
            dict_of_agents_configs=eval_agent_configs,
            world_config=World_EnvironmentConfig(
                seed=config["SEED"],
                timePeriod=config["EvalTimePeriod"],
                # Only override parameters that exist in both config and World_EnvironmentConfig
                **{k.lower(): v for k, v in config.items() 
                   if hasattr(World_EnvironmentConfig(), k.lower()) and k not in ["SEED", "EvalTimePeriod"]}
            )
        )
   

    env : MARLEnv = MARLEnv(key=init_key, multi_agent_config=ma_config)
    if config["CALC_EVAL"]:
        eval_env: MARLEnv = MARLEnv(key=init_key,multi_agent_config=eval_ma_config)

    agent_type_names = list(env.type_names)

    config["NUM_ACTORS_PERTYPE"] = [n * config["NUM_ENVS"] for n in config["NUM_AGENTS_PER_TYPE"]]  # Should be a list.
    config["NUM_ACTORS_TOTAL"] = env.num_agents * config["NUM_ENVS"]

    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZES"] = [
        nact * config["NUM_STEPS"] // config["NUM_MINIBATCHES"] for i,nact in enumerate(config["NUM_ACTORS_PERTYPE"])
    ]
    execution_index = next(
        (
            idx
            for idx, agent_config in enumerate(env.list_of_agents_configs)
            if _is_execution_agent(agent_config)
        ),
        None,
    )
    execution_actor_count = (
        None
        if execution_index is None
        else config["NUM_ACTORS_PERTYPE"][execution_index]
    )
    phasic_settings = resolve_phasic_reliability_settings(
        config,
        execution_index=execution_index,
        execution_actor_count=execution_actor_count,
    )
    phasic_mode = phasic_settings.enabled
    print(
        "RELIABILITY_OPTIMIZATION_CONFIG",
        f"mode={phasic_settings.mode}",
        f"aux_epochs={phasic_settings.num_epochs}",
        f"aux_minibatches={phasic_settings.num_minibatches}",
        f"aux_learning_rate={phasic_settings.learning_rate}",
        f"aux_max_grad_norm={phasic_settings.max_grad_norm}",
    )
    # config["CLIP_EPS"] = (
    #     config["CLIP_EPS"] / env.num_agents
    #     if config["SCALE_CLIP_EPS"]
    #     else config["CLIP_EPS"]
    # )

    # env = SMAXLogWrapper(env)

    def linear_schedule(lr,count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return lr * frac

    def train(rng, run: wandb.sdk.wandb_run.Run = None):
        # INIT NETWORK


        # For a given agent type (instance) we need the following inputs:
        # Action space, obs space, 

        # The outputs that depends on these and are kept seperate are;
        # - network, init_x, init_hstate, network_params, train_state
        hstates = []
        network_params_list = []
        train_states = []
        aux_tx = (
            make_auxiliary_optimizer(
                phasic_settings,
                total_updates=config["NUM_UPDATES"],
            )
            if phasic_mode
            else None
        )
        aux_opt_state = None
        num_agents_of_instance_list = []
        init_dones_agents = []
        for i, instance in enumerate(env.instance_list):
            # print("Action space dimension for network i ",env.action_spaces[i])
            network = ActorCriticRNN(env.action_spaces[i], config=config)
            rng, _rng = jax.random.split(rng)
            if hasattr(env.observation_spaces[i], "spaces"):
                obs_shape = env.observation_spaces[i].spaces['exec_obs'].shape[0]
                init_obs = {
                    'exec_obs': jnp.zeros((1, config["NUM_ENVS"], obs_shape)), 
                    'vision_obs': jnp.zeros((1, config["NUM_ENVS"], 10, 3, 2)), # Shape của LOB
                    'mid_context': jnp.zeros((1, config["NUM_ENVS"], 4)),
                }
            elif isinstance(env.observation_spaces[i], dict):
                obs_shape = env.observation_spaces[i]['exec_obs'].shape[0]
                init_obs = {
                    'exec_obs': jnp.zeros((1, config["NUM_ENVS"], obs_shape)), 
                    'vision_obs': jnp.zeros((1, config["NUM_ENVS"], 10, 3, 2)), # Shape của LOB
                    'mid_context': jnp.zeros((1, config["NUM_ENVS"], 4)),
                }
            else:
                init_obs = jnp.zeros((1, config["NUM_ENVS"], env.observation_spaces[i].shape[0]))

            init_x = (
                init_obs,
                jnp.zeros((1, config["NUM_ENVS"])) # dones
                # jnp.zeros((1, config["NUM_ENVS"], env.action_spaces[i].n)), #     avail_actions
            )

            # FIXME: very unsure about this, why is it NUM_ENVS and not NUM_ACTORS?
            init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
            network_params = network.init(_rng, init_hstate, init_x)
            if (
                (grad_diag_enabled or phasic_mode)
                and config.get("use_reliability_head", False)
                and config.get("use_survival_loss", False)
                and _is_execution_agent(env.list_of_agents_configs[i])
            ):
                group_counts = validate_required_parameter_groups(
                    network_params,
                    required_groups=GRADIENT_GROUPS,
                )
                for group in GRADIENT_GROUPS:
                    print(
                        "GRAD_DIAG_GROUP_RULE",
                        f"agent=EXE group={group}",
                        f"param_leaf_count={group_counts[group]}",
                        f"rule={PARAMETER_GROUP_RULES[group]}",
                    )
            if phasic_mode and i == execution_index:
                aux_opt_state = aux_tx.init(network_params)
            if config["ANNEAL_LR"][i]:
                tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"][i]),
                    optax.adam(learning_rate=functools.partial(linear_schedule,config["LR"][i]), eps=1e-5),
                )
            else:
                tx = optax.chain(
                    optax.clip_by_global_norm(config["MAX_GRAD_NORM"][i]),
                    optax.adam(config["LR"][i], eps=1e-5),
                )
            train_state = TrainState.create(
                apply_fn=network.apply,
                params=network_params,
                tx=tx,
            )
            init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS_PERTYPE"][i], config["GRU_HIDDEN_DIM"])

            # Instead of appending dicts, maintain separate lists for each attribute
            hstates.append(init_hstate)
            network_params_list.append(network_params)
            train_states.append(train_state)
            num_agents_of_instance_list.append(env.multi_agent_config.number_of_agents_per_type[i])
            init_dones_agents.append(jnp.zeros((config["NUM_ACTORS_PERTYPE"][i]), dtype=bool))

        if phasic_mode and aux_opt_state is None:
            raise ValueError("Failed to initialize the Execution auxiliary optimizer state.")

        def _unpack_runner_state(runner_state):
            if phasic_mode:
                (
                    runner_train_states,
                    runner_env_state,
                    runner_last_obs,
                    runner_last_done,
                    runner_hstates,
                    runner_rng,
                    runner_aux_opt_state,
                    runner_exe_episode_return,
                ) = runner_state
            else:
                (
                    runner_train_states,
                    runner_env_state,
                    runner_last_obs,
                    runner_last_done,
                    runner_hstates,
                    runner_rng,
                    runner_exe_episode_return,
                ) = runner_state
                runner_aux_opt_state = None
            return (
                runner_train_states,
                runner_env_state,
                runner_last_obs,
                runner_last_done,
                runner_hstates,
                runner_rng,
                runner_aux_opt_state,
                runner_exe_episode_return,
            )

        def _pack_runner_state(
            runner_train_states,
            runner_env_state,
            runner_last_obs,
            runner_last_done,
            runner_hstates,
            runner_rng,
            runner_aux_opt_state,
            runner_exe_episode_return,
        ):
            base_state = (
                runner_train_states,
                runner_env_state,
                runner_last_obs,
                runner_last_done,
                runner_hstates,
                runner_rng,
            )
            if phasic_mode:
                return base_state + (
                    runner_aux_opt_state,
                    runner_exe_episode_return,
                )
            return base_state + (runner_exe_episode_return,)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        env_params=env.default_params
        if config["CALC_EVAL"]:
            eval_env_params=eval_env.default_params # type: ignore
        else:
            eval_env_params = None
        # env_params=jax.device_put(env_params)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,None))(reset_rng,env_params)
        # TRAIN LOOP
        

        def _update_step(update_runner_state,env_params,eval_env_params, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state
            # FIXME: APPLY VISION
            def _env_step(runner_state, unused):
                (
                    train_states,
                    env_state,
                    last_obs,
                    last_done,
                    h_states,
                    rng,
                    current_aux_opt_state,
                    running_exe_episode_return,
                ) = _unpack_runner_state(runner_state)

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                
                # Ignore getting the available actions for now, assume all actions are available.
                # avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                # avail_actions = jax.lax.stop_gradient(
                #     batchify(avail_actions, env.agents, config["NUM_ACTORS"])
                # )
                # obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                actions=[]
                pre_tanh_actions=[]
                values=[]
                log_probs=[]
                '''
                Duyệt qua các agent trong môi trường, lấy ra hành động và trạng thái tại bước thời gian đó
                '''
                for i, train_state in enumerate(train_states):
                    obs_i= last_obs[i]
                    obs_i=batchify(obs_i,config["NUM_ACTORS_PERTYPE"][i])  # Reshape to match the input shape of the network

                    obs_i_batched = jax.tree.map(lambda x: x[jnp.newaxis, :], obs_i)
                    ac_in = (
                        obs_i_batched,
                        last_done[i][jnp.newaxis, :],
                        # avail_actions,
                    )
                    h_states[i], pi, value, _, policy_aux_info = train_state.apply_fn(
                        train_state.params,
                        h_states[i],
                        ac_in,
                    )
                    values.append(value)
                    action_space = env.action_spaces[i]
                    sample_kwargs = {}
                    if isinstance(action_space, spaces.Box):
                        sample_kwargs = {
                            "action_low": action_space.low,
                            "action_high": action_space.high,
                        }
                    policy_sample = sample_policy_action(
                        pi,
                        policy_aux_info,
                        _rng,
                        **sample_kwargs,
                    )
                    log_probs.append(policy_sample.log_prob)
                    action = unbatchify(
                        policy_sample.action,
                        config["NUM_ENVS"],
                        env.multi_agent_config.number_of_agents_per_type[i],
                    )
                    pre_tanh_action = unbatchify(
                        policy_sample.pre_tanh_action,
                        config["NUM_ENVS"],
                        env.multi_agent_config.number_of_agents_per_type[i],
                    )
                    actions.append(action.squeeze())
                    pre_tanh_actions.append(pre_tanh_action.squeeze())
                    # env_act = unbatchify(
                    #     action, env.agents, config["NUM_ENVS"], env.num_agents
                    # )
                    # env_act = {k: v.squeeze() for k, v in env_act.items()}
                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                '''
                Cho các agent tương tác với môi trường, nhận lại obs mới, trạng thái và rewards
                '''
                pre_step_env_state = env_state
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0,None)
                )(rng_step, env_state, actions,env_params)

                # info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                '''
                Ghi lại nhật kí 
                '''
                done_batch=done
                transitions=[]
                for i,train_state in enumerate(train_states):
                    done_batch['agents'][i] = batchify(done["agents"][i],config["NUM_ACTORS_PERTYPE"][i]).squeeze()
                    obs_batch = batchify(last_obs[i],config["NUM_ACTORS_PERTYPE"][i])
                    action_batch = batchify_action(actions[i],config["NUM_ACTORS_PERTYPE"][i])
                    pre_tanh_action_batch = batchify_action(
                        pre_tanh_actions[i],
                        config["NUM_ACTORS_PERTYPE"][i],
                    )
                    value = values[i]
                    log_prob = log_probs[i]

                    info_world_i = {
                        **info["world"],
                        "obs_mid_price": pre_step_env_state.world_state.mid_price,
                        "obs_ask_raw_orders": pre_step_env_state.world_state.ask_raw_orders,
                        "obs_bid_raw_orders": pre_step_env_state.world_state.bid_raw_orders,
                    }
                    info_i={"world":info_world_i,"agent":jax.tree.map(lambda x: x.reshape(config["NUM_ACTORS_PERTYPE"][i],-1),info["agents"][i])}
                    # print(f"info for agenttype {i}:", info_i)


                    transitions.append(Transition(
                        jnp.tile(done["__all__"], config["NUM_AGENTS_PER_TYPE"][i]),
                        last_done[i],
                        action_batch.squeeze(),
                        pre_tanh_action_batch.squeeze(),
                        value.squeeze(),
                        batchify(reward[i], config["NUM_ACTORS_PERTYPE"][i]).squeeze(),
                        log_prob.squeeze(),
                        obs_batch,
                        info_i,
                        # avail_actions,
                    ))
                runner_state = _pack_runner_state(
                    train_states,
                    env_state,
                    obsv,
                    done_batch['agents'],
                    h_states,
                    rng,
                    current_aux_opt_state,
                    running_exe_episode_return,
                )
                return runner_state, transitions
            initial_hstates = _unpack_runner_state(runner_state)[4]

            survival_delta_steps = int(config.get("survival_delta_steps", 10))
            if survival_delta_steps < 1:
                raise ValueError("survival_delta_steps must be at least one.")
            reliability_future_padding = survival_delta_steps
            total_rollout_steps = config["NUM_STEPS"] + reliability_future_padding

            def scan_body(carry, step_idx):
                current_runner_state, stashed_runner_state = carry
                
                # Tiến 1 bước
                next_runner_state, transition = _env_step(current_runner_state, None)
                
                # MA THUẬT: Chụp ảnh trạng thái ở đúng vạch đích NUM_STEPS - 1 (bước 128)
                is_step_128 = (step_idx == config["NUM_STEPS"] - 1)
                new_stashed_state = jax.tree_util.tree_map(
                    lambda next_s, stash_s: jnp.where(is_step_128, next_s, stash_s),
                    next_runner_state, stashed_runner_state
                )
                
                return (next_runner_state, new_stashed_state), transition

            # Chạy 138 bước
            (final_runner_state, stashed_runner_state), traj_batch_padded = jax.lax.scan(
                scan_body, 
                (runner_state, runner_state), 
                jnp.arange(total_rollout_steps)
            )

            # ==========================================================
            # [PHASE 2]: BUILD LIQUIDITY RELIABILITY TARGETS
            # ==========================================================
            survival_labels = []
            survival_masks = []
            survival_task_side_masks = []
            survival_target_diags = []
            if not hasattr(env.multi_agent_config.world_config, "tick_size"):
                raise ValueError(
                    "Cannot build liquidity survival labels: "
                    "env.multi_agent_config.world_config.tick_size is required."
                )
            tick_size = env.multi_agent_config.world_config.tick_size
            if tick_size is None:
                raise ValueError(
                    "Cannot build liquidity survival labels: "
                    "env.multi_agent_config.world_config.tick_size is None."
                )

            _surv_dist_groups = (
                "CURRENT_RANK_TOP3_TASK_SIDE",
                "CURRENT_RANK_TOP3_OPPOSITE_SIDE",
                "CURRENT_RANK_FAR_TASK_SIDE",
                "CURRENT_RANK_FAR_OPPOSITE_SIDE",
            )
            _surv_dist_stats = (
                "count",
                "mean",
                "std",
                "min",
                "p10",
                "p25",
                "p50",
                "p75",
                "p90",
                "p95",
                "p99",
                "max",
                "zero_rate",
                "pos_rate_0p1",
                "pos_rate_0p3",
                "pos_rate_0p5",
                "pos_rate_0p7",
            )

            def _rel_dist_key(component_name, group_name, stat_name):
                return f"rel_dist_{component_name}_{group_name}_{stat_name}"

            def _zero_reliability_alignment_diag():
                diag = {
                    "score_shape_ok": jnp.array(0.0, dtype=jnp.float32),
                    "mask_shape_ok": jnp.array(0.0, dtype=jnp.float32),
                    "score_ndim": jnp.array(0.0, dtype=jnp.float32),
                    "label_ndim": jnp.array(0.0, dtype=jnp.float32),
                    "mask_ndim": jnp.array(0.0, dtype=jnp.float32),
                    "score_t0_b0_ask": jnp.zeros((10,), dtype=jnp.float32),
                    "score_t0_b0_bid": jnp.zeros((10,), dtype=jnp.float32),
                    "target_t0_b0_ask": jnp.zeros((10,), dtype=jnp.float32),
                    "target_t0_b0_bid": jnp.zeros((10,), dtype=jnp.float32),
                    "mask_t0_b0_ask": jnp.zeros((10,), dtype=jnp.float32),
                    "mask_t0_b0_bid": jnp.zeros((10,), dtype=jnp.float32),
                }
                for prefix in ("score", "label", "mask"):
                    for axis in range(5):
                        diag[f"{prefix}_dim{axis}"] = jnp.array(-1.0, dtype=jnp.float32)
                for component_name in ("score", "target", "abs_error", "signed_error"):
                    for group_name in _surv_dist_groups:
                        for stat_name in _surv_dist_stats:
                            diag[_rel_dist_key(component_name, group_name, stat_name)] = jnp.array(
                                0.0,
                                dtype=jnp.float32,
                            )
                return diag

            def _build_reliability_alignment_diag(
                reliability_scores,
                surv_labels,
                surv_mask,
                surv_task_side,
            ):
                score_shape = reliability_scores.shape
                label_shape = surv_labels.shape
                mask_shape = surv_mask.shape
                if score_shape == label_shape + (1,):
                    aligned_scores = jnp.squeeze(reliability_scores, axis=-1)
                elif score_shape == label_shape:
                    aligned_scores = reliability_scores
                else:
                    raise ValueError(
                        "Reliability/target shape mismatch in diagnostics: "
                        f"scores={score_shape}, labels={label_shape}."
                    )
                if mask_shape != label_shape:
                    raise ValueError(
                        "Reliability mask/target shape mismatch in diagnostics: "
                        f"mask={mask_shape}, labels={label_shape}."
                    )

                def _shape_dim(shape, axis):
                    return jnp.array(shape[axis] if axis < len(shape) else -1, dtype=jnp.float32)

                def _sample_t0_b0_side(x, side):
                    vals = jnp.asarray(x[0, 0, :10, side], dtype=jnp.float32)
                    return jnp.pad(vals, (0, 10 - vals.shape[0]), constant_values=0.0)

                def _masked_stats(x, mask, eps_value=1e-8):
                    x = jnp.asarray(x, dtype=jnp.float32)
                    mask = jnp.asarray(mask, dtype=jnp.float32)
                    count = jnp.sum(mask)
                    safe_count = jnp.maximum(count, eps_value)
                    mean = jnp.sum(x * mask) / safe_count
                    centered = (x - mean) * mask
                    std = jnp.sqrt(jnp.sum(jnp.square(centered)) / safe_count)
                    flat_x = jnp.reshape(x, (-1,))
                    flat_mask = jnp.reshape(mask > 0, (-1,))
                    sorted_vals = jnp.sort(jnp.where(flat_mask, flat_x, jnp.inf))
                    count_int = jnp.maximum(count.astype(jnp.int32), 1)
                    has_values = count > 0

                    def _percentile(q):
                        idx = jnp.floor(q * (count_int.astype(jnp.float32) - 1.0)).astype(jnp.int32)
                        return jnp.take(sorted_vals, idx)

                    min_val = jnp.min(jnp.where(flat_mask, flat_x, jnp.inf))
                    max_val = jnp.max(jnp.where(flat_mask, flat_x, -jnp.inf))
                    zero_rate = jnp.sum(((x <= 1e-6).astype(jnp.float32)) * mask) / safe_count
                    pos_rate_0p1 = jnp.sum(((x > 0.1).astype(jnp.float32)) * mask) / safe_count
                    pos_rate_0p3 = jnp.sum(((x > 0.3).astype(jnp.float32)) * mask) / safe_count
                    pos_rate_0p5 = jnp.sum(((x > 0.5).astype(jnp.float32)) * mask) / safe_count
                    pos_rate_0p7 = jnp.sum(((x > 0.7).astype(jnp.float32)) * mask) / safe_count
                    return {
                        "count": count,
                        "mean": jnp.where(has_values, mean, 0.0),
                        "std": jnp.where(has_values, std, 0.0),
                        "min": jnp.where(has_values, min_val, 0.0),
                        "p10": jnp.where(has_values, _percentile(0.10), 0.0),
                        "p25": jnp.where(has_values, _percentile(0.25), 0.0),
                        "p50": jnp.where(has_values, _percentile(0.50), 0.0),
                        "p75": jnp.where(has_values, _percentile(0.75), 0.0),
                        "p90": jnp.where(has_values, _percentile(0.90), 0.0),
                        "p95": jnp.where(has_values, _percentile(0.95), 0.0),
                        "p99": jnp.where(has_values, _percentile(0.99), 0.0),
                        "max": jnp.where(has_values, max_val, 0.0),
                        "zero_rate": jnp.where(has_values, zero_rate, 0.0),
                        "pos_rate_0p1": jnp.where(has_values, pos_rate_0p1, 0.0),
                        "pos_rate_0p3": jnp.where(has_values, pos_rate_0p3, 0.0),
                        "pos_rate_0p5": jnp.where(has_values, pos_rate_0p5, 0.0),
                        "pos_rate_0p7": jnp.where(has_values, pos_rate_0p7, 0.0),
                    }

                component_mask = jnp.asarray(surv_mask, dtype=jnp.float32)
                base_mask = component_mask > 0
                level_ids_for_diag = jnp.arange(surv_labels.shape[2])
                top_level_mask = (level_ids_for_diag < 3)[
                    None,
                    None,
                    :,
                    None,
                ]
                far_level_mask = (level_ids_for_diag >= 3)[
                    None,
                    None,
                    :,
                    None,
                ]
                task_side_mask = surv_task_side > 0.5
                opposite_side_mask = surv_task_side <= 0.5
                group_masks = {
                    "CURRENT_RANK_TOP3_TASK_SIDE": base_mask & top_level_mask & task_side_mask,
                    "CURRENT_RANK_TOP3_OPPOSITE_SIDE": base_mask & top_level_mask & opposite_side_mask,
                    "CURRENT_RANK_FAR_TASK_SIDE": base_mask & far_level_mask & task_side_mask,
                    "CURRENT_RANK_FAR_OPPOSITE_SIDE": base_mask & far_level_mask & opposite_side_mask,
                }
                rel_components = {
                    "score": aligned_scores,
                    "target": surv_labels,
                    "abs_error": jnp.abs(aligned_scores - surv_labels),
                    "signed_error": aligned_scores - surv_labels,
                }

                diag = {
                    "score_shape_ok": jnp.array(1.0, dtype=jnp.float32),
                    "mask_shape_ok": jnp.array(1.0, dtype=jnp.float32),
                    "score_ndim": jnp.array(len(score_shape), dtype=jnp.float32),
                    "label_ndim": jnp.array(len(label_shape), dtype=jnp.float32),
                    "mask_ndim": jnp.array(len(mask_shape), dtype=jnp.float32),
                    "score_t0_b0_ask": _sample_t0_b0_side(aligned_scores, 0),
                    "score_t0_b0_bid": _sample_t0_b0_side(aligned_scores, 1),
                    "target_t0_b0_ask": _sample_t0_b0_side(surv_labels, 0),
                    "target_t0_b0_bid": _sample_t0_b0_side(surv_labels, 1),
                    "mask_t0_b0_ask": _sample_t0_b0_side(component_mask, 0),
                    "mask_t0_b0_bid": _sample_t0_b0_side(component_mask, 1),
                }
                for prefix, shape in (
                    ("score", score_shape),
                    ("label", label_shape),
                    ("mask", mask_shape),
                ):
                    for axis in range(5):
                        diag[f"{prefix}_dim{axis}"] = _shape_dim(shape, axis)

                for component_name in ("score", "target", "abs_error", "signed_error"):
                    for group_name in _surv_dist_groups:
                        stats = _masked_stats(rel_components[component_name], group_masks[group_name])
                        for stat_name in _surv_dist_stats:
                            diag[_rel_dist_key(component_name, group_name, stat_name)] = stats[
                                stat_name
                            ]
                return diag


            for i in range(len(stashed_runner_state[0])): # Lặp qua train_states
                end_mid_prices = traj_batch_padded[i].info["world"]["end_mid_price"]
                obs_mid_prices = traj_batch_padded[i].info["world"].get("obs_mid_price", end_mid_prices)
                obs_ask_raw_orders = traj_batch_padded[i].info["world"].get("obs_ask_raw_orders", None)
                obs_bid_raw_orders = traj_batch_padded[i].info["world"].get("obs_bid_raw_orders", None)
                
                # Quét cửa sổ tương lai (Logic giữ nguyên, chạy thẳng trên mid_prices chuẩn)
                
                # Chỉ tính nhãn cho 128 bước đầu
                
                # Gán nhãn

                if (
                    config.get("use_survival_loss", False)
                    and _is_execution_agent(env.list_of_agents_configs[i])
                    and isinstance(traj_batch_padded[i].obs, dict)
                    and "vision_obs" in traj_batch_padded[i].obs
                ):
                    execution_task = getattr(env.list_of_agents_configs[i], "task", None)
                    is_sell_task = resolve_rollout_is_sell_task(
                        traj_batch_padded[i].info["agent"],
                        task=execution_task,
                        num_steps=config["NUM_STEPS"],
                        batch_size=traj_batch_padded[i].obs["vision_obs"].shape[1],
                    )
                    surv_label, surv_mask, surv_target_diag = build_liquidity_survival_targets(
                        traj_batch_padded[i].obs["vision_obs"],
                        obs_mid_prices,
                        tick_size=tick_size,
                        survival_delta_steps=survival_delta_steps,
                        survival_min_volume=config.get("survival_min_volume", 1.0),
                        ask_raw_orders=obs_ask_raw_orders,
                        bid_raw_orders=obs_bid_raw_orders,
                        new_trades=traj_batch_padded[i].info["world"]["new_trades"],
                        trade_valid_mask=traj_batch_padded[i].info["world"]["trade_valid_mask"],
                        trade_buffer_saturated=traj_batch_padded[i].info["world"]["trade_buffer_saturated"],
                        num_steps=config["NUM_STEPS"],
                        episode_done=traj_batch_padded[i].global_done,
                        return_diagnostics=True,
                        eps=config.get("survival_eps", 1e-8),
                    )
                    task_side = jnp.stack(
                        [is_sell_task, 1.0 - is_sell_task],
                        axis=-1,
                    )
                    surv_task_side = jnp.broadcast_to(
                        task_side[:, :, None, :],
                        surv_label.shape,
                    )
                else:
                    reward_shape = traj_batch_padded[i].reward.shape
                    surv_label = jnp.zeros((config["NUM_STEPS"], reward_shape[1], 10, 2), dtype=jnp.float32)
                    surv_mask = jnp.zeros_like(surv_label)
                    surv_task_side = jnp.zeros_like(surv_label)
                    surv_target_diag = empty_liquidity_survival_diagnostics(
                        surv_label.shape[2]
                    )
                survival_labels.append(surv_label)
                survival_masks.append(surv_mask)
                survival_task_side_masks.append(surv_task_side)
                survival_target_diags.append(surv_target_diag)

            # ==========================================================
            # [PHASE 3]: CƯA ĐUÔI DATA VÀ KHÔI PHỤC DÒNG THỜI GIAN
            # ==========================================================
            # 3.1 Cưa bỏ 10 bước padding, trả về traj_batch chuẩn 128 bước
            traj_batch = jax.tree_util.tree_map(
                lambda x: x[:config["NUM_STEPS"]], traj_batch_padded
            )

            # 3.2 Khôi phục bộ nhớ ở bước 128 cho vòng lặp sau
            (
                t_states,
                e_state,
                l_obs,
                l_dones,
                h_states,
                _,
                aux_opt_state,
                running_exe_episode_return,
            ) = _unpack_runner_state(stashed_runner_state)

            if execution_index is not None:
                execution_trajectory = traj_batch[execution_index]
                execution_reward_shape = execution_trajectory.reward.shape
                execution_quant_left = jnp.reshape(
                    execution_trajectory.info["agent"]["quant_left"],
                    execution_reward_shape,
                )
                execution_task_size = jnp.reshape(
                    execution_trajectory.info["agent"]["denom_task"],
                    execution_reward_shape,
                )
                execution_info = execution_trajectory.info["agent"]

                def reshape_execution_info(key):
                    return jnp.reshape(
                        execution_info[key],
                        execution_reward_shape,
                    )
                (
                    running_exe_episode_return,
                    execution_episode_metrics,
                ) = accumulate_execution_episode_metrics(
                    running_exe_episode_return,
                    execution_trajectory.reward,
                    execution_trajectory.global_done,
                    execution_quant_left,
                    execution_task_size,
                    full_completion=reshape_execution_info(
                        "terminal_full_completion"
                    ),
                    realized_is_bps=reshape_execution_info(
                        "terminal_realized_is_bps"
                    ),
                    realized_is_valid=reshape_execution_info(
                        "terminal_realized_is_valid"
                    ),
                    forced_liquidation_is_bps=reshape_execution_info(
                        "terminal_forced_liquidation_is_bps"
                    ),
                    forced_liquidation_is_valid=reshape_execution_info(
                        "terminal_forced_liquidation_is_valid"
                    ),
                    twap_forced_liquidation_is_bps=reshape_execution_info(
                        "terminal_twap_forced_liquidation_is_bps"
                    ),
                    twap_forced_liquidation_is_valid=reshape_execution_info(
                        "terminal_twap_forced_liquidation_is_valid"
                    ),
                    twap_advantage_bps=reshape_execution_info(
                        "terminal_twap_advantage_bps"
                    ),
                    twap_comparison_valid=reshape_execution_info(
                        "terminal_twap_comparison_valid"
                    ),
                    twap_win=reshape_execution_info("terminal_twap_win"),
                )
            else:
                execution_episode_metrics = empty_execution_episode_metrics()
            
            # Chôm chìa khóa RNG từ bước 138 (final_runner_state).
            fresh_rng = _unpack_runner_state(final_runner_state)[5]
            
            # Gắn lại vào runner_state
            runner_state = _pack_runner_state(
                t_states,
                e_state,
                l_obs,
                l_dones,
                h_states,
                fresh_rng,
                aux_opt_state,
                running_exe_episode_return,
            )

            # CALCULATE ADVANTAGE
            (
                train_states,
                env_state,
                last_obs,
                last_dones,
                hstates_new,
                rng,
                aux_opt_state,
                running_exe_episode_return,
            ) = _unpack_runner_state(runner_state)

            def _calculate_gae(gamma,gae_lambda,traj_batch, last_val):
                    def _get_advantages(gae_and_next_value, transition):
                        gae, next_value = gae_and_next_value
                        done, value, reward = (
                            transition.global_done,
                            transition.value,
                            transition.reward,
                        )
                        delta = reward + gamma * next_value * (1 - done) - value
                        gae = (
                            delta
                            + gamma * gae_lambda * (1 - done) * gae
                        )
                        return (gae, value), gae

                    _, advantages = jax.lax.scan(
                        _get_advantages,
                        (jnp.zeros_like(last_val), last_val),
                        traj_batch,
                        reverse=True,
                        unroll=16,
                    )
                    return advantages, advantages + traj_batch.value

            advantages=[]
            targets=[]
            for i, train_state in enumerate(train_states):
                last_obs_batch = batchify(last_obs[i], config["NUM_ACTORS_PERTYPE"][i])
                last_obs_batch_expanded = jax.tree.map(lambda x: x[jnp.newaxis, :], last_obs_batch)
                # avail_actions = jnp.ones(
                #     (config["NUM_ACTORS"], env.action_space(env.agents[0]).n)
                # )
                ac_in = (
                    last_obs_batch_expanded,
                    last_dones[i][jnp.newaxis, :],
                    # avail_actions,
                )
                _, _, last_val, _, _ = train_state.apply_fn(train_state.params, hstates_new[i], ac_in)
                last_val = last_val.squeeze()

                advantages_i, targets_i = _calculate_gae(config["GAMMA"][i],config["GAE_LAMBDA"][i],traj_batch[i], last_val)
                advantages.append(advantages_i)
                targets.append(targets_i)

            # UPDATE NETWORKS
            # FIXME: APPLY VISION, GATED-FUSION
            loss_infos = []
            grad_interaction_diags = []
            phasic_aux_diags = []
            ppo_safety_diags = []
            box_ppo_numerics_diags = []
            execution_post_ppo_rng = rng
            for i, train_state in enumerate(train_states):
                agent_is_execution = _is_execution_agent(env.list_of_agents_configs[i])
                agent_is_box = isinstance(env.action_spaces[i], spaces.Box)
                ppo_objective_survival_weight = ppo_survival_loss_weight(
                    phasic_settings,
                    config.get("lambda_surv", 0.0),
                )
                grad_diag_applicable = bool(
                    agent_is_execution
                    and config.get("use_reliability_head", False)
                    and config.get("use_survival_loss", False)
                    and isinstance(traj_batch[i].obs, dict)
                )

                def _update_epoch(update_state, epoch_index):
                    def _update_minbatch(update_carry, scan_input):
                        (
                            train_state,
                            grad_interaction_diag,
                            ppo_safety_state,
                        ) = update_carry
                        batch_info, minibatch_index = scan_input
                        (
                            init_hstate,
                            traj_batch,
                            advantages,
                            targets,
                            surv_labels,
                            surv_mask,
                        ) = batch_info

                        def _compute_loss_components(
                            params,
                            init_hstate,
                            traj_batch,
                            gae,
                            targets,
                            surv_labels,
                            surv_mask,
                            objective_survival_weight,
                        ):
                            # RERUN NETWORK
                            _, pi, value, _z_vision, aux_info = train_state.apply_fn(
                                params,
                                init_hstate.squeeze(),
                                (traj_batch.obs, traj_batch.done),
                            )
                            replay_kwargs = {}
                            if isinstance(env.action_spaces[i], spaces.Box):
                                replay_kwargs = {
                                    "action_low": env.action_spaces[i].low,
                                    "action_high": env.action_spaces[i].high,
                                }
                            log_prob = policy_log_prob_from_transition(
                                pi,
                                aux_info,
                                traj_batch.action,
                                traj_batch.pre_tanh_action,
                                **replay_kwargs,
                            )

                            # CALCULATE VALUE LOSS
                            value_pred_clipped = traj_batch.value + (
                                value - traj_batch.value
                            ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                            value_losses = jnp.square(value - targets)
                            value_losses_clipped = jnp.square(value_pred_clipped - targets)
                            value_loss = 0.5 * jnp.maximum(
                                value_losses, value_losses_clipped
                            ).mean()

                            # CALCULATE ACTOR LOSS
                            logratio = log_prob - traj_batch.log_prob
                            ratio = jnp.exp(logratio)
                            unnormalized_gae = gae
                            gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                            loss_actor1 = ratio * gae
                            loss_actor2 = (
                                jnp.clip(
                                    ratio,
                                    1.0 - config["CLIP_EPS"],
                                    1.0 + config["CLIP_EPS"],
                                )
                                * gae
                            )
                            loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                            loss_actor = loss_actor.mean()
                            if isinstance(env.action_spaces[i], spaces.Box):
                                entropy = -log_prob.mean()
                            else:
                                entropy = pi.entropy().mean()

                            # TỔNG HỢP PPO LOSS
                            weighted_value_loss = config["VF_COEF"][i] * value_loss
                            weighted_entropy_term = -config["ENT_COEF"][i] * entropy
                            ppo_loss = loss_actor + weighted_value_loss + weighted_entropy_term

                            reliability_scores = aux_info["reliability_scores"]
                            reliability_logits = aux_info["reliability_logits"]
                            if reliability_scores.ndim == 4:
                                reliability_scores = reliability_scores[..., None]
                            if reliability_logits.ndim == 4:
                                reliability_logits = reliability_logits[..., None]

                            reliability_mean = jnp.mean(reliability_scores)
                            reliability_std = jnp.std(reliability_scores)
                            reliability_min = jnp.min(reliability_scores)
                            reliability_max = jnp.max(reliability_scores)

                            reliability_level_mean = jnp.mean(reliability_scores, axis=(0, 1, 3, 4))
                            reliability_level_mean = reliability_level_mean[:10]
                            reliability_level_mean = jnp.pad(
                                reliability_level_mean,
                                (0, 10 - reliability_level_mean.shape[0]),
                                constant_values=0.0,
                            )
                            reliability_side_mean = jnp.mean(reliability_scores, axis=(0, 1, 2, 4))
                            reliability_side_mean = reliability_side_mean[:2]
                            reliability_side_mean = jnp.pad(
                                reliability_side_mean,
                                (0, 2 - reliability_side_mean.shape[0]),
                                constant_values=0.0,
                            )

                            z_tokens_norm = aux_info["z_tokens_norm"]
                            filtered_tokens_norm = aux_info["filtered_tokens_norm"]
                            z_tokens_norm_mean = jnp.mean(z_tokens_norm)
                            z_tokens_norm_std = jnp.std(z_tokens_norm)
                            filtered_tokens_norm_mean = jnp.mean(filtered_tokens_norm)
                            filtered_tokens_norm_std = jnp.std(filtered_tokens_norm)
                            filtering_ratio = filtered_tokens_norm_mean / jnp.maximum(z_tokens_norm_mean, 1e-8)
                            exec_obs_norm_mean = jnp.mean(aux_info["exec_obs_norm"])
                            vision_token_pooled_norm_mean = jnp.mean(aux_info["vision_token_pooled_norm"])
                            fusion_output_norm_mean = jnp.mean(aux_info["fusion_output_norm"])
                            pre_rnn_embedding_norm_mean = jnp.mean(aux_info["pre_rnn_embedding_norm"])
                            actor_input_norm_mean = jnp.mean(aux_info["actor_input_norm"])
                            rel_z_shape = aux_info.get("rel_z_shape", jnp.full((5,), -1.0, dtype=jnp.float32))
                            rel_side_id_shape = aux_info.get("rel_side_id_shape", jnp.full((5,), -1.0, dtype=jnp.float32))
                            rel_mid_context_shape = aux_info.get("rel_mid_context_shape", jnp.full((3,), -1.0, dtype=jnp.float32))
                            mid_return_from_init_mean = jnp.mean(aux_info.get("mid_return_from_init", jnp.array(0.0, dtype=jnp.float32)))
                            spread_ticks_mean = jnp.mean(aux_info.get("spread_ticks", jnp.array(0.0, dtype=jnp.float32)))
                            mid_delta_ticks_mean = jnp.mean(aux_info.get("mid_delta_ticks", jnp.array(0.0, dtype=jnp.float32)))
                            mid_volatility_ticks_mean = jnp.mean(aux_info.get("mid_volatility_ticks", jnp.array(0.0, dtype=jnp.float32)))
                            h_prev_reliability_norm_mean = jnp.mean(
                                aux_info.get(
                                    "h_prev_reliability_norm",
                                    jnp.array(0.0, dtype=jnp.float32),
                                )
                            )
                            use_h_prev_in_reliability_value = jnp.mean(
                                aux_info.get(
                                    "use_h_prev_in_reliability",
                                    jnp.array(
                                        float(config.get("use_h_prev_in_reliability", True)),
                                        dtype=jnp.float32,
                                    ),
                                )
                            )
                            h_prev_used_in_reliability_value = jnp.mean(
                                aux_info.get(
                                    "h_prev_used_in_reliability",
                                    jnp.array(
                                        float(config.get("use_h_prev_in_reliability", True)),
                                        dtype=jnp.float32,
                                    ),
                                )
                            )
                            h_prev_reliability_zeroed_value = jnp.mean(
                                aux_info.get(
                                    "h_prev_reliability_zeroed",
                                    jnp.array(
                                        float(not config.get("use_h_prev_in_reliability", True)),
                                        dtype=jnp.float32,
                                    ),
                                )
                            )

                            # FINAL LOSS: PPO plus reliability survival auxiliary loss.
                            if (
                                config.get("use_survival_loss", False)
                                and config.get("use_reliability_head", False)
                                and agent_is_execution
                                and isinstance(traj_batch.obs, dict)
                            ):
                                # Soft targets use the same auxiliary slot as the legacy survival loss.
                                if not (
                                    reliability_scores.shape == surv_labels.shape
                                    or reliability_scores.shape == surv_labels.shape + (1,)
                                ):
                                    raise ValueError(
                                        "Reliability scores and survival labels are not aligned: "
                                        f"scores={reliability_scores.shape}, labels={surv_labels.shape}."
                                    )
                                if surv_mask.shape != surv_labels.shape:
                                    raise ValueError(
                                        "Survival mask and labels are not aligned: "
                                        f"mask={surv_mask.shape}, labels={surv_labels.shape}."
                                    )
                                survival_loss = masked_reliability_loss(
                                    reliability_scores,
                                    surv_labels,
                                    surv_mask,
                                    loss_type=config.get("reliability_loss_type", "bce"),
                                    eps=config.get("survival_eps", 1e-8),
                                    reliability_logits=reliability_logits,
                                )
                                lambda_surv = config.get("lambda_surv", 0.0)
                                survival_mask_ratio = jnp.mean(surv_mask.astype(jnp.float32))
                            else:
                                survival_loss = jnp.array(0.0)
                                lambda_surv = jnp.array(0.0)
                                survival_mask_ratio = jnp.array(0.0)

                            weighted_survival_loss = (
                                objective_survival_weight * survival_loss
                            )
                            aux_loss = weighted_survival_loss
                            total_loss = (
                                ppo_loss
                                + objective_survival_weight * survival_loss
                            )

                            # debug
                            approx_kl = ((ratio - 1) - logratio).mean()
                            clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])

                            return {
                                "total_loss": total_loss,
                                "ppo_loss": ppo_loss,
                                "survival_loss": survival_loss,
                                "weighted_survival_loss": weighted_survival_loss,
                                "ppo_numerics_inputs": {
                                    "loc": aux_info.get(
                                        "policy_loc",
                                        jnp.zeros(
                                            traj_batch.action.shape + (1,),
                                            dtype=jnp.float32,
                                        ),
                                    ),
                                    "log_std": aux_info.get(
                                        "policy_log_std",
                                        jnp.zeros((1,), dtype=jnp.float32),
                                    ),
                                    "pre_tanh_action": traj_batch.pre_tanh_action,
                                    "action": traj_batch.action,
                                    "old_log_prob": traj_batch.log_prob,
                                    "new_log_prob": log_prob,
                                    "logratio": logratio,
                                    "ratio": ratio,
                                    "advantage": unnormalized_gae,
                                    "value": value,
                                },
                                "metrics": (
                                value_loss,
                                loss_actor,
                                entropy,
                                ratio,
                                approx_kl,
                                clip_frac,
                                survival_loss,
                                weighted_survival_loss,
                                survival_mask_ratio,
                                reliability_mean,
                                ppo_loss,
                                weighted_value_loss,
                                weighted_entropy_term,
                                aux_loss,
                                reliability_std,
                                reliability_min,
                                reliability_max,
                                reliability_level_mean[0],
                                reliability_level_mean[1],
                                reliability_level_mean[2],
                                reliability_level_mean[3],
                                reliability_level_mean[4],
                                reliability_level_mean[5],
                                reliability_level_mean[6],
                                reliability_level_mean[7],
                                reliability_level_mean[8],
                                reliability_level_mean[9],
                                reliability_side_mean[0],
                                reliability_side_mean[1],
                                z_tokens_norm_mean,
                                z_tokens_norm_std,
                                filtered_tokens_norm_mean,
                                filtered_tokens_norm_std,
                                filtering_ratio,
                                exec_obs_norm_mean,
                                vision_token_pooled_norm_mean,
                                fusion_output_norm_mean,
                                pre_rnn_embedding_norm_mean,
                                actor_input_norm_mean,
                                rel_z_shape,
                                rel_side_id_shape,
                                rel_mid_context_shape,
                                mid_return_from_init_mean,
                                spread_ticks_mean,
                                mid_delta_ticks_mean,
                                mid_volatility_ticks_mean,
                                h_prev_reliability_norm_mean,
                                use_h_prev_in_reliability_value,
                                h_prev_used_in_reliability_value,
                                h_prev_reliability_zeroed_value,
                                ),
                            }

                        def _loss_fn(
                            params,
                            init_hstate,
                            traj_batch,
                            gae,
                            targets,
                            surv_labels,
                            surv_mask,
                            objective_survival_weight,
                        ):
                            components = _compute_loss_components(
                                params,
                                init_hstate,
                                traj_batch,
                                gae,
                                targets,
                                surv_labels,
                                surv_mask,
                                objective_survival_weight,
                            )
                            return components["total_loss"], (
                                components["metrics"],
                                components["ppo_numerics_inputs"],
                            )

                        def _ppo_objective(
                            params,
                            init_hstate,
                            traj_batch,
                            gae,
                            targets,
                            surv_labels,
                            surv_mask,
                        ):
                            return _compute_loss_components(
                                params,
                                init_hstate,
                                traj_batch,
                                gae,
                                targets,
                                surv_labels,
                                surv_mask,
                                0.0,
                            )["ppo_loss"]

                        def _survival_objective(
                            params,
                            init_hstate,
                            traj_batch,
                            gae,
                            targets,
                            surv_labels,
                            surv_mask,
                        ):
                            return _compute_loss_components(
                                params,
                                init_hstate,
                                traj_batch,
                                gae,
                                targets,
                                surv_labels,
                                surv_mask,
                                0.0,
                            )["survival_loss"]

                        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                        total_loss, grads = grad_fn(
                            train_state.params,
                            init_hstate,
                            traj_batch,
                            advantages,
                            targets,
                            surv_labels,
                            surv_mask,
                            ppo_objective_survival_weight,
                        )
                        loss_value, loss_aux = total_loss
                        loss_metrics, ppo_numerics_inputs = loss_aux
                        if grad_diag_enabled and grad_diag_applicable:
                            should_compute_grad_diag = gradient_diag_should_run(
                                update_steps,
                                grad_diag_cadence,
                                epoch_index,
                                minibatch_index,
                            )

                            def _compute_grad_interaction_diag(_):
                                if phasic_mode:
                                    ppo_grads = jax.grad(_ppo_objective)(
                                        train_state.params,
                                        init_hstate,
                                        traj_batch,
                                        advantages,
                                        targets,
                                        surv_labels,
                                        surv_mask,
                                    )
                                    survival_grads = jax.grad(
                                        _survival_objective
                                    )(
                                        train_state.params,
                                        init_hstate,
                                        traj_batch,
                                        advantages,
                                        targets,
                                        surv_labels,
                                        surv_mask,
                                    )
                                    return summarize_phasic_gradient_interaction(
                                        train_state.params,
                                        ppo_grads,
                                        survival_grads,
                                        survival_loss_pre_ppo,
                                    )
                                _, ppo_grads = jax.value_and_grad(
                                    _loss_fn,
                                    has_aux=True,
                                )(
                                    train_state.params,
                                    init_hstate,
                                    traj_batch,
                                    advantages,
                                    targets,
                                    surv_labels,
                                    surv_mask,
                                    0.0,
                                )
                                _, joint_unit_grads = jax.value_and_grad(
                                    _loss_fn,
                                    has_aux=True,
                                )(
                                    train_state.params,
                                    init_hstate,
                                    traj_batch,
                                    advantages,
                                    targets,
                                    surv_labels,
                                    surv_mask,
                                    1.0,
                                )
                                survival_grads = subtract_gradient_trees(
                                    joint_unit_grads,
                                    ppo_grads,
                                )
                                return summarize_gradient_interaction(
                                    train_state.params,
                                    grads,
                                    ppo_grads,
                                    survival_grads,
                                    config.get("lambda_surv", 0.0),
                                    survival_loss_pre_ppo,
                                )

                            grad_interaction_diag = jax.lax.cond(
                                should_compute_grad_diag,
                                _compute_grad_interaction_diag,
                                lambda _: grad_interaction_diag,
                                operand=None,
                            )
                        guarded_update = guarded_ppo_apply_gradients(
                            train_state,
                            grads,
                            total_loss=loss_value,
                            new_log_prob=ppo_numerics_inputs["new_log_prob"],
                            logratio=ppo_numerics_inputs["logratio"],
                            ratio=ppo_numerics_inputs["ratio"],
                        )
                        train_state_after = select_guarded_train_state(
                            train_state,
                            guarded_update,
                            ppo_safety_state,
                        )
                        attempt_active = ~ppo_safety_state["stopped"]
                        diagnostic_guard = guarded_update._replace(
                            accepted=attempt_active & guarded_update.accepted,
                            rejected_nonfinite=(
                                attempt_active & guarded_update.rejected_nonfinite
                            ),
                        )
                        ppo_safety_state = update_ppo_safety_state(
                            ppo_safety_state,
                            guarded_update,
                            epoch_index=epoch_index,
                            minibatch_index=minibatch_index,
                        )
                        if box_ppo_diag_enabled and agent_is_execution and agent_is_box:
                            box_ppo_diag = build_box_ppo_numerics_diagnostics(
                                enabled=True,
                                loc=ppo_numerics_inputs["loc"],
                                log_std=ppo_numerics_inputs["log_std"],
                                pre_tanh_action=ppo_numerics_inputs["pre_tanh_action"],
                                action=ppo_numerics_inputs["action"],
                                action_low=env.action_spaces[i].low,
                                action_high=env.action_spaces[i].high,
                                old_log_prob=ppo_numerics_inputs["old_log_prob"],
                                new_log_prob=ppo_numerics_inputs["new_log_prob"],
                                advantage=ppo_numerics_inputs["advantage"],
                                value=ppo_numerics_inputs["value"],
                                grads=grads,
                                total_loss=loss_value,
                                candidate_update=diagnostic_guard,
                                epoch_index=epoch_index,
                                minibatch_index=minibatch_index,
                            )
                        else:
                            action_dim = (
                                env.action_spaces[i].shape[-1]
                                if agent_is_box
                                else 1
                            )
                            box_ppo_diag = empty_box_ppo_numerics_diagnostics(
                                action_dim
                            )
                        total_loss = (
                            loss_value,
                            loss_metrics + (box_ppo_diag,),
                        )
                        return (
                            train_state_after,
                            grad_interaction_diag,
                            ppo_safety_state,
                        ), total_loss
                    (
                        train_state,
                        init_hstate,
                        traj_batch,
                        advantages,
                        targets,
                        surv_labels,
                        surv_mask,
                        rng,
                        grad_interaction_diag,
                        ppo_safety_state,
                    ) = update_state
                    rng, _rng = jax.random.split(rng)

                    # adding an additional "fake" dimensionality to perform minibatching correctly
                    init_hstate = jnp.reshape(
                        init_hstate, (1, config["NUM_ACTORS_PERTYPE"][i], -1)
                    )
                    batch = (
                        init_hstate,
                        traj_batch,
                        advantages.squeeze(),
                        targets.squeeze(),
                        surv_labels,
                        surv_mask,
                    )
                    permutation = jax.random.permutation(_rng, config["NUM_ACTORS_PERTYPE"][i])

                    shuffled_batch = jax.tree.map(
                        lambda x: jnp.take(x, permutation, axis=1), batch
                    )

                    minibatches = jax.tree.map(
                        lambda x: jnp.swapaxes(
                            jnp.reshape(
                                x,
                                [x.shape[0], config["NUM_MINIBATCHES"], -1]
                                + list(x.shape[2:]),
                            ),
                            1,
                            0,
                        ),
                        shuffled_batch,
                    )

                    (
                        train_state,
                        grad_interaction_diag,
                        ppo_safety_state,
                    ), total_loss = jax.lax.scan(
                        _update_minbatch,
                        (train_state, grad_interaction_diag, ppo_safety_state),
                        (
                            minibatches,
                            jnp.arange(config["NUM_MINIBATCHES"], dtype=jnp.int32),
                        ),
                    )
                    update_state = (
                        train_state,
                        init_hstate.squeeze(),
                        traj_batch,
                        advantages,
                        targets,
                        surv_labels,
                        surv_mask,
                        rng,
                        grad_interaction_diag,
                        ppo_safety_state,
                    )
                    return update_state, total_loss

                cadence_due = update_steps % grad_diag_cadence == 0
                survival_loss_pre_ppo = jnp.array(0.0, dtype=jnp.float32)
                if grad_diag_enabled and grad_diag_applicable and phasic_mode:
                    def _compute_survival_loss_pre_ppo(_):
                        pre_ppo_outputs = build_rollout_outputs(
                            train_state.apply_fn,
                            train_state.params,
                            initial_hstates[i],
                            traj_batch[i].obs,
                            traj_batch[i].done,
                            is_discrete=isinstance(
                                env.action_spaces[i],
                                spaces.Discrete,
                            ),
                        )
                        return masked_reliability_loss(
                            pre_ppo_outputs.reliability_scores,
                            survival_labels[i],
                            survival_masks[i],
                            loss_type=config.get("reliability_loss_type", "bce"),
                            eps=config.get("survival_eps", 1e-8),
                            reliability_logits=pre_ppo_outputs.reliability_logits,
                        )

                    survival_loss_pre_ppo = jax.lax.cond(
                        cadence_due,
                        _compute_survival_loss_pre_ppo,
                        lambda _: jnp.array(0.0, dtype=jnp.float32),
                        operand=None,
                    )
                initial_grad_interaction_diag = empty_gradient_interaction_diagnostics(
                    train_state.params,
                    enabled=grad_diag_enabled,
                    skipped_by_cadence=(
                        grad_diag_enabled
                        and grad_diag_applicable
                        and ~cadence_due
                    ),
                    not_applicable=(grad_diag_enabled and not grad_diag_applicable),
                    reason_not_execution=(
                        grad_diag_enabled and not agent_is_execution
                    ),
                    reason_reliability_disabled=(
                        grad_diag_enabled
                        and not config.get("use_reliability_head", False)
                    ),
                    reason_survival_disabled=(
                        grad_diag_enabled
                        and not config.get("use_survival_loss", False)
                    ),
                    survival_loss_pre_ppo=survival_loss_pre_ppo,
                )
                update_state = (
                    train_state,
                    initial_hstates[i],
                    traj_batch[i],
                    advantages[i],
                    targets[i],
                    survival_labels[i],
                    survival_masks[i],
                    rng,
                    initial_grad_interaction_diag,
                    empty_ppo_safety_state(),
                )
                update_state, loss_info = jax.lax.scan(
                    _update_epoch,
                    update_state,
                    jnp.arange(config["UPDATE_EPOCHS"], dtype=jnp.int32),
                )
                train_states[i] = update_state[0]
                loss_infos.append(loss_info)
                grad_interaction_diags.append(update_state[8])
                ppo_safety_diags.append(update_state[9])
                box_ppo_numerics_diags.append(loss_info[1][-1])
                phasic_aux_diags.append(
                    empty_phasic_aux_diagnostics(
                        is_discrete=isinstance(env.action_spaces[i], spaces.Discrete),
                        settings=phasic_settings,
                    )
                )
                if phasic_mode and i == execution_index:
                    execution_post_ppo_rng = update_state[7]

            if phasic_mode:
                execution_train_state = train_states[execution_index]
                (
                    phasic_params,
                    aux_opt_state,
                    rng,
                    phasic_aux_diag,
                ) = run_phasic_auxiliary_phase(
                    apply_fn=execution_train_state.apply_fn,
                    params=execution_train_state.params,
                    aux_opt_state=aux_opt_state,
                    aux_tx=aux_tx,
                    init_hstate=initial_hstates[execution_index],
                    obs=traj_batch[execution_index].obs,
                    done=traj_batch[execution_index].done,
                    labels=survival_labels[execution_index],
                    mask=survival_masks[execution_index],
                    rng=execution_post_ppo_rng,
                    settings=phasic_settings,
                    is_discrete=isinstance(
                        env.action_spaces[execution_index],
                        spaces.Discrete,
                    ),
                    reliability_loss_type=config.get("reliability_loss_type", "bce"),
                    survival_eps=config.get("survival_eps", 1e-8),
                )
                train_states[execution_index] = execution_train_state.replace(
                    params=phasic_params
                )
                phasic_aux_diags[execution_index] = phasic_aux_diag

            reliability_alignment_diags = []
            for i, train_state in enumerate(train_states):
                if (
                    config.get("use_survival_loss", False)
                    and config.get("use_reliability_head", False)
                    and _is_execution_agent(env.list_of_agents_configs[i])
                    and isinstance(traj_batch[i].obs, dict)
                    and "vision_obs" in traj_batch[i].obs
                ):
                    diag_outputs = build_rollout_outputs(
                        train_state.apply_fn,
                        train_state.params,
                        initial_hstates[i],
                        traj_batch[i].obs,
                        traj_batch[i].done,
                        is_discrete=isinstance(
                            env.action_spaces[i],
                            spaces.Discrete,
                        ),
                    )
                    reliability_alignment_diags.append(
                        _build_reliability_alignment_diag(
                            diag_outputs.reliability_scores,
                            survival_labels[i],
                            survival_masks[i],
                            survival_task_side_masks[i],
                        )
                    )
                else:
                    reliability_alignment_diags.append(_zero_reliability_alignment_diag())


            callback_world_exclusions = {
                "new_trades",
                "trade_valid_mask",
                "obs_ask_raw_orders",
                "obs_bid_raw_orders",
            }
            callback_agent_exclusions = {
                "terminal_full_completion",
                "terminal_realized_is_bps",
                "terminal_realized_is_valid",
                "terminal_forced_liquidation_is_bps",
                "terminal_forced_liquidation_is_valid",
                "terminal_twap_forced_liquidation_is_bps",
                "terminal_twap_forced_liquidation_is_valid",
                "terminal_twap_advantage_bps",
                "terminal_twap_comparison_valid",
                "terminal_twap_win",
            }
            callback_traj_batch = [
                transition._replace(
                    info={
                        "world": {
                            key: value
                            for key, value in transition.info["world"].items()
                            if key not in callback_world_exclusions
                        },
                        "agent": {
                            key: value
                            for key, value in transition.info["agent"].items()
                            if key not in callback_agent_exclusions
                        },
                    }
                )
                for transition in traj_batch
            ]
            metrics= {}
            metrics['agents'] = [jax.tree.map(
                lambda x: x.reshape(
                    (config["NUM_STEPS"], config["NUM_ENVS"], config["NUM_AGENTS_PER_TYPE"][i])
                ),
                trjbtch.info['agent']) for i, trjbtch in enumerate(traj_batch)]
            metrics['world'] = [
                transition.info['world'] for transition in callback_traj_batch
            ]
            metrics["execution_target_diag"] = survival_target_diags
            metrics["reliability_alignment_diag"] = reliability_alignment_diags
            metrics["grad_interaction_diag"] = grad_interaction_diags
            metrics["phasic_aux_diag"] = phasic_aux_diags
            metrics["ppo_safety_diag"] = ppo_safety_diags
            metrics["box_ppo_numerics_diag"] = box_ppo_numerics_diags
            metrics["loss"]=[]
            for i,loss_info in enumerate(loss_infos):
                ratio_0 = loss_info[1][3].at[0,0].get().mean()
                loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
                loss_metrics = {
                    "total_loss": loss_info[0],
                    "value_loss": loss_info[1][0],
                    "actor_loss": loss_info[1][1],
                    "policy_loss": loss_info[1][1],
                    "entropy": loss_info[1][2],
                    "entropy_loss": -loss_info[1][2],
                    "ratio": loss_info[1][3],
                    "ratio_0": ratio_0,
                    "approx_kl": loss_info[1][4],
                    "clip_frac": loss_info[1][5],
                    "survival_loss": loss_info[1][6],
                    "reliability_loss": loss_info[1][6],
                    "weighted_survival_loss": loss_info[1][7],
                    "survival_mask_ratio": loss_info[1][8],
                    "reliability_mean": loss_info[1][9],
                    "ppo_loss": loss_info[1][10],
                    "weighted_value_loss": loss_info[1][11],
                    "weighted_entropy_term": loss_info[1][12],
                    "aux_loss": loss_info[1][13],
                    "reliability_std": loss_info[1][14],
                    "reliability_min": loss_info[1][15],
                    "reliability_max": loss_info[1][16],
                    "reliability_level_mean_0": loss_info[1][17],
                    "reliability_level_mean_1": loss_info[1][18],
                    "reliability_level_mean_2": loss_info[1][19],
                    "reliability_level_mean_3": loss_info[1][20],
                    "reliability_level_mean_4": loss_info[1][21],
                    "reliability_level_mean_5": loss_info[1][22],
                    "reliability_level_mean_6": loss_info[1][23],
                    "reliability_level_mean_7": loss_info[1][24],
                    "reliability_level_mean_8": loss_info[1][25],
                    "reliability_level_mean_9": loss_info[1][26],
                    "reliability_side0_mean": loss_info[1][27],
                    "reliability_side1_mean": loss_info[1][28],
                    "z_tokens_norm_mean": loss_info[1][29],
                    "z_tokens_norm_std": loss_info[1][30],
                    "filtered_tokens_norm_mean": loss_info[1][31],
                    "filtered_tokens_norm_std": loss_info[1][32],
                    "filtering_ratio": loss_info[1][33],
                    "exec_obs_norm_mean": loss_info[1][34],
                    "vision_token_pooled_norm_mean": loss_info[1][35],
                    "fusion_output_norm_mean": loss_info[1][36],
                    "pre_rnn_embedding_norm_mean": loss_info[1][37],
                    "actor_input_norm_mean": loss_info[1][38],
                    "rel_z_shape": loss_info[1][39],
                    "rel_side_id_shape": loss_info[1][40],
                    "rel_mid_context_shape": loss_info[1][41],
                    "mid_return_from_init_mean": loss_info[1][42],
                    "spread_ticks_mean": loss_info[1][43],
                    "mid_delta_ticks_mean": loss_info[1][44],
                    "mid_volatility_ticks_mean": loss_info[1][45],
                    "h_prev_reliability_norm_mean": loss_info[1][46],
                    "use_h_prev_in_reliability": loss_info[1][47],
                    "h_prev_used_in_reliability": loss_info[1][48],
                    "h_prev_reliability_zeroed": loss_info[1][49],
                    "total_loss_with_aux": loss_info[0],
                    "weighted_entropy_loss": loss_info[1][2] * config["ENT_COEF"][i],
                    "lambda_surv": jnp.array(config.get("lambda_surv", 0.0), dtype=jnp.float32),
                    "use_survival_loss": jnp.array(
                        float(config.get("use_survival_loss", False)),
                        dtype=jnp.float32,
                    ),
                }
                ratio_eps = 1e-8
                abs_ppo_loss = jnp.abs(loss_metrics["ppo_loss"])
                abs_aux_loss = jnp.abs(loss_metrics["aux_loss"])
                loss_metrics.update({
                    "abs_ppo_loss": abs_ppo_loss,
                    "abs_aux_loss": abs_aux_loss,
                    "aux_to_ppo_ratio": abs_aux_loss / (abs_ppo_loss + ratio_eps),
                    "survival_to_ppo_ratio": jnp.abs(loss_metrics["weighted_survival_loss"]) / (abs_ppo_loss + ratio_eps),
                })
                metrics["loss"].append(loss_metrics)


            #jax.debug.print(f"traj_batch: {len(traj_batch)}")
            #for i, tr in enumerate(traj_batch):
            #    jax.debug.print(f"traj_batch {i} reward shape: {tr.reward.shape}")
            #    jax.debug.print(f"current mean: {jnp.mean(tr.reward)}")
            #    jax.debug.print("flattened mean: ", jnp.mean(tr.reward.flatten()))

            metrics['avg_reward'] = [jnp.mean(tr.reward) for tr in traj_batch]
            metrics['avg_reward_flattened'] = [jnp.mean(tr.reward.flatten()) for tr in traj_batch]
            metrics["execution_episode_metrics"] = execution_episode_metrics
            metrics["traj_batch"] = callback_traj_batch


            if config["CALC_EVAL"]:
                def _eval_step(eval_runner_state, unused):
                    train_states, eval_env_state, last_obs, last_done,h_states, rng = eval_runner_state
                    rng, _rng = jax.random.split(rng)
                
                    actions=[]
                    pre_tanh_actions=[]
                    values=[]
                    log_probs=[]

                    for i, train_state in enumerate(train_states):
                        obs_i= last_obs[i]
                        obs_i=batchify(obs_i,config["NUM_ACTORS_PERTYPE"][i])  # Reshape to match the input shape of the network
                        obs_i_batched = jax.tree.map(lambda x: x[jnp.newaxis, :], obs_i)
                        ac_in = (
                            obs_i_batched,
                            last_done[i][jnp.newaxis, :],
                            # avail_actions,
                        )
                        h_states[i], pi, value, _, policy_aux_info = train_state.apply_fn(
                            train_state.params,
                            h_states[i],
                            ac_in,
                        )
                        values.append(value)
                        action_space = env.action_spaces[i]
                        sample_kwargs = {}
                        if isinstance(action_space, spaces.Box):
                            sample_kwargs = {
                                "action_low": action_space.low,
                                "action_high": action_space.high,
                            }
                        policy_sample = sample_policy_action(
                            pi,
                            policy_aux_info,
                            _rng,
                            **sample_kwargs,
                        )
                        log_probs.append(policy_sample.log_prob)
                        action = unbatchify(
                            policy_sample.action,
                            config["NUM_ENVS"],
                            env.multi_agent_config.number_of_agents_per_type[i],
                        )
                        pre_tanh_action = unbatchify(
                            policy_sample.pre_tanh_action,
                            config["NUM_ENVS"],
                            env.multi_agent_config.number_of_agents_per_type[i],
                        )
                        actions.append(action.squeeze())
                        pre_tanh_actions.append(pre_tanh_action.squeeze())

                        rng, _rng = jax.random.split(rng)
                        rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                





                    # STEP ENV
                    rng, _rng = jax.random.split(rng)
                    rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                    obsv, eval_env_state, reward, done, info = jax.vmap(
                        eval_env.step, in_axes=(0, 0, 0, None) # type: ignore
                    )(rng_step, eval_env_state, actions, eval_env_params)
                    done_batch=done
                    transitions=[]    

                    for i, train_state in enumerate(train_states):
                        done_batch['agents'][i] = batchify(done["agents"][i],config["NUM_ACTORS_PERTYPE"][i]).squeeze()
                        obs_batch = batchify(last_obs[i],config["NUM_ACTORS_PERTYPE"][i])
                        action_batch = batchify_action(actions[i],config["NUM_ACTORS_PERTYPE"][i])
                        pre_tanh_action_batch = batchify_action(
                            pre_tanh_actions[i],
                            config["NUM_ACTORS_PERTYPE"][i],
                        )
                        value = values[i]
                        log_prob = log_probs[i]

                        info_i={"world":info["world"],"agent":jax.tree.map(lambda x: x.reshape(config["NUM_ACTORS_PERTYPE"][i],-1),info["agents"][i])}
                        # print(f"info for agenttype {i}:", info_i)


                        transitions.append(Transition(
                            jnp.tile(done["__all__"], config["NUM_AGENTS_PER_TYPE"][i]),
                            last_done[i],
                            action_batch.squeeze(),
                            pre_tanh_action_batch.squeeze(),
                            value.squeeze(),
                            batchify(reward[i], config["NUM_ACTORS_PERTYPE"][i]).squeeze(),
                            log_prob.squeeze(),
                            obs_batch,
                            info_i,
                            # avail_actions,
                        ))
                    eval_runner_state = (train_states, eval_env_state, obsv, done_batch['agents'], h_states, rng)
                    return eval_runner_state, transitions

                rng, _rng = jax.random.split(rng)
                reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
                eval_obsv, eval_env_state = jax.vmap(eval_env.reset, in_axes=(0, None))(reset_rng, eval_env_params) # type: ignore


                eval_hstates=[]
                init_dones_agents_eval=[]
                for i, train_state in enumerate(train_states):
                    eval_hstates.append(ScannedRNN.initialize_carry(config["NUM_ACTORS_PERTYPE"][i], config["GRU_HIDDEN_DIM"]))
                    init_dones_agents_eval.append(jnp.zeros((config["NUM_ACTORS_PERTYPE"][i]), dtype=bool))


                
                eval_runner_state = (
                train_states,
                eval_env_state,
                eval_obsv,
                init_dones_agents_eval,
                eval_hstates,
                _rng,
                )
                eval_runner_state, eval_traj_batch = jax.lax.scan(
                    _eval_step, eval_runner_state, None,  config["NUM_STEPS_EVAL"]
                )
                callback_eval_traj_batch = [
                    transition._replace(
                        info={
                            "world": {
                                key: value
                                for key, value in transition.info["world"].items()
                                if key not in callback_world_exclusions
                            },
                            "agent": {
                                key: value
                                for key, value in transition.info["agent"].items()
                                if key not in callback_agent_exclusions
                            },
                        }
                    )
                    for transition in eval_traj_batch
                ]
                metrics['agents_eval'] = [jax.tree.map(
                    lambda x: x.reshape(
                        (config["NUM_STEPS_EVAL"], config["NUM_ENVS"], config["NUM_AGENTS_PER_TYPE"][i])
                    ),
                    trjbtch.info['agent']) for i, trjbtch in enumerate(callback_eval_traj_batch)]
                metrics['world_eval'] = [
                    trjbtch.info['world'] for trjbtch in callback_eval_traj_batch
                ]
                if config["CALC_EVAL"]:
                    metrics['avg_reward_eval'] = [jnp.mean(tr.reward) for tr in callback_eval_traj_batch]
                    metrics["traj_batch_eval"] = callback_eval_traj_batch

            def callback(metric):
                update_idx = int(metric["update_steps"])
                print(f"\n==================== UPDATE {update_idx} / {config['NUM_UPDATES']} ====================")
                # for loss_idx, m in enumerate(metric["loss"]):
                #         logging_dict[f"agent_{agent_name}/loss_{loss_idx}"] = m
                # Needed?

                def _loss_value(loss_metrics, key, default=0.0):
                    if key in loss_metrics:
                        return float(np.nanmean(np.array(loss_metrics[key])))
                    return float(default)

                def _loss_bool(loss_metrics, key, default=False):
                    default_value = 1.0 if default else 0.0
                    return bool(_loss_value(loss_metrics, key, default_value) >= 0.5)

                def _fmt_values(values):
                    return "[" + ", ".join(f"{float(v):.4g}" for v in np.array(values).reshape(-1)) + "]"

                exe_loss_metrics = None
                exe_agent_index = None
                print("[TRAINING LOSS]")
                for loss_agent_index, loss_metrics in enumerate(metric["loss"]):
                    loss_agent_name = agent_type_names[loss_agent_index]
                    if loss_agent_name == "EXE":
                        exe_loss_metrics = loss_metrics
                        exe_agent_index = loss_agent_index
                    print(" ".join([
                        "LOSS_DIAG",
                        f"update={update_idx}",
                        f"agent={loss_agent_name}",
                        f"{loss_agent_name}_total_loss={_loss_value(loss_metrics, 'total_loss'):.6g}",
                        f"{loss_agent_name}_ppo_loss={_loss_value(loss_metrics, 'ppo_loss'):.6g}",
                        f"{loss_agent_name}_policy_loss={_loss_value(loss_metrics, 'policy_loss'):.6g}",
                        f"{loss_agent_name}_value_loss={_loss_value(loss_metrics, 'value_loss'):.6g}",
                        f"{loss_agent_name}_entropy_loss={_loss_value(loss_metrics, 'entropy_loss'):.6g}",
                        f"{loss_agent_name}_aux_loss={_loss_value(loss_metrics, 'aux_loss'):.6g}",
                        f"{loss_agent_name}_approx_kl={_loss_value(loss_metrics, 'approx_kl'):.6g}",
                        f"{loss_agent_name}_clip_frac={_loss_value(loss_metrics, 'clip_frac'):.6g}",
                    ]))

                print("[PPO UPDATE SAFETY]")
                ppo_safety_wandb_metrics = [
                    {} for _ in metric["ppo_safety_diag"]
                ]
                for safety_agent_index, safety_diag in enumerate(
                    metric["ppo_safety_diag"]
                ):
                    stage_index = int(
                        np.asarray(safety_diag["first_nonfinite_stage"])
                    )
                    stage_name = FIRST_NONFINITE_STAGE_NAME.get(
                        stage_index,
                        f"unknown_{stage_index}",
                    )
                    print(" ".join([
                        "PPO_SAFETY",
                        f"update={update_idx}",
                        f"agent={agent_type_names[safety_agent_index]}",
                        "ppo_candidate_accepted="
                        f"{str(bool(np.asarray(safety_diag['ppo_candidate_accepted']))).lower()}",
                        "ppo_candidate_rejected_nonfinite="
                        f"{str(bool(np.asarray(safety_diag['ppo_candidate_rejected_nonfinite']))).lower()}",
                        "accepted_minibatch_count="
                        f"{int(np.asarray(safety_diag['accepted_minibatch_count']))}",
                        f"first_nonfinite_stage={stage_name}",
                        "rejected_epoch_index="
                        f"{int(np.asarray(safety_diag['rejected_epoch_index']))}",
                        "rejected_minibatch_index="
                        f"{int(np.asarray(safety_diag['rejected_minibatch_index']))}",
                    ]))

                    safety_prefix = (
                        f"agent_{agent_type_names[safety_agent_index]}/ppo_safety"
                    )
                    ppo_safety_wandb_metrics[safety_agent_index] = {
                        f"{safety_prefix}/candidate_accepted": float(
                            np.asarray(safety_diag["ppo_candidate_accepted"])
                        ),
                        f"{safety_prefix}/candidate_rejected_nonfinite": float(
                            np.asarray(
                                safety_diag["ppo_candidate_rejected_nonfinite"]
                            )
                        ),
                        f"{safety_prefix}/accepted_minibatch_count": float(
                            np.asarray(safety_diag["accepted_minibatch_count"])
                        ),
                        f"{safety_prefix}/first_nonfinite_stage_code": float(
                            stage_index
                        ),
                        f"{safety_prefix}/rejected_epoch_index": float(
                            np.asarray(safety_diag["rejected_epoch_index"])
                        ),
                        f"{safety_prefix}/rejected_minibatch_index": float(
                            np.asarray(safety_diag["rejected_minibatch_index"])
                        ),
                    }

                box_ppo_wandb_metrics = {}
                if box_ppo_diag_enabled and exe_agent_index is not None:
                    box_diag_tree = metric["box_ppo_numerics_diag"][exe_agent_index]
                    epoch_count = np.asarray(box_diag_tree["active"]).shape[0]
                    minibatch_count = np.asarray(box_diag_tree["active"]).shape[1]

                    def _box_diag_value(key, epoch, minibatch):
                        return float(np.asarray(box_diag_tree[key])[epoch, minibatch])

                    def _box_diag_values(key, epoch, minibatch):
                        return np.asarray(box_diag_tree[key])[epoch, minibatch]

                    for epoch_index in range(epoch_count):
                        for minibatch_index in range(minibatch_count):
                            stage_index = int(
                                _box_diag_value(
                                    "first_nonfinite_stage",
                                    epoch_index,
                                    minibatch_index,
                                )
                            )
                            stage_name = FIRST_NONFINITE_STAGE_NAME.get(
                                stage_index,
                                f"unknown_{stage_index}",
                            )
                            print(" ".join([
                                "BOX_PPO_NUMERICS",
                                f"update={update_idx}",
                                f"epoch={epoch_index}",
                                f"minibatch={minibatch_index}",
                                "agent=EXE",
                                f"loc_mean={_fmt_values(_box_diag_values('loc_mean', epoch_index, minibatch_index))}",
                                f"loc_std={_fmt_values(_box_diag_values('loc_std', epoch_index, minibatch_index))}",
                                f"loc_min={_fmt_values(_box_diag_values('loc_min', epoch_index, minibatch_index))}",
                                f"loc_max={_fmt_values(_box_diag_values('loc_max', epoch_index, minibatch_index))}",
                                f"log_std={_fmt_values(_box_diag_values('log_std', epoch_index, minibatch_index))}",
                                f"std={_fmt_values(_box_diag_values('std', epoch_index, minibatch_index))}",
                                f"old_log_prob_mean={_box_diag_value('old_log_prob_mean', epoch_index, minibatch_index):.6g}",
                                f"old_log_prob_min={_box_diag_value('old_log_prob_min', epoch_index, minibatch_index):.6g}",
                                f"old_log_prob_max={_box_diag_value('old_log_prob_max', epoch_index, minibatch_index):.6g}",
                                f"new_log_prob_mean={_box_diag_value('new_log_prob_mean', epoch_index, minibatch_index):.6g}",
                                f"new_log_prob_min={_box_diag_value('new_log_prob_min', epoch_index, minibatch_index):.6g}",
                                f"new_log_prob_max={_box_diag_value('new_log_prob_max', epoch_index, minibatch_index):.6g}",
                                f"logratio_mean={_box_diag_value('logratio_mean', epoch_index, minibatch_index):.6g}",
                                f"logratio_std={_box_diag_value('logratio_std', epoch_index, minibatch_index):.6g}",
                                f"logratio_p95={_box_diag_value('logratio_p95', epoch_index, minibatch_index):.6g}",
                                f"logratio_p99={_box_diag_value('logratio_p99', epoch_index, minibatch_index):.6g}",
                                f"logratio_min={_box_diag_value('logratio_min', epoch_index, minibatch_index):.6g}",
                                f"logratio_max={_box_diag_value('logratio_max', epoch_index, minibatch_index):.6g}",
                                f"ratio_mean={_box_diag_value('ratio_mean', epoch_index, minibatch_index):.6g}",
                                f"ratio_std={_box_diag_value('ratio_std', epoch_index, minibatch_index):.6g}",
                                f"ratio_p95={_box_diag_value('ratio_p95', epoch_index, minibatch_index):.6g}",
                                f"ratio_p99={_box_diag_value('ratio_p99', epoch_index, minibatch_index):.6g}",
                                f"ratio_min={_box_diag_value('ratio_min', epoch_index, minibatch_index):.6g}",
                                f"ratio_max={_box_diag_value('ratio_max', epoch_index, minibatch_index):.6g}",
                            ]))
                            print(" ".join([
                                "BOX_PPO_ACTION_DIAG",
                                f"update={update_idx}",
                                f"epoch={epoch_index}",
                                f"minibatch={minibatch_index}",
                                f"pre_tanh_min={_fmt_values(_box_diag_values('pre_tanh_min', epoch_index, minibatch_index))}",
                                f"pre_tanh_max={_fmt_values(_box_diag_values('pre_tanh_max', epoch_index, minibatch_index))}",
                                f"action_min={_fmt_values(_box_diag_values('action_min', epoch_index, minibatch_index))}",
                                f"action_max={_fmt_values(_box_diag_values('action_max', epoch_index, minibatch_index))}",
                                f"exact_low_rate={_fmt_values(_box_diag_values('exact_low_rate', epoch_index, minibatch_index))}",
                                f"exact_high_rate={_fmt_values(_box_diag_values('exact_high_rate', epoch_index, minibatch_index))}",
                                f"exact_low_count={_fmt_values(_box_diag_values('exact_low_count', epoch_index, minibatch_index))}",
                                f"exact_high_count={_fmt_values(_box_diag_values('exact_high_count', epoch_index, minibatch_index))}",
                                f"near_low_rate={_fmt_values(_box_diag_values('near_low_rate', epoch_index, minibatch_index))}",
                                f"near_high_rate={_fmt_values(_box_diag_values('near_high_rate', epoch_index, minibatch_index))}",
                                f"advantage_mean={_box_diag_value('advantage_mean', epoch_index, minibatch_index):.6g}",
                                f"advantage_std={_box_diag_value('advantage_std', epoch_index, minibatch_index):.6g}",
                                f"advantage_min={_box_diag_value('advantage_min', epoch_index, minibatch_index):.6g}",
                                f"advantage_max={_box_diag_value('advantage_max', epoch_index, minibatch_index):.6g}",
                                f"value_mean={_box_diag_value('value_mean', epoch_index, minibatch_index):.6g}",
                                f"value_std={_box_diag_value('value_std', epoch_index, minibatch_index):.6g}",
                                f"value_min={_box_diag_value('value_min', epoch_index, minibatch_index):.6g}",
                                f"value_max={_box_diag_value('value_max', epoch_index, minibatch_index):.6g}",
                            ]))
                            print(" ".join([
                                "BOX_PPO_FINITE_DIAG",
                                f"update={update_idx}",
                                f"epoch={epoch_index}",
                                f"minibatch={minibatch_index}",
                                f"actor_loc_grad_norm={_box_diag_value('actor_loc_grad_norm', epoch_index, minibatch_index):.6g}",
                                f"log_std_grad_norm={_box_diag_value('log_std_grad_norm', epoch_index, minibatch_index):.6g}",
                                f"total_grad_norm={_box_diag_value('total_grad_norm', epoch_index, minibatch_index):.6g}",
                                f"candidate_accepted={str(bool(_box_diag_value('ppo_candidate_accepted', epoch_index, minibatch_index))).lower()}",
                                f"candidate_rejected_nonfinite={str(bool(_box_diag_value('ppo_candidate_rejected_nonfinite', epoch_index, minibatch_index))).lower()}",
                                f"first_nonfinite_stage={stage_name}",
                                f"total_loss_finite={str(bool(_box_diag_value('total_loss_finite', epoch_index, minibatch_index))).lower()}",
                                f"loc_finite={str(bool(_box_diag_value('loc_finite', epoch_index, minibatch_index))).lower()}",
                                f"log_std_finite={str(bool(_box_diag_value('log_std_finite', epoch_index, minibatch_index))).lower()}",
                                f"pre_tanh_finite={str(bool(_box_diag_value('pre_tanh_finite', epoch_index, minibatch_index))).lower()}",
                                f"action_finite={str(bool(_box_diag_value('action_finite', epoch_index, minibatch_index))).lower()}",
                                f"old_log_prob_finite={str(bool(_box_diag_value('old_log_prob_finite', epoch_index, minibatch_index))).lower()}",
                                f"new_log_prob_finite={str(bool(_box_diag_value('new_log_prob_finite', epoch_index, minibatch_index))).lower()}",
                                f"logratio_finite={str(bool(_box_diag_value('logratio_finite', epoch_index, minibatch_index))).lower()}",
                                f"ratio_finite={str(bool(_box_diag_value('ratio_finite', epoch_index, minibatch_index))).lower()}",
                                f"advantage_finite={str(bool(_box_diag_value('advantage_finite', epoch_index, minibatch_index))).lower()}",
                                f"value_finite={str(bool(_box_diag_value('value_finite', epoch_index, minibatch_index))).lower()}",
                                f"gradients_finite={str(bool(_box_diag_value('gradients_finite', epoch_index, minibatch_index))).lower()}",
                                f"candidate_params_finite={str(bool(_box_diag_value('candidate_params_finite', epoch_index, minibatch_index))).lower()}",
                                f"candidate_optimizer_state_finite={str(bool(_box_diag_value('candidate_optimizer_state_finite', epoch_index, minibatch_index))).lower()}",
                            ]))
                            box_prefix = (
                                "agent_EXE/box_ppo_numerics/"
                                f"epoch_{epoch_index}/minibatch_{minibatch_index}"
                            )
                            scalar_box_metrics = (
                                "old_log_prob_mean",
                                "new_log_prob_mean",
                                "logratio_mean",
                                "logratio_std",
                                "logratio_p95",
                                "logratio_p99",
                                "logratio_min",
                                "logratio_max",
                                "ratio_mean",
                                "ratio_std",
                                "ratio_p95",
                                "ratio_p99",
                                "ratio_min",
                                "ratio_max",
                                "actor_loc_grad_norm",
                                "log_std_grad_norm",
                                "total_grad_norm",
                                "ppo_candidate_accepted",
                                "ppo_candidate_rejected_nonfinite",
                                "first_nonfinite_stage",
                            )
                            box_ppo_wandb_metrics.update({
                                f"{box_prefix}/{key}": _box_diag_value(
                                    key,
                                    epoch_index,
                                    minibatch_index,
                                )
                                for key in scalar_box_metrics
                            })
                            for vector_key in (
                                "loc_mean",
                                "loc_std",
                                "log_std",
                                "pre_tanh_min",
                                "pre_tanh_max",
                                "action_min",
                                "action_max",
                                "exact_low_count",
                                "exact_high_count",
                                "near_low_rate",
                                "near_high_rate",
                            ):
                                for dim_index, dim_value in enumerate(
                                    _box_diag_values(
                                        vector_key,
                                        epoch_index,
                                        minibatch_index,
                                    )
                                ):
                                    box_ppo_wandb_metrics[
                                        f"{box_prefix}/{vector_key}_dim_{dim_index}"
                                    ] = float(dim_value)

                print("[AUX LOSS]")
                for loss_agent_index, loss_metrics in enumerate(metric["loss"]):
                    loss_agent_name = agent_type_names[loss_agent_index]
                    print(" ".join([
                        "AUX_DIAG",
                        f"update={update_idx}",
                        f"agent={loss_agent_name}",
                        f"{loss_agent_name}_lambda_surv={_loss_value(loss_metrics, 'lambda_surv'):.6g}",
                        f"{loss_agent_name}_use_survival_loss={_loss_bool(loss_metrics, 'use_survival_loss')}",
                        f"{loss_agent_name}_survival_loss={_loss_value(loss_metrics, 'survival_loss'):.6g}",
                        f"{loss_agent_name}_weighted_survival_loss={_loss_value(loss_metrics, 'weighted_survival_loss'):.6g}",
                        f"{loss_agent_name}_reliability_loss={_loss_value(loss_metrics, 'reliability_loss'):.6g}",
                        f"{loss_agent_name}_abs_ppo_loss={_loss_value(loss_metrics, 'abs_ppo_loss'):.6g}",
                        f"{loss_agent_name}_abs_aux_loss={_loss_value(loss_metrics, 'abs_aux_loss'):.6g}",
                        f"{loss_agent_name}_aux_to_ppo_ratio={_loss_value(loss_metrics, 'aux_to_ppo_ratio'):.6g}",
                        f"{loss_agent_name}_survival_to_ppo_ratio={_loss_value(loss_metrics, 'survival_to_ppo_ratio'):.6g}",
                    ]))

                print("[RVD FORWARD]")
                if exe_loss_metrics is not None:
                    reliability_levels = [
                        _loss_value(exe_loss_metrics, f"reliability_level_mean_{idx}")
                        for idx in range(10)
                    ]
                    print(" ".join([
                        "RVD_DIAG",
                        f"update={update_idx}",
                        f"reliability_mean={_loss_value(exe_loss_metrics, 'reliability_mean'):.6g}",
                        f"reliability_std={_loss_value(exe_loss_metrics, 'reliability_std'):.6g}",
                        f"reliability_min={_loss_value(exe_loss_metrics, 'reliability_min'):.6g}",
                        f"reliability_max={_loss_value(exe_loss_metrics, 'reliability_max'):.6g}",
                        f"reliability_side0_mean={_loss_value(exe_loss_metrics, 'reliability_side0_mean'):.6g}",
                        f"reliability_side1_mean={_loss_value(exe_loss_metrics, 'reliability_side1_mean'):.6g}",
                        "tick_shift_removed=true",
                    ]))
                    print(
                        "RVD_DIAG_LEVELS "
                        f"update={update_idx} "
                        f"reliability_level_mean={_fmt_values(reliability_levels)}"
                    )
                    print(" ".join([
                        "RVD_TOKEN_DIAG",
                        f"update={update_idx}",
                        f"z_tokens_norm_mean={_loss_value(exe_loss_metrics, 'z_tokens_norm_mean'):.6g}",
                        f"z_tokens_norm_std={_loss_value(exe_loss_metrics, 'z_tokens_norm_std'):.6g}",
                        f"filtered_tokens_norm_mean={_loss_value(exe_loss_metrics, 'filtered_tokens_norm_mean'):.6g}",
                        f"filtered_tokens_norm_std={_loss_value(exe_loss_metrics, 'filtered_tokens_norm_std'):.6g}",
                        f"filtering_ratio={_loss_value(exe_loss_metrics, 'filtering_ratio'):.6g}",
                    ]))
                    print(" ".join([
                        "RVD_FUSION_DIAG",
                        f"update={update_idx}",
                        f"exec_obs_norm_mean={_loss_value(exe_loss_metrics, 'exec_obs_norm_mean'):.6g}",
                        f"vision_token_pooled_norm_mean={_loss_value(exe_loss_metrics, 'vision_token_pooled_norm_mean'):.6g}",
                        f"fusion_output_norm_mean={_loss_value(exe_loss_metrics, 'fusion_output_norm_mean'):.6g}",
                        f"pre_rnn_embedding_norm_mean={_loss_value(exe_loss_metrics, 'pre_rnn_embedding_norm_mean'):.6g}",
                        f"actor_input_norm_mean={_loss_value(exe_loss_metrics, 'actor_input_norm_mean'):.6g}",
                    ]))
                    exe_actor_count = config["NUM_ENVS"]
                    if "NUM_ACTORS_PERTYPE" in config and exe_agent_index is not None:
                        exe_actor_count = config["NUM_ACTORS_PERTYPE"][exe_agent_index]
                    z_tokens_shape = f"({config['NUM_STEPS']},{exe_actor_count},10,2,{config['FC_DIM_SIZE']})"
                    side_id_shape = f"({config['NUM_STEPS']},{exe_actor_count},10,2,1)"
                    mid_context_shape = f"({config['NUM_STEPS']},{exe_actor_count},4)"
                    use_h_prev_flag = _loss_value(exe_loss_metrics, "use_h_prev_in_reliability") >= 0.5
                    h_prev_used_flag = _loss_value(exe_loss_metrics, "h_prev_used_in_reliability") >= 0.5
                    h_prev_zeroed_flag = _loss_value(exe_loss_metrics, "h_prev_reliability_zeroed") >= 0.5
                    print(" ".join([
                        "REL_INPUT_DIAG",
                        f"update={update_idx}",
                        f"z_tokens_shape={z_tokens_shape}",
                        f"side_id_shape={side_id_shape}",
                        "side_id_order=[Ask:+1,Bid:-1]",
                        f"mid_context_shape={mid_context_shape}",
                        f"mid_return_from_init_mean={_loss_value(exe_loss_metrics, 'mid_return_from_init_mean'):.6g}",
                        f"spread_ticks_mean={_loss_value(exe_loss_metrics, 'spread_ticks_mean'):.6g}",
                        f"mid_delta_ticks_mean={_loss_value(exe_loss_metrics, 'mid_delta_ticks_mean'):.6g}",
                        f"mid_volatility_ticks_mean={_loss_value(exe_loss_metrics, 'mid_volatility_ticks_mean'):.6g}",
                        f"use_h_prev_in_reliability={str(use_h_prev_flag).lower()}",
                        f"h_prev_used_in_reliability={str(h_prev_used_flag).lower()}",
                        f"h_prev_reliability_norm_mean={_loss_value(exe_loss_metrics, 'h_prev_reliability_norm_mean'):.6g}",
                        f"h_prev_reliability_zeroed={str(h_prev_zeroed_flag).lower()}",
                        "obs_exec_used_in_reliability=false",
                        "tick_shift_removed=true",
                    ]))
                else:
                    print(f"RVD_DIAG update={update_idx} status=no_execution_loss_metrics")

                if exe_agent_index is not None and "execution_target_diag" in metric:
                    target_diag = metric["execution_target_diag"][exe_agent_index]

                    def _target_diag_value(key, default=0.0):
                        if key in target_diag:
                            return float(np.nanmean(np.array(target_diag[key])))
                        return float(default)

                    def _target_diag_values(key):
                        if key in target_diag:
                            return np.array(target_diag[key]).reshape(-1)
                        return np.zeros((10,), dtype=np.float32)

                    print("[EXECUTION-AWARE RELIABILITY TARGET]")
                    print(" ".join([
                        "EXEC_AWARE_TARGET_DIAG",
                        f"update={update_idx}",
                        f"valid_target_count={_target_diag_value('valid_target_count'):.0f}",
                        f"valid_target_rate={_target_diag_value('valid_target_rate'):.6g}",
                        f"done_masked_rate={_target_diag_value('done_masked_rate'):.6g}",
                        f"trade_buffer_saturated_rate={_target_diag_value('trade_buffer_saturated_rate'):.6g}",
                        f"q0_mean={_target_diag_value('q0_mean'):.6g}",
                        f"q0_min={_target_diag_value('q0_min'):.6g}",
                        f"q0_max={_target_diag_value('q0_max'):.6g}",
                        f"q_tau_mean={_target_diag_value('q_tau_mean'):.6g}",
                        f"cumulative_executed_mean={_target_diag_value('cumulative_executed_mean'):.6g}",
                        f"cancel_star_mean={_target_diag_value('cancel_star_mean'):.6g}",
                        f"target_mean={_target_diag_value('target_mean'):.6g}",
                        f"target_std={_target_diag_value('target_std'):.6g}",
                        f"target_min={_target_diag_value('target_min'):.6g}",
                        f"target_max={_target_diag_value('target_max'):.6g}",
                    ]))
                    print(" ".join([
                        "EXEC_AWARE_TARGET_SIDE_DIAG",
                        f"update={update_idx}",
                        f"ask_valid_count={_target_diag_value('ask_valid_count'):.0f}",
                        f"ask_mean={_target_diag_value('ask_target_mean'):.6g}",
                        f"ask_std={_target_diag_value('ask_target_std'):.6g}",
                        f"ask_min={_target_diag_value('ask_target_min'):.6g}",
                        f"ask_max={_target_diag_value('ask_target_max'):.6g}",
                        f"bid_valid_count={_target_diag_value('bid_valid_count'):.0f}",
                        f"bid_mean={_target_diag_value('bid_target_mean'):.6g}",
                        f"bid_std={_target_diag_value('bid_target_std'):.6g}",
                        f"bid_min={_target_diag_value('bid_target_min'):.6g}",
                        f"bid_max={_target_diag_value('bid_target_max'):.6g}",
                    ]))
                    print(
                        "EXEC_AWARE_TARGET_LEVELS "
                        f"update={update_idx} aggregation=current_rank_diagnostic_only "
                        f"target_ask={_fmt_values(_target_diag_values('target_level_mean_ask'))} "
                        f"target_bid={_fmt_values(_target_diag_values('target_level_mean_bid'))} "
                        f"q0_ask={_fmt_values(_target_diag_values('q0_level_mean_ask'))} "
                        f"q0_bid={_fmt_values(_target_diag_values('q0_level_mean_bid'))} "
                        f"q_tau_ask={_fmt_values(_target_diag_values('q_tau_level_mean_ask'))} "
                        f"q_tau_bid={_fmt_values(_target_diag_values('q_tau_level_mean_bid'))} "
                        f"cumulative_executed_ask={_fmt_values(_target_diag_values('cumulative_executed_level_mean_ask'))} "
                        f"cumulative_executed_bid={_fmt_values(_target_diag_values('cumulative_executed_level_mean_bid'))} "
                        f"cancel_star_ask={_fmt_values(_target_diag_values('cancel_star_level_mean_ask'))} "
                        f"cancel_star_bid={_fmt_values(_target_diag_values('cancel_star_level_mean_bid'))}"
                    )

                if exe_agent_index is not None and "reliability_alignment_diag" in metric:
                    exe_rel_diag = metric["reliability_alignment_diag"][exe_agent_index]

                    def _rel_value(key, default=0.0):
                        if key in exe_rel_diag:
                            return float(np.nanmean(np.array(exe_rel_diag[key])))
                        return float(default)

                    def _rel_values(key):
                        if key in exe_rel_diag:
                            values = np.array(exe_rel_diag[key]).reshape(-1)
                        else:
                            values = np.zeros((10,), dtype=np.float32)
                        if values.size < 10:
                            values = np.pad(values, (0, 10 - values.size), constant_values=0.0)
                        return values[:10]

                    def _shape_text(prefix):
                        ndim = int(_rel_value(f"{prefix}_ndim", 0))
                        dims = [
                            int(_rel_value(f"{prefix}_dim{axis}", -1))
                            for axis in range(max(ndim, 0))
                        ]
                        return "(" + ",".join(str(dim) for dim in dims) + ")"

                    print("[RELIABILITY ALIGNMENT]")
                    print(
                        "SURV_LOSS_SCORE_SHAPE "
                        f"update={update_idx} "
                        f"shape={_shape_text('score')} "
                        f"shape_ok={bool(_rel_value('score_shape_ok') >= 0.5)}"
                    )
                    print(
                        "SURV_LOSS_LABEL_SHAPE "
                        f"update={update_idx} "
                        f"shape={_shape_text('label')}"
                    )
                    print(
                        "SURV_LOSS_MASK_SHAPE "
                        f"update={update_idx} "
                        f"shape={_shape_text('mask')} "
                        f"shape_ok={bool(_rel_value('mask_shape_ok') >= 0.5)}"
                    )
                    if exe_agent_index is not None and "execution_target_diag" in metric:
                        exe_mid_diag_for_rel = metric["execution_target_diag"][exe_agent_index]

                        def _rel_mid_values(key):
                            if key in exe_mid_diag_for_rel:
                                values = np.array(exe_mid_diag_for_rel[key]).reshape(-1)
                            else:
                                values = np.zeros((10,), dtype=np.float32)
                            if values.size < 10:
                                values = np.pad(values, (0, 10 - values.size), constant_values=0.0)
                            return values[:10]
                    else:
                        def _rel_mid_values(_key):
                            return np.zeros((10,), dtype=np.float32)

                    print(
                        "REL_ALIGNMENT_SAMPLE "
                        f"update={update_idx} t=0 b=0 side=Ask "
                        f"price_key={_fmt_values(_rel_mid_values('ask_key_t0_b0'))} "
                        f"target={_fmt_values(_rel_values('target_t0_b0_ask'))} "
                        f"score={_fmt_values(_rel_values('score_t0_b0_ask'))} "
                        f"mask={_fmt_values(_rel_values('mask_t0_b0_ask'))}"
                    )
                    print(
                        "REL_ALIGNMENT_SAMPLE "
                        f"update={update_idx} t=0 b=0 side=Bid "
                        f"price_key={_fmt_values(_rel_mid_values('bid_key_t0_b0'))} "
                        f"target={_fmt_values(_rel_values('target_t0_b0_bid'))} "
                        f"score={_fmt_values(_rel_values('score_t0_b0_bid'))} "
                        f"mask={_fmt_values(_rel_values('mask_t0_b0_bid'))}"
                    )
                    rel_dist_components = (
                        "score",
                        "target",
                        "abs_error",
                        "signed_error",
                    )
                    rel_dist_groups = (
                        "CURRENT_RANK_TOP3_TASK_SIDE",
                        "CURRENT_RANK_TOP3_OPPOSITE_SIDE",
                        "CURRENT_RANK_FAR_TASK_SIDE",
                        "CURRENT_RANK_FAR_OPPOSITE_SIDE",
                    )
                    rel_dist_stats = (
                        "count",
                        "mean",
                        "std",
                        "min",
                        "p10",
                        "p25",
                        "p50",
                        "p75",
                        "p90",
                        "p95",
                        "p99",
                        "max",
                        "zero_rate",
                        "pos_rate_0p1",
                        "pos_rate_0p3",
                        "pos_rate_0p5",
                        "pos_rate_0p7",
                    )
                    for component_name in rel_dist_components:
                        for group_name in rel_dist_groups:
                            fields = [
                                "REL_DIST",
                                f"update={update_idx}",
                                f"component={component_name}",
                                f"group={group_name}",
                            ]
                            fields.extend(
                                f"{stat_name}={_rel_value(f'rel_dist_{component_name}_{group_name}_{stat_name}'):.6g}"
                                for stat_name in rel_dist_stats
                            )
                            print(" ".join(fields))

                print("[GRADIENTS]")
                grad_wandb_metrics = {}
                phasic_wandb_metrics = {}
                if exe_agent_index is not None:
                    execution_grad_diag = metric["grad_interaction_diag"][
                        exe_agent_index
                    ]
                    grad_lines, grad_values = format_gradient_interaction_diagnostics(
                        execution_grad_diag,
                        update=update_idx,
                        agent="EXE",
                        optimization_mode=phasic_settings.mode,
                    )
                    for line in grad_lines:
                        print(line)
                    grad_wandb_metrics = {
                        f"agent_EXE/gradient_interaction/{key}": value
                        for key, value in grad_values.items()
                    }
                    survival_loss_pre_ppo = None
                    if (
                        phasic_settings.mode == "phasic"
                        and float(
                            np.mean(
                                np.asarray(
                                    execution_grad_diag["grad_diag_active"]
                                )
                            )
                        )
                        >= 0.5
                    ):
                        survival_loss_pre_ppo = execution_grad_diag[
                            "survival_loss_pre_ppo"
                        ]
                    phasic_line, phasic_values = format_phasic_aux_diagnostics(
                        metric["phasic_aux_diag"][exe_agent_index],
                        update=update_idx,
                        mode=phasic_settings.mode,
                        survival_loss_pre_ppo=survival_loss_pre_ppo,
                    )
                    print(phasic_line)
                    phasic_wandb_metrics = {
                        f"agent_EXE/phasic_aux/{key}": value
                        for key, value in phasic_values.items()
                    }
                else:
                    print(
                        f"GRAD_DIAG update={update_idx} status=not_applicable "
                        "reason=no_execution_agent"
                    )

                for agent_index, tr in enumerate(metric["traj_batch"]):
                    agent_name = agent_type_names[agent_index]

                    action_distribution = {}
                    actions = np.array(tr.action).flatten()
                    if isinstance(env.action_spaces[agent_index], spaces.Discrete):
                        unique_actions, counts = np.unique(actions, return_counts=True)
                        tot_counts=sum(counts)
                        # Add each action count to the dictionary with a unique key
                        for a, c in zip(unique_actions, counts):
                            action_distribution[f"agent_{agent_name}/action_{int(a)}"] = c/tot_counts*100
                    else:
                        action_distribution[f"agent_{agent_name}/action_mean"] = float(np.mean(actions))
                        action_distribution[f"agent_{agent_name}/action_std"] = float(np.std(actions))
                    logging_dict = {
                        # TODO: Log the quantities of interest. Keep it trivial for now.
                        "env_step": (metric["update_steps"]+1)
                        * config["NUM_ENVS"]
                        * config["NUM_STEPS"],
                        **{f"agent_{agent_name}/{j}": m for j, m in metric["loss"][agent_index].items()},
                        **{f"agent_{agent_name}/reward": metric["avg_reward"][agent_index]},
                        **action_distribution
                    }
                    if agent_name == "EXE":
                        logging_dict.update(grad_wandb_metrics)
                        logging_dict.update(phasic_wandb_metrics)
                        logging_dict.update(box_ppo_wandb_metrics)
                        exe_episode_metrics = metric["execution_episode_metrics"]
                        logging_dict.update({
                            "agent_EXE/episode_count": int(
                                np.asarray(exe_episode_metrics.episode_count)
                            ),
                            "agent_EXE/episode_return_mean": float(
                                np.asarray(exe_episode_metrics.episode_return_mean)
                            ),
                            "agent_EXE/terminal_quant_left_mean": float(
                                np.asarray(exe_episode_metrics.terminal_quant_left_mean)
                            ),
                            "agent_EXE/terminal_fill_ratio_mean": float(
                                np.asarray(exe_episode_metrics.terminal_fill_ratio_mean)
                            ),
                            "agent_EXE/full_completion_rate": float(
                                np.asarray(exe_episode_metrics.full_completion_rate)
                            ),
                            "agent_EXE/realized_is_bps_mean": float(
                                np.asarray(exe_episode_metrics.realized_is_bps_mean)
                            ),
                            "agent_EXE/forced_liquidation_is_bps_mean": float(
                                np.asarray(
                                    exe_episode_metrics.forced_liquidation_is_bps_mean
                                )
                            ),
                            "agent_EXE/twap_forced_liquidation_is_bps_mean": float(
                                np.asarray(
                                    exe_episode_metrics.twap_forced_liquidation_is_bps_mean
                                )
                            ),
                            "agent_EXE/twap_advantage_bps_mean": float(
                                np.asarray(exe_episode_metrics.twap_advantage_bps_mean)
                            ),
                            "agent_EXE/twap_win_rate": float(
                                np.asarray(exe_episode_metrics.twap_win_rate)
                            ),
                        })
                    logging_dict.update(
                        ppo_safety_wandb_metrics[agent_index]
                    )
                
                    
                    for key, value in tr.info['agent'].items():
                    # Check if value is a numpy array or jax array and has elements
                        if isinstance(value, (jnp.ndarray, np.ndarray)) and value.size > 0:
                            flat_value = np.array(value).flatten()
                            if flat_value.size > 0:
                                # Get agent short_name from config
                                logging_dict[f"agent_{agent_name}/{key}_mean"] = float(np.mean(flat_value))
                                logging_dict[f"agent_{agent_name}/{key}_std"] = float(np.std(flat_value))

                    if agent_name == "EXE":
                        agent_info = tr.info['agent']
                        rewards = np.array(tr.reward).reshape(-1)
                        if "execution_target_diag" in metric:
                            target_diag = metric["execution_target_diag"][agent_index]
                            for key in (
                                "valid_target_count",
                                "valid_target_rate",
                                "done_masked_rate",
                                "trade_buffer_saturated_rate",
                                "q0_mean",
                                "q_tau_mean",
                                "cumulative_executed_mean",
                                "cancel_star_mean",
                                "target_mean",
                                "target_std",
                                "target_min",
                                "target_max",
                                "ask_valid_count",
                                "ask_target_mean",
                                "ask_target_std",
                                "bid_valid_count",
                                "bid_target_mean",
                                "bid_target_std",
                            ):
                                logging_dict[f"agent_{agent_name}/reliability_target/{key}"] = float(
                                    np.nanmean(np.asarray(target_diag[key]))
                                )

                        def _flat_info(key, default=0.0):
                            if key in agent_info:
                                return np.array(agent_info[key]).reshape(-1)
                            return np.full(rewards.shape, default, dtype=np.float32)

                        def _masked_mean(values, mask):
                            return float(np.mean(values[mask])) if np.any(mask) else float("nan")

                        def _fmt(values):
                            return "[" + ",".join(f"{float(v):.4g}" for v in np.array(values).reshape(-1)) + "]"

                        def _stats_text(prefix, values):
                            return (
                                f"{prefix}_mean={float(np.nanmean(values)):.6g} "
                                f"{prefix}_min={float(np.nanmin(values)):.6g} "
                                f"{prefix}_max={float(np.nanmax(values)):.6g}"
                            )

                        doom_quant = _flat_info("doom_quant")
                        quant_left = _flat_info("quant_left")
                        agent_quant = _flat_info("agentQuant", np.nan)
                        agent_quant_step = _flat_info("agentQuant_step", np.nan)
                        V_RL_k = _flat_info("V_RL_k", np.nan)
                        V_base_k = _flat_info("V_base_k", np.nan)
                        r_comp_raw = _flat_info("r_comp_raw", np.nan)
                        r_comp = _flat_info("r_comp", np.nan)
                        r_mimic = _flat_info("r_mimic", np.nan)
                        r_terminal = _flat_info("r_terminal", np.nan)
                        reward_main = _flat_info("reward_main", np.nan)
                        reward_info_values = _flat_info("reward", np.nan)
                        quant_left_before_unwind = _flat_info("quant_left_before_unwind", np.nan)
                        denom_comp = _flat_info("denom_comp", np.nan)
                        denom_base = _flat_info("denom_base", np.nan)
                        denom_task = _flat_info("denom_task", np.nan)
                        reward_window_count = _flat_info("reward_window_count", np.nan)
                        target_quants_l1 = _flat_info("target_quants_l1", np.nan)
                        target_quants_l2 = _flat_info("target_quants_l2", np.nan)
                        target_quants_l3 = _flat_info("target_quants_l3", np.nan)
                        target_quants_sum = _flat_info("target_quants_sum", np.nan)
                        action_msg_volume = _flat_info("action_msg_volume", np.nan)
                        cancel_msg_count = _flat_info("cancel_msg_count", np.nan)
                        cancel_msg_volume = _flat_info("cancel_msg_volume", np.nan)
                        is_sell_task = _flat_info("is_sell_task").astype(bool)
                        action_values = np.array(tr.action)
                        if isinstance(env.action_spaces[agent_index], spaces.Discrete):
                            action_2d = action_values.reshape(-1, 1)
                        else:
                            action_2d = action_values.reshape(-1, action_values.shape[-1])

                        avg_reward_exe = float(np.array(metric['avg_reward'][agent_index]))
                        action_mean = np.mean(action_2d, axis=0)
                        action_min = np.min(action_2d, axis=0)
                        action_max = np.max(action_2d, axis=0)

                        print("[EXECUTION REWARD]")
                        print(
                            "EXE_REWARD_DIAG "
                            f"update={update_idx} "
                            f"avg_reward_EXE={avg_reward_exe:.6g} "
                            f"V_RL_k_mean={float(np.nanmean(V_RL_k)):.6g} "
                            f"V_base_k_mean={float(np.nanmean(V_base_k)):.6g} "
                            f"r_comp_mean={float(np.nanmean(r_comp)):.6g} "
                            f"r_mimic_mean={float(np.nanmean(r_mimic)):.6g} "
                            f"r_terminal_mean={float(np.nanmean(r_terminal)):.6g} "
                            f"reward_main_mean={float(np.nanmean(reward_main)):.6g} "
                            f"reward_mean={float(np.nanmean(reward_info_values)):.6g} "
                            f"buy_reward={_masked_mean(rewards, ~is_sell_task):.6g} "
                            f"sell_reward={_masked_mean(rewards, is_sell_task):.6g}"
                        )
                        print(
                            "EXE_REWARD_RANGE_DIAG "
                            f"update={update_idx} "
                            f"{_stats_text('r_comp_raw', r_comp_raw)} "
                            f"{_stats_text('r_comp', r_comp)} "
                            f"{_stats_text('r_mimic', r_mimic)} "
                            f"{_stats_text('r_terminal', r_terminal)} "
                            f"{_stats_text('reward_main', reward_main)} "
                            f"{_stats_text('reward', reward_info_values)} "
                            f"{_stats_text('denom_comp', denom_comp)} "
                            f"{_stats_text('denom_base', denom_base)} "
                            f"{_stats_text('denom_task', denom_task)} "
                            f"{_stats_text('reward_window_count', reward_window_count)}"
                        )

                        print("[ACTION / EXECUTION]")
                        print(
                            "EXE_ACTION_DIAG "
                            f"update={update_idx} "
                            f"action_mean={_fmt(action_mean)} "
                            f"action_min={_fmt(action_min)} "
                            f"action_max={_fmt(action_max)} "
                            f"target_quants_mean={float(np.nanmean(target_quants_sum / 3.0)):.6g} "
                            f"target_quants_l1_mean={float(np.nanmean(target_quants_l1)):.6g} "
                            f"target_quants_l2_mean={float(np.nanmean(target_quants_l2)):.6g} "
                            f"target_quants_l3_mean={float(np.nanmean(target_quants_l3)):.6g} "
                            f"action_msg_volume_mean={float(np.nanmean(action_msg_volume)):.6g} "
                            f"agentQuant_step_mean={float(np.nanmean(agent_quant_step)):.6g} "
                            f"cancel_msg_count_mean={float(np.nanmean(cancel_msg_count)):.6g} "
                            f"cancel_msg_volume_mean={float(np.nanmean(cancel_msg_volume)):.6g}"
                        )
                        print(
                            "EXE_DIAG "
                            f"update={update_idx} "
                            f"avg_reward={avg_reward_exe:.6g} "
                            f"doom_count={int(np.sum(doom_quant > 0))} "
                            f"doom_quant_mean={float(np.mean(doom_quant)):.6g} "
                            f"doom_quant_max={float(np.max(doom_quant)):.6g} "
                            f"quant_left_mean={float(np.mean(quant_left)):.6g} "
                            f"quant_left_max={float(np.max(quant_left)):.6g} "
                            f"terminal_left_before_unwind_mean={float(np.nanmean(quant_left_before_unwind)):.6g} "
                            f"agentQuant_mean={float(np.nanmean(agent_quant)):.6g} "
                            f"agentQuant_step_mean={float(np.nanmean(agent_quant_step)):.6g}"
                        )
                    
                    # Process world info if available
                    if 'world' in tr.info and tr.info['world']:
                        for key, value in tr.info['world'].items():
                            if isinstance(value, (jnp.ndarray, np.ndarray)) and value.size > 0:
                                flat_value = np.array(value).flatten()
                                if flat_value.size > 0:
                                    logging_dict[f"world/{key}_mean"] = float(np.mean(flat_value))

                    # Add evaluation metrics if available
                    if config["CALC_EVAL"] and "traj_batch_eval" in metric:
                        tr= metric["traj_batch_eval"][agent_index]
                        agent_name = agent_type_names[agent_index]
                        action_distribution = {}
                        actions = np.array(tr.action).flatten()
                        if isinstance(env.action_spaces[agent_index], spaces.Discrete):
                            unique_actions, counts = np.unique(actions, return_counts=True)
                            tot_counts=sum(counts)
                            # Add each action count to the dictionary with a unique key
                            for a, c in zip(unique_actions, counts):
                                action_distribution[f"eval_agent_{agent_name}/action_{int(a)}"] = c/tot_counts*100
                        else:
                            action_distribution[f"eval_agent_{agent_name}/action_mean"] = float(np.mean(actions))
                            action_distribution[f"eval_agent_{agent_name}/action_std"] = float(np.std(actions))
                        logging_dict.update(action_distribution)
                        for key, value in tr.info['agent'].items():
                            if isinstance(value, (jnp.ndarray, np.ndarray)) and value.size > 0:
                                flat_value = np.array(value).flatten()
                                if flat_value.size > 0:
                                    logging_dict[f"eval_agent_{agent_name}/{key}_mean"] = float(np.mean(flat_value))
                                    logging_dict[f"eval_agent_{agent_name}/{key}_std"] = float(np.std(flat_value))
                        
                        # Process world eval info if available
                        if 'world' in tr.info and tr.info['world']:
                            for key, value in tr.info['world'].items():
                                if isinstance(value, (jnp.ndarray, np.ndarray)) and value.size > 0:
                                    flat_value = np.array(value).flatten()
                                    if flat_value.size > 0:
                                        logging_dict[f"eval_world/{key}_mean"] = float(np.mean(flat_value))

                        logging_dict.update({
                            **{f"eval_agent_{agent_name}/reward": metric["avg_reward_eval"][agent_index]},
                        })
                    if config["WANDB_MODE"]!= "disabled":
                        wandb.log(logging_dict)


                for agent_index, agent_value in enumerate(metric["avg_reward"]):
                    agent_name = agent_type_names[agent_index]
                    print(f"avg_reward_{agent_name} {agent_value}")
                summary_parts = [
                    "SUMMARY_DIAG",
                    f"update={update_idx}",
                    "status=update_completed",
                ]
                for agent_index, agent_value in enumerate(metric["avg_reward"]):
                    agent_name = agent_type_names[agent_index]
                    summary_parts.append(f"avg_reward_{agent_name}={float(np.array(agent_value)):.6g}")
                exe_episode_metrics = metric["execution_episode_metrics"]
                summary_parts.extend([
                    f"exe_episode_count={int(np.asarray(exe_episode_metrics.episode_count))}",
                    f"exe_episode_return_mean={float(np.asarray(exe_episode_metrics.episode_return_mean)):.6g}",
                    f"exe_terminal_quant_left_mean={float(np.asarray(exe_episode_metrics.terminal_quant_left_mean)):.6g}",
                    f"exe_terminal_fill_ratio_mean={float(np.asarray(exe_episode_metrics.terminal_fill_ratio_mean)):.6g}",
                    f"exe_full_completion_rate={float(np.asarray(exe_episode_metrics.full_completion_rate)):.6g}",
                    f"exe_realized_is_bps_mean={float(np.asarray(exe_episode_metrics.realized_is_bps_mean)):.6g}",
                    f"exe_forced_liquidation_is_bps_mean={float(np.asarray(exe_episode_metrics.forced_liquidation_is_bps_mean)):.6g}",
                    f"exe_twap_forced_liquidation_is_bps_mean={float(np.asarray(exe_episode_metrics.twap_forced_liquidation_is_bps_mean)):.6g}",
                    f"exe_twap_advantage_bps_mean={float(np.asarray(exe_episode_metrics.twap_advantage_bps_mean)):.6g}",
                    f"exe_twap_win_rate={float(np.asarray(exe_episode_metrics.twap_win_rate)):.6g}",
                ])
                print("[SUMMARY]")
                print(" ".join(summary_parts))

            metrics["update_steps"] = update_steps
            jax.experimental.io_callback(callback, None, metrics)
            update_steps = update_steps + 1
            runner_state = _pack_runner_state(
                train_states,
                env_state,
                last_obs,
                last_dones,
                hstates_new,
                rng,
                aux_opt_state,
                running_exe_episode_return,
            )

            print("Finished compiling")
            # jax.profiler.save_device_memory_profile(f"memory_{update_steps}.prof")
            return (runner_state, update_steps), metrics

        rng, _rng = jax.random.split(rng)
        running_exe_episode_return = jnp.zeros(
            (execution_actor_count or 0,),
            dtype=jnp.float32,
        )
        runner_state = _pack_runner_state(
            train_states,
            env_state,
            obsv,
            init_dones_agents,
            hstates,
            _rng,
            aux_opt_state,
            running_exe_episode_return,
        )

        jitted_update_step = jax.jit(_update_step)
        
        orbax_checkpointer = oxcp.PyTreeCheckpointer()
        keep_period = max(1, config["NUM_UPDATES"] // 2)
        options = oxcp.CheckpointManagerOptions(max_to_keep=2, create=True, keep_period=keep_period)
        checkpoint_path = os.path.abspath(
            f'./checkpoints/MARLCheckpoints/{config["PROJECT"]}/{(run.name if run.name else run.id) if run else "GENERIC_RUN"}'
        )

        checkpoint_manager = oxcp.CheckpointManager(
            checkpoint_path, # Dùng biến path vừa tạo
            orbax_checkpointer, 
            options
        )


        
        updates=0
        for i in range(config["NUM_UPDATES"]):
            print(f"Update step {i+1}/{config['NUM_UPDATES']}")
            # Run the update step:
            if config["world_config"]["debug_mode"] == True:
                if i>2 and i<4:
                    jax.profiler.start_trace("/tmp/profile-data")
            (runner_state,updates),metrics=jitted_update_step((runner_state,updates),env_params,eval_env_params,None)
            if config["world_config"]["debug_mode"] == True:
                if i>2 and i<4:
                    jax.block_until_ready((runner_state,updates,metrics))
                    jax.profiler.stop_trace()
            print(f"Update step {updates} completed with metrics {metrics['avg_reward']}")
            if config["CALC_EVAL"]:
                ckpt = {
                    'model': runner_state[0],  # train_states
                    # 'config': config if isinstance(config, dict) else config.as_dict(),
                    'metrics': {
                        'train_rewards': metrics["avg_reward"],
                        'eval_rewards': metrics["avg_reward_eval"],
                        }
                }
            else:
                ckpt = {
                    'model': runner_state[0],  # train_states
                    # 'config': config if isinstance(config, dict) else config.as_dict(),
                    'metrics': {
                        'train_rewards': metrics["avg_reward"],
                        }
                }
            if phasic_mode:
                ckpt["aux_optimizer_state"] = _unpack_runner_state(runner_state)[6]
            print(f"Saving checkpoint {updates} with metrics {metrics['avg_reward']}")
            save_args = orbax_utils.save_args_from_target(ckpt)
            checkpoint_manager.save(updates, ckpt, save_kwargs={"save_args": save_args})
            del metrics
            gc.collect()
        

        checkpoint_manager.wait_until_finished()

        # runner_state, metrics = jax.lax.scan(
        #     _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        # )
        
        
        return {"runner_state": runner_state}

    return train


@hydra.main(version_base=None, config_path="config", config_name="ippo_rnn_JAXMARL_2player")
def main(config):
    print("MultiAgentConfig", MultiAgentConfig().world_config)
    env_config = OmegaConf.structured(
        MultiAgentConfig(
            number_of_agents_per_type=config["NUM_AGENTS_PER_TYPE"]
        )
    )

    env_config_plain = OmegaConf.create(
        OmegaConf.to_container(env_config, resolve=True)
    )
    hydra_config_plain = OmegaConf.create(
        OmegaConf.to_container(config, resolve=True)
    )

    final_config = OmegaConf.merge(
        env_config_plain,
        hydra_config_plain,
    )
    config = OmegaConf.to_container(final_config, resolve=True)

    print(config)

    def sweep_fun():
        print(f"WANDB CONFIG PRIOR {wandb.config}")

        run=wandb.init(
            entity=config["ENTITY"], # type: ignore
            project=config["PROJECT"], # type: ignore
            tags=["IPPO", "RNN"], # type: ignore
            mode=config["WANDB_MODE"], # type: ignore
            allow_val_change=True,
        )

        sweep_overrides = OmegaConf.to_container(OmegaConf.create(dict(wandb.config)), resolve=True)
        active_config = OmegaConf.to_container(
            OmegaConf.merge(OmegaConf.create(copy.deepcopy(config)), OmegaConf.create(sweep_overrides)),
            resolve=True,
        )
        run.config.update(active_config, allow_val_change=True)

        
        # params_file_name = f'params_file_{wandb.run.name}_{datetime.datetime.now().strftime("%m-%d_%H-%M")}'
        
        
        # print(f"WANDB CONFIG {wandb.config}")
        # +++++ Single GPU +++++
        

        rng = jax.random.PRNGKey(active_config["SEED"])

        print("wandb.config", active_config)

        
        # print("+++++++++++ Training turned off whilst debugging wandb ++++++++++++")
        


        if active_config["Timing"]:
            #print("Start compilation")
            #train_jit = jax.jit(make_train(wandb.config)).lower(rng).compile()
            print("Start training")
            start_time = time.time()


        train_fun = make_train(active_config)
        out = train_fun(rng,run)
        # train_state = out['runner_state'][0] # runner_state.train_state
        # params = train_state.params

        if active_config["Timing"]:
            end_time = time.time()
            elapsed = end_time - start_time
            total_steps = active_config["TOTAL_TIMESTEPS"]
            agents_per_type = active_config["NUM_AGENTS_PER_TYPE"]
            num_data_msgs = active_config.get("n_data_msg_per_step", None)
            num_envs = active_config["NUM_ENVS"]

            # Print results
            print(f"Total steps: {total_steps}")
            print(f"Elapsed time: {elapsed} seconds")
            print(f"Steps per second: {total_steps / elapsed}")
            print(f"Agents per type: {agents_per_type}")
            print(f"Num data messages: {num_data_msgs}")
            print(f"Num envs: {num_envs}")

            # Save to CSV
            results = {
                "total_steps": [total_steps],
                "elapsed_seconds": [elapsed],
                "steps_per_second": [total_steps / elapsed],
                "agents_per_type": [str(agents_per_type)],
                "num_data_msgs": [num_data_msgs],
                "num_envs": [num_envs],
            }
            df = pd.DataFrame(results)
            csv_path = "timing_results.csv"
            # Append if file exists, else write header
            with open(csv_path, "w", newline="") as f:
                df.to_csv(f, index=False)
        # # Save the params to a file using flax.serialization.to_bytes
        # with open(params_file_name, 'wb') as f:
        #     f.write(flax.serialization.to_bytes(params))
        #     print(f"params saved")

        # Load the params from the file using flax.serialization.from_bytes
        # with open(params_file_name, 'rb') as f:
        #     restored_params = flax.serialization.from_bytes(flax.core.frozen_dict.FrozenDict, f.read())
        #     print(f"params restored")
        # Clean up resources after training
        del out
        gc.collect()

        # Force JAX to release memory
        jax.clear_caches()
        jax.local_devices()  # This can help trigger cleanup of device buffers
        run.finish()

    if config["WANDB_MODE"] == "disabled":
        print("WANDB_MODE=disabled: running single-run without wandb.sweep()/wandb.agent().")
        rng = jax.random.PRNGKey(config["SEED"])
        train_fun = make_train(config)
        out = train_fun(rng, None)
        del out
        gc.collect()
        jax.clear_caches()
        return

    # NOTE: Sweep Parameters will override the config file, but cannot be used to override any environment params currently. 
    # This latter option will require some careful thought on how best to implement - due to to variable number of agent types.
    sweep_parameters = {
        # "LR": {"values": [config["LR"]]},
        # "NUM_STEPS": {"values": [32,config["NUM_STEPS"], 512]},
        #"GAMMA": {"values": [config["GAMMA"], [0.99,0.99]]},
        # "LR": {"values": [config["LR"], [0.004,0.004], [0.00004,0.00004]]},
        #"ENT_COEF": {"values": [config["ENT_COEF"], [0.1,0.1], [0.05,0.05]]},
        # "UPDATE_EPOCHS": {"values": [config["UPDATE_EPOCHS"], 8]},
        #"CLIP_EPS": {"values": [config["CLIP_EPS"], 0.3, 0.1]},
        #"VF_COEF": {"values": [config["VF_COEF"], [1e-6,1e-7], [1e-9,1e-8]]},
        #"FC_DIM_SIZE": {"values": [config["FC_DIM_SIZE"], 256]},
       # "NUM_AGENTS_PER_TYPE": {"values": [config["NUM_AGENTS_PER_TYPE"], [2,2], [10,10]]},
        "SEED": {"values": [config["SEED"],34]},
       #"NUM_ENVS": {"values": [config["NUM_ENVS"]]},
       #"NUM_STEPS": {"values": [config["NUM_STEPS"], 32, 4]},
       
        
        "AGENT_CONFIGS" : {"parameters": {
                        "MarketMaking" : {"parameters":
                                        {"inv_penalty": {"values":['quadratic']}, # "none" "linear "quadratic"
                                        "skew_multiplier": {"values":[10]},
                                        "action_space": {"values":["fixed_quants"]}, #"spread_skew",,"fixed_quants"simple
                                        "reward_space" : {"values":["spooner","buy_sell_pnl"]}, # "spooner"buy_sell_pnl
                                        "reference_price_portfolio_value":{"values":["best_bid_ask"]}, #best_bid_ask "mid"
                        }},
                        "Execution" : {"parameters": {"reward_lambda": {"values":[0.5]},
                                                      "observation_space": {"values": ["execution_policy"]}, #20 on fixed quants
                                                      "action_space": {"values":["policy_blending"]}, #fixed_quants,fixed_quants_complex
                                                      "task_size": {"values":[600]},
                                                      "doom_price_penalty": {"values":[0.1]},
                        }},
        }}
    }


    sweep_config={
        "method": "grid",
        "parameters": sweep_parameters,
    }
    print(sweep_config)
    sweep_id = wandb.sweep(sweep=sweep_config, project=config["PROJECT"],entity=config["ENTITY"])
    print(sweep_id)
    wandb.agent(sweep_id, function=sweep_fun, count=500,)


    sys.exit(0)

@hydra.main(version_base=None, config_path="config", config_name="ippo_rnn_JAXMARL_2player")
def seperate_main(config):
    print("MultiAgentConfig", MultiAgentConfig().world_config)
    env_config=OmegaConf.structured(MultiAgentConfig(number_of_agents_per_type=config["NUM_AGENTS_PER_TYPE"]))
    final_config=OmegaConf.merge(config,env_config)
    config = OmegaConf.to_container(final_config)

    # jax.profiler.start_trace("/tmp/profile-data")

    
    rng = jax.random.PRNGKey(0)

    train_fun = make_train(config)
    # print("+++++++++++ Training turned off whilst debugging wandb ++++++++++++")
    out = train_fun(rng)



    # out=jax.block_until_ready(out)  # Ensure the computation is complete before proceeding
    # (dummy * dummy).block_until_ready()
    # jax.profiler.stop_trace()


if __name__ == "__main__":
    main()
