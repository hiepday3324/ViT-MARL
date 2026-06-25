"""
Based on PureJaxRL Implementation of PPO
"""

import os
import sys
# =====================================================================
# THÊM KHỐI NÀY VÀO ĐẦU FILE (DƯỚI IMPORT OS, SYS)
# =====================================================================
if "JAX_COORDINATOR" in os.environ:
    import jax
    jax.distributed.initialize(
        coordinator_address=os.environ["JAX_COORDINATOR"],
        num_processes=int(os.environ["JAX_NUM_PROCESSES"]),
        process_id=int(os.environ["JAX_PROCESS_ID"]),
        initialization_timeout=600,
    )
    print(f"--- [Host {jax.process_index()}] Khởi tạo Cluster thành công! Total devices: {jax.device_count()} ---", flush=True)
# =====================================================================
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
from flax import serialization, jax_utils
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
from gymnax_exchange.networks.gate_fusion import EMASmoothing, StableGatedCrossAttention
from gymnax_exchange.networks.reliability_head import LevelWiseReliabilityHead
from gymnax_exchange.networks.vision_agent import VisionAgent, supervised_contrastive_loss
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
        obs_exec_t, obs_exec_smoothed_t, z_tokens_t, done_t, tick_shift_t = x
        rnn_state = jnp.where(done_t[:, jnp.newaxis], jnp.zeros_like(carry), carry)

        reliability = LevelWiseReliabilityHead(
            hidden_dim=self.config.get("reliability_hidden_dim", self.config["FC_DIM_SIZE"]),
            gate_epsilon=self.config.get("reliability_gate_epsilon", 0.1),
        )
        reliability_scores_t, filtered_tokens_t = reliability(
            z_tokens=z_tokens_t,
            obs_exec=obs_exec_t,
            h_prev=rnn_state,
            tick_shift=tick_shift_t,
        )

        fusion = StableGatedCrossAttention(d_model=self.config["FC_DIM_SIZE"])
        fused_t = fusion(obs_exec_smoothed_t, filtered_tokens_t)
        embedding_t = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(fused_t)
        embedding_t = nn.relu(embedding_t)

        new_rnn_state, y_t = nn.GRUCell(features=self.config["FC_DIM_SIZE"])(rnn_state, embedding_t)
        return new_rnn_state, (y_t, reliability_scores_t)

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

            vision_encoder = VisionAgent(embed_dim=self.config["FC_DIM_SIZE"])
            z_tokens = vision_encoder(obs_vision, return_tokens=True)
            z_vision = jnp.mean(z_tokens, axis=(-3, -2))

            ema_module = EMASmoothing(alpha = 0.5)
            obs_exec_smoothed = ema_module(obs_exec)

            use_reliability_head = self.config.get("use_reliability_head", False)
            if use_reliability_head:
                use_tick_shift = self.config.get("use_tick_shift", False)
                tick_shift = obs.get("tick_shift", None) if use_tick_shift else None
                if tick_shift is None:
                    tick_shift = jnp.zeros((*obs_exec.shape[:2], 1), dtype=obs_exec.dtype)

                hidden, (embedding, reliability_scores) = ReliabilityFusionRNN(config=self.config)(
                    hidden,
                    (obs_exec, obs_exec_smoothed, z_tokens, dones, tick_shift),
                )
                aux_info = {"reliability_scores": reliability_scores}
            else:
                fusion = StableGatedCrossAttention(d_model=self.config["FC_DIM_SIZE"])
                fused_obs = fusion(obs_exec_smoothed, z_tokens)
                embedding = nn.Dense(
                    self.config["FC_DIM_SIZE"], kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
                )(fused_obs)
                embedding = nn.relu(embedding)

                rnn_in = (embedding, dones)

                hidden, embedding = ScannedRNN()(hidden, rnn_in)
                aux_info = {
                    "reliability_scores": jnp.zeros((*z_tokens.shape[:-1], 1), dtype=z_tokens.dtype)
                }
        else:
            fused_obs = obs
            z_vision = jnp.zeros((1,))
            embedding = nn.Dense(
                self.config["FC_DIM_SIZE"], kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
            )(fused_obs)
            embedding = nn.relu(embedding)

            rnn_in = (embedding, dones)

            hidden, embedding = ScannedRNN()(hidden, rnn_in)
            aux_info = {
                "reliability_scores": jnp.zeros((*embedding.shape[:-1], 1, 1), dtype=embedding.dtype)
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


def masked_bce_loss(reliability_scores, labels, mask, eps=1e-8):
    if reliability_scores.ndim == labels.ndim + 1:
        reliability_scores = jnp.squeeze(reliability_scores, axis=-1)

    r = jnp.clip(reliability_scores, eps, 1.0 - eps)
    y = labels.astype(jnp.float32)
    m = mask.astype(jnp.float32)

    bce = -(y * jnp.log(r) + (1.0 - y) * jnp.log(1.0 - r))
    return jnp.sum(bce * m) / jnp.maximum(jnp.sum(m), eps)


def build_liquidity_survival_targets(
    vision_obs,
    mid_prices,
    *,
    tick_size,
    survival_delta_steps,
    survival_min_volume,
    survival_ratio,
    num_steps,
):
    """Build side-aware liquidity survival labels from LOB vision frames.

    The returned labels and masks are shaped ``(time, batch, levels, sides)``.
    Side 0 is Ask and side 1 is Bid; Ask/Bid are preserved as separate
    reliability targets.
    """
    vision_obs = jnp.asarray(vision_obs, dtype=jnp.float32)
    mid_prices = jnp.asarray(mid_prices, dtype=jnp.float32)

    if mid_prices.ndim == 1:
        mid_prices = mid_prices[:, None]
    if mid_prices.shape[1] != vision_obs.shape[1]:
        if vision_obs.shape[1] % mid_prices.shape[1] != 0:
            raise ValueError(
                "Cannot broadcast world mid_prices to actor vision observations: "
                f"mid_prices batch={mid_prices.shape[1]}, vision batch={vision_obs.shape[1]}."
            )
        repeat_factor = vision_obs.shape[1] // mid_prices.shape[1]
        mid_prices = jnp.repeat(mid_prices, repeat_factor, axis=1)

    current_obs = vision_obs[:num_steps]
    future_obs = vision_obs[survival_delta_steps:survival_delta_steps + num_steps]
    current_mid = mid_prices[:num_steps, :, None]
    future_mid = mid_prices[survival_delta_steps:survival_delta_steps + num_steps, :, None]

    ask_gap = current_obs[..., 0, 0]
    bid_gap = current_obs[..., 0, 1]
    ask_volume = jnp.expm1(current_obs[..., 1, 0])
    bid_volume = jnp.expm1(current_obs[..., 1, 1])

    future_ask_gap = future_obs[..., 0, 0]
    future_bid_gap = future_obs[..., 0, 1]
    future_ask_volume = jnp.expm1(future_obs[..., 1, 0])
    future_bid_volume = jnp.expm1(future_obs[..., 1, 1])

    tick_size = jnp.asarray(tick_size, dtype=jnp.float32)
    ask_key = jnp.rint((current_mid + ask_gap * tick_size) / tick_size)
    bid_key = jnp.rint((current_mid - bid_gap * tick_size) / tick_size)
    future_ask_key = jnp.rint((future_mid + future_ask_gap * tick_size) / tick_size)
    future_bid_key = jnp.rint((future_mid - future_bid_gap * tick_size) / tick_size)

    ask_mask = ask_volume >= survival_min_volume
    bid_mask = bid_volume >= survival_min_volume
    ask_matches = future_ask_key[..., None, :] == ask_key[..., :, None]
    bid_matches = future_bid_key[..., None, :] == bid_key[..., :, None]
    matched_future_ask_volume = jnp.max(
        jnp.where(ask_matches, future_ask_volume[..., None, :], 0.0),
        axis=-1,
    )
    matched_future_bid_volume = jnp.max(
        jnp.where(bid_matches, future_bid_volume[..., None, :], 0.0),
        axis=-1,
    )
    ask_label = matched_future_ask_volume >= survival_ratio * ask_volume
    bid_label = matched_future_bid_volume >= survival_ratio * bid_volume

    side_mask = jnp.stack([ask_mask, bid_mask], axis=-1)
    side_label = jnp.stack([ask_label & ask_mask, bid_label & bid_mask], axis=-1)

    return side_label.astype(jnp.float32), side_mask.astype(jnp.float32)


def make_train(config):
    # scenario = map_name_to_scenario(config["MAP_NAME"])
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
    else:
        agent_configs = {
            agent_type: agent_cfg()
            for agent_type, agent_cfg in config_dict.items()
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
        else:
            eval_agent_configs = {
                agent_type: agent_cfg()
                for agent_type, agent_cfg in config_dict.items()
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
        num_devices = int(config["N_DEVICES"])
        if config["NUM_ENVS"] % num_devices != 0:
            raise ValueError(
                f"NUM_ENVS={config['NUM_ENVS']} must be divisible by N_DEVICES={num_devices}."
            )
        local_num_envs = config["NUM_ENVS"] // num_devices
        local_num_actors_per_type = [
            n * local_num_envs for n in config["NUM_AGENTS_PER_TYPE"]
        ]


        # For a given agent type (instance) we need the following inputs:
        # Action space, obs space, 

        # The outputs that depends on these and are kept seperate are;
        # - network, init_x, init_hstate, network_params, train_state
        hstates = []
        network_params_list = []
        train_states = []
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
                    'tick_shift': jnp.zeros((1, config["NUM_ENVS"], 1)),
                }
            elif isinstance(env.observation_spaces[i], dict):
                obs_shape = env.observation_spaces[i]['exec_obs'].shape[0]
                init_obs = {
                    'exec_obs': jnp.zeros((1, config["NUM_ENVS"], obs_shape)), 
                    'vision_obs': jnp.zeros((1, config["NUM_ENVS"], 10, 3, 2)), # Shape của LOB
                    'tick_shift': jnp.zeros((1, config["NUM_ENVS"], 1)),
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

        train_states = jax_utils.replicate(train_states)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        host_env_rng = jax.random.fold_in(_rng, jax.process_index())
        reset_rng = jax.random.split(host_env_rng, config["NUM_ENVS"])
        env_params=env.default_params
        if config["CALC_EVAL"]:
            eval_env_params=eval_env.default_params # type: ignore
        else:
            eval_env_params = None
        # env_params=jax.device_put(env_params)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,None))(reset_rng,env_params)

        def reshape_pytree_leading_dim(pytree, devices):
            def _reshape_leaf(leaf):
                if not isinstance(leaf, jnp.ndarray) or leaf.ndim == 0:
                    return leaf
                leading_dim = leaf.shape[0]
                local_dim = leading_dim // devices
                if leading_dim != devices * local_dim:
                    raise ValueError(
                        f"Leading dimension {leading_dim} cannot be split across {devices} devices."
                    )
                return leaf.reshape((devices, local_dim) + leaf.shape[1:])

            return jax.tree_util.tree_map(_reshape_leaf, pytree)

        env_state = reshape_pytree_leading_dim(env_state, num_devices)
        obsv = reshape_pytree_leading_dim(obsv, num_devices)
        init_dones_agents = reshape_pytree_leading_dim(init_dones_agents, num_devices)
        hstates = reshape_pytree_leading_dim(hstates, num_devices)
        # TRAIN LOOP
        

        def _update_step(update_runner_state,env_params,eval_env_params, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state
            # FIXME: APPLY VISION
            def _env_step(runner_state, unused):
                train_states, env_state, last_obs, last_done,h_states, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                
                # Ignore getting the available actions for now, assume all actions are available.
                # avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                # avail_actions = jax.lax.stop_gradient(
                #     batchify(avail_actions, env.agents, config["NUM_ACTORS"])
                # )
                # obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                actions=[]
                values=[]
                log_probs=[]
                '''
                Duyệt qua các agent trong môi trường, lấy ra hành động và trạng thái tại bước thời gian đó
                '''
                for i, train_state in enumerate(train_states):
                    obs_i= last_obs[i]
                    obs_i=batchify(obs_i,local_num_actors_per_type[i])  # Reshape to match the input shape of the network

                    obs_i_batched = jax.tree.map(lambda x: x[jnp.newaxis, :], obs_i)
                    ac_in = (
                        obs_i_batched,
                        last_done[i][jnp.newaxis, :],
                        # avail_actions,
                    )
                    h_states[i], pi, value, _, _ = train_state.apply_fn(train_state.params, h_states[i], ac_in)
                    values.append(value)
                    action = pi.sample(seed=_rng)
                    log_probs.append(pi.log_prob(action))
                    action=unbatchify(action, local_num_envs, env.multi_agent_config.number_of_agents_per_type[i])  # Reshape to match the action shape
                    actions.append(action.squeeze())
                    # env_act = unbatchify(
                    #     action, env.agents, config["NUM_ENVS"], env.num_agents
                    # )
                    # env_act = {k: v.squeeze() for k, v in env_act.items()}
                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, local_num_envs)
                '''
                Cho các agent tương tác với môi trường, nhận lại obs mới, trạng thái và rewards
                '''
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
                    done_batch['agents'][i] = batchify(done["agents"][i],local_num_actors_per_type[i]).squeeze()
                    obs_batch = batchify(obsv[i],local_num_actors_per_type[i])
                    action_batch = batchify_action(actions[i],local_num_actors_per_type[i])
                    value = values[i]
                    log_prob = log_probs[i]

                    info_i={"world":info["world"],"agent":jax.tree.map(lambda x: x.reshape(local_num_actors_per_type[i],-1),info["agents"][i])}
                    # print(f"info for agenttype {i}:", info_i)


                    transitions.append(Transition(
                        jnp.tile(done["__all__"], config["NUM_AGENTS_PER_TYPE"][i]),
                        last_done[i],
                        action_batch.squeeze(),
                        value.squeeze(),
                        batchify(reward[i], local_num_actors_per_type[i]).squeeze(),
                        log_prob.squeeze(),
                        obs_batch,
                        info_i,
                        # avail_actions,
                    ))
                runner_state = (train_states, env_state, obsv, done_batch['agents'], h_states, rng)
                return runner_state, transitions
            initial_hstates = runner_state[-2]

            window_size = 10
            survival_delta_steps = int(config.get("survival_delta_steps", window_size))
            if config.get("use_survival_loss", False) and survival_delta_steps > window_size:
                raise ValueError(
                    "survival_delta_steps must be <= rollout window_size. "
                    f"Got survival_delta_steps={survival_delta_steps}, window_size={window_size}."
                )
            total_rollout_steps = config["NUM_STEPS"] + window_size

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
            # [PHASE 2]: TRÍCH XUẤT NHÃN VOLATILITY TỪ 10 BƯỚC TƯƠNG LAI
            # ==========================================================
            volatility_labels = []
            survival_labels = []
            survival_masks = []
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
            for i in range(len(stashed_runner_state[0])): # Lặp qua train_states
                mid_prices = traj_batch_padded[i].info["world"]["end_mid_price"]
                
                # Quét cửa sổ tương lai (Logic giữ nguyên, chạy thẳng trên mid_prices chuẩn)
                def calc_future_std(t):
                    future_window = jax.lax.dynamic_slice_in_dim(mid_prices, t + 1, window_size, axis=0)
                    return jnp.std(future_window, axis=0)
                
                # Chỉ tính nhãn cho 128 bước đầu
                timesteps = jnp.arange(config["NUM_STEPS"])
                future_vol = jax.vmap(calc_future_std)(timesteps)
                
                # Gán nhãn
                labels = jnp.where(future_vol > config.get("VOL_HIGH", 0.02), 2, 
                         jnp.where(future_vol > config.get("VOL_LOW", 0.005), 1, 0))
                volatility_labels.append(labels)

                if isinstance(traj_batch_padded[i].obs, dict) and "vision_obs" in traj_batch_padded[i].obs:
                    surv_label, surv_mask = build_liquidity_survival_targets(
                        traj_batch_padded[i].obs["vision_obs"],
                        mid_prices,
                        tick_size=tick_size,
                        survival_delta_steps=survival_delta_steps,
                        survival_min_volume=config.get("survival_min_volume", 1.0),
                        survival_ratio=config.get("survival_ratio", 0.5),
                        num_steps=config["NUM_STEPS"],
                    )
                else:
                    reward_shape = traj_batch_padded[i].reward.shape
                    surv_label = jnp.zeros((config["NUM_STEPS"], reward_shape[1], 10, 2), dtype=jnp.float32)
                    surv_mask = jnp.zeros_like(surv_label)
                survival_labels.append(surv_label)
                survival_masks.append(surv_mask)

            # ==========================================================
            # [PHASE 3]: CƯA ĐUÔI DATA VÀ KHÔI PHỤC DÒNG THỜI GIAN
            # ==========================================================
            # 3.1 Cưa bỏ 10 bước padding, trả về traj_batch chuẩn 128 bước
            traj_batch = jax.tree_util.tree_map(
                lambda x: x[:config["NUM_STEPS"]], traj_batch_padded
            )

            # 3.2 Khôi phục bộ nhớ ở bước 128 cho vòng lặp sau
            t_states, e_state, l_obs, l_dones, h_states, _ = stashed_runner_state
            
            # Chôm chìa khóa RNG từ bước 138 (final_runner_state).
            fresh_rng = final_runner_state[-1] 
            
            # Gắn lại vào runner_state
            runner_state = (t_states, e_state, l_obs, l_dones, h_states, fresh_rng)

            # CALCULATE ADVANTAGE
            train_states, env_state, last_obs, last_dones, hstates_new, rng = runner_state

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
                last_obs_batch = batchify(last_obs[i], local_num_actors_per_type[i])
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
            for i, train_state in enumerate(train_states):
                def _update_epoch(update_state, unused):
                    def _update_minbatch(train_state, batch_info):
                        init_hstate, traj_batch, advantages, targets, vol_labels, surv_labels, surv_mask = batch_info

                        def _loss_fn(params, init_hstate, traj_batch, gae, targets, vol_labels, surv_labels, surv_mask):
                            # RERUN NETWORK
                            _, pi, value, z_vision, aux_info = train_state.apply_fn(
                                params,
                                init_hstate.squeeze(),
                                (traj_batch.obs, traj_batch.done),
                            )
                            log_prob = pi.log_prob(traj_batch.action)

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
                            ppo_loss = (
                                loss_actor
                                + config["VF_COEF"][i] * value_loss
                                - config["ENT_COEF"][i] * entropy
                            )

                            # TÍNH SUPCON LOSS CHO VISION AGENT
                            if isinstance(traj_batch.obs, dict):
                                z_flat = z_vision.reshape(-1, z_vision.shape[-1])
                                labels_flat = vol_labels.reshape(-1)
                                supcon_loss = supervised_contrastive_loss(z_flat, labels_flat, temperature=0.1)
                                alpha = config.get("SUPCON_ALPHA", 0.1)
                            else:
                                supcon_loss = jnp.array(0.0)
                                alpha = jnp.array(0.0)

                            # LOSS CUỐI CÙNG (Cộng PPO và SupCon có hệ số)
                            if (
                                config.get("use_survival_loss", False)
                                and config.get("use_reliability_head", False)
                                and isinstance(traj_batch.obs, dict)
                            ):
                                reliability_scores = aux_info["reliability_scores"]
                                survival_loss = masked_bce_loss(reliability_scores, surv_labels, surv_mask)
                                lambda_surv = config.get("lambda_surv", 0.0)
                                survival_mask_ratio = jnp.mean(surv_mask.astype(jnp.float32))
                                reliability_mean = jnp.mean(reliability_scores)
                            else:
                                survival_loss = jnp.array(0.0)
                                lambda_surv = jnp.array(0.0)
                                survival_mask_ratio = jnp.array(0.0)
                                reliability_mean = jnp.array(0.0)

                            weighted_survival_loss = lambda_surv * survival_loss
                            total_loss = ppo_loss + alpha * supcon_loss + weighted_survival_loss

                            # debug
                            approx_kl = ((ratio - 1) - logratio).mean()
                            clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])

                            return total_loss, (
                                value_loss,
                                loss_actor,
                                entropy,
                                ratio,
                                approx_kl,
                                clip_frac,
                                supcon_loss,
                                survival_loss,
                                weighted_survival_loss,
                                survival_mask_ratio,
                                reliability_mean,
                            )
                        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                        total_loss, grads = grad_fn(
                            train_state.params,
                            init_hstate,
                            traj_batch,
                            advantages,
                            targets,
                            vol_labels,
                            surv_labels,
                            surv_mask,
                        )
                        total_loss = jax.lax.pmean(total_loss, axis_name="device_batch")
                        grads = jax.lax.pmean(grads, axis_name="device_batch")
                        train_state = train_state.apply_gradients(grads=grads)
                        return train_state, total_loss
                    (
                        train_state,
                        init_hstate,
                        traj_batch,
                        advantages,
                        targets,
                        vol_labels,
                        surv_labels,
                        surv_mask,
                        rng,
                    ) = update_state
                    rng, _rng = jax.random.split(rng)

                    # adding an additional "fake" dimensionality to perform minibatching correctly
                    init_hstate = jnp.reshape(
                        init_hstate, (1, local_num_actors_per_type[i], -1)
                    )
                    batch = (
                        init_hstate,
                        traj_batch,
                        advantages.squeeze(),
                        targets.squeeze(),
                        vol_labels,
                        surv_labels,
                        surv_mask,
                    )
                    permutation = jax.random.permutation(_rng, local_num_actors_per_type[i])

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

                    train_state, total_loss = jax.lax.scan(
                        _update_minbatch, train_state, minibatches
                    )
                    update_state = (
                        train_state,
                        init_hstate.squeeze(),
                        traj_batch,
                        advantages,
                        targets,
                        vol_labels,
                        surv_labels,
                        surv_mask,
                        rng,
                    )
                    return update_state, total_loss

                update_state = (
                    train_state,
                    initial_hstates[i],
                    traj_batch[i],
                    advantages[i],
                    targets[i],
                    volatility_labels[i],
                    survival_labels[i],
                    survival_masks[i],
                    rng,
                )
                update_state, loss_info = jax.lax.scan(
                    _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
                )
                train_states[i] = update_state[0]
                loss_infos.append(loss_info)


            metrics= {}
            metrics['agents'] = [jax.tree.map(
                lambda x: x.reshape(
                    (config["NUM_STEPS"], local_num_envs, config["NUM_AGENTS_PER_TYPE"][i])
                ),
                trjbtch.info['agent']) for i, trjbtch in enumerate(traj_batch)]
            metrics['world'] = [traj_batch.info['world'] for i, traj_batch in enumerate(traj_batch)]
            metrics["loss"]=[]
            for i,loss_info in enumerate(loss_infos):
                ratio_0 = loss_info[1][3].at[0,0].get().mean()
                loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
                metrics["loss"].append({
                    "total_loss": loss_info[0],
                    "value_loss": loss_info[1][0],
                    "actor_loss": loss_info[1][1],
                    "entropy": loss_info[1][2],
                    "ratio": loss_info[1][3],
                    "ratio_0": ratio_0,
                    "approx_kl": loss_info[1][4],
                    "clip_frac": loss_info[1][5],
                    "survival_loss": loss_info[1][7],
                    "weighted_survival_loss": loss_info[1][8],
                    "survival_mask_ratio": loss_info[1][9],
                    "reliability_mean": loss_info[1][10],
                    "weighted_entropy_loss": loss_info[1][2] * config["ENT_COEF"][i],
                    "weighted_value_loss": loss_info[1][0] * config["VF_COEF"][i],
                })


            #jax.debug.print(f"traj_batch: {len(traj_batch)}")
            #for i, tr in enumerate(traj_batch):
            #    jax.debug.print(f"traj_batch {i} reward shape: {tr.reward.shape}")
            #    jax.debug.print(f"current mean: {jnp.mean(tr.reward)}")
            #    jax.debug.print("flattened mean: ", jnp.mean(tr.reward.flatten()))

            metrics['avg_reward'] = [jnp.mean(tr.reward) for tr in traj_batch]
            metrics['avg_reward_flattened'] = [jnp.mean(tr.reward.flatten()) for tr in traj_batch]
            metrics["traj_batch"] = traj_batch


            if config["CALC_EVAL"]:
                def _eval_step(eval_runner_state, unused):
                    train_states, eval_env_state, last_obs, last_done,h_states, rng = eval_runner_state
                    rng, _rng = jax.random.split(rng)
                
                    actions=[]
                    values=[]
                    log_probs=[]

                    for i, train_state in enumerate(train_states):
                        obs_i= last_obs[i]
                        obs_i=batchify(obs_i,local_num_actors_per_type[i])  # Reshape to match the input shape of the network
                        obs_i_batched = jax.tree.map(lambda x: x[jnp.newaxis, :], obs_i)
                        ac_in = (
                            obs_i_batched,
                            last_done[i][jnp.newaxis, :],
                            # avail_actions,
                        )
                        h_states[i], pi, value, _, _ = train_state.apply_fn(train_state.params, h_states[i], ac_in)
                        values.append(value)
                        action = pi.sample(seed=_rng)
                        log_probs.append(pi.log_prob(action))
                        action=unbatchify(action, local_num_envs, env.multi_agent_config.number_of_agents_per_type[i])  # Reshape to match the action shape
                        actions.append(action.squeeze())

                        rng, _rng = jax.random.split(rng)
                        rng_step = jax.random.split(_rng, local_num_envs)

                





                    # STEP ENV
                    rng, _rng = jax.random.split(rng)
                    rng_step = jax.random.split(_rng, local_num_envs)
                    obsv, eval_env_state, reward, done, info = jax.vmap(
                        eval_env.step, in_axes=(0, 0, 0, None) # type: ignore
                    )(rng_step, eval_env_state, actions, eval_env_params)
                    done_batch=done
                    transitions=[]    

                    for i, train_state in enumerate(train_states):
                        done_batch['agents'][i] = batchify(done["agents"][i],local_num_actors_per_type[i]).squeeze()
                        obs_batch = batchify(obsv[i],local_num_actors_per_type[i])
                        action_batch = batchify_action(actions[i],local_num_actors_per_type[i])
                        value = values[i]
                        log_prob = log_probs[i]

                        info_i={"world":info["world"],"agent":jax.tree.map(lambda x: x.reshape(local_num_actors_per_type[i],-1),info["agents"][i])}
                        # print(f"info for agenttype {i}:", info_i)


                        transitions.append(Transition(
                            jnp.tile(done["__all__"], config["NUM_AGENTS_PER_TYPE"][i]),
                            last_done[i],
                            action_batch.squeeze(),
                            value.squeeze(),
                            batchify(reward[i], local_num_actors_per_type[i]).squeeze(),
                            log_prob.squeeze(),
                            obs_batch,
                            info_i,
                            # avail_actions,
                        ))
                    eval_runner_state = (train_states, eval_env_state, obsv, done_batch['agents'], h_states, rng)
                    return eval_runner_state, transitions

                rng, _rng = jax.random.split(rng)
                reset_rng = jax.random.split(_rng, local_num_envs)
                eval_obsv, eval_env_state = jax.vmap(eval_env.reset, in_axes=(0, None))(reset_rng, eval_env_params) # type: ignore


                eval_hstates=[]
                init_dones_agents_eval=[]
                for i, train_state in enumerate(train_states):
                    eval_hstates.append(ScannedRNN.initialize_carry(local_num_actors_per_type[i], config["GRU_HIDDEN_DIM"]))
                    init_dones_agents_eval.append(jnp.zeros((local_num_actors_per_type[i]), dtype=bool))


                
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
                metrics['agents_eval'] = [jax.tree.map(
                    lambda x: x.reshape(
                        (config["NUM_STEPS_EVAL"], local_num_envs, config["NUM_AGENTS_PER_TYPE"][i])
                    ),
                    trjbtch.info['agent']) for i, trjbtch in enumerate(eval_traj_batch)]
                metrics['world_eval'] = [trjbtch.info['world'] for i, trjbtch in enumerate(eval_traj_batch)]
                if config["CALC_EVAL"]:
                    metrics['avg_reward_eval'] = [jnp.mean(tr.reward) for tr in eval_traj_batch]
                    metrics["traj_batch_eval"] = eval_traj_batch

            def callback(metric):
                print("Update step:", metric["update_steps"])
                # for loss_idx, m in enumerate(metric["loss"]):
                #         logging_dict[f"agent_{agent_name}/loss_{loss_idx}"] = m
                # Needed?

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
                
                    
                    for key, value in tr.info['agent'].items():
                    # Check if value is a numpy array or jax array and has elements
                        if isinstance(value, (jnp.ndarray, np.ndarray)) and value.size > 0:
                            flat_value = np.array(value).flatten()
                            if flat_value.size > 0:
                                # Get agent short_name from config
                                logging_dict[f"agent_{agent_name}/{key}_mean"] = float(np.mean(flat_value))
                                logging_dict[f"agent_{agent_name}/{key}_std"] = float(np.std(flat_value))
                    
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

            metrics["update_steps"] = update_steps
            update_steps = update_steps + 1
            runner_state = (train_states, env_state, last_obs, last_dones, hstates_new, rng)

            print("Finished compiling")
            # jax.profiler.save_device_memory_profile(f"memory_{update_steps}.prof")
            return (runner_state, update_steps), metrics

        rng, _rng = jax.random.split(rng)
        host_device_rng = jax.random.fold_in(_rng, jax.process_index())
        device_rng = jax.random.split(host_device_rng, num_devices)
        runner_state = (
            train_states,
            env_state,
            obsv,
            init_dones_agents, # last_done
            hstates,  # initial hidden states for RNN
            device_rng,
        )

        jitted_update_step = jax.jit(_update_step)
        pmapped_update_step = jax.pmap(
            jitted_update_step,
            axis_name="device_batch",
            in_axes=(((0, 0, 0, 0, 0, 0), None), None, None, None),
            out_axes=(((0, 0, 0, 0, 0, 0), None), 0),
        )
        
        checkpoint_manager = None
        if config.get("SAVE_CHECKPOINT", False):
            orbax_checkpointer = oxcp.PyTreeCheckpointer()
            keep_period = max(1, config["NUM_UPDATES"] // 2)
            options = oxcp.CheckpointManagerOptions(max_to_keep=2, create=True, keep_period=keep_period)
            checkpoint_path = os.path.abspath(
                f'./checkpoints/MARLCheckpoints/{config["PROJECT"]}/{(run.name if run.name else run.id) if run else "GENERIC_RUN"}'
            )

            checkpoint_manager = oxcp.CheckpointManager(
                checkpoint_path,
                orbax_checkpointer,
                options,
            )
        
        updates=0
        for i in range(config["NUM_UPDATES"]):
            print(f"Update step {i+1}/{config['NUM_UPDATES']}")
            # Run the update step:
            if config["world_config"]["debug_mode"] == True:
                if i>2 and i<4:
                    jax.profiler.start_trace("/tmp/profile-data")
            (runner_state,updates),metrics=pmapped_update_step((runner_state,updates),env_params,eval_env_params,None)
            if config["world_config"]["debug_mode"] == True:
                if i>2 and i<4:
                    jax.block_until_ready((runner_state,updates,metrics))
                    jax.profiler.stop_trace()
            print(f"Update step {updates} completed with metrics {metrics['avg_reward']}")
            if checkpoint_manager is not None:
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
                print(f"Saving checkpoint {updates} with metrics {metrics['avg_reward']}")
                save_args = orbax_utils.save_args_from_target(ckpt)
                checkpoint_manager.save(updates, ckpt, save_kwargs={"save_args": save_args})
            del metrics
            gc.collect()
        

        if checkpoint_manager is not None:
            checkpoint_manager.wait_until_finished()

        # runner_state, metrics = jax.lax.scan(
        #     _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        # )
        
        
        return {"runner_state": runner_state}

    return train


@hydra.main(version_base=None, config_path="config", config_name="PMAP_ippo_rnn_JAXMARL_2player")
def main(config):
    print("MultiAgentConfig", MultiAgentConfig().world_config)
    env_config=OmegaConf.structured(MultiAgentConfig(number_of_agents_per_type=config["NUM_AGENTS_PER_TYPE"]))
    final_config=OmegaConf.merge(config,env_config)
    config = OmegaConf.to_container(final_config)

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

@hydra.main(version_base=None, config_path="config", config_name="PMAP_ippo_rnn_JAXMARL_2player")
def seperate_main(config):
    print("MultiAgentConfig", MultiAgentConfig().world_config)
    env_config=OmegaConf.structured(MultiAgentConfig(number_of_agents_per_type=config["NUM_AGENTS_PER_TYPE"]))
    final_config=OmegaConf.merge(config,env_config)
    config = OmegaConf.to_container(final_config)

    # =====================================================================
    # BỐC CONFIG TỪ HÀM SWEEP CHO VÀO ĐÂY ĐỂ VÁ LỖI KEYERROR
    # =====================================================================
    config["AGENT_CONFIGS"] = {
        "MarketMaking": {
            "inv_penalty": "quadratic", 
            "skew_multiplier": 10,
            "action_space": "fixed_quants",
            "reward_space": "spooner",
            "reference_price_portfolio_value": "best_bid_ask"
        },
        "Execution": {
            "reward_lambda": 0.5,
            "observation_space": "execution_policy",
            "action_space": "policy_blending",
            "task_size": 600,
            "doom_price_penalty": 0.1
        }
    }
    # =====================================================================

    # jax.profiler.start_trace("/tmp/profile-data")

    rng = jax.random.PRNGKey(0)

    train_fun = make_train(config)
    # print("+++++++++++ Training turned off whilst debugging wandb ++++++++++++")
    out = train_fun(rng)

    # out=jax.block_until_ready(out)  # Ensure the computation is complete before proceeding
    # (dummy * dummy).block_until_ready()
    # jax.profiler.stop_trace()


if __name__ == "__main__":
    seperate_main()
