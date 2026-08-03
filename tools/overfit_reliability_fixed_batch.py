"""Fixed-batch supervised overfit diagnostic for the Reliability Head.

This script collects one rollout batch, builds the same liquidity reliability
targets used by PPO training, then optimizes only the masked reliability loss on
that frozen batch. It fails fast when the Execution network does not actually
run the Reliability path or when gradients cannot update the Reliability Head.
"""

from __future__ import annotations

import copy
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, NamedTuple

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

from gymnax_exchange.jaxen.marl_env import MARLEnv
from gymnax_exchange.jaxob.jaxob_config import (
    Execution_EnvironmentConfig,
    MarketMaking_EnvironmentConfig,
    MultiAgentConfig,
    World_EnvironmentConfig,
)
from gymnax_exchange.jaxrl.MARL.ippo_rnn_JAXMARL import (
    ActorCriticRNN,
    ScannedRNN,
    Transition,
    batchify,
    batchify_action,
    unbatchify,
)
from gymnax_exchange.jaxrl.MARL.gradient_diagnostics import (
    flatten_gradient_tree,
    flatten_tree_with_paths,
    gradient_l2_norm,
    matching_parameter_paths,
    subtract_gradient_trees,
    tree_l2_norm,
)
from gymnax_exchange.jaxrl.MARL.box_ppo import sample_policy_action
from gymnax_exchange.jaxrl.MARL.reliability_targets import (
    build_liquidity_survival_targets,
    masked_reliability_loss,
    resolve_rollout_is_sell_task,
)


LOG_STEPS_DEFAULT = (0, 1, 5, 10, 25, 50, 100, 200, 300, 500, 750, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000)
CONFIG_PATH = (
    REPO_ROOT
    / "gymnax_exchange"
    / "jaxrl"
    / "MARL"
    / "config"
    / "ippo_rnn_JAXMARL_2player.yaml"
)


@dataclass
class FixedBatch:
    train_state: TrainState
    init_hstate: jax.Array
    obs: Dict[str, jax.Array]
    done: jax.Array
    labels: jax.Array
    mask: jax.Array
    is_sell_task: jax.Array
    target_diag: Dict[str, jax.Array]
    config: Dict[str, Any]


class RolloutBundle(NamedTuple):
    train_states: list[TrainState]
    initial_hstates: list[jax.Array]
    traj_batch_padded: list[Transition]
    env: MARLEnv
    config: Dict[str, Any]


def _to_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _load_config(argv: list[str]) -> Dict[str, Any]:
    base_cfg = OmegaConf.load(CONFIG_PATH)
    cli_cfg = OmegaConf.from_cli(argv)
    cli_values = OmegaConf.to_container(cli_cfg, resolve=True)
    merged = OmegaConf.merge(base_cfg, cli_cfg)
    env_cfg = OmegaConf.structured(
        MultiAgentConfig(number_of_agents_per_type=merged["NUM_AGENTS_PER_TYPE"])
    )
    merged = OmegaConf.merge(merged, env_cfg)
    config = OmegaConf.to_container(merged, resolve=True)

    config["WANDB_MODE"] = "disabled"
    config["CALC_EVAL"] = False
    # This diagnostic is specifically for fitting the Reliability Head. The
    # training YAML defaults this flag to false, so use an explicit script-local
    # default instead of silently inheriting the fallback-zero forward path.
    config["use_reliability_head"] = _to_bool(
        cli_values.get("use_reliability_head", True)
    )
    config["use_survival_loss"] = _to_bool(
        cli_values.get("use_survival_loss", True)
    )
    config["reliability_loss_type"] = str(
        config.get("reliability_loss_type", "bce")
    ).lower()
    config["use_h_prev_in_reliability"] = _to_bool(
        config.get("use_h_prev_in_reliability", True)
    )
    config["overfit_steps"] = int(config.get("overfit_steps", 300))
    config["overfit_lr"] = float(config.get("overfit_lr", 1e-3))
    config["overfit_mode"] = config.get("overfit_mode", "all_params")

    # Reliability overfit requires the dict observation path. Keep this local to
    # the diagnostic script so normal training defaults remain untouched.
    if "AGENT_CONFIGS" not in config:
        exec_cfg = config.setdefault("dict_of_agents_configs", {}).setdefault("Execution", {})
        exec_cfg.update(
            {
                "action_space": "policy_blending",
                "observation_space": "execution_policy",
                "task_size": int(exec_cfg.get("task_size", 600)),
                "reward_lambda": float(exec_cfg.get("reward_lambda", 0.5)),
                "doom_price_penalty": float(exec_cfg.get("doom_price_penalty", 0.1)),
            }
        )
        mm_cfg = config.setdefault("dict_of_agents_configs", {}).setdefault("MarketMaking", {})
        mm_cfg.setdefault("action_space", "fixed_quants")
        mm_cfg.setdefault("inv_penalty", "quadratic")
        mm_cfg.setdefault("reference_price_portfolio_value", "best_bid_ask")
        mm_cfg.setdefault("reward_space", "buy_sell_pnl")
        mm_cfg.setdefault("skew_multiplier", 10)

    if "ANNEAL_LR" in config:
        config["ANNEAL_LR"] = [_to_bool(x) for x in config["ANNEAL_LR"]]
    return config


def _make_env(config: Dict[str, Any]) -> MARLEnv:
    config_dict = {
        "MarketMaking": MarketMaking_EnvironmentConfig,
        "Execution": Execution_EnvironmentConfig,
    }
    if "AGENT_CONFIGS" in config:
        agent_configs = {
            agent_type: config_dict[agent_type](**{k.lower(): v for k, v in agent_cfg.items()})
            for agent_type, agent_cfg in config["AGENT_CONFIGS"].items()
            if agent_type in config_dict
        }
    else:
        agent_configs = {
            agent_type: config_dict[agent_type](**agent_cfg)
            for agent_type, agent_cfg in config.get("dict_of_agents_configs", {}).items()
            if agent_type in config_dict
        }

    ma_config = MultiAgentConfig(
        number_of_agents_per_type=config["NUM_AGENTS_PER_TYPE"],
        dict_of_agents_configs=agent_configs,
        world_config=World_EnvironmentConfig(
            seed=config["SEED"],
            **{
                k.lower(): v
                for k, v in config.items()
                if hasattr(World_EnvironmentConfig(), k.lower()) and k != "SEED"
            },
        ),
    )
    return MARLEnv(key=jax.random.PRNGKey(config["SEED"]), multi_agent_config=ma_config)


def _init_models(env: MARLEnv, config: Dict[str, Any], rng: jax.Array):
    config["NUM_ACTORS_PERTYPE"] = [
        n * config["NUM_ENVS"] for n in config["NUM_AGENTS_PER_TYPE"]
    ]
    config["NUM_ACTORS_TOTAL"] = env.num_agents * config["NUM_ENVS"]

    train_states = []
    hstates = []
    init_done_agents = []
    for i, _instance in enumerate(env.instance_list):
        agent_cfg = env.list_of_agents_configs[i]
        is_execution = isinstance(agent_cfg, Execution_EnvironmentConfig)
        network_config = dict(config)
        if is_execution:
            use_reliability_head = network_config.get("use_reliability_head")
            print(
                "OVERFIT_RUNTIME_CONFIG",
                f"agent_index={i}",
                f"use_reliability_head={use_reliability_head}",
                f"use_reliability_head_type={type(use_reliability_head).__name__}",
                f"use_survival_loss={network_config.get('use_survival_loss')}",
                f"reliability_loss_type={network_config.get('reliability_loss_type')}",
            )
            if type(use_reliability_head) is not bool or not use_reliability_head:
                raise ValueError(
                    "Fixed-batch Reliability overfit requires the lowercase "
                    "config key use_reliability_head=true."
                )
        network = ActorCriticRNN(env.action_spaces[i], config=network_config)
        if is_execution and network.config.get("use_reliability_head") is not True:
            raise AssertionError(
                "Execution ActorCriticRNN did not retain use_reliability_head=True."
            )
        rng, init_rng = jax.random.split(rng)
        if hasattr(env.observation_spaces[i], "spaces"):
            obs_shape = env.observation_spaces[i].spaces["exec_obs"].shape[0]
            init_obs = {
                "exec_obs": jnp.zeros((1, config["NUM_ENVS"], obs_shape)),
                "vision_obs": jnp.zeros((1, config["NUM_ENVS"], 10, 3, 2)),
                "mid_context": jnp.zeros((1, config["NUM_ENVS"], 4)),
            }
        else:
            init_obs = jnp.zeros((1, config["NUM_ENVS"], env.observation_spaces[i].shape[0]))

        init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
        )
        init_x = (init_obs, jnp.zeros((1, config["NUM_ENVS"])))
        params = network.init(init_rng, init_hstate, init_x)
        tx = optax.adam(config["LR"][i], eps=1e-5)
        train_states.append(
            TrainState.create(
                apply_fn=network.apply,
                params=params,
                tx=tx,
            )
        )
        hstates.append(
            ScannedRNN.initialize_carry(
                config["NUM_ACTORS_PERTYPE"][i], config["GRU_HIDDEN_DIM"]
            )
        )
        init_done_agents.append(jnp.zeros((config["NUM_ACTORS_PERTYPE"][i]), dtype=bool))
    return train_states, hstates, init_done_agents


def _stack_transitions(transitions_per_agent: list[list[Transition]]) -> list[Transition]:
    return [
        jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *agent_transitions)
        for agent_transitions in transitions_per_agent
    ]


def _collect_rollout(config: Dict[str, Any]) -> RolloutBundle:
    rng = jax.random.PRNGKey(config["SEED"])
    env = _make_env(config)
    train_states, hstates, init_done_agents = _init_models(env, config, rng)
    initial_hstates = [h.copy() for h in hstates]

    rng, reset_rng = jax.random.split(rng)
    reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
    env_params = env.default_params
    last_obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rngs, env_params)
    last_done = init_done_agents

    total_rollout_steps = config["NUM_STEPS"] + int(config.get("survival_delta_steps", 10))
    transitions_per_agent: list[list[Transition]] = [
        [] for _ in range(len(train_states))
    ]

    for _step in range(total_rollout_steps):
        actions = []
        pre_tanh_actions = []
        values = []
        log_probs = []
        for i, train_state in enumerate(train_states):
            obs_i = batchify(last_obs[i], config["NUM_ACTORS_PERTYPE"][i])
            obs_i_batched = jax.tree.map(lambda x: x[jnp.newaxis, :], obs_i)
            ac_in = (obs_i_batched, last_done[i][jnp.newaxis, :])
            hstates[i], pi, value, _, policy_aux_info = train_state.apply_fn(
                train_state.params,
                hstates[i],
                ac_in,
            )
            rng, action_rng = jax.random.split(rng)
            action_space = env.action_spaces[i]
            sample_kwargs = {}
            if hasattr(action_space, "low") and hasattr(action_space, "high"):
                sample_kwargs = {
                    "action_low": action_space.low,
                    "action_high": action_space.high,
                }
            policy_sample = sample_policy_action(
                pi,
                policy_aux_info,
                action_rng,
                **sample_kwargs,
            )
            values.append(value)
            log_probs.append(policy_sample.log_prob)
            actions.append(
                unbatchify(
                    policy_sample.action,
                    config["NUM_ENVS"],
                    env.multi_agent_config.number_of_agents_per_type[i],
                ).squeeze()
            )
            pre_tanh_actions.append(
                unbatchify(
                    policy_sample.pre_tanh_action,
                    config["NUM_ENVS"],
                    env.multi_agent_config.number_of_agents_per_type[i],
                ).squeeze()
            )

        rng, step_rng = jax.random.split(rng)
        rng_step = jax.random.split(step_rng, config["NUM_ENVS"])
        pre_step_env_state = env_state
        obsv, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(rng_step, env_state, actions, env_params)

        done_batch = copy.copy(done)
        for i, _train_state in enumerate(train_states):
            done_batch["agents"][i] = batchify(
                done["agents"][i], config["NUM_ACTORS_PERTYPE"][i]
            ).squeeze()
            obs_batch = batchify(last_obs[i], config["NUM_ACTORS_PERTYPE"][i])
            action_batch = batchify_action(actions[i], config["NUM_ACTORS_PERTYPE"][i])
            pre_tanh_action_batch = batchify_action(
                pre_tanh_actions[i],
                config["NUM_ACTORS_PERTYPE"][i],
            )
            info_world_i = {
                **info["world"],
                "obs_mid_price": pre_step_env_state.world_state.mid_price,
                "obs_ask_raw_orders": pre_step_env_state.world_state.ask_raw_orders,
                "obs_bid_raw_orders": pre_step_env_state.world_state.bid_raw_orders,
            }
            info_i = {
                "world": info_world_i,
                "agent": jax.tree.map(
                    lambda x: x.reshape(config["NUM_ACTORS_PERTYPE"][i], -1),
                    info["agents"][i],
                ),
            }
            transitions_per_agent[i].append(
                Transition(
                    jnp.tile(done["__all__"], config["NUM_AGENTS_PER_TYPE"][i]),
                    last_done[i],
                    action_batch.squeeze(),
                    pre_tanh_action_batch.squeeze(),
                    values[i].squeeze(),
                    batchify(reward[i], config["NUM_ACTORS_PERTYPE"][i]).squeeze(),
                    log_probs[i].squeeze(),
                    obs_batch,
                    info_i,
                )
            )

        last_obs = obsv
        last_done = done_batch["agents"]

    return RolloutBundle(
        train_states=train_states,
        initial_hstates=initial_hstates,
        traj_batch_padded=_stack_transitions(transitions_per_agent),
        env=env,
        config=config,
    )


def _slice_obs(obs, num_steps):
    return jax.tree_util.tree_map(lambda x: x[:num_steps], obs)


def _make_fixed_batch(bundle: RolloutBundle) -> FixedBatch:
    config = bundle.config
    env = bundle.env
    exec_idx = None
    for idx, agent_cfg in enumerate(env.list_of_agents_configs):
        class_name = type(agent_cfg).__name__
        short_name = getattr(agent_cfg, "short_name", None)
        if "Execution" in class_name or short_name == "EXE":
            exec_idx = idx
            break
    if exec_idx is None:
        if "Execution" in list(env.type_names):
            exec_idx = list(env.type_names).index("Execution")
        elif "EXE" in list(env.type_names):
            exec_idx = list(env.type_names).index("EXE")
    if exec_idx is None:
        raise ValueError(f"Could not find Execution agent in {env.type_names}")

    traj = bundle.traj_batch_padded[exec_idx]
    required_obs_keys = {"exec_obs", "vision_obs", "mid_context"}
    if not isinstance(traj.obs, dict) or not required_obs_keys.issubset(traj.obs):
        raise ValueError(
            "Execution observation is missing one of "
            f"{sorted(required_obs_keys)}. "
            "Use execution_policy observation space."
        )
    agent_cfg = env.list_of_agents_configs[exec_idx]
    print(
        "OVERFIT_MODEL",
        f"exec_idx={exec_idx}",
        f"env_type_names={list(env.type_names)}",
        f"agent_config_class={type(agent_cfg).__name__}",
        f"observation_type={type(traj.obs).__name__}",
        f"observation_keys={sorted(traj.obs.keys())}",
        f"vision_obs_shape={tuple(traj.obs['vision_obs'].shape)}",
    )
    if not isinstance(agent_cfg, Execution_EnvironmentConfig):
        raise AssertionError(
            f"Selected overfit model is not Execution: {type(agent_cfg).__name__}."
        )

    tick_size = env.multi_agent_config.world_config.tick_size
    survival_delta_steps = int(config.get("survival_delta_steps", 10))
    execution_task = getattr(env.list_of_agents_configs[exec_idx], "task", None)
    is_sell_task = resolve_rollout_is_sell_task(
        traj.info["agent"],
        task=execution_task,
        num_steps=config["NUM_STEPS"],
        batch_size=traj.obs["vision_obs"].shape[1],
    )
    labels, mask, target_diag = build_liquidity_survival_targets(
        traj.obs["vision_obs"],
        traj.info["world"].get("obs_mid_price", traj.info["world"]["end_mid_price"]),
        tick_size=tick_size,
        survival_delta_steps=survival_delta_steps,
        survival_min_volume=config.get("survival_min_volume", 1.0),
        ask_raw_orders=traj.info["world"].get("obs_ask_raw_orders", None),
        bid_raw_orders=traj.info["world"].get("obs_bid_raw_orders", None),
        new_trades=traj.info["world"]["new_trades"],
        trade_valid_mask=traj.info["world"]["trade_valid_mask"],
        trade_buffer_saturated=traj.info["world"]["trade_buffer_saturated"],
        num_steps=config["NUM_STEPS"],
        episode_done=traj.global_done,
        return_diagnostics=True,
        eps=config.get("survival_eps", 1e-8),
    )
    labels_host = np.asarray(jax.device_get(labels))
    mask_host = np.asarray(jax.device_get(mask))
    valid_target_count = float(mask_host.sum())
    if valid_target_count <= 0:
        raise ValueError("Fixed overfit batch contains no valid reliability targets.")
    if not np.isfinite(labels_host).all():
        raise ValueError("Fixed overfit labels contain NaN or Inf.")
    if np.any(labels_host < 0.0) or np.any(labels_host > 1.0):
        raise ValueError("Fixed overfit labels must be within [0, 1].")
    print(
        "OVERFIT_TARGET_DIAG",
        f"valid_target_count={valid_target_count:.0f}",
        f"valid_target_rate={_float(target_diag['valid_target_rate']):.6g}",
        f"target_min={_float(target_diag['target_min']):.6g}",
        f"target_mean={_float(target_diag['target_mean']):.6g}",
        f"target_max={_float(target_diag['target_max']):.6g}",
        f"target_std={_float(target_diag['target_std']):.6g}",
        f"q0_mean={_float(target_diag['q0_mean']):.6g}",
        f"q0_min={_float(target_diag['q0_min']):.6g}",
        f"q0_max={_float(target_diag['q0_max']):.6g}",
        f"q_tau_mean={_float(target_diag['q_tau_mean']):.6g}",
        f"cumulative_E_tau_mean={_float(target_diag['cumulative_executed_mean']):.6g}",
        f"net_missing_liquidity_mean={_float(target_diag['net_missing_liquidity_mean']):.6g}",
        f"trade_buffer_saturated_rate={_float(target_diag['trade_buffer_saturated_rate']):.6g}",
        f"done_masked_rate={_float(target_diag['done_masked_rate']):.6g}",
        f"ask_valid_count={_float(target_diag['ask_valid_count']):.0f}",
        f"ask_target_mean={_float(target_diag['ask_target_mean']):.6g}",
        f"ask_target_std={_float(target_diag['ask_target_std']):.6g}",
        f"bid_valid_count={_float(target_diag['bid_valid_count']):.0f}",
        f"bid_target_mean={_float(target_diag['bid_target_mean']):.6g}",
        f"bid_target_std={_float(target_diag['bid_target_std']):.6g}",
    )
    return FixedBatch(
        train_state=bundle.train_states[exec_idx],
        init_hstate=bundle.initial_hstates[exec_idx],
        obs=_slice_obs(traj.obs, config["NUM_STEPS"]),
        done=traj.done[: config["NUM_STEPS"]],
        labels=labels,
        mask=mask,
        is_sell_task=is_sell_task,
        target_diag=target_diag,
        config=config,
    )


def _safe_mean(x, mask, eps=1e-8):
    x = jnp.asarray(x, dtype=jnp.float32)
    mask = jnp.asarray(mask, dtype=jnp.float32)
    return jnp.sum(x * mask) / jnp.maximum(jnp.sum(mask), eps)


def _safe_corr(score, target, mask, eps=1e-8):
    score_mean = _safe_mean(score, mask, eps)
    target_mean = _safe_mean(target, mask, eps)
    score_centered = (score - score_mean) * mask
    target_centered = (target - target_mean) * mask
    cov = jnp.sum(score_centered * target_centered) / jnp.maximum(jnp.sum(mask), eps)
    score_var = jnp.sum(jnp.square(score_centered)) / jnp.maximum(jnp.sum(mask), eps)
    target_var = jnp.sum(jnp.square(target_centered)) / jnp.maximum(jnp.sum(mask), eps)
    return cov / jnp.maximum(jnp.sqrt(score_var * target_var), eps)


def _group_masks(labels, mask, is_sell_task, rank_depth=3):
    time_steps, batch_size, levels, sides = labels.shape
    level_ids = jnp.arange(levels)[None, None, :, None]
    top = jnp.broadcast_to(level_ids < rank_depth, labels.shape)
    far = jnp.broadcast_to(level_ids >= rank_depth, labels.shape)
    is_sell_task = jnp.asarray(is_sell_task, dtype=jnp.bool_)[:, :, None]
    ask_actionable = jnp.broadcast_to(is_sell_task, (time_steps, batch_size, levels))
    bid_actionable = jnp.broadcast_to(~is_sell_task, (time_steps, batch_size, levels))
    actionable = jnp.stack([ask_actionable, bid_actionable], axis=-1)
    nonactionable = ~actionable
    valid = mask > 0.0
    return {
        "CURRENT_RANK_TOP3_TASK_SIDE": valid & top & actionable,
        "CURRENT_RANK_TOP3_OPPOSITE_SIDE": valid & top & nonactionable,
        "CURRENT_RANK_FAR_TASK_SIDE": valid & far & actionable,
        "CURRENT_RANK_FAR_OPPOSITE_SIDE": valid & far & nonactionable,
    }


def _level_means(x, mask):
    return jnp.asarray([_safe_mean(x[:, :, k], mask[:, :, k]) for k in range(x.shape[2])])


def _float(x):
    return float(jax.device_get(x))


def _fmt(values):
    arr = np.asarray(jax.device_get(values), dtype=np.float32).reshape(-1)
    return "[" + ",".join(f"{v:.6g}" for v in arr) + "]"


def _top_level_param_keys(params):
    return sorted({path[0] for path in flatten_tree_with_paths(params) if path})


def _matching_param_paths(params):
    paths = set()
    for group in ("reliability_head", "vision_encoder", "fusion_shared_trunk"):
        paths.update(matching_parameter_paths(params, group))
    return ["/".join(path) for path in sorted(paths)]


def _score_grad_norm(grads, leaf_name):
    selected = {
        path: value
        for path, value in flatten_gradient_tree(grads, "reliability_head").items()
        if len(path) >= 2 and path[-2] == "score" and path[-1] == leaf_name
    }
    return tree_l2_norm(selected)


def _parameter_delta_norm(new_params, old_params):
    return tree_l2_norm(subtract_gradient_trees(new_params, old_params))


def _make_loss_fn(batch: FixedBatch):
    config = batch.config

    def loss_fn(params):
        hidden, pi, value, _z_vision, aux_info = batch.train_state.apply_fn(
            params,
            batch.init_hstate,
            (batch.obs, batch.done),
        )
        del hidden, pi, value
        logits = aux_info["reliability_logits"]
        scores = aux_info["reliability_scores"]
        reliability_loss = masked_reliability_loss(
            scores,
            batch.labels,
            batch.mask,
            loss_type=config.get("reliability_loss_type", "bce"),
            eps=config.get("survival_eps", 1e-8),
            reliability_logits=logits,
        )
        aligned_scores = jnp.squeeze(scores, axis=-1) if scores.ndim == 5 else scores
        aligned_logits = jnp.squeeze(logits, axis=-1) if logits.ndim == 5 else logits
        total_loss = reliability_loss
        diag = {
            "score": aligned_scores,
            "logits": aligned_logits,
            "reliability_path_active": jnp.mean(
                aux_info["reliability_path_active"].astype(jnp.float32)
            ),
            "total_loss": total_loss,
            "reliability_loss": reliability_loss,
            "mae": _safe_mean(jnp.abs(aligned_scores - batch.labels), batch.mask),
            "corr": _safe_corr(aligned_scores, batch.labels, batch.mask),
            "score_mean": _safe_mean(aligned_scores, batch.mask),
            "target_mean": _safe_mean(batch.labels, batch.mask),
            "score_std": jnp.sqrt(
                _safe_mean(jnp.square(aligned_scores - _safe_mean(aligned_scores, batch.mask)), batch.mask)
            ),
            "target_std": jnp.sqrt(
                _safe_mean(jnp.square(batch.labels - _safe_mean(batch.labels, batch.mask)), batch.mask)
            ),
        }
        return total_loss, diag

    return loss_fn


def _raw_output_grad_norm(batch: FixedBatch, logits):
    loss_type = batch.config.get("reliability_loss_type", "bce")
    if loss_type == "bce":
        def output_loss(candidate_logits):
            return masked_reliability_loss(
                jax.nn.sigmoid(candidate_logits),
                batch.labels,
                batch.mask,
                loss_type="bce",
                eps=batch.config.get("survival_eps", 1e-8),
                reliability_logits=candidate_logits,
            )

        return tree_l2_norm(jax.grad(output_loss)(logits))

    def output_loss(candidate_scores):
        return masked_reliability_loss(
            candidate_scores,
            batch.labels,
            batch.mask,
            loss_type=loss_type,
            eps=batch.config.get("survival_eps", 1e-8),
        )

    return tree_l2_norm(jax.grad(output_loss)(jax.nn.sigmoid(logits)))


def _print_diagnostics(
    step,
    loss,
    diag,
    total_grads,
    batch: FixedBatch,
    parameter_update_norm,
):
    score = diag["score"]
    labels = batch.labels
    mask = batch.mask
    print(
        "OVERFIT_FORWARD",
        f"step={step}",
        f"reliability_path_active={_float(diag['reliability_path_active']):.6g}",
        f"reliability_scores_shape={tuple(score.shape)}",
        f"score_min={_float(jnp.min(score)):.6g}",
        f"score_max={_float(jnp.max(score)):.6g}",
        f"score_mean={_float(jnp.mean(score)):.6g}",
        f"score_std={_float(jnp.std(score)):.6g}",
        f"score_finite_rate={_float(jnp.mean(jnp.isfinite(score))):.6g}",
    )
    print(
        "OVERFIT_DIAG",
        f"step={step}",
        f"total_loss={_float(diag['total_loss']):.6g}",
        f"reliability_loss={_float(diag['reliability_loss']):.6g}",
        f"mae={_float(diag['mae']):.6g}",
        f"corr_score_target={_float(diag['corr']):.6g}",
        f"score_mean={_float(diag['score_mean']):.6g}",
        f"target_mean={_float(diag['target_mean']):.6g}",
        f"score_std={_float(diag['score_std']):.6g}",
        f"target_std={_float(diag['target_std']):.6g}",
    )

    groups = _group_masks(
        labels,
        mask,
        batch.is_sell_task,
    )
    for group_name, group_mask in groups.items():
        abs_error = jnp.abs(score - labels)
        print(
            "OVERFIT_GROUP",
            f"step={step}",
            f"group={group_name}",
            f"score_mean={_float(_safe_mean(score, group_mask)):.6g}",
            f"target_mean={_float(_safe_mean(labels, group_mask)):.6g}",
            f"abs_error={_float(_safe_mean(abs_error, group_mask)):.6g}",
        )

    print(
        "OVERFIT_LEVEL",
        f"step={step}",
        f"score_level_mean={_fmt(_level_means(score, mask))}",
        f"target_level_mean={_fmt(_level_means(labels, mask))}",
    )
    print(
        "OVERFIT_GRAD",
        f"step={step}",
        f"grad_norm_total={_float(tree_l2_norm(total_grads)):.6g}",
        f"grad_norm_vision_agent={_float(gradient_l2_norm(total_grads, 'vision_encoder')):.6g}",
        f"grad_norm_reliability_head={_float(gradient_l2_norm(total_grads, 'reliability_head')):.6g}",
        f"grad_norm_score_kernel={_float(_score_grad_norm(total_grads, 'kernel')):.6g}",
        f"grad_norm_score_bias={_float(_score_grad_norm(total_grads, 'bias')):.6g}",
        f"grad_norm_raw_reliability_output={_float(_raw_output_grad_norm(batch, diag['logits'])):.6g}",
        f"parameter_update_norm={_float(parameter_update_norm):.6g}",
    )


def _validate_initial_forward_and_gradients(diag, grads, parameter_update_norm):
    scores = np.asarray(jax.device_get(diag["score"]), dtype=np.float32)
    if not np.isfinite(scores).all():
        raise FloatingPointError("Reliability scores contain NaN or Inf at initialization.")
    if np.all(scores == 0.0):
        raise RuntimeError(
            "Reliability scores are all zero; ActorCriticRNN used the fallback path."
        )
    if np.all(scores == 1.0):
        raise RuntimeError("Reliability scores are all one at initialization.")
    if _float(diag["reliability_path_active"]) < 1.0:
        raise RuntimeError("Forward did not pass through ReliabilityFusionRNN.")

    total_grad_norm = _float(tree_l2_norm(grads))
    reliability_grad_norm = _float(
        gradient_l2_norm(grads, "reliability_head")
    )
    update_norm = _float(parameter_update_norm)
    if total_grad_norm <= 1e-12:
        raise RuntimeError(f"Total gradient is zero: {total_grad_norm}.")
    if reliability_grad_norm <= 1e-12:
        raise RuntimeError(
            f"Reliability Head gradient is zero: {reliability_grad_norm}."
        )
    if update_norm <= 1e-12:
        raise RuntimeError(f"Optimizer parameter update is zero: {update_norm}.")


def run_overfit(batch: FixedBatch):
    mode = batch.config.get("overfit_mode", "all_params")
    if mode != "all_params":
        raise NotImplementedError(
            "overfit_mode='reliability_head_only' is not implemented; "
            "use overfit_mode=all_params."
        )
    steps = int(batch.config.get("overfit_steps", 300))
    if steps < 1:
        raise ValueError("overfit_steps must be at least 1 for gradient validation.")
    lr = float(batch.config.get("overfit_lr", 1e-3))
    log_steps = {s for s in LOG_STEPS_DEFAULT if s <= steps}
    log_steps.add(steps)

    params = batch.train_state.params
    tx = optax.adam(lr)
    opt_state = tx.init(params)
    loss_fn = _make_loss_fn(batch)
    value_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    @jax.jit
    def update_step(params, opt_state):
        (loss, diag), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, diag, tree_l2_norm(updates)

    (initial_loss, initial_diag), initial_grads = value_and_grad_fn(params)
    updates, opt_state = tx.update(initial_grads, opt_state, params)
    updated_params = optax.apply_updates(params, updates)
    initial_update_norm = _parameter_delta_norm(updated_params, params)
    _print_diagnostics(
        0,
        initial_loss,
        initial_diag,
        initial_grads,
        batch,
        initial_update_norm,
    )
    _validate_initial_forward_and_gradients(
        initial_diag,
        initial_grads,
        initial_update_norm,
    )

    params = updated_params
    (loss, diag), total_grads = value_and_grad_fn(params)
    if abs(_float(loss) - _float(initial_loss)) <= 1e-12:
        raise RuntimeError("Loss did not change after the first optimizer update.")
    if 1 in log_steps:
        _print_diagnostics(
            1,
            loss,
            diag,
            total_grads,
            batch,
            initial_update_norm,
        )

    final = (loss, diag, total_grads)
    for step in range(2, steps + 1):
        params, opt_state, _pre_update_loss, _pre_update_diag, update_norm = update_step(
            params,
            opt_state,
        )
        if step in log_steps:
            (loss, diag), total_grads = value_and_grad_fn(params)
            _print_diagnostics(
                step,
                loss,
                diag,
                total_grads,
                batch,
                update_norm,
            )
            final = (loss, diag, total_grads)
    if steps >= 20 and _float(final[0]) >= _float(initial_loss):
        raise RuntimeError(
            "Fixed-batch overfit failed: final loss did not improve over initial loss "
            f"({_float(final[0]):.6g} >= {_float(initial_loss):.6g})."
        )
    return final


def main(argv: list[str] | None = None):
    argv = sys.argv[1:] if argv is None else argv
    config = _load_config(argv)
    if config.get("overfit_mode", "all_params") != "all_params":
        print("OVERFIT_MODE reliability_head_only implemented=false")
        raise SystemExit(2)

    print(
        "OVERFIT_CONFIG",
        "mode=all_params",
        f"steps={config['overfit_steps']}",
        f"lr={config['overfit_lr']}",
        f"num_envs={config['NUM_ENVS']}",
        f"num_steps={config['NUM_STEPS']}",
        f"survival_delta_steps={config.get('survival_delta_steps', 10)}",
        f"use_reliability_head={config.get('use_reliability_head')}",
        f"use_reliability_head_type={type(config.get('use_reliability_head')).__name__}",
        f"use_survival_loss={config.get('use_survival_loss')}",
        f"reliability_loss_type={config.get('reliability_loss_type')}",
        "book_source=fullbook_raw_orders",
        "target_semantics=same_side_price_execution_aware",
        "fullbook_match=same_side_absolute_price",
        "execution_source=transition_new_trades",
        f"use_h_prev_in_reliability={str(config.get('use_h_prev_in_reliability', True)).lower()}",
    )
    bundle = _collect_rollout(config)
    batch = _make_fixed_batch(bundle)
    print(
        "OVERFIT_PARAM_KEYS",
        "top_level=" + "[" + ",".join(_top_level_param_keys(batch.train_state.params)) + "]",
    )
    matching_paths = _matching_param_paths(batch.train_state.params)
    for path in matching_paths:
        print("OVERFIT_PARAM_PATH", f"path={path}")
    has_reliability_head = any(
        "LevelWiseReliabilityHead" in path
        or ("ReliabilityFusionRNN" in path and "/score/" in path)
        for path in matching_paths
    )
    if not has_reliability_head:
        raise RuntimeError(
            "Execution parameter tree does not contain LevelWiseReliabilityHead/score params."
        )
    run_overfit(batch)


if __name__ == "__main__":
    main()
