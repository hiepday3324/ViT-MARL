"""Fixed-batch supervised overfit diagnostic for the Reliability Head.

This script collects one rollout batch, builds the same liquidity reliability
targets used by PPO training, then optimizes only the masked reliability loss on
that frozen batch. It does not save checkpoints and does not change default
training behavior.
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
from flax.traverse_util import flatten_dict
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
from gymnax_exchange.jaxrl.MARL.reliability_targets import (
    build_liquidity_survival_targets,
    masked_reliability_loss,
    resolve_rollout_is_sell_task,
)
from gymnax_exchange.networks.vision_agent import supervised_contrastive_loss


LOG_STEPS_DEFAULT = (0, 1, 5, 10, 25, 50, 100, 200, 300)
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
    vol_labels: jax.Array
    raw_future_vol: jax.Array
    future_vol_ticks: jax.Array
    is_sell_task: jax.Array
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
    merged = OmegaConf.merge(base_cfg, cli_cfg)
    env_cfg = OmegaConf.structured(
        MultiAgentConfig(number_of_agents_per_type=merged["NUM_AGENTS_PER_TYPE"])
    )
    merged = OmegaConf.merge(merged, env_cfg)
    config = OmegaConf.to_container(merged, resolve=True)

    config["WANDB_MODE"] = "disabled"
    config["CALC_EVAL"] = False
    config["use_reliability_head"] = _to_bool(config.get("use_reliability_head", True))
    config["use_survival_loss"] = _to_bool(config.get("use_survival_loss", True))
    config["use_supcon_loss"] = _to_bool(config.get("use_supcon_loss", False))
    config["use_h_prev_in_reliability"] = _to_bool(
        config.get("use_h_prev_in_reliability", True)
    )
    config["overfit_steps"] = int(config.get("overfit_steps", 300))
    config["overfit_lr"] = float(config.get("overfit_lr", 1e-3))
    config["overfit_mode"] = config.get("overfit_mode", "all_params")
    config["overfit_use_supcon"] = _to_bool(
        config.get("overfit_use_supcon", config.get("use_supcon_loss", False))
    )
    config["overfit_lambda_supcon"] = float(
        config.get("overfit_lambda_supcon", config.get("lambda_supcon", 0.0))
    )
    config["overfit_supcon_temperature"] = float(
        config.get("overfit_supcon_temperature", 0.1)
    )
    config["VOL_LOW"] = float(config.get("VOL_LOW", 1.0))
    config["VOL_HIGH"] = float(config.get("VOL_HIGH", 3.0))
    survival_delta_steps = max(1, int(config.get("survival_delta_steps", 10)))
    requested_vol_window = int(
        config.get(
            "overfit_vol_window",
            config.get("SUPCON_VOL_WINDOW", survival_delta_steps),
        )
    )
    config["overfit_vol_window"] = max(
        1,
        min(requested_vol_window, survival_delta_steps),
    )

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
        network = ActorCriticRNN(env.action_spaces[i], config=config)
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
        values = []
        log_probs = []
        for i, train_state in enumerate(train_states):
            obs_i = batchify(last_obs[i], config["NUM_ACTORS_PERTYPE"][i])
            obs_i_batched = jax.tree.map(lambda x: x[jnp.newaxis, :], obs_i)
            ac_in = (obs_i_batched, last_done[i][jnp.newaxis, :])
            hstates[i], pi, value, _, _ = train_state.apply_fn(
                train_state.params,
                hstates[i],
                ac_in,
            )
            rng, action_rng = jax.random.split(rng)
            action = pi.sample(seed=action_rng)
            values.append(value)
            log_probs.append(pi.log_prob(action))
            actions.append(
                unbatchify(
                    action,
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


def _broadcast_mid_prices_to_batch(mid_prices, *, batch_size):
    mid_prices = jnp.asarray(mid_prices, dtype=jnp.float32)
    while mid_prices.ndim > 2 and mid_prices.shape[-1] == 1:
        mid_prices = jnp.squeeze(mid_prices, axis=-1)
    if mid_prices.ndim == 1:
        mid_prices = mid_prices[:, None]
    if mid_prices.ndim != 2:
        raise ValueError(
            "mid_prices for volatility labels must be shaped (time,), "
            f"(time, batch), or (time, batch, 1); got {mid_prices.shape}."
        )
    if mid_prices.shape[1] != batch_size:
        if batch_size % mid_prices.shape[1] != 0:
            raise ValueError(
                "Cannot broadcast mid_prices to actor batch for volatility labels: "
                f"mid batch={mid_prices.shape[1]}, actor batch={batch_size}."
            )
        mid_prices = jnp.repeat(mid_prices, batch_size // mid_prices.shape[1], axis=1)
    return mid_prices


def _make_volatility_labels(
    traj: Transition,
    config: Dict[str, Any],
    *,
    tick_size,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    batch_size = traj.obs["vision_obs"].shape[1]
    window_size = int(config["overfit_vol_window"])
    required_steps = int(config["NUM_STEPS"]) + window_size
    tick_size = jnp.asarray(tick_size, dtype=jnp.float32)
    if float(jax.device_get(tick_size)) <= 0:
        raise ValueError(f"tick_size must be positive for volatility labels, got {tick_size}.")
    mid_prices = _broadcast_mid_prices_to_batch(
        traj.info["world"]["end_mid_price"],
        batch_size=batch_size,
    )
    if mid_prices.shape[0] < required_steps:
        raise ValueError(
            "end_mid_price must include enough padded future frames for SupCon "
            f"volatility labels: need {required_steps}, got {mid_prices.shape[0]}."
        )

    def calc_future_std(t):
        future_window = jax.lax.dynamic_slice_in_dim(
            mid_prices,
            t + 1,
            window_size,
            axis=0,
        )
        return jnp.std(future_window, axis=0)

    timesteps = jnp.arange(config["NUM_STEPS"])
    raw_future_vol = jax.vmap(calc_future_std)(timesteps)
    future_vol_ticks = raw_future_vol / tick_size
    vol_labels = jnp.where(
        future_vol_ticks > config["VOL_HIGH"],
        2,
        jnp.where(future_vol_ticks > config["VOL_LOW"], 1, 0),
    ).astype(jnp.int32)
    return vol_labels, raw_future_vol.astype(jnp.float32), future_vol_ticks.astype(jnp.float32)


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
    if not isinstance(traj.obs, dict) or "vision_obs" not in traj.obs:
        raise ValueError(
            "Execution observation is not dict/vision-enabled. "
            "Use execution_policy observation space."
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
    labels, mask = build_liquidity_survival_targets(
        traj.obs["vision_obs"],
        traj.info["world"].get("obs_mid_price", traj.info["world"]["end_mid_price"]),
        tick_size=tick_size,
        survival_delta_steps=survival_delta_steps,
        survival_min_volume=config.get("survival_min_volume", 1.0),
        survival_ratio=config.get("survival_ratio", 0.5),
        survival_availability_temperature=config.get(
            "survival_availability_temperature", 0.15
        ),
        ask_raw_orders=traj.info["world"].get("obs_ask_raw_orders", None),
        bid_raw_orders=traj.info["world"].get("obs_bid_raw_orders", None),
        num_steps=config["NUM_STEPS"],
        episode_done=traj.info["agent"]["done"],
    )
    vol_labels, raw_future_vol, future_vol_ticks = _make_volatility_labels(
        traj,
        config,
        tick_size=tick_size,
    )
    return FixedBatch(
        train_state=bundle.train_states[exec_idx],
        init_hstate=bundle.initial_hstates[exec_idx],
        obs=_slice_obs(traj.obs, config["NUM_STEPS"]),
        done=traj.done[: config["NUM_STEPS"]],
        labels=labels,
        mask=mask,
        vol_labels=vol_labels,
        raw_future_vol=raw_future_vol,
        future_vol_ticks=future_vol_ticks,
        is_sell_task=is_sell_task,
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


VISION_GRAD_FILTERS = ("VisionAgent", "vision_agent", "vision", "encoder")
RELIABILITY_GRAD_FILTERS = (
    "LevelWiseReliabilityHead",
    "ReliabilityHead",
    "reliability_head",
    "reliability",
)


def _tree_norm(tree):
    leaves = [x for x in jax.tree_util.tree_leaves(tree) if x is not None]
    if not leaves:
        return jnp.array(0.0, dtype=jnp.float32)
    return jnp.sqrt(sum(jnp.sum(jnp.square(jnp.asarray(x))) for x in leaves))


def _flatten_grad_subtree(grads, key_filter):
    filters = (key_filter,) if isinstance(key_filter, str) else tuple(key_filter)
    flat = flatten_dict(grads)
    return [
        value
        for key, value in flat.items()
        if any(token in "/".join(str(part) for part in key) for token in filters)
    ]


def _grad_norm(flat_grad):
    if not flat_grad:
        return jnp.array(0.0, dtype=jnp.float32)
    return _tree_norm(flat_grad)


def _grad_cosine(flat_grad_a, flat_grad_b, eps=1e-8):
    if not flat_grad_a or not flat_grad_b:
        return jnp.array(0.0, dtype=jnp.float32)
    if len(flat_grad_a) != len(flat_grad_b):
        return jnp.array(0.0, dtype=jnp.float32)
    dot = sum(
        jnp.sum(jnp.asarray(a) * jnp.asarray(b))
        for a, b in zip(flat_grad_a, flat_grad_b)
    )
    norm_a = _grad_norm(flat_grad_a)
    norm_b = _grad_norm(flat_grad_b)
    return dot / (norm_a * norm_b + eps)


def _filtered_grad_norm(grads, key_filter):
    selected = _flatten_grad_subtree(grads, key_filter)
    if not selected:
        return jnp.array(0.0, dtype=jnp.float32)
    return _grad_norm(selected)


def _top_level_param_keys(params):
    flat = flatten_dict(params)
    return sorted({str(key[0]) for key in flat.keys() if key})


def _make_loss_fns(batch: FixedBatch):
    config = batch.config
    overfit_use_supcon = bool(config.get("overfit_use_supcon", False))
    overfit_lambda_supcon = float(config.get("overfit_lambda_supcon", 0.0))
    overfit_supcon_temperature = float(config.get("overfit_supcon_temperature", 0.1))

    def _forward_losses(params):
        hidden, pi, value, z_vision, aux_info = batch.train_state.apply_fn(
            params,
            batch.init_hstate,
            (batch.obs, batch.done),
        )
        del hidden, pi, value
        scores = aux_info["reliability_scores"]
        reliability_loss = masked_reliability_loss(
            scores,
            batch.labels,
            batch.mask,
            loss_type=config.get("reliability_loss_type", "bce"),
            eps=config.get("survival_eps", 1e-8),
        )
        z_flat = z_vision.reshape(-1, z_vision.shape[-1])
        labels_flat = batch.vol_labels.reshape(-1)
        supcon_loss = supervised_contrastive_loss(
            z_flat,
            labels_flat,
            temperature=overfit_supcon_temperature,
        )
        aligned_scores = jnp.squeeze(scores, axis=-1) if scores.ndim == 5 else scores
        z_norm = jnp.linalg.norm(z_vision, axis=-1)
        return reliability_loss, supcon_loss, aligned_scores, z_norm

    def reliability_loss_only_fn(params):
        reliability_loss, _supcon_loss, _aligned_scores, _z_norm = _forward_losses(params)
        return reliability_loss

    def supcon_loss_only_fn(params):
        _reliability_loss, supcon_loss, _aligned_scores, _z_norm = _forward_losses(params)
        return supcon_loss

    def total_loss_fn(params):
        reliability_loss, supcon_loss, aligned_scores, z_norm = _forward_losses(params)
        weighted_supcon_loss = jnp.where(
            jnp.asarray(float(overfit_use_supcon), dtype=jnp.float32) > 0.0,
            overfit_lambda_supcon * supcon_loss,
            jnp.array(0.0, dtype=jnp.float32),
        )
        total_loss = reliability_loss + weighted_supcon_loss
        diag = {
            "score": aligned_scores,
            "total_loss": total_loss,
            "reliability_loss": reliability_loss,
            "supcon_loss": supcon_loss,
            "weighted_supcon_loss": weighted_supcon_loss,
            "lambda_supcon": jnp.asarray(overfit_lambda_supcon, dtype=jnp.float32),
            "use_supcon_loss": jnp.asarray(float(overfit_use_supcon), dtype=jnp.float32),
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
            "z_vision_norm_mean": jnp.mean(z_norm),
            "z_vision_norm_std": jnp.std(z_norm),
            "future_vol_ticks_mean": jnp.mean(batch.future_vol_ticks),
            "future_vol_ticks_std": jnp.std(batch.future_vol_ticks),
            "future_vol_ticks_min": jnp.min(batch.future_vol_ticks),
            "future_vol_ticks_max": jnp.max(batch.future_vol_ticks),
            "raw_future_vol_mean": jnp.mean(batch.raw_future_vol),
            "raw_future_vol_std": jnp.std(batch.raw_future_vol),
            "vol_label_count_0": jnp.sum(batch.vol_labels == 0),
            "vol_label_count_1": jnp.sum(batch.vol_labels == 1),
            "vol_label_count_2": jnp.sum(batch.vol_labels == 2),
        }
        return total_loss, diag

    return total_loss_fn, reliability_loss_only_fn, supcon_loss_only_fn


def _print_diagnostics(
    step,
    loss,
    diag,
    total_grads,
    rel_grads,
    supcon_grads,
    batch: FixedBatch,
):
    score = diag["score"]
    labels = batch.labels
    mask = batch.mask
    config = batch.config
    overfit_use_supcon = bool(config.get("overfit_use_supcon", False))
    overfit_lambda_supcon = float(config.get("overfit_lambda_supcon", 0.0))
    effective_lambda_supcon = overfit_lambda_supcon if overfit_use_supcon else 0.0
    print(
        "OVERFIT_DIAG",
        f"step={step}",
        f"total_loss={_float(diag['total_loss']):.6g}",
        f"reliability_loss={_float(diag['reliability_loss']):.6g}",
        f"supcon_loss={_float(diag['supcon_loss']):.6g}",
        f"weighted_supcon_loss={_float(diag['weighted_supcon_loss']):.6g}",
        f"lambda_supcon={_float(diag['lambda_supcon']):.6g}",
        f"use_supcon={str(bool(_float(diag['use_supcon_loss']))).lower()}",
        f"mae={_float(diag['mae']):.6g}",
        f"corr_score_target={_float(diag['corr']):.6g}",
        f"score_mean={_float(diag['score_mean']):.6g}",
        f"target_mean={_float(diag['target_mean']):.6g}",
        f"score_std={_float(diag['score_std']):.6g}",
        f"target_std={_float(diag['target_std']):.6g}",
        f"z_vision_norm_mean={_float(diag['z_vision_norm_mean']):.6g}",
        f"z_vision_norm_std={_float(diag['z_vision_norm_std']):.6g}",
        f"future_vol_ticks_mean={_float(diag['future_vol_ticks_mean']):.6g}",
        f"future_vol_ticks_std={_float(diag['future_vol_ticks_std']):.6g}",
        f"future_vol_ticks_min={_float(diag['future_vol_ticks_min']):.6g}",
        f"future_vol_ticks_max={_float(diag['future_vol_ticks_max']):.6g}",
        f"raw_future_vol_mean={_float(diag['raw_future_vol_mean']):.6g}",
        f"raw_future_vol_std={_float(diag['raw_future_vol_std']):.6g}",
        f"vol_label_count_0={int(_float(diag['vol_label_count_0']))}",
        f"vol_label_count_1={int(_float(diag['vol_label_count_1']))}",
        f"vol_label_count_2={int(_float(diag['vol_label_count_2']))}",
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
    total_vision = _flatten_grad_subtree(total_grads, VISION_GRAD_FILTERS)
    total_reliability = _flatten_grad_subtree(total_grads, RELIABILITY_GRAD_FILTERS)
    rel_vision = _flatten_grad_subtree(rel_grads, VISION_GRAD_FILTERS)
    supcon_vision = _flatten_grad_subtree(supcon_grads, VISION_GRAD_FILTERS)
    rel_reliability = _flatten_grad_subtree(rel_grads, RELIABILITY_GRAD_FILTERS)
    supcon_reliability = _flatten_grad_subtree(supcon_grads, RELIABILITY_GRAD_FILTERS)
    rel_vision_norm = _grad_norm(rel_vision)
    supcon_vision_norm = _grad_norm(supcon_vision)
    rel_reliability_norm = _grad_norm(rel_reliability)
    supcon_reliability_norm = _grad_norm(supcon_reliability)
    eps = jnp.array(1e-8, dtype=jnp.float32)

    print(
        "OVERFIT_GRAD",
        f"step={step}",
        f"overfit_use_supcon={str(overfit_use_supcon).lower()}",
        f"effective_lambda_supcon={effective_lambda_supcon:.6g}",
        f"grad_norm_total={_float(_tree_norm(total_grads)):.6g}",
        f"grad_norm_vision_agent={_float(_grad_norm(total_vision)):.6g}",
        f"grad_norm_reliability_head={_float(_grad_norm(total_reliability)):.6g}",
        f"grad_norm_rel_total={_float(_tree_norm(rel_grads)):.6g}",
        f"grad_norm_supcon_total={_float(_tree_norm(supcon_grads)):.6g}",
        f"grad_norm_rel_vision_agent={_float(rel_vision_norm):.6g}",
        f"grad_norm_supcon_vision_agent={_float(supcon_vision_norm):.6g}",
        f"grad_ratio_supcon_to_rel_vision_agent={_float(effective_lambda_supcon * supcon_vision_norm / (rel_vision_norm + eps)):.6g}",
        f"grad_cos_rel_supcon_vision_agent={_float(_grad_cosine(rel_vision, supcon_vision)):.6g}",
        f"grad_norm_rel_reliability_head={_float(rel_reliability_norm):.6g}",
        f"grad_norm_supcon_reliability_head={_float(supcon_reliability_norm):.6g}",
        f"grad_ratio_supcon_to_rel_reliability_head={_float(effective_lambda_supcon * supcon_reliability_norm / (rel_reliability_norm + eps)):.6g}",
        f"grad_cos_rel_supcon_reliability_head={_float(_grad_cosine(rel_reliability, supcon_reliability)):.6g}",
    )


def run_overfit(batch: FixedBatch):
    mode = batch.config.get("overfit_mode", "all_params")
    if mode != "all_params":
        raise NotImplementedError(
            "overfit_mode='reliability_head_only' is not implemented; "
            "use overfit_mode=all_params."
        )
    steps = int(batch.config.get("overfit_steps", 300))
    lr = float(batch.config.get("overfit_lr", 1e-3))
    log_steps = {s for s in LOG_STEPS_DEFAULT if s <= steps}
    log_steps.add(steps)

    params = batch.train_state.params
    tx = optax.adam(lr)
    opt_state = tx.init(params)
    total_loss_fn, reliability_loss_only_fn, supcon_loss_only_fn = _make_loss_fns(batch)
    total_value_and_grad_fn = jax.jit(jax.value_and_grad(total_loss_fn, has_aux=True))
    rel_grad_fn = jax.jit(jax.grad(reliability_loss_only_fn))
    supcon_grad_fn = jax.jit(jax.grad(supcon_loss_only_fn))

    @jax.jit
    def update_step(params, opt_state):
        (loss, diag), grads = jax.value_and_grad(total_loss_fn, has_aux=True)(params)
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, diag

    (loss, diag), total_grads = total_value_and_grad_fn(params)
    rel_grads = rel_grad_fn(params)
    supcon_grads = supcon_grad_fn(params)
    _print_diagnostics(0, loss, diag, total_grads, rel_grads, supcon_grads, batch)

    final = (loss, diag, total_grads)
    for step in range(1, steps + 1):
        params, opt_state, loss, diag = update_step(params, opt_state)
        if step in log_steps:
            (loss, diag), total_grads = total_value_and_grad_fn(params)
            rel_grads = rel_grad_fn(params)
            supcon_grads = supcon_grad_fn(params)
            _print_diagnostics(
                step,
                loss,
                diag,
                total_grads,
                rel_grads,
                supcon_grads,
                batch,
            )
            final = (loss, diag, total_grads)
    return final


def main(argv: list[str] | None = None):
    argv = sys.argv[1:] if argv is None else argv
    config = _load_config(argv)
    if config.get("overfit_mode", "all_params") != "all_params":
        print("OVERFIT_MODE reliability_head_only implemented=false")
        raise SystemExit(2)
    config_tick_size = config.get("tick_size", World_EnvironmentConfig().tick_size)

    print(
        "OVERFIT_CONFIG",
        "mode=all_params",
        f"steps={config['overfit_steps']}",
        f"lr={config['overfit_lr']}",
        f"num_envs={config['NUM_ENVS']}",
        f"num_steps={config['NUM_STEPS']}",
        f"survival_delta_steps={config.get('survival_delta_steps', 10)}",
        f"overfit_use_supcon={str(config.get('overfit_use_supcon', False)).lower()}",
        f"overfit_lambda_supcon={config.get('overfit_lambda_supcon', 0.0)}",
        f"overfit_supcon_temperature={config.get('overfit_supcon_temperature', 0.1)}",
        f"overfit_vol_window={config.get('overfit_vol_window', config.get('survival_delta_steps', 10))}",
        "supcon_vol_unit=ticks",
        f"VOL_LOW={config.get('VOL_LOW', 1.0)}",
        f"VOL_HIGH={config.get('VOL_HIGH', 3.0)}",
        f"tick_size={config_tick_size}",
        "book_source=fullbook_raw_orders",
        "fullbook_match=absolute_price_sum",
        f"use_h_prev_in_reliability={str(config.get('use_h_prev_in_reliability', True)).lower()}",
    )
    bundle = _collect_rollout(config)
    batch = _make_fixed_batch(bundle)
    print(
        "OVERFIT_PARAM_KEYS",
        "top_level=" + "[" + ",".join(_top_level_param_keys(batch.train_state.params)) + "]",
    )
    run_overfit(batch)


if __name__ == "__main__":
    main()
