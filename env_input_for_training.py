"""
Builds a complete, ready-to-use BaseLOBEnv input bundle for training.

This file focuses on:
- Correct World_EnvironmentConfig (paths + episode settings)
- Environment initialization
- A simple rollout collector
- A stub loss hook you can replace with your real loss
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp

from gymnax_exchange.jaxen.base_env import BaseLOBEnv
from gymnax_exchange.jaxob.jaxob_config import World_EnvironmentConfig


@dataclass
class EnvInput:
    """Container for all inputs typically needed in a training step."""

    env: BaseLOBEnv
    env_params: Any
    rng: jax.Array
    obs: Any
    state: Any


def _resolve_workspace_root(start: Optional[Path] = None) -> Path:
    """Find repo root (folder containing gymnax_exchange)."""
    base = start or Path(__file__).resolve().parent
    for p in [base, *base.parents]:
        if (p / "gymnax_exchange").exists():
            return p
    return base


def make_world_config(
    *,
    workspace_root: Optional[Path] = None,
    stock: str = "AMZN",
    time_period: str = "2012-06-21",
    ep_type: str = "fixed_time",
    episode_time: int = 1800,
    n_data_msg_per_step: int = 100,
    day_start: int = 34200,
    day_end: int = 57600,
    start_resolution: int = 60,
    book_depth: int = 10,
    window_selector: int = -1,
    use_pickles_for_init: bool = True,
    **overrides: Any,
) -> World_EnvironmentConfig:
    """Create a fully specified World_EnvironmentConfig.

    IMPORTANT: dataPath and alphatradePath must point to your workspace root.
    The loader will look for: <dataPath>/<stock>/<time_period>/
    """

    root = _resolve_workspace_root(workspace_root)
    params = dict(
        dataPath=str(root),
        alphatradePath=str(root),
        stock=stock,
        timePeriod=time_period,
        ep_type=ep_type,
        episode_time=episode_time,
        n_data_msg_per_step=n_data_msg_per_step,
        day_start=day_start,
        day_end=day_end,
        start_resolution=start_resolution,
        book_depth=book_depth,
        window_selector=window_selector,
        use_pickles_for_init=use_pickles_for_init,
    )
    params.update(overrides)
    return World_EnvironmentConfig(**params)


def build_env_input(
    *,
    seed: int = 0,
    config: Optional[World_EnvironmentConfig] = None,
) -> EnvInput:
    """Initialize BaseLOBEnv + default params + a reset state.

    Returns a ready-to-use EnvInput object you can pass into training.
    """

    cfg = config or make_world_config()

    rng = jax.random.PRNGKey(seed)
    rng, key_init, key_reset = jax.random.split(rng, 3)

    env = BaseLOBEnv(cfg=cfg, key=key_init)
    env_params = env.default_params
    obs, state = env.reset(key_reset, env_params)

    return EnvInput(env=env, env_params=env_params, rng=rng, obs=obs, state=state)


def step_env_input(
    env_input: EnvInput,
    *,
    action: Optional[Dict[str, jnp.ndarray]] = None,
) -> EnvInput:
    """Advance the environment by one step and return updated EnvInput."""

    action = action or {}
    rng, key_step = jax.random.split(env_input.rng, 2)
    obs, state, reward, done, info = env_input.env.step_env(
        key_step, env_input.state, action, env_input.env_params
    )

    if done:
        rng, key_reset = jax.random.split(rng, 2)
        obs, state = env_input.env.reset(key_reset, env_input.env_params)

    return EnvInput(
        env=env_input.env,
        env_params=env_input.env_params,
        rng=rng,
        obs=obs,
        state=state,
    )


def collect_rollout(
    env_input: EnvInput,
    *,
    num_steps: int = 8,
    action_fn: Optional[Callable[[EnvInput], Dict[str, jnp.ndarray]]] = None,
) -> Dict[str, jnp.ndarray]:
    """Collect a short rollout for loss computation.

    Returns a dict with stacked arrays: obs, rewards, dones.
    """

    obs_list: List[jnp.ndarray] = []
    reward_list: List[jnp.ndarray] = []
    done_list: List[jnp.ndarray] = []

    current = env_input
    for _ in range(num_steps):
        action = action_fn(current) if action_fn else {}
        rng, key_step = jax.random.split(current.rng, 2)
        obs, state, reward, done, _info = current.env.step_env(
            key_step, current.state, action, current.env_params
        )

        obs_list.append(jnp.asarray(obs))
        reward_list.append(jnp.asarray(reward))
        done_list.append(jnp.asarray(done))

        if done:
            rng, key_reset = jax.random.split(rng, 2)
            obs, state = current.env.reset(key_reset, current.env_params)

        current = EnvInput(
            env=current.env,
            env_params=current.env_params,
            rng=rng,
            obs=obs,
            state=state,
        )

    return {
        "obs": jnp.stack(obs_list),
        "rewards": jnp.stack(reward_list),
        "dones": jnp.stack(done_list),
    }


def dummy_loss(rollout: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """A placeholder loss to plug into your training loop.

    Replace with your real loss function.
    """

    # Example: maximize mean reward (so loss is negative reward)
    return -jnp.mean(rollout["rewards"])


if __name__ == "__main__":
    # Quick sanity check (no training): build input + collect a rollout
    env_input = build_env_input(seed=0)
    rollout = collect_rollout(env_input, num_steps=4)
    loss_val = dummy_loss(rollout)
    print("Rollout shapes:", {k: v.shape for k, v in rollout.items()})
    print("Dummy loss:", loss_val)
