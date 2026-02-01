                                                                            """
Standalone environment-input scaffold.

This file is self-contained and does NOT modify the repo. It provides:
- A complete env-input bundle (env, env_params, rng, obs, state)
- A minimal rollout collector
- Clear examples showing where to plug your own loss

If the real environment is available in the workspace, it will use it.
Otherwise, it falls back to a tiny replica for testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp


# ----------------------------------------------------------------------------
# Try real environment first; fall back to a replica
# ----------------------------------------------------------------------------
try:
    from gymnax_exchange.jaxen.base_env import BaseLOBEnv as _RealBaseLOBEnv
    from gymnax_exchange.jaxob.jaxob_config import World_EnvironmentConfig as _RealWorldConfig
    REAL_ENV_AVAILABLE = True
except Exception:
    _RealBaseLOBEnv = None
    _RealWorldConfig = None
    REAL_ENV_AVAILABLE = False


@dataclass(frozen=True)
class WorldConfigReplica:
    """Replica of key fields needed for environment initialization.

    Use this only if the real config class is unavailable.

    Example:
        cfg = WorldConfigReplica(
            dataPath=r"F:/path/to/workspace",
            alphatradePath=r"F:/path/to/workspace",
            stock="AMZN",
            timePeriod="2012-06-21",
        )
    """

    dataPath: str
    alphatradePath: str
    stock: str = "AMZN"
    timePeriod: str = "2012-06-21"
    ep_type: str = "fixed_time"
    episode_time: int = 1800
    n_data_msg_per_step: int = 100
    day_start: int = 34200
    day_end: int = 57600
    start_resolution: int = 60
    book_depth: int = 10
    window_selector: int = -1
    use_pickles_for_init: bool = True


class BaseLOBEnvReplica:
    """Tiny stand-in environment (reset/step) for loss wiring tests.

    This is a functional replica with the same call surface as the real env.

    Example:
        env = BaseLOBEnvReplica(cfg, key)
        params = env.default_params
        obs, state = env.reset(key, params)
        obs, state, reward, done, info = env.step_env(key, state, {}, params)
    """

    def __init__(self, cfg: WorldConfigReplica, key: jax.Array):
        self.cfg = cfg
        self._rng = key
        self._t = 0

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"message_data": jnp.zeros((1, 8), dtype=jnp.int32)}

    def reset(self, key: jax.Array, params: Dict[str, Any]):
        self._t = 0
        obs = jnp.zeros((1,), dtype=jnp.float32)
        state = {"t": 0}
        return obs, state

    def step_env(self, key: jax.Array, state: Dict[str, Any], action: Dict[str, Any], params: Dict[str, Any]):
        self._t += 1
        obs = jnp.array([self._t], dtype=jnp.float32)
        reward = jnp.array(0.0, dtype=jnp.float32)
        done = self._t >= 4
        info = {"info": 0}
        return obs, {"t": self._t}, reward, done, info


def _resolve_workspace_root(start: Optional[Path] = None) -> Path:
    """Find the workspace root containing gymnax_exchange.

    Example:
        root = _resolve_workspace_root()
    """
    base = start or Path(__file__).resolve().parent
    for p in [base, *base.parents]:
        if (p / "gymnax_exchange").exists():
            return p
    return base


def make_world_config(*, workspace_root: Optional[Path] = None, **overrides: Any):
    """Create a complete config with correct dataPath/alphatradePath.

    Example:
        cfg = make_world_config(
            stock="AMZN",
            timePeriod="2012-06-21",
            ep_type="fixed_time",
            episode_time=1800,
            n_data_msg_per_step=100,
        )
    """
    root = _resolve_workspace_root(workspace_root)
    params = dict(
        dataPath=str(root),
        alphatradePath=str(root),
    )
    params.update(overrides)

    if REAL_ENV_AVAILABLE:
        return _RealWorldConfig(**params)
    return WorldConfigReplica(**params)


@dataclass
class EnvInput:
    """Bundle of objects required by a training step.

    Example:
        env_input = build_env_input(seed=0)
        env_input.obs  # current observation
    """
    env: Any
    env_params: Any
    rng: jax.Array
    obs: Any
    state: Any


def build_env_input(seed: int = 0, config: Optional[Any] = None) -> EnvInput:
    """Initialize env + params + first reset into an EnvInput bundle.

    Example:
        env_input = build_env_input(seed=0)
    """
    cfg = config or make_world_config()
    rng = jax.random.PRNGKey(seed)
    rng, key_init, key_reset = jax.random.split(rng, 3)

    env = _RealBaseLOBEnv(cfg=cfg, key=key_init) if REAL_ENV_AVAILABLE else BaseLOBEnvReplica(cfg, key_init)
    env_params = env.default_params
    obs, state = env.reset(key_reset, env_params)
    return EnvInput(env=env, env_params=env_params, rng=rng, obs=obs, state=state)


def collect_rollout(
    env_input: EnvInput,
    *,
    num_steps: int = 8,
    action_fn: Optional[Callable[[EnvInput], Dict[str, jnp.ndarray]]] = None,
) -> Dict[str, jnp.ndarray]:
    """Collect a short rollout for your loss function.

    Returns a dict of stacked arrays: obs, rewards, dones.

    Example:
        env_input = build_env_input(seed=0)
        batch = collect_rollout(env_input, num_steps=16)
        # Your loss function can use batch["obs"], batch["rewards"], batch["dones"]
    """
    obs_list = []
    reward_list = []
    done_list = []

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


if __name__ == "__main__":
    # Example usage (no loss function here):
    env_input = build_env_input(seed=0)
    rollout = collect_rollout(env_input, num_steps=4)
    print("Rollout shapes:", {k: v.shape for k, v in rollout.items()})
