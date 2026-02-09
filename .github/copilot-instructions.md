# ViT-MARL Copilot Instructions

## Architecture

```
jaxrl/ (Training)          → MARLEnv, Hydra YAML configs, WandB, PPO+GRU
  jaxen/ (Environments)    → MARLEnv orchestrates BaseLOBEnv + agent list
    jaxob/ (Order Book)    → Pure JAX functions (no classes), aliased as `job`
  networks/                → Flax linen modules (VisionAgent)
  jaxlobster/              → LOBSTER CSV loading (LoadLOBSTER_resample)
```

- **`jaxob/JaxOrderBookArrays.py`** — All orderbook logic as pure `@jax.jit` functions. Always imported as `job`.
- **`jaxen/vision_env.py`** — `ExecutionAgent` class (misnamed; "vision" = observation type, not a separate env).
- **`jaxen/marl_env.py`** — `MARLEnv` orchestrates world-level OB + multiple agent types.
- **`jaxen/base_env.py`** — `BaseLOBEnv` manages orderbook state, data replay, and resets.

## Config System (Critical)

```
JAXLOB_Configuration          ← base: maxint, nOrders, seed, paths
  └─ World_EnvironmentConfig   ← extends: tick_size, episode_time, book_depth
Execution_EnvironmentConfig    ← standalone: action_space, task_size (NO maxint!)
```

`ExecutionAgent` stores **two** configs: `self.cfg` (Execution) and `self.world_config` (World).
- Pass `self.world_config` to any `job.*` function — they need `JAXLOB_Configuration` fields like `maxint`, `tick_size`.
- Pass `self.cfg` only for agent-specific settings (action_space, task_size, reward_lambda).
- All configs are `@dataclass(frozen=True)`. Use `__post_init__` with `object.__setattr__` for derived fields.

## Data Flow: CSV → Vision Features

1. **CSV loading**: `LoadLOBSTER_resample(datapath, atpath, stock, time_period, ...)` → `.run_loading()` returns `(msgs, starts, ends, obs, max_msgs_arr)`. Caches to `saved_npz/`.
2. **OB init**: `job.init_msgs_from_l2(cfg, obs[i], time)` → `job.scan_through_entire_array(cfg, key, init_orders, (asks, bids, trades))`
3. **Raw vision**: `job.get_vision_L2_state(asks, bids, 10, cfg)` → shape `(10, 2, 2)` = `[levels, (price/vol), (ask/bid)]`
4. **Normalized**: `agent.normalize_vision_obs(raw, world_state)` → shape `(10, 3, 2)` = `[levels, (gap/logvol/cumvol), (ask/bid)]`

## JAX Patterns

- **State is immutable**: All states use `@flax.struct.dataclass`. Update via `state.replace(field=new_val)`.
- **`jax.lax.scan`** for: message processing through OB, RL trajectory collection, GAE.
- **`jax.vmap`** for: multi-agent parallelism, volume aggregation at price levels.
- **`jax.lax.cond`** for: buy/sell branching, order type dispatch.
- **`jax.debug.print`** for debugging inside JIT (regular `print` won't work).
- **`@partial(jax.jit, static_argnums=(2, 3))`** on `get_vision_L2_state` — `n_levels` and `cfg` are static.

## Domain Conventions

- **Prices in hundredths of cents**: `tick_size=100` means 1-cent ticks. Mid price ~2238500 = $223.85.
- **Order IDs count downward** from negative start (e.g., `-200`). Positive IDs = historical data.
- **Trader IDs are negative**: Start from `trader_id_range_start=-100`, decrement per agent. `-1` is reserved.
- **Message format**: 8 features `[Type, Side, Quant, Price, OID, TID, Time_s, Time_ns]`.
- **Cancel-before-action**: Combined messages = `cancel_msgs + action_msgs` (cancels process first).

## Code Conventions

- Comments are bilingual: English docstrings, Vietnamese inline comments (especially vision pipeline).
- Duplicate files exist at root (`lobster_loader.py`, `data_loading.py`) and in `gymnax_exchange/jaxlobster/` — package version is canonical for env imports; root version works for standalone notebooks.
- Agent files use bracket naming: `[marketmaker]mm_env.py`.
- Hardcoded Windows paths in `jaxob_config.py` are marked `#FIXME`.
- `[marketmaker]mm_env.py` forces `jax.config.update('jax_platform_name', 'cpu')` — beware if importing alongside GPU code.

## Training Workflow

```bash
python gymnax_exchange/jaxrl/MARL/ippo_rnn_JAXMARL.py  # Hydra config from config/
```

- Hydra YAML in `jaxrl/MARL/config/` → builds `MultiAgentConfig`.
- One `ActorCriticRNN` (GRU) per agent **type**, vmapped across instances.
- Per-type hyperparams: `LR`, `GAMMA`, `ENT_COEF` etc. are lists indexed by agent type.
