import jax
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from gymnax_exchange.jaxob import JaxOrderBookArrays as job
from gymnax_exchange.jaxob.jaxob_config import JAXLOB_Configuration
from lobster_loader import LoadLOBSTER_resample

# 1. Load CSV files
loader = LoadLOBSTER_resample(
    datapath="/home/doanduchieu001/GitHub/ViT-MARL",
    atpath="/home/doanduchieu001/GitHub/ViT-MARL",  # Fixed: was 'alphatradepath'
    stock="AMZN",
    time_period="2012-06-21",  # Fixed: was 'timePeriod'
    n_Levels=10,
    n_data_msg_per_step=100,
    day_start=34200,
    day_end=57600,
    type_="fixed_time",
    window_length=1800,
    window_resolution=60
)

msgs, starts, ends, obs, max_msgs_arr = loader.run_loading()

# 2. Initialize orderbook from first window
cfg = JAXLOB_Configuration()
key = jax.random.PRNGKey(0)

init_orders = job.init_msgs_from_l2(cfg, obs[0], time=jnp.array([34200, 0]))
asks = job.init_orderside(cfg.nOrders)
bids = job.init_orderside(cfg.nOrders)
trades = (jnp.ones((cfg.nTrades, 8)) * -1).astype(jnp.int32)

asks, bids, trades = job.scan_through_entire_array(
    cfg, key, init_orders, (asks, bids, trades)
)

# 3. Process some messages to update orderbook dynamically
start_idx = starts[0]
batch_messages = msgs[start_idx:start_idx+100]  # First 100 messages

asks, bids, trades = job.scan_through_entire_array(
    cfg, key, batch_messages, (asks, bids, trades)
)

print("✓ Orderbook loaded and processed from CSV")
print(f"  Ask orders shape: {asks.shape}")
print(f"  Bid orders shape: {bids.shape}")






# Create ExecutionAgent and WorldState (your central data structure)
from gymnax_exchange.jaxen.vision_env import ExecutionAgent
from gymnax_exchange.jaxob.jaxob_config import Execution_EnvironmentConfig, World_EnvironmentConfig
from gymnax_exchange.jaxen.StatesandParams import WorldState

# Initialize configs
exec_config = Execution_EnvironmentConfig(
    action_space="simplest_case",
    observation_space="execution_policy"
)
world_config = World_EnvironmentConfig()

# Create ExecutionAgent instance
agent = ExecutionAgent(cfg=exec_config, world_config=world_config)

# ----- CREATE WORLD_STATE (Your Central Data Structure) -----
# Calculate best bid/ask from raw orderbook
best_bid, best_ask = job.get_best_bid_and_ask_inclQuants(cfg, asks, bids)
mid_price = (best_ask[0] + best_bid[0]) / 2

world_state = WorldState(
    ask_raw_orders=asks,
    bid_raw_orders=bids,
    trades=trades,
    init_time=jnp.array([34200, 0]),
    window_index=0,
    max_steps_in_episode=100,
    start_index=0,
    step_counter=0,
    best_bids=jnp.array([best_bid]),
    best_asks=jnp.array([best_ask]),
    time=jnp.array([34200, 0]),
    order_id_counter=0,
    mid_price=mid_price,
    delta_time=0.0
)

print("\n✓ WorldState created (your central data structure)")
print(f"  Mid Price: {mid_price:.2f}")
print(f"  Best Ask: {best_ask[0]}, Vol: {best_ask[1]}")
print(f"  Best Bid: {best_bid[0]}, Vol: {best_bid[1]}")

# ----- USE BUILTIN _get_obs_vision -----
vision_obs_normalized = agent._get_obs_vision(world_state, normalize=True)

print(f"\n✓ Vision features extracted using builtin _get_obs_vision")
print(f"  Normalized tensor shape: {vision_obs_normalized.shape}  # (10, 3, 2)")
print(f"  Features: [Gap, LogVol, CumVol] x [Ask, Bid]")
print(f"\nAsk side (top 3 levels):\n{vision_obs_normalized[:3, :, 0]}")
print(f"\nBid side (top 3 levels):\n{vision_obs_normalized[:3, :, 1]}")




# Build vision features for ALL windows (expected shape: [n_windows, 10, 3, 2])
from gymnax_exchange.jaxen.StatesandParams import WorldState

n_windows_total = min(len(starts), len(ends), len(obs))
max_windows =n_windows_total # set to n_windows_total for full 390 windows
n_windows = min(n_windows_total, max_windows)
print(f"Building {n_windows} / {n_windows_total} windows...")

vision_obs_all = []
key_all = jax.random.PRNGKey(0)

# Keep these consistent with loader settings
_day_start = 34200
_window_resolution = 60

# Toggle heavy processing of full window messages (slow)
process_window_msgs = False
max_msgs_per_window = 100  # only used if process_window_msgs=True

# JAX_TRACEBACK_FILTERING=off
hehe=0

for i in range(n_windows):
    key_all, subkey = jax.random.split(key_all)

    # Init orderbook from window i snapshot
    init_orders_i = job.init_msgs_from_l2(cfg, obs[i], time=jnp.array([_day_start + i * _window_resolution, 0]))
    asks_i = job.init_orderside(cfg.nOrders)
    bids_i = job.init_orderside(cfg.nOrders)
    trades_i = (jnp.ones((cfg.nTrades, 8)) * -1).astype(jnp.int32)

    asks_i, bids_i, trades_i = job.scan_through_entire_array(
        cfg, subkey, init_orders_i, (asks_i, bids_i, trades_i)
    )

    # Optionally process messages in this window
    if process_window_msgs:
        start_i = int(starts[i])
        end_i = int(ends[i])
        if end_i > start_i:
            key_all, subkey = jax.random.split(key_all)
            
            # --- PAD WINDOW MSGS TO A FIXED SHAPE ---
            actual_len = min(end_i - start_i, max_msgs_per_window if max_msgs_per_window else 1000)
            raw_slice = msgs[start_i:start_i + actual_len]
            
            # Chêm (Pad) array để luôn có độ dài cố định là max_msgs_per_window
            pad_len = max_msgs_per_window - actual_len
            if pad_len > 0:
                # Tạo dummy array có cùng số cột (thường là 8 cột) filled với 0
                dummy_padding = jnp.zeros((pad_len, raw_slice.shape[1]), dtype=raw_slice.dtype)
                window_msgs = jnp.concatenate([raw_slice, dummy_padding], axis=0)
            else:
                window_msgs = raw_slice
                
            asks_i, bids_i, trades_i = job.scan_through_entire_array(
                cfg, subkey, window_msgs, (asks_i, bids_i, trades_i)
            )

    # Build WorldState for this window
    best_bid_i, best_ask_i = job.get_best_bid_and_ask_inclQuants(cfg, asks_i, bids_i)
    mid_price_i = (best_ask_i[0] + best_bid_i[0]) / 2

    world_state_i = WorldState(
        ask_raw_orders=asks_i,
        bid_raw_orders=bids_i,
        trades=trades_i,
        init_time=jnp.array([_day_start + i * _window_resolution, 0]),
        window_index=i,
        max_steps_in_episode=100,
        start_index=int(starts[i]),
        step_counter=0,
        best_bids=jnp.array([best_bid_i]),
        best_asks=jnp.array([best_ask_i]),
        time=jnp.array([_day_start + i * _window_resolution, 0]),
        order_id_counter=0,
        mid_price=mid_price_i,
        delta_time=0.0
    )

    # Lấy đặc trưng (features) và DÙNG .block_until_ready() ĐỂ ÉP JAX THỰC THI NGAY
    # Điều này sẽ dọn sạch memory queue của JAX sau mỗi vòng lặp
    obs_i = agent._get_obs_vision(world_state_i, normalize=True)
    obs_i = obs_i.block_until_ready()
    vision_obs_all.append(obs_i)
    
    hehe = hehe+1
    print(f"[{hehe}/{n_windows}] Đã xử lý xong window, dọn dẹp bộ nhớ.")

vision_obs_all = jnp.stack(vision_obs_all, axis=0)
print(f"✓ vision_obs_all shape: {vision_obs_all.shape} (expected ~{n_windows}, 10, 3, 2)")