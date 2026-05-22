import jax
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from gymnax_exchange.jaxob import JaxOrderBookArrays as job
from gymnax_exchange.jaxob.jaxob_config import JAXLOB_Configuration
from lobster_loader import LoadLOBSTER_resample

loader = LoadLOBSTER_resample(
    datapath="/home/doanduchieu001/GitHub/ViT-MARL",
    atpath="/home/doanduchieu001/GitHub/ViT-MARL",
    stock="AMZN",
    time_period="2012-06-21",
    n_Levels=10,
    n_data_msg_per_step=100,
    day_start=34200,
    day_end=57600,
    type_="fixed_time",
    window_length=1800,
    window_resolution=60
)
msgs, starts, ends, obs, max_msgs_arr = loader.run_loading()
cfg = JAXLOB_Configuration()

# JIT the scan_through_entire_array
jitted_scan = jax.jit(job.scan_through_entire_array, static_argnums=(0,))

import gc

for i in range(100):
   init_orders_i = job.init_msgs_from_l2(cfg, obs[i], time=jnp.array([34200 + i * 60, 0]))
   asks_i = job.init_orderside(cfg.nOrders)
   bids_i = job.init_orderside(cfg.nOrders)
   trades_i = (jnp.ones((cfg.nTrades, 8)) * -1).astype(jnp.int32)
   key = jax.random.PRNGKey(i)
   asks_i, bids_i, trades_i = jitted_scan(cfg, key, init_orders_i, (asks_i, bids_i, trades_i))
   asks_i.block_until_ready()
   print(f"[{i+1}/100] memory leak test")
   # Clear JAX cache to be safe? 

