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
shapes = set()
for i in range(10):
    init_orders_i = job.init_msgs_from_l2(cfg, obs[i], time=jnp.array([34200 + i * 60, 0]))
    shapes.add(init_orders_i.shape)
print("Shapes of init_orders_i:", shapes)
