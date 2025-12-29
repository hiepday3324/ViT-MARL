"""
State and Parameter Definitions for Multi-Agent Limit Order Book Trading

University of Oxford
Corresponding Author: 
Valentin Mohl (valentin.mohl@cs.ox.ac.uk)
Reuben Leyland (Reuben.leyland@sky.com)
Sascha Frey (sascha.frey@st-hughs.ox.ac.uk)


Module Description
This module defines all state and parameter classes used across the multi-agent 
reinforcement learning environment for limit order book trading. It provides 
structured data classes for managing environment states, agent states, and 
configuration parameters using JAX-compatible dataclasses.

Key Components
LoadedEnvState:    Base state class for the loaded environment, containing
                   raw order book data, trades, and timing information, which is loaded from the data files.
WorldState:        Extended state class with market information like best bids/asks,
                   mid price, and order ID counter etc.
MultiAgentState:   Combined state class managing the shared world state and
                   individual agent states for multi-agent coordination.
LoadedEnvParams:   Base parameters class for environment data and initialization.
MultiAgentParams:  Combined parameters class for multi-agent coordination.
MMEnvParams:       Market making and directional trading agent-specific parameters.
ExecEnvParams:     Execution agent-specific parameters.
MMEnvState:        Market making and directional trading agent state with inventory and position tracking.
ExecEnvState:      Execution agent state with task-specific information.

State Hierarchy
- LoadedEnvState: Base environment state
  - WorldState: Extended with market information
    - MultiAgentState: Combined with agent states
- MMEnvState: Market making agent state
- ExecEnvState: Execution agent state

Parameter Hierarchy  
- LoadedEnvParams: Base environment parameters
  - MultiAgentParams: Combined with agent parameters
- MMEnvParams: Market making agent parameters
- ExecEnvParams: Execution agent parameters
"""

import jax.numpy as jnp
from flax import struct
from typing import Any
import chex



########################################################################################
########################################################################################
# States
########################################################################################
########################################################################################

@struct.dataclass
class LoadedEnvState:
    ask_raw_orders: chex.Array
    bid_raw_orders: chex.Array
    trades: chex.Array
    init_time: chex.Array
    window_index:int
    max_steps_in_episode: int
    start_index: int # This should be here because its the same for all agents, but it changes for all agents when resetting (this is why its not in Params)
    step_counter: int
    


@struct.dataclass
class WorldState(LoadedEnvState):
    # But everything here that is not loaded from the base config but shared by all agents
    best_bids: jnp.ndarray
    best_asks: jnp.ndarray
    time: chex.Array
    order_id_counter: int
    mid_price:float
    delta_time: float


# Define a combined (multi–agent) state that extends the base order book state
@struct.dataclass
class MultiAgentState():
    # Sub–state for market maker and execution agent.
    world_state: WorldState

    agent_states: list[Any]



@struct.dataclass
class MMEnvState():
    inventory: int
    total_PnL: float
    cash_balance: float


@struct.dataclass
class ExecEnvState():
    init_price: int
    task_to_execute: int
    quant_executed: int
    # rewards
    total_revenue: float
    drift_return: float
    advantage_return: float
    slippage_rm: float
    price_adv_rm: float
    price_drift_rm: float
    vwap_rm: float
    is_sell_task: int
    trade_duration: float






########################################################################################
########################################################################################
# Params
########################################################################################
########################################################################################

@struct.dataclass
class LoadedEnvParams:
    message_data: chex.Array
    book_data: chex.Array
    init_states_array: chex.Array



# Define a combined parameters class.
# Logic: All the data is in BaseParams. All the things that depend on all agents are added to it (e.g. num_msgs_per_step). The rest stays in the config
@struct.dataclass
class MultiAgentParams():
    loaded_params: LoadedEnvParams

    # Put everything here that is shared by all agents, and will be determined by the world config (if its hard encoded but it in world config)
    # However put static things on the self if they have to be calculated (num_msg_per_step) or in the config if not
    #num_msgs_per_step: int

    agent_params: list[Any] # List of either MMEnvParams or ExecEnvParams


@struct.dataclass
class MMEnvParams():
    trader_id: chex.Array
    time_delay_obs_act: chex.Array
    normalize: chex.Array



@struct.dataclass
class ExecEnvParams():
    trader_id: chex.Array
    task_size: chex.Array 
    reward_lambda: chex.Array
    time_delay_obs_act: chex.Array
    normalize: chex.Array


