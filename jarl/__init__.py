from typing import NamedTuple

import jax

class Transition(NamedTuple):
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    next_obs: jax.Array
    done: jax.Array

from .replay_buffer import ReplayBuffer
from .env_wrappers import JaxEnv

from . import loss
