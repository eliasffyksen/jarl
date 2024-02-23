import gymnasium as gym

import jax.numpy as jnp

class JaxEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        obs, info = self.env.reset()

        return jnp.array(obs), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        return \
            jnp.array(obs), \
            jnp.array(reward), \
            jnp.array(done), \
            jnp.array(truncated), \
            info
